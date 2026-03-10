"""
Node Evidence Matcher

Maps ingested documents to causal graph nodes using semantic similarity.

Flow:
  1. Load causal nodes for an event from the DB.
  2. Load documents for the event (from documents table or JSON files).
  3. Embed node descriptions + document summaries with sentence-transformers.
  4. Cosine similarity matrix → assign each doc to its best-matching node.
  5. Write mapped_node_id + similarity_score back to the documents table.

Usage:
    matcher = NodeEvidenceMatcher.from_env()
    stats = matcher.match_event("gemini-3pt5-release-2026", graph_id=uuid)
"""

import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

_DEFAULT_DB_URL = "postgresql://causal:causal@localhost:5432/causal_interface"
_SIMILARITY_THRESHOLD = 0.40    # Minimum cosine similarity to assign a doc to a node
_MAX_DOCS_PER_NODE = 10         # Hard cap: keep only best-matching docs
_EMBED_MODEL = "all-MiniLM-L6-v2"  # Already in conda env via sentence-transformers


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class NodeRecord:
    node_id: str
    label: str
    event_type: str
    description: str      # from raw_data_json.description
    query: str = ""       # pre-built query string for embedding


@dataclass
class DocumentRecord:
    doc_id: str
    title: Optional[str]
    url: str
    source: str
    summary: Optional[str]
    raw_text: Optional[str]
    published_at: Optional[str]
    db_id: Optional[str] = None         # UUID primary key in documents table


@dataclass
class MatchResult:
    doc_id: str
    node_id: str
    similarity: float


# ─── Embedder ─────────────────────────────────────────────────────────────────

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading sentence-transformer: {_EMBED_MODEL}")
        _encoder = SentenceTransformer(_EMBED_MODEL)
        logger.info("Encoder loaded")
    return _encoder


def _embed(texts: List[str]) -> np.ndarray:
    """Embed a list of strings → (N, D) float32 array, L2-normalised."""
    enc = _get_encoder()
    vecs = enc.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return np.array(vecs, dtype=np.float32)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between every row in a and every row in b.
    Assumes both are already L2-normalised. Returns (len(a), len(b)) matrix."""
    return a @ b.T


# ─── DB helpers ───────────────────────────────────────────────────────────────

class _DB:
    def __init__(self, db_url: str):
        import psycopg2
        from psycopg2.extras import RealDictCursor
        self._conn = psycopg2.connect(db_url)
        self._RealDictCursor = RealDictCursor

    def fetch_nodes(self, graph_id: str) -> List[NodeRecord]:
        """Load causal nodes for a graph."""
        sql = """
            SELECT node_id, label, event_type, raw_data_json
            FROM causal_nodes
            WHERE graph_id = %s
        """
        with self._conn.cursor(cursor_factory=self._RealDictCursor) as cur:
            cur.execute(sql, (graph_id,))
            rows = cur.fetchall()

        nodes = []
        for row in rows:
            raw = row["raw_data_json"] or {}
            desc = raw.get("description", "")
            nodes.append(NodeRecord(
                node_id=row["node_id"],
                label=row["label"] or "",
                event_type=row["event_type"] or "",
                description=desc,
            ))
        return nodes

    def fetch_documents(self, event_id: str) -> List[DocumentRecord]:
        """Load documents (feed items) for an event from the feed_items table."""
        sql = """
            SELECT id, headline as title, url, source, summary, raw_text, timestamp as published_at
            FROM feed_items
            WHERE event_id = %s AND mapped_node_id IS NULL AND headline IS NOT NULL
        """
        with self._conn.cursor(cursor_factory=self._RealDictCursor) as cur:
            cur.execute(sql, (event_id,))
            rows = cur.fetchall()

        return [
            DocumentRecord(
                doc_id=str(row["id"]),
                title=row["title"],
                url=row["url"],
                source=row["source"],
                summary=row["summary"],
                raw_text=row["raw_text"],
                published_at=str(row["published_at"]) if row["published_at"] else None,
                db_id=str(row["id"]),
            )
            for row in rows
        ]

    def write_mappings(self, matches: List[MatchResult], graph_id: Optional[str] = None) -> int:
        """Update mapped_node_id on feed_items AND evidence_count on causal_nodes."""
        if not matches:
            return 0
        sql = """
            UPDATE feed_items
            SET mapped_node_id = %s
            WHERE id = %s::uuid
        """
        with self._conn.cursor() as cur:
            cur.executemany(sql, [
                (m.node_id, m.doc_id) for m in matches
            ])

            # Update evidence_count on causal_nodes
            if graph_id:
                # Count how many feed_items point to each node_id
                from collections import Counter
                node_counts = Counter(m.node_id for m in matches)
                for node_id, count in node_counts.items():
                    cur.execute("""
                        UPDATE causal_nodes
                        SET evidence_count = %s
                        WHERE graph_id = %s AND node_id = %s
                    """, (count, graph_id, node_id))

        self._conn.commit()
        return len(matches)

    def upsert_document(self, doc: DocumentRecord, event_id: str, market_id: Optional[str]) -> None:
        """Insert or update a document record (used when syncing from JSON files)."""
        sql = """
            INSERT INTO documents
                (doc_id, event_id, market_id, source, url, title, summary, raw_text, published_at, content_hash)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET
                summary = EXCLUDED.summary,
                mapped_node_id = documents.mapped_node_id  -- preserve existing mapping
        """
        import hashlib
        content_hash = hashlib.sha256(
            ((doc.title or "") + "|" + (doc.raw_text or "")[:200]).encode()
        ).hexdigest()[:64]
        with self._conn.cursor() as cur:
            cur.execute(sql, (
                doc.doc_id, event_id, market_id,
                doc.source, doc.url, doc.title,
                doc.summary, doc.raw_text,
                doc.published_at, content_hash
            ))
        self._conn.commit()

    def close(self):
        self._conn.close()


# ─── Artifact-based document loader ──────────────────────────────────────────

def _load_docs_from_artifacts(event_id: str) -> List[DocumentRecord]:
    """
    Load ingested documents from the PostgreSQL pipeline_artifacts table.
    Used to bootstrap the backend documents DB table from the pipeline stage.
    """
    from utils.db_storage import load_artifact
    
    docs = []
    artifacts = load_artifact(event_id, "normalized_documents")
    if not artifacts:
        return docs

    for d in artifacts:
        try:
            docs.append(DocumentRecord(
                doc_id=d["doc_id"],
                title=d.get("title"),
                url=d.get("url", ""),
                source=d.get("source", "web"),
                summary=None,       # Will be generated
                raw_text=d.get("raw_text"),
                published_at=d.get("timestamp"),
            ))
        except Exception as e:
            logger.debug(f"Skipping artifact doc {d.get('doc_id')}: {e}")

    return docs


# ─── Main matcher ─────────────────────────────────────────────────────────────

class NodeEvidenceMatcher:
    """
    Matches ingested documents to causal graph nodes via sentence-transformer embeddings.

    Steps:
      1. Build query strings for each node: "{label}: {description}"
      2. Build content strings for each doc: summary (or truncated raw_text)
      3. Compute cosine similarity matrix
      4. Assign each document to its best-matching node (if above threshold)
      5. Cap each node at MAX_DOCS_PER_NODE (keep highest similarity)
      6. Write mappings to the documents table
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        similarity_threshold: float = _SIMILARITY_THRESHOLD,
        max_docs_per_node: int = _MAX_DOCS_PER_NODE,
    ):
        self.db_url = db_url or os.getenv("CAUSAL_DB_URL", _DEFAULT_DB_URL)
        self.threshold = similarity_threshold
        self.max_docs_per_node = max_docs_per_node

    @classmethod
    def from_env(cls, **kwargs) -> "NodeEvidenceMatcher":
        return cls(db_url=os.getenv("CAUSAL_DB_URL", _DEFAULT_DB_URL), **kwargs)

    def match(
        self,
        node_queries: List[str],
        doc_texts: List[str],
    ) -> List[Tuple[int, int, float]]:
        """
        Low-level similarity match.

        Args:
            node_queries: List of node query strings (label: description).
            doc_texts: List of document content strings.

        Returns:
            List of (doc_idx, node_idx, similarity) tuples above threshold.
        """
        if not node_queries or not doc_texts:
            return []

        node_vecs = _embed(node_queries)    # (N_nodes, D)
        doc_vecs = _embed(doc_texts)        # (N_docs, D)
        sim = cosine_similarity_matrix(doc_vecs, node_vecs)  # (N_docs, N_nodes)

        results = []
        for doc_idx in range(len(doc_texts)):
            best_node_idx = int(np.argmax(sim[doc_idx]))
            best_sim = float(sim[doc_idx, best_node_idx])
            if best_sim >= self.threshold:
                results.append((doc_idx, best_node_idx, best_sim))

        return results

    def match_event(
        self,
        event_id: str,
        graph_id: str,
        market_id: Optional[str] = None,
        sync_from_files: bool = True,
    ) -> Dict:
        """
        Full pipeline: load nodes + docs, match, write mappings.

        Args:
            event_id: e.g. "gemini-3pt5-release-2026"
            graph_id: UUID of the causal_graph to match against
            market_id: Optional UUID for documents.market_id FK
            sync_from_files: If True, import JSON-file docs into the DB first

        Returns:
            Stats dict {total_docs, matched_docs, per_node_counts}
        """
        db = _DB(self.db_url)
        try:
            # 1. Load causal nodes
            nodes = db.fetch_nodes(graph_id)
            if not nodes:
                logger.warning(f"No nodes found for graph_id={graph_id}")
                return {"total_docs": 0, "matched_docs": 0, "per_node_counts": {}}

            logger.info(f"Loaded {len(nodes)} causal nodes")

            # 2. (Skipped file sync because we rely on ingestion feeding feed_items natively)

            # 3. Load documents from DB
            docs = db.fetch_documents(event_id)
            if not docs:
                # Fallback: load from pipeline artifacts (normalized_documents)
                # This is needed because feed_items are populated by backend sync (Step 2)
                # which runs AFTER the enrichment step (Step 1c)
                logger.info(f"No feed_items found for event_id={event_id}, trying pipeline artifacts...")
                docs = _load_docs_from_artifacts(event_id)
                if docs:
                    logger.info(f"Loaded {len(docs)} documents from pipeline artifacts")
                    # Sync them to the DB so write_mappings can reference them
                    if sync_from_files:
                        self._summarize_and_sync(docs, event_id, market_id, db)
                else:
                    logger.warning(f"No documents found for event_id={event_id}")
                    return {"total_docs": 0, "matched_docs": 0, "per_node_counts": {}}

            logger.info(f"Loaded {len(docs)} documents for matching")

            # 4. Build node query strings
            for node in nodes:
                node.query = f"{node.label}: {node.description}".strip(": ")

            node_queries = [n.query for n in nodes]
            doc_texts = [
                doc.summary or ((doc.raw_text or "")[:600])
                for doc in docs
            ]

            # 5. Compute matches
            raw_matches = self.match(node_queries, doc_texts)
            logger.info(f"Raw matches above threshold ({self.threshold}): {len(raw_matches)}")

            # 6. Cap docs per node (keep highest similarity)
            node_to_matches: Dict[int, List[Tuple[int, float]]] = {}
            for doc_idx, node_idx, sim in raw_matches:
                node_to_matches.setdefault(node_idx, []).append((doc_idx, sim))

            final_matches: List[MatchResult] = []
            per_node_counts: Dict[str, int] = {}

            for node_idx, doc_sims in node_to_matches.items():
                # Sort by similarity desc, cap at max_docs_per_node
                doc_sims.sort(key=lambda x: x[1], reverse=True)
                capped = doc_sims[: self.max_docs_per_node]
                node_id = nodes[node_idx].node_id
                per_node_counts[node_id] = len(capped)
                for doc_idx, sim in capped:
                    final_matches.append(MatchResult(
                        doc_id=docs[doc_idx].doc_id,
                        node_id=node_id,
                        similarity=sim,
                    ))

            # 7. Write mappings to DB (also updates evidence_count on causal_nodes)
            written = db.write_mappings(final_matches, graph_id=graph_id)
            logger.info(f"Wrote {written} doc→node mappings to DB")

            stats = {
                "total_docs": len(docs),
                "matched_docs": written,
                "per_node_counts": per_node_counts,
            }
            logger.info(f"Match stats: {stats}")
            return stats

        finally:
            db.close()

    def _summarize_and_sync(
        self,
        docs: List[DocumentRecord],
        event_id: str,
        market_id: Optional[str],
        db: _DB,
    ) -> None:
        """
        Generate summaries for file-based docs and upsert into DB.
        """
        from pipeline.ingestion.article_summarizer import ArticleSummarizer
        summarizer = ArticleSummarizer()

        texts = [doc.raw_text or "" for doc in docs]
        titles = [doc.title for doc in docs]
        logger.info(f"Generating summaries for {len(docs)} docs...")
        summaries = summarizer.summarize_batch(texts, titles)

        for doc, summary in zip(docs, summaries):
            doc.summary = summary
            try:
                db.upsert_document(doc, event_id, market_id)
            except Exception as e:
                logger.debug(f"Failed to upsert doc {doc.doc_id}: {e}")
                # MUST rollback - psycopg2 leaves the connection in an aborted
                # transaction state after any exception, which breaks all subsequent queries
                try:
                    db._conn.rollback()
                except Exception:
                    pass

        logger.info(f"Synced {len(docs)} docs to DB with summaries")

    def fetch_node_evidence(
        self,
        event_id: str,
        node_id: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Fetch top-K evidence documents for a specific node (for grounded detail generation).

        Returns:
            List of dicts with {title, url, source, summary, similarity_score, published_at}
        """
        db = _DB(self.db_url)
        try:
            sql = """
                SELECT 
                    fi.headline            AS title,
                    fi.url,
                    fi.source,
                    fi.summary,
                    1.0                    AS similarity_score,
                    fi.timestamp           AS published_at
                FROM feed_items fi
                JOIN markets m ON m.id = fi.market_id
                WHERE m.event_id = %s
                  AND fi.mapped_node_id = %s
                  AND fi.summary IS NOT NULL
                ORDER BY fi.timestamp DESC NULLS LAST
                LIMIT %s
            """
            import psycopg2
            from psycopg2.extras import RealDictCursor
            conn = psycopg2.connect(self.db_url)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (event_id, node_id, top_k))
                rows = cur.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        finally:
            db.close()
