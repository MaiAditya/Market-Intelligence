"""
Node Detail Generator

Generates rich node detail data (thesis, caseForHigher/Lower, impactOnMarket, etc.)
for each causal node — grounded in real evidence documents.

Flow per node:
  1. Fetch top-K evidence summaries from DB (mapped by NodeEvidenceMatcher)
  2. Build a grounded prompt: node description + causal context + evidence summaries
  3. Call LLMClient.generate_json() → rich detail JSON
  4. Write detail_data to causal_nodes.raw_data_json

Falls back to LLM-only (world knowledge) mode if no evidence docs are available.
"""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from integrations.llm_client import LLMClient, LLMError

logger = logging.getLogger(__name__)

_DEFAULT_DB_URL = "postgresql://causal:causal@localhost:5432/causal_interface"
_TOP_K_EVIDENCE = 5          # Evidence summaries to include per node
_MAX_PARALLEL_NODES = 3      # Concurrent LLM calls (rate limit safe)
_MAX_SUMMARY_CHARS = 400     # Truncate each evidence summary

# ─── Prompt ────────────────────────────────────────────────────────────────────

DETAIL_SYSTEM = """You are an expert analyst for prediction market causal graphs.
Generate structured JSON analysis for a single causal node, grounded strictly in
the provided evidence summaries. Do not invent facts not supported by the evidence.
Return ONLY valid JSON, no markdown."""

DETAIL_PROMPT_WITH_EVIDENCE = """Analyze this causal node for a prediction market:

NODE:
  label: {label}
  type: {event_type}
  description: {description}
  current_probability: {probability}%
  direction: {direction}  (positive = supports the market outcome, negative = blocks it)

CAUSAL CONTEXT:
  This node affects: {affects}
  This node is caused by: {caused_by}

EVIDENCE SUMMARIES (from recent news/articles — use these to ground your analysis):
{evidence_block}

Generate the analysis JSON with this exact schema:
{{
  "thesis": "2-3 sentence analysis grounded in the evidence above.",
  "numericalSummary": "1 sentence with a specific number or statistic from the evidence.",
  "caseForHigher": ["reason1 (cite evidence)", "reason2", "reason3"],
  "caseForLower": ["reason1 (cite evidence)", "reason2"],
  "counterEvidence": "1 sentence: the strongest counterargument from the evidence.",
  "impactOnMarket": {{
    "ifResolvesYes": 18,
    "ifResolvesNo": -12
  }},
  "probabilityBreakdown": [
    {{"label": "Base rate", "contribution": 35, "contributionDir": "positive"}},
    {{"label": "Evidence signal", "contribution": 15, "contributionDir": "positive"}},
    {{"label": "Key risk", "contribution": -10, "contributionDir": "negative"}}
  ],
  "historicalInference": "1-2 sentences about historical precedents for this type of event.",
  "historicalPrecedent": {{
    "cases": [
      {{
        "id": "case_1",
        "title": "Historical case name",
        "year": 2023,
        "similarity": 75,
        "outcome": "What happened",
        "result": "passed|failed|mixed",
        "attributes": ["Attribute1", "Attribute2"],
        "timeline": "Timeframe"
      }}
    ],
    "patternAnalysis": [
      {{"label": "Similar pattern", "passageRate": 65, "totalCases": 20, "matchingCases": 13}}
    ],
    "modelPrediction": {{
      "probability": {probability},
      "range": 15,
      "explanation": "Brief explanation based on historical patterns."
    }}
  }},
  "rumorEvidence": [
    {{"name": "Source name from evidence", "reliability": 3, "description": "What they reported."}}
  ],
  "sensitivityAnalysis": [
    {{"nodeId": "dependent_node_id", "nodeLabel": "Dependent Node Label", "currentProb": 60, "projectedRange": [45, 75]}}
  ]
}}

RULES:
- impactOnMarket.ifResolvesYes must be positive integer if direction=positive, negative if direction=negative.
- impactOnMarket.ifResolvesNo must be the opposite sign.
- probabilityBreakdown items should approximately sum to {probability}%.
- rumorEvidence: extract actual source names from the evidence summaries.
- If evidence is limited, extrapolate carefully using world knowledge.
"""

DETAIL_PROMPT_NO_EVIDENCE = """Analyze this causal node for a prediction market (no evidence available — use world knowledge):

NODE:
  label: {label}
  type: {event_type}
  description: {description}
  current_probability: {probability}%
  direction: {direction}

CAUSAL CONTEXT:
  This node affects: {affects}
  This node is caused by: {caused_by}

Generate the same JSON schema as described. Mark rumorEvidence as [] since no sources are available.
"""


# ─── DB helpers ────────────────────────────────────────────────────────────────

class _DB:
    def __init__(self, db_url: str):
        import psycopg2
        from psycopg2.extras import RealDictCursor
        self._conn = psycopg2.connect(db_url)
        self._RDC = RealDictCursor

    def fetch_nodes_with_edges(self, graph_id: str) -> List[Dict]:
        """Load all nodes plus their edge context for a graph."""
        sql = """
            SELECT n.node_id, n.label, n.event_type, n.probability,
                   n.raw_data_json, n.confidence_level
            FROM causal_nodes n
            WHERE n.graph_id = %s
        """
        with self._conn.cursor(cursor_factory=self._RDC) as cur:
            cur.execute(sql, (graph_id,))
            nodes = [dict(r) for r in cur.fetchall()]

        edge_sql = """
            SELECT source_node_id, target_node_id, mechanism_type, direction
            FROM causal_edges
            WHERE graph_id = %s
        """
        with self._conn.cursor(cursor_factory=self._RDC) as cur:
            cur.execute(edge_sql, (graph_id,))
            edges = cur.fetchall()

        # Build adjacency for context
        affects: Dict[str, List[str]] = {}
        caused_by: Dict[str, List[str]] = {}
        for e in edges:
            src, tgt = e["source_node_id"], e["target_node_id"]
            affects.setdefault(src, []).append(tgt)
            caused_by.setdefault(tgt, []).append(src)

        for node in nodes:
            nid = node["node_id"]
            node["affects"] = affects.get(nid, [])
            node["caused_by"] = caused_by.get(nid, [])

        return nodes

    def fetch_evidence(self, event_id: str, node_id: str, top_k: int) -> List[Dict]:
        """Fetch top-K evidence feed items for a node (mapped during pipeline ingest)."""
        sql = """
            SELECT
                fi.headline            AS title,
                fi.url,
                fi.source,
                fi.summary,
                fi.timestamp           AS published_at,
                1.0                    AS similarity_score
            FROM feed_items fi
            JOIN markets m ON m.id = fi.market_id
            WHERE m.event_id = %s
              AND fi.mapped_node_id = %s
              AND fi.headline IS NOT NULL
              AND length(fi.headline) > 10
            ORDER BY fi.timestamp DESC NULLS LAST
            LIMIT %s
        """
        with self._conn.cursor(cursor_factory=self._RDC) as cur:
            cur.execute(sql, (event_id, node_id, top_k))
            return [dict(r) for r in cur.fetchall()]

    def update_node_detail(self, graph_id: str, node_id: str, detail_data: Dict) -> None:
        """Merge detail_data into causal_nodes.raw_data_json.
        raw_data_json is type `json` (not jsonb), so we cast explicitly.
        """
        sql = """
            UPDATE causal_nodes
            SET raw_data_json = (
                COALESCE(raw_data_json::jsonb, '{}'::jsonb)
                || jsonb_build_object('detail_data', %s::jsonb)
            )::json
            WHERE graph_id = %s AND node_id = %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (json.dumps(detail_data), graph_id, node_id))
        self._conn.commit()

    def close(self):
        self._conn.close()


# ─── Core generator ────────────────────────────────────────────────────────────

class NodeDetailGenerator:
    """
    Generates evidence-grounded detail data for each causal node.

    Usage:
        gen = NodeDetailGenerator.from_env()
        stats = gen.generate_all(event_id, graph_id)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        db_url: Optional[str] = None,
        top_k_evidence: int = _TOP_K_EVIDENCE,
        max_parallel: int = _MAX_PARALLEL_NODES,
    ):
        self.client = llm_client
        self.db_url = db_url or os.getenv("DATABASE_URL_SYNC") or _DEFAULT_DB_URL
        self.top_k = top_k_evidence
        self.max_parallel = max_parallel

    @classmethod
    def from_env(cls, **kwargs) -> "NodeDetailGenerator":
        llm_client = LLMClient.from_env()
        return cls(llm_client=llm_client, **kwargs)

    def _build_evidence_block(self, docs: List[Dict]) -> str:
        """Format evidence docs into a prompt-friendly block."""
        if not docs:
            return "(No evidence available)"
        lines = []
        for i, doc in enumerate(docs, 1):
            title = doc.get("title") or "Untitled"
            source = doc.get("source") or "unknown"
            summary = (doc.get("summary") or "")[:_MAX_SUMMARY_CHARS]
            lines.append(
                f"[{i}] {title} ({source})\n"
                f"    Summary: {summary}"
            )
        return "\n\n".join(lines)

    def _overlay_evidence_urls(
        self,
        detail: Dict,
        evidence_docs: List[Dict],
    ) -> None:
        """
        After the LLM generates rumorEvidence sources (which lack URLs), inject
        the real document URLs and published_at from the matched evidence docs.

        Strategy: pair each rumorEvidence source with the corresponding evidence doc
        (by index), skipping any video/social domains. Any sources beyond the doc
        count get no URL.
        """
        from urllib.parse import urlparse
        _VIDEO_DOMAINS = frozenset({
            "youtube.com", "youtu.be", "tiktok.com",
            "vimeo.com", "instagram.com", "facebook.com",
            "fb.com", "twitch.tv",
        })

        def _is_video_url(url: str) -> bool:
            try:
                host = urlparse(url).netloc.lower().lstrip("www.")
                return any(host == d or host.endswith("." + d) for d in _VIDEO_DOMAINS)
            except Exception:
                return False

        sources = detail.get("rumorEvidence", [])
        if not isinstance(sources, list):
            return

        # Build a list of valid (non-video) docs with real URLs
        valid_docs = [
            doc for doc in evidence_docs
            if doc.get("url") and not _is_video_url(doc["url"])
        ]

        for i, source in enumerate(sources):
            if not isinstance(source, dict):
                continue
            if i < len(valid_docs):
                doc = valid_docs[i]
                url = doc.get("url")
                if url:
                    source["url"] = url
                pub = doc.get("published_at")
                if pub:
                    # Convert datetime to ISO string if needed
                    source["published_at"] = str(pub)[:19]  # "YYYY-MM-DDTHH:MM:SS"


    def _generate_node_detail(
        self,
        node: Dict,
        event_id: str,
        db_url: str,           # Pass URL instead of shared connection
        dry_run: bool = False,
    ) -> Optional[Dict]:
        """Generate detail data for a single node. Uses its own DB connection (thread-safe)."""
        node_id = node["node_id"]
        label = node.get("label", "")
        raw = node.get("raw_data_json") or {}

        # Skip belief root node — it uses the graph-level thesis
        if node.get("event_type") == "belief":
            return None

        # Each thread gets its own connection
        db = _DB(db_url)
        try:
            # Fetch evidence
            evidence_docs = db.fetch_evidence(event_id, node_id, self.top_k)
            has_evidence = len(evidence_docs) > 0
            logger.info(f"  Node {node_id}: {len(evidence_docs)} evidence docs ({label[:40]})")

            # Build prompt
            affects_labels = node.get("affects", [])
            caused_by_labels = node.get("caused_by", [])
            evidence_block = self._build_evidence_block(evidence_docs)

            prompt_template = DETAIL_PROMPT_WITH_EVIDENCE if has_evidence else DETAIL_PROMPT_NO_EVIDENCE
            prompt = prompt_template.format(
                label=label,
                event_type=node.get("event_type", "signal"),
                description=raw.get("description", ""),
                probability=int(node.get("probability") or 50),
                direction=raw.get("direction", "ambiguous"),
                affects=", ".join(affects_labels) or "belief (market outcome)",
                caused_by=", ".join(caused_by_labels) or "root causes",
                evidence_block=evidence_block,
            )

            # Generate
            try:
                detail = self.client.generate_json(
                    prompt=prompt,
                    system_instruction=DETAIL_SYSTEM,
                )
            except LLMError as e:
                logger.warning(f"  Failed to generate detail for {node_id}: {e}")
                return None

            if not isinstance(detail, dict):
                logger.warning(f"  Non-dict response for {node_id}")
                return None

            # Tag whether this was evidence-grounded
            detail["_grounded"] = has_evidence
            detail["_evidence_count"] = len(evidence_docs)

            # Overlay real document URLs + published_at into rumorEvidence sources
            # so the frontend can show actual site names and timestamps
            if has_evidence:
                self._overlay_evidence_urls(detail, evidence_docs)

            if not dry_run:
                db.update_node_detail(node.get("_graph_id"), node_id, detail)
                logger.info(f"  ✓ Wrote detail for {node_id} (grounded={has_evidence})")

            return detail

        finally:
            db.close()

    def regenerate_single_node(
        self,
        event_id: str,
        graph_id: str,
        node_id: str,
        dry_run: bool = False,
    ) -> Optional[Dict]:
        """
        Regenerate detail_data for a single node using latest evidence.

        Called by the RSS pipeline when a new high-confidence cluster maps
        to a node, providing updated evidence for the node's analysis.

        Args:
            event_id: Market event ID (e.g. "fed-rate-hike-q2-2026")
            graph_id: UUID of the causal_graph
            node_id: The specific node to regenerate
            dry_run: If True, generate but don't write to DB

        Returns:
            Generated detail_data dict, or None if failed.
        """
        db = _DB(self.db_url)
        try:
            nodes = db.fetch_nodes_with_edges(graph_id)
            target = None
            for n in nodes:
                if n["node_id"] == node_id:
                    target = n
                    break

            if not target:
                logger.warning(f"Node {node_id} not found in graph {graph_id}")
                return None

            if target.get("event_type") == "belief":
                logger.debug(f"Skipping belief node {node_id}")
                return None

            target["_graph_id"] = graph_id
            logger.info(f"Regenerating detail for node {node_id} ({target.get('label', '')[:40]})")
            return self._generate_node_detail(target, event_id, self.db_url, dry_run)

        finally:
            db.close()

    def generate_all(
        self,
        event_id: str,
        graph_id: str,
        dry_run: bool = False,
    ) -> Dict:
        """
        Generate or update detail data for all non-belief nodes in a graph.

        Args:
            event_id: e.g. "gemini-3pt5-release-2026"
            graph_id: UUID of the causal_graph
            dry_run: If True, generate but don't write to DB

        Returns:
            Stats dict
        """
        db = _DB(self.db_url)
        try:
            nodes = db.fetch_nodes_with_edges(graph_id)
            # Inject graph_id for update call
            for n in nodes:
                n["_graph_id"] = graph_id

            # Filter out belief node
            target_nodes = [n for n in nodes if n.get("event_type") != "belief"]
            logger.info(f"Generating grounded details for {len(target_nodes)} nodes...")

            results = {"success": 0, "failed": 0, "no_evidence": 0, "skipped": 0}

            # Process in parallel — each thread opens its own DB connection
            with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
                futures = {
                    executor.submit(
                        self._generate_node_detail, node, event_id, self.db_url, dry_run
                    ): node["node_id"]
                    for node in target_nodes
                }
                for future in as_completed(futures):
                    node_id = futures[future]
                    try:
                        detail = future.result()
                        if detail is None:
                            results["skipped"] += 1
                        elif detail.get("_grounded"):
                            results["success"] += 1
                        else:
                            results["no_evidence"] += 1
                    except Exception as e:
                        logger.error(f"  Error for node {node_id}: {e}")
                        results["failed"] += 1

            logger.info(f"Node detail generation complete: {results}")
            return results

        finally:
            db.close()
