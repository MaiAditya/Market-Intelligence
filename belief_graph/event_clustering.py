"""
Event Clustering Module

Groups similar events (documents about the same real-world occurrence) using
semantic similarity. This prevents duplicate events from cluttering the graph.

Key features:
- Semantic similarity clustering using sentence-transformers
- Canonical event selection (best representative per cluster)
- Event consolidation (merge actors, aggregate certainty)
- Maintains links to all source documents

Example: 5 articles about "EU AI Act passes committee vote" become 1 clustered event
with references to all 5 source documents.
"""

import logging
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from belief_graph.models import EventNode, EventType
from pipeline.normalizer import DocumentNormalizer

logger = logging.getLogger(__name__)


@dataclass
class ClusteredEvent:
    """
    Represents a cluster of similar events.
    
    Contains a canonical (representative) event and links to all
    source documents that describe the same real-world event.
    """
    # The representative event for this cluster
    canonical_event: EventNode
    
    # All event IDs in this cluster
    member_event_ids: List[str] = field(default_factory=list)
    
    # All source document IDs
    source_doc_ids: List[str] = field(default_factory=list)
    
    # All source URLs
    source_urls: List[str] = field(default_factory=list)
    
    # Aggregated actors from all members
    all_actors: Set[str] = field(default_factory=set)
    
    # Cluster confidence (higher with more corroborating sources)
    cluster_confidence: float = 0.0
    
    # Average similarity within cluster
    avg_similarity: float = 0.0
    
    # Number of sources
    num_sources: int = 1
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "canonical_event": self.canonical_event.to_dict(),
            "member_event_ids": self.member_event_ids,
            "source_doc_ids": self.source_doc_ids,
            "source_urls": self.source_urls,
            "all_actors": list(self.all_actors),
            "cluster_confidence": self.cluster_confidence,
            "avg_similarity": self.avg_similarity,
            "num_sources": self.num_sources,
        }


class EventClusterer:
    """
    Clusters similar events using semantic similarity.
    
    Uses sentence-transformers to compute embeddings and groups events
    via Hierarchical Agglomerative Clustering (average linkage) with a
    temporal window constraint to prevent chronologically distant events
    from merging.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.75,
        model_name: str = "all-mpnet-base-v2",
        time_window_hours: float = 96,
        temporal_decay_tau: float = 12.0,
        entity_weight: float = 0.20,
        normalizer: Optional[DocumentNormalizer] = None,
    ):
        """
        Initialize the event clusterer.
        
        Args:
            similarity_threshold: Minimum cosine similarity to cluster events (0.0-1.0)
            model_name: Sentence transformer model to use
            time_window_hours: Time window (hours) for full-similarity clustering.
                              Events within this window keep full similarity.
                              Events beyond it get exponentially decayed similarity
                              instead of being hard-cut to 0.  Default 96 = ±2 days.
            temporal_decay_tau: Decay constant (hours) for the soft temporal falloff.
                               Controls how quickly similarity drops beyond the window.
                               Smaller = sharper dropoff.  Default 12h.
            entity_weight: Weight for entity-overlap Jaccard boost (0.0-1.0).
                          Final similarity = (1 - entity_weight) * semantic + entity_weight * jaccard.
            normalizer: Optional document normalizer for loading body text.
        """
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self.time_window_hours = time_window_hours
        self.temporal_decay_tau = temporal_decay_tau
        self.entity_weight = entity_weight
        self._model = None
        self._model_loaded = False
        self._normalizer = normalizer
        self._doc_cache: Dict[str, Optional[object]] = {}  # cache loaded docs
        
        logger.info(
            f"EventClusterer initialized with threshold={similarity_threshold}, "
            f"model={model_name}, time_window={time_window_hours}h, "
            f"decay_tau={temporal_decay_tau}h, entity_weight={entity_weight}"
        )
    
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self._model_loaded:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._model_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers model: {e}")
            self._model = None
            self._model_loaded = True  # Mark as attempted
    
    def _get_event_text(
        self,
        event: EventNode,
        market_question: Optional[str] = None,
    ) -> str:
        """
        Extract text representation of an event for embedding.
        
        Uses market context + title + body snippet + actors for a comprehensive,
        semantically rich representation that produces better clustering.
        """
        parts = []
        
        # Prepend market question as context anchor
        if market_question:
            parts.append(f"Market: {market_question}.")
        
        # Use raw title if available
        if event.raw_title:
            parts.append(event.raw_title)
        else:
            # Fall back to action + object
            if event.action:
                parts.append(event.action)
            if event.object:
                parts.append(event.object)
        
        # Add body text snippet from source document (first ~300 chars)
        body_snippet = self._get_body_snippet(event)
        if body_snippet:
            parts.append(body_snippet)
        
        # Add key actors
        if event.actors:
            actors_str = ", ".join(event.actors[:5])
            parts.append(f"Actors: {actors_str}")
        
        return " ".join(parts)
    
    def _load_doc_cached(self, doc_id: str):
        """Load a document from disk, using cache to avoid repeated I/O."""
        if doc_id in self._doc_cache:
            return self._doc_cache[doc_id]
        
        if self._normalizer is None:
            try:
                self._normalizer = DocumentNormalizer()
            except Exception:
                self._doc_cache[doc_id] = None
                return None
        
        try:
            doc = self._normalizer.load(doc_id)
            self._doc_cache[doc_id] = doc
            return doc
        except Exception as e:
            logger.debug(f"Could not load doc {doc_id}: {e}")
            self._doc_cache[doc_id] = None
            return None
    
    def _get_body_snippet(self, event: EventNode, max_chars: int = 300) -> str:
        """
        Load the first N characters of the source document body text.
        Uses document cache to avoid repeated disk I/O.
        """
        if not event.source_doc_id:
            return ""
        
        doc = self._load_doc_cached(event.source_doc_id)
        if doc and doc.raw_text:
            text = doc.raw_text
            if doc.title and text.startswith(doc.title):
                text = text[len(doc.title):].lstrip()
            return text[:max_chars]
        
        return ""
    
    def _compute_embeddings(
        self,
        events: List[EventNode],
        market_question: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Compute embeddings for a list of events.
        
        Args:
            events: List of EventNode objects
            market_question: Optional market context to anchor embeddings
        
        Returns:
            Numpy array of embeddings or None if model unavailable
        """
        self._load_model()
        
        if self._model is None:
            logger.warning("Model not available, cannot compute embeddings")
            return None
        
        texts = [
            self._get_event_text(event, market_question=market_question)
            for event in events
        ]
        
        try:
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            return None
    
    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.
        
        Args:
            embeddings: Numpy array of embeddings (n_events x embedding_dim)
        
        Returns:
            Similarity matrix (n_events x n_events)
        """
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        
        # Compute cosine similarity
        similarity = np.dot(normalized, normalized.T)
        
        return similarity
    
    def _apply_temporal_decay(
        self,
        events: List[EventNode],
        similarity_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Apply soft temporal decay using vectorized numpy operations.

        Uses numpy broadcasting instead of O(n²) Python loop for speed.
        Events with timestamp=None are treated conservatively (no decay).
        """
        n = len(events)
        window_seconds = self.time_window_hours * 3600
        tau_seconds = self.temporal_decay_tau * 3600

        # Extract timestamps as epoch seconds; use NaN for None
        epoch = np.full(n, np.nan)
        for i, ev in enumerate(events):
            if ev.timestamp is not None:
                epoch[i] = ev.timestamp.timestamp()

        # Vectorized pairwise time difference matrix
        diff = np.abs(epoch[:, None] - epoch[None, :])  # n×n, NaN where unknown

        # Compute decay factors: 1.0 within window, exp(-overshoot/tau) beyond
        overshoot = np.maximum(0.0, diff - window_seconds)
        decay_matrix = np.exp(-overshoot / tau_seconds)

        # Where either timestamp is NaN, keep decay = 1.0 (no penalty)
        nan_mask = np.isnan(diff)
        decay_matrix[nan_mask] = 1.0

        # Diagonal stays 1.0
        np.fill_diagonal(decay_matrix, 1.0)

        return similarity_matrix * decay_matrix

    def _compute_entity_overlap_matrix(
        self,
        events: List[EventNode],
    ) -> np.ndarray:
        """
        Compute pairwise entity Jaccard similarity matrix.

        Uses document cache to avoid repeated disk I/O.
        Pre-computes entity sets, then vectorizes the Jaccard computation.
        """
        n = len(events)
        jaccard = np.zeros((n, n))

        # Build entity sets (uses cached docs — no extra disk I/O)
        entity_sets: List[Set[str]] = []
        for event in events:
            entities: Set[str] = set()
            if event.actors:
                entities.update(a.lower().strip() for a in event.actors)
            if event.source_doc_id:
                doc = self._load_doc_cached(event.source_doc_id)
                if doc and hasattr(doc, 'extracted_entities') and doc.extracted_entities:
                    entities.update(
                        e.get("text", "").lower().strip()
                        for e in doc.extracted_entities
                        if e.get("text")
                    )
            entity_sets.append(entities)

        # Pre-compute set sizes for vectorized Jaccard
        for i in range(n):
            jaccard[i, i] = 1.0
            if not entity_sets[i]:
                continue
            for j in range(i + 1, n):
                if not entity_sets[j]:
                    continue
                intersection = len(entity_sets[i] & entity_sets[j])
                if intersection == 0:
                    continue
                union = len(entity_sets[i] | entity_sets[j])
                score = intersection / union
                jaccard[i, j] = score
                jaccard[j, i] = score

        return jaccard

    def _cluster_by_similarity(
        self,
        events: List[EventNode],
        similarity_matrix: np.ndarray,
    ) -> List[List[int]]:
        """
        Group events into clusters using Hierarchical Agglomerative
        Clustering (HAC) with **average linkage**.

        Pipeline:
        1. Compute entity-overlap Jaccard matrix
        2. Fuse: final_sim = (1-w)*semantic + w*entity_jaccard
        3. Apply soft temporal decay (exponential falloff beyond window)
        4. HAC average-linkage clustering on the fused matrix

        Args:
            events: List of events
            similarity_matrix: Semantic pairwise similarity matrix

        Returns:
            List of clusters (each cluster is a list of event indices)
        """
        n = len(events)

        if n <= 1:
            return [[i] for i in range(n)]

        # --- Phase 1: Entity overlap Jaccard boost ---
        # Use max(semantic, blended) so entity overlap can only BOOST
        # similarity, never dilute it when entity data is sparse/absent.
        entity_jaccard = self._compute_entity_overlap_matrix(events)
        w = self.entity_weight
        blended = (1 - w) * similarity_matrix + w * entity_jaccard
        fused_sim = np.maximum(similarity_matrix, blended)

        # --- Phase 2: Soft temporal decay ---
        decayed_sim = self._apply_temporal_decay(events, fused_sim)

        # --- Phase 3: HAC with average linkage ---
        # Convert similarity → distance (sklearn expects distance)
        distance_matrix = 1.0 - decayed_sim
        np.fill_diagonal(distance_matrix, 0.0)  # self-distance = 0
        # Clamp any floating-point noise
        distance_matrix = np.clip(distance_matrix, 0.0, 2.0)

        distance_threshold = 1.0 - self.similarity_threshold

        try:
            from sklearn.cluster import AgglomerativeClustering

            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage="average",
                distance_threshold=distance_threshold,
            )
            labels = clustering.fit_predict(distance_matrix)
        except Exception as e:
            logger.warning(
                f"HAC clustering failed ({e}), falling back to greedy merge"
            )
            return self._cluster_greedy_fallback(events, decayed_sim)

        # Convert labels → list of clusters
        cluster_dict: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            cluster_dict[int(label)].append(idx)

        return list(cluster_dict.values())

    # ------------------------------------------------------------------ #
    #  Legacy fallback (kept for safety if sklearn is unavailable)        #
    # ------------------------------------------------------------------ #
    def _cluster_greedy_fallback(
        self,
        events: List[EventNode],
        similarity_matrix: np.ndarray,
    ) -> List[List[int]]:
        """Greedy single-linkage fallback used if HAC is unavailable."""
        n = len(events)
        cluster_assignment = list(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    old_cluster = cluster_assignment[j]
                    new_cluster = cluster_assignment[i]
                    if old_cluster != new_cluster:
                        for k in range(n):
                            if cluster_assignment[k] == old_cluster:
                                cluster_assignment[k] = new_cluster

        cluster_dict: Dict[int, List[int]] = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_assignment):
            cluster_dict[cluster_id].append(idx)

        return list(cluster_dict.values())
    
    def _select_canonical_event(
        self,
        events: List[EventNode],
        indices: List[int]
    ) -> Tuple[EventNode, int]:
        """
        Select the best representative event from a cluster.
        
        Selection criteria (in order of priority):
        1. Most entities extracted
        2. Highest source credibility
        3. Most recent timestamp
        
        Args:
            events: Full list of events
            indices: Indices of events in this cluster
        
        Returns:
            Tuple of (canonical event, index in original list)
        """
        if len(indices) == 1:
            return events[indices[0]], indices[0]
        
        # Score each event
        def score_event(idx: int) -> Tuple[int, float, datetime]:
            event = events[idx]
            
            # Entity count (from actors as proxy)
            entity_count = len(event.actors) if event.actors else 0
            
            # Source credibility
            credibility_map = {
                "official": 0.9,
                "research": 0.8,
                "journalist": 0.7,
                "forum": 0.4,
                "social": 0.3,
            }
            # Try to infer source type from source domain
            source_lower = (event.source or "").lower()
            if any(x in source_lower for x in [".gov", "official", "europa.eu"]):
                credibility = 0.9
            elif any(x in source_lower for x in ["reuters", "bloomberg", "bbc", "nytimes"]):
                credibility = 0.8
            elif any(x in source_lower for x in [".edu", "research", "arxiv"]):
                credibility = 0.8
            elif any(x in source_lower for x in ["reddit", "twitter", "x.com"]):
                credibility = 0.3
            else:
                credibility = 0.5
            
            # Timestamp (prefer more recent)
            timestamp = event.timestamp if event.timestamp else datetime.min
            
            return (entity_count, credibility, timestamp)
        
        # Sort by score (descending)
        scored = [(idx, score_event(idx)) for idx in indices]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        best_idx = scored[0][0]
        return events[best_idx], best_idx
    
    # ------------------------------------------------------------------ #
    #  Phase 4: Canonical Event Synthesis                                 #
    # ------------------------------------------------------------------ #
    def _synthesize_canonical_event(
        self,
        events: List[EventNode],
        cluster_indices: List[int],
    ) -> EventNode:
        """
        Build a synthesised "super node" that represents the whole cluster.

        The synthesis picks consensus values from all members rather than
        choosing a single random article as the representative.

        Rules:
        - **timestamp**: mode day, tie-break to median timestamp
        - **actors**: union of all unique actors across members
        - **certainty**: min(0.95, avg_certainty + log(num_sources) * 0.1)
        - **raw_title**: longest title from members (richest information)
        - **source**: source of the highest-credibility member
        - **other fields**: taken from the member selected by
          ``_select_canonical_event`` (event_type, scope, etc.)

        Args:
            events: Full list of events
            cluster_indices: Indices belonging to this cluster

        Returns:
            A new EventNode with synthesised metadata
        """
        if len(cluster_indices) == 1:
            return events[cluster_indices[0]]

        # Start from the canonical selection (best source / quality)
        base, _ = self._select_canonical_event(events, cluster_indices)

        # --- Consensus timestamp (mode-day, tie-break median) ---
        timestamps = [
            events[i].timestamp
            for i in cluster_indices
            if events[i].timestamp is not None
        ]
        consensus_ts = base.timestamp
        if timestamps:
            # Find mode day
            day_counts = Counter(ts.date() for ts in timestamps)
            mode_day = day_counts.most_common(1)[0][0]
            same_day = [ts for ts in timestamps if ts.date() == mode_day]
            # Median of timestamps on the mode day
            same_day.sort()
            consensus_ts = same_day[len(same_day) // 2]

        # --- Union of actors ---
        all_actors: Set[str] = set()
        for idx in cluster_indices:
            if events[idx].actors:
                all_actors.update(events[idx].actors)

        # --- Boosted certainty ---
        avg_certainty = np.mean(
            [events[i].certainty for i in cluster_indices]
        )
        num_sources = len(cluster_indices)
        synthesised_certainty = min(
            0.95, float(avg_certainty) + math.log(num_sources) * 0.1
        )

        # --- Best title (longest = most descriptive) ---
        best_title = base.raw_title or ""
        for idx in cluster_indices:
            candidate = events[idx].raw_title or ""
            if len(candidate) > len(best_title):
                best_title = candidate

        # Build synthesised node (copy immutable fields from base)
        return EventNode(
            event_id=base.event_id,
            event_type=base.event_type,
            timestamp=consensus_ts,
            actors=list(all_actors)[:15],
            action=base.action,
            object=base.object,
            certainty=round(synthesised_certainty, 4),
            source=base.source,
            scope=base.scope,
            source_doc_id=base.source_doc_id,
            source_signal_id=base.source_signal_id,
            raw_title=best_title or base.raw_title,
            url=base.url,
        )

    def cluster_events(
        self,
        events: List[EventNode],
        min_cluster_size: int = 1,
        market_question: Optional[str] = None,
    ) -> List[ClusteredEvent]:
        """
        Cluster similar events together.
        
        Args:
            events: List of EventNode objects to cluster
            min_cluster_size: Minimum events to form a cluster (default 1)
            market_question: Optional market question text to anchor embeddings.
                           Significantly improves clustering quality by giving
                           all articles a shared context prefix.
        
        Returns:
            List of ClusteredEvent objects
        """
        if not events:
            return []
        
        logger.info(
            f"Clustering {len(events)} events with threshold {self.similarity_threshold}"
            f"{f' (market: {market_question[:50]}...)' if market_question else ''}"
        )
        
        # Compute embeddings (with optional market context)
        embeddings = self._compute_embeddings(events, market_question=market_question)
        
        if embeddings is None:
            # Fall back to no clustering (each event is its own cluster)
            logger.warning("Falling back to no clustering (model unavailable)")
            return self._create_singleton_clusters(events)
        
        # Compute semantic similarity matrix
        similarity_matrix = self._compute_similarity_matrix(embeddings)
        
        # Find clusters (entity boost + temporal decay + HAC average linkage)
        clusters = self._cluster_by_similarity(events, similarity_matrix)
        
        logger.info(f"Found {len(clusters)} clusters from {len(events)} events")
        
        # Create ClusteredEvent objects
        clustered_events = []
        
        for cluster_indices in clusters:
            if len(cluster_indices) < min_cluster_size:
                continue
            
            # Phase 4: Synthesise canonical event instead of picking one
            canonical = self._synthesize_canonical_event(events, cluster_indices)
            
            # Collect all member information
            member_ids = [events[i].event_id for i in cluster_indices]
            doc_ids = [events[i].source_doc_id for i in cluster_indices if events[i].source_doc_id]
            urls = [events[i].url for i in cluster_indices if events[i].url]
            
            # Aggregate actors (already in canonical, but keep set for ClusteredEvent)
            all_actors: Set[str] = set()
            for idx in cluster_indices:
                if events[idx].actors:
                    all_actors.update(events[idx].actors)
            
            # Calculate cluster metrics
            avg_sim = 0.0
            if len(cluster_indices) > 1:
                # Average pairwise similarity
                sims = []
                for i, idx_i in enumerate(cluster_indices):
                    for idx_j in cluster_indices[i+1:]:
                        sims.append(similarity_matrix[idx_i, idx_j])
                avg_sim = np.mean(sims) if sims else 0.0
            
            clustered = ClusteredEvent(
                canonical_event=canonical,
                member_event_ids=member_ids,
                source_doc_ids=doc_ids,
                source_urls=urls,
                all_actors=all_actors,
                cluster_confidence=canonical.certainty,
                avg_similarity=float(avg_sim),
                num_sources=len(cluster_indices),
            )
            
            clustered_events.append(clustered)
        
        logger.info(
            f"Created {len(clustered_events)} clustered events "
            f"(reduction: {len(events)} -> {len(clustered_events)})"
        )
        
        return clustered_events
    
    def _create_singleton_clusters(self, events: List[EventNode]) -> List[ClusteredEvent]:
        """Create single-event clusters when clustering is not available."""
        return [
            ClusteredEvent(
                canonical_event=event,
                member_event_ids=[event.event_id],
                source_doc_ids=[event.source_doc_id] if event.source_doc_id else [],
                source_urls=[event.url] if event.url else [],
                all_actors=set(event.actors) if event.actors else set(),
                cluster_confidence=event.certainty,
                avg_similarity=1.0,
                num_sources=1,
            )
            for event in events
        ]


def cluster_events(
    events: List[EventNode],
    similarity_threshold: float = 0.75,
    min_cluster_size: int = 1,
    time_window_hours: float = 96,
    market_question: Optional[str] = None,
) -> List[ClusteredEvent]:
    """
    Convenience function to cluster events.
    
    Args:
        events: List of events to cluster
        similarity_threshold: Minimum similarity to group (0.0-1.0)
        min_cluster_size: Minimum events per cluster
        time_window_hours: Maximum temporal distance for clustering (default 96 = ±2 days)
        market_question: Optional market question to anchor embeddings
    
    Returns:
        List of ClusteredEvent objects
    """
    clusterer = EventClusterer(
        similarity_threshold=similarity_threshold,
        time_window_hours=time_window_hours,
    )
    return clusterer.cluster_events(
        events, min_cluster_size=min_cluster_size, market_question=market_question
    )


if __name__ == "__main__":
    # Test the clustering
    import argparse
    
    parser = argparse.ArgumentParser(description="Test event clustering")
    parser.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold")
    args = parser.parse_args()
    
    # Create some test events
    test_events = [
        EventNode(
            event_id="test_1",
            event_type="policy",
            timestamp=datetime.now(),
            actors=["EU", "European Commission"],
            action="EU AI Act",
            object="passes committee vote",
            certainty=0.7,
            source="reuters.com",
            scope="global",
            raw_title="EU AI Act passes key committee vote",
            url="https://reuters.com/eu-ai-act-vote"
        ),
        EventNode(
            event_id="test_2",
            event_type="policy",
            timestamp=datetime.now(),
            actors=["European Parliament", "EU"],
            action="AI Act",
            object="approved by committee",
            certainty=0.65,
            source="bbc.com",
            scope="global",
            raw_title="European Parliament committee approves AI Act",
            url="https://bbc.com/eu-ai-vote"
        ),
        EventNode(
            event_id="test_3",
            event_type="economic",
            timestamp=datetime.now(),
            actors=["Apple", "Tim Cook"],
            action="Apple announces",
            object="new AI features",
            certainty=0.8,
            source="apple.com",
            scope="global",
            raw_title="Apple announces new AI features at WWDC",
            url="https://apple.com/wwdc"
        ),
    ]
    
    clusterer = EventClusterer(similarity_threshold=args.threshold)
    clusters = clusterer.cluster_events(test_events)
    
    print(f"\nClustered {len(test_events)} events into {len(clusters)} clusters:\n")
    
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}:")
        print(f"  Canonical: {cluster.canonical_event.raw_title}")
        print(f"  Members: {len(cluster.member_event_ids)}")
        print(f"  Sources: {cluster.num_sources}")
        print(f"  Confidence: {cluster.cluster_confidence:.2f}")
        print(f"  Avg Similarity: {cluster.avg_similarity:.2f}")
        print()
