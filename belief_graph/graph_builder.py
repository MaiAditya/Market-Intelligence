"""
Graph Builder

Main orchestrator for building belief update graphs.
Assembles all components:
1. Event extraction
2. Candidate generation (rule-based)
3. Mechanism classification (LLM-bounded)
4. Evidence scoring (empirical)
5. Confidence calculation
6. DAG assembly with cycle detection

The output is a complete BeliefGraph with nodes, edges, and metadata.
"""

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.event_registry import EventRegistry, get_registry, Event
from pipeline.normalizer import DocumentNormalizer
from pipeline.signal_extractor import SignalExtractor
from integrations.polymarket_client import PolymarketClient, get_polymarket_client

from belief_graph.models import (
    EventNode, BeliefNode, BeliefEdge, EvidenceScores, BeliefGraph,
    MechanismType, DirectionType, LatencyType
)
from belief_graph.event_extractor import EventExtractor
from belief_graph.candidate_generator import generate_candidates
from belief_graph.mechanism_classifier import MechanismClassifier
from belief_graph.evidence_scorer import EvidenceScorer
from belief_graph.confidence_calculator import (
    ConfidenceCalculator,
    calculate_confidence,
    CONFIDENCE_THRESHOLD
)
from belief_graph.event_clustering import EventClusterer, ClusteredEvent
from models.signal_classifier import DirectionClassifier

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Get current UTC time as a naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class GraphBuilder:
    """
    Builds belief update graphs for target market events.
    
    Orchestrates the full pipeline:
    1. Create BeliefNode from event registry
    2. Extract EventNodes from documents
    3. Generate candidate edges (rule-based)
    4. Classify mechanisms (LLM-bounded)
    5. Score evidence (empirical)
    6. Calculate confidence and filter
    7. Assemble DAG (remove cycles)
    """
    
    def __init__(
        self,
        registry: Optional[EventRegistry] = None,
        normalizer: Optional[DocumentNormalizer] = None,
        polymarket_client: Optional[PolymarketClient] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD
    ):
        """
        Initialize graph builder.
        
        Args:
            registry: Event registry
            normalizer: Document normalizer
            polymarket_client: Polymarket client
            confidence_threshold: Minimum confidence for edges
        """
        logger.info("Initializing GraphBuilder...")
        
        self.registry = registry or get_registry()
        self.normalizer = normalizer or DocumentNormalizer()
        self.pm_client = polymarket_client or get_polymarket_client()
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.signal_extractor = SignalExtractor(
            registry=self.registry,
            normalizer=self.normalizer
        )
        self.event_extractor = EventExtractor(
            normalizer=self.normalizer,
            signal_extractor=self.signal_extractor
        )
        self.mechanism_classifier = MechanismClassifier()
        self.evidence_scorer = EvidenceScorer(
            polymarket_client=self.pm_client,
            normalizer=self.normalizer
        )
        self.confidence_calculator = ConfidenceCalculator(
            threshold=confidence_threshold
        )
        
        # Direction classifier for content-based sentiment
        self.direction_classifier = DirectionClassifier()
        
        # Event clusterer for deduplication
        self.event_clusterer = EventClusterer(similarity_threshold=0.75)
        
        # Cache for document directions
        self._direction_cache: Dict[str, Tuple[str, float]] = {}
        
        logger.info("GraphBuilder initialized")
    
    def _create_belief_node(self, event: Event) -> BeliefNode:
        """
        Create BeliefNode from event registry entry.
        
        Args:
            event: Event from registry
        
        Returns:
            BeliefNode
        """
        logger.debug(f"Creating BeliefNode for event: {event.event_id}")
        
        # Get market data from Polymarket
        market = None
        if event.polymarket_slug:
            market = self.pm_client.get_market_by_slug(event.polymarket_slug)
        
        # Determine resolution time (Event uses 'deadline' not 'resolution_date')
        resolution_time = event.deadline
        if market and market.end_date:
            resolution_time = market.end_date
        if resolution_time is None:
            resolution_time = _utc_now()
        
        return BeliefNode(
            belief_id=f"belief_{event.event_id}",
            question=event.event_title,
            resolution_time=resolution_time,
            current_price=market.probability if market else 0.5,
            liquidity=market.liquidity if market else 0.0,
            event_id=event.event_id,
            polymarket_slug=event.polymarket_slug or ""
        )
    
    def _infer_direction(
        self,
        event: EventNode,
        belief: BeliefNode
    ) -> DirectionType:
        """
        Infer edge direction using content-based sentiment analysis.
        
        Uses:
        1. Document content sentiment via DirectionClassifier
        2. Belief-relevant keyword matching
        3. Signal direction if available
        
        NO arbitrary heuristics based on event type.
        
        Args:
            event: Source event
            belief: Target belief
        
        Returns:
            Direction type (positive, negative, ambiguous)
        """
        doc_id = event.source_doc_id
        
        # Check cache first
        if doc_id and doc_id in self._direction_cache:
            cached_dir, cached_conf = self._direction_cache[doc_id]
            if cached_conf > 0.55:
                return cached_dir if cached_dir != "neutral" else "ambiguous"
        
        # Method 1: Check signal direction if available
        if event.source_signal_id:
            signals = self.signal_extractor.load_signals_for_event(belief.event_id)
            for sig in signals:
                if sig.signal_id == event.source_signal_id:
                    if sig.direction == "positive" and sig.confidence > 0.5:
                        return "positive"
                    elif sig.direction == "negative" and sig.confidence > 0.5:
                        return "negative"
        
        # Method 2: Content-based sentiment classification
        if doc_id:
            doc = self.normalizer.load(doc_id)
            if doc and doc.raw_text:
                # Use first 2000 chars for efficiency
                text = doc.raw_text[:2000]
                try:
                    direction, confidence = self.direction_classifier.get_direction(text)
                    self._direction_cache[doc_id] = (direction, confidence)
                    
                    if confidence > 0.55:
                        if direction == "positive":
                            return "positive"
                        elif direction == "negative":
                            return "negative"
                except Exception as e:
                    logger.debug(f"Direction classification failed: {e}")
        
        # Method 3: Belief-specific keyword matching
        # Check if event content suggests positive or negative for the belief
        event_text = f"{event.action} {event.object}".lower()
        belief_question = belief.question.lower()
        
        # Positive keywords for "will X happen" type beliefs
        positive_indicators = [
            "pass", "approve", "advance", "succeed", "progress", "achieve",
            "implement", "enforce", "support", "commit", "confirm", "agree",
            "breakthrough", "milestone", "completion"
        ]
        
        # Negative keywords
        negative_indicators = [
            "fail", "reject", "delay", "block", "oppose", "concern",
            "challenge", "obstacle", "setback", "risk", "problem", "halt",
            "suspend", "postpone", "uncertainty", "doubt"
        ]
        
        pos_count = sum(1 for kw in positive_indicators if kw in event_text)
        neg_count = sum(1 for kw in negative_indicators if kw in event_text)
        
        if pos_count > neg_count + 1:
            return "positive"
        elif neg_count > pos_count + 1:
            return "negative"
        
        return "ambiguous"
    
    def _infer_latency(
        self,
        event: EventNode,
        target_id: str,
        events: Dict[str, EventNode],
        belief: BeliefNode
    ) -> LatencyType:
        """
        Infer latency type for an edge.
        
        Args:
            event: Source event
            target_id: Target event/belief ID
            events: All events
            belief: Target belief
        
        Returns:
            Latency type
        """
        if not event.timestamp:
            return "uncertain"
        
        # Get target timestamp
        if target_id == belief.belief_id:
            target_time = belief.resolution_time
        else:
            target_event = events.get(target_id)
            if target_event and target_event.timestamp:
                target_time = target_event.timestamp
            else:
                return "uncertain"
        
        if not target_time:
            return "uncertain"
        
        # Calculate time difference
        diff_hours = (target_time - event.timestamp).total_seconds() / 3600
        
        if diff_hours < 24:
            return "immediate"
        elif diff_hours < 168:  # 1 week
            return "delayed"
        else:
            return "uncertain"
    
    def _remove_cycles(
        self,
        edges: List[BeliefEdge],
        belief_id: str
    ) -> List[BeliefEdge]:
        """
        Remove edges that would create cycles.
        
        Uses topological sort to detect and remove cycle-causing edges.
        
        Args:
            edges: List of edges
            belief_id: ID of belief node (sink)
        
        Returns:
            Filtered list of edges (DAG)
        """
        if not edges:
            return edges
        
        # Build adjacency list
        graph: Dict[str, Set[str]] = defaultdict(set)
        in_degree: Dict[str, int] = defaultdict(int)
        
        all_nodes: Set[str] = set()
        for edge in edges:
            all_nodes.add(edge.from_event_id)
            all_nodes.add(edge.to_event_id)
        
        for edge in edges:
            graph[edge.from_event_id].add(edge.to_event_id)
            in_degree[edge.to_event_id] += 1
        
        # Initialize in_degree for nodes with no incoming edges
        for node in all_nodes:
            if node not in in_degree:
                in_degree[node] = 0
        
        # Kahn's algorithm for topological sort
        queue = [node for node in all_nodes if in_degree[node] == 0]
        sorted_nodes = []
        
        while queue:
            node = queue.pop(0)
            sorted_nodes.append(node)
            
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If we couldn't sort all nodes, there are cycles
        if len(sorted_nodes) != len(all_nodes):
            logger.warning(
                f"Detected cycles in graph. {len(all_nodes) - len(sorted_nodes)} "
                f"nodes in cycles."
            )
            
            # Remove edges that cause cycles
            # Keep only edges where from comes before to in sorted order
            sorted_set = set(sorted_nodes)
            valid_edges = []
            removed = 0
            
            for edge in edges:
                # If both nodes are in sorted, check order
                if edge.from_event_id in sorted_set and edge.to_event_id in sorted_set:
                    from_idx = sorted_nodes.index(edge.from_event_id)
                    to_idx = sorted_nodes.index(edge.to_event_id)
                    if from_idx < to_idx:
                        valid_edges.append(edge)
                    else:
                        removed += 1
                elif edge.from_event_id in sorted_set:
                    # to_event is in a cycle, check if edge goes to belief
                    if edge.to_event_id == belief_id:
                        valid_edges.append(edge)
                    else:
                        removed += 1
                else:
                    removed += 1
            
            logger.info(f"Removed {removed} cycle-causing edges")
            return valid_edges
        
        return edges
    
    def _generate_explanation(
        self,
        event: EventNode,
        mechanism: MechanismType,
        confidence: float,
        direction: DirectionType
    ) -> str:
        """
        Generate rich explanation string for an edge.
        
        Includes:
        - Event title/description
        - Key actors
        - Direction with semantic meaning
        - Mechanism type
        - Source URL
        
        Args:
            event: Source event
            mechanism: Mechanism type
            confidence: Confidence score
            direction: Direction type
        
        Returns:
            Rich explanation string
        """
        # Get event title (prefer raw_title)
        title = event.raw_title or f"{event.action} {event.object}"
        # Truncate if too long
        if len(title) > 80:
            title = title[:77] + "..."
        
        # Get actors
        actors = ""
        if event.actors:
            actors = ", ".join(event.actors[:3])
            if len(event.actors) > 3:
                actors += f" +{len(event.actors) - 3} more"
        
        # Direction text with semantic meaning
        direction_text = {
            "positive": "increases likelihood of",
            "negative": "decreases likelihood of",
            "ambiguous": "has uncertain effect on"
        }.get(direction, "influences")
        
        # Mechanism description
        mechanism_text = mechanism.replace('_', ' ')
        
        # Build explanation
        parts = [f'"{title}"']
        
        if actors:
            parts.append(f"involving {actors}")
        
        parts.append(f"{direction_text} the outcome")
        parts.append(f"via {mechanism_text}")
        parts.append(f"(confidence: {confidence:.0%})")
        
        explanation = " ".join(parts)
        
        # Add source URL if available
        if event.url:
            explanation += f" | Source: {event.url}"
        
        return explanation
    
    def build(
        self,
        belief_event_id: str,
        depth: int = 2,
        max_events: int = 100,
        max_edges: int = 200
    ) -> BeliefGraph:
        """
        Build belief update graph for a target event.
        
        Args:
            belief_event_id: Event ID from registry
            depth: Graph depth (not currently used for filtering)
            max_events: Maximum number of events to include
            max_edges: Maximum number of edges
        
        Returns:
            Complete BeliefGraph
        """
        logger.info(f"Building belief graph for event: {belief_event_id}")
        
        # Step 1: Get event from registry and create BeliefNode
        event = self.registry.get_event(belief_event_id)
        if event is None:
            raise ValueError(f"Event not found: {belief_event_id}")
        
        belief = self._create_belief_node(event)
        logger.info(f"Created BeliefNode: {belief.belief_id}")
        
        # Step 2: Extract EventNodes from documents
        raw_event_nodes = self.event_extractor.extract_events_for_belief(
            belief_event_id,
            max_events=max_events * 2  # Extract more, then cluster
        )
        logger.info(f"Extracted {len(raw_event_nodes)} raw EventNodes")
        
        if not raw_event_nodes:
            logger.warning("No events extracted, returning empty graph")
            return BeliefGraph(
                belief_node=belief,
                event_nodes={},
                edges=[],
                depth=depth
            )
        
        # Step 2b: Cluster similar events to reduce duplicates
        clustered = self.event_clusterer.cluster_events(raw_event_nodes)
        logger.info(f"Clustered into {len(clustered)} unique events")
        
        # Use canonical events from clusters
        event_nodes = [c.canonical_event for c in clustered]
        
        # Update certainty with cluster corroboration
        for i, cluster in enumerate(clustered):
            if cluster.num_sources > 1:
                # Boost certainty for corroborated events
                event_nodes[i].certainty = cluster.cluster_confidence
        
        # Limit to max_events
        if len(event_nodes) > max_events:
            # Sort by certainty and keep top
            event_nodes.sort(key=lambda e: e.certainty, reverse=True)
            event_nodes = event_nodes[:max_events]
            logger.info(f"Limited to top {max_events} events by certainty")
        
        # Build event lookup
        events: Dict[str, EventNode] = {e.event_id: e for e in event_nodes}
        
        # Step 3: Generate candidate edges (rule-based, NO LLM)
        candidates = generate_candidates(event_nodes, belief)
        logger.info(f"Generated {len(candidates)} candidate edges")
        
        if not candidates:
            logger.warning("No candidate edges, returning graph with nodes only")
            return BeliefGraph(
                belief_node=belief,
                event_nodes=events,
                edges=[],
                depth=depth
            )
        
        # Limit candidates if too many
        if len(candidates) > max_edges * 2:
            logger.info(f"Limiting candidates to {max_edges * 2}")
            candidates = candidates[:max_edges * 2]
        
        # Step 4-6: Process candidates
        edges: List[BeliefEdge] = []
        
        for from_id, to_id in candidates:
            from_event = events.get(from_id)
            if not from_event:
                continue
            
            to_event = events.get(to_id) if to_id != belief.belief_id else None
            
            try:
                # Step 4: Classify mechanism (LLM-bounded)
                mechanism, mech_confidence = self.mechanism_classifier.classify(
                    from_event,
                    to_event,
                    belief if to_id == belief.belief_id else None
                )
                
                # Step 5: Score evidence (No LLM)
                evidence = self.evidence_scorer.score(
                    from_event,
                    to_id,
                    belief,
                    to_event
                )
                
                # Step 6: Calculate confidence
                confidence = self.confidence_calculator.calculate(
                    evidence,
                    from_event.event_type,
                    to_event.event_type if to_event else "belief"
                )
                
                # Apply threshold
                if not self.confidence_calculator.should_include(confidence):
                    continue
                
                # Infer direction and latency
                direction = self._infer_direction(from_event, belief)
                latency = self._infer_latency(from_event, to_id, events, belief)
                
                # Generate rich explanation with event context
                explanation = self._generate_explanation(
                    from_event, mechanism, confidence, direction
                )
                
                # Create edge
                edge = BeliefEdge(
                    edge_id=f"edge_{from_id[:12]}_{to_id[:12]}",
                    from_event_id=from_id,
                    to_event_id=to_id,
                    mechanism_type=mechanism,
                    direction=direction,
                    latency=latency,
                    confidence=round(confidence, 4),
                    evidence=evidence,
                    explanation=explanation
                )
                
                edges.append(edge)
                
            except Exception as e:
                logger.warning(f"Error processing edge {from_id} → {to_id}: {e}")
                continue
        
        logger.info(f"Created {len(edges)} edges after confidence filtering")
        
        # Step 7: Ensure DAG (remove cycles)
        edges = self._remove_cycles(edges, belief.belief_id)
        
        # Limit edges if still too many
        if len(edges) > max_edges:
            # Keep highest confidence edges
            edges = sorted(edges, key=lambda e: e.confidence, reverse=True)[:max_edges]
            logger.info(f"Limited to top {max_edges} edges by confidence")
        
        # Build final graph
        graph = BeliefGraph(
            belief_node=belief,
            event_nodes=events,
            edges=edges,
            depth=depth
        )
        
        logger.info(
            f"Built belief graph: {len(graph.event_nodes)} nodes, "
            f"{len(graph.edges)} edges"
        )
        
        return graph
    
    def get_ranked_upstream_events(
        self,
        graph: BeliefGraph,
        limit: int = 10
    ) -> List[Tuple[EventNode, float]]:
        """
        Get upstream events ranked by belief impact.
        
        Args:
            graph: Built graph
            limit: Maximum events to return
        
        Returns:
            List of (EventNode, impact_score) tuples
        """
        upstream = graph.get_upstream_events(limit=limit)
        
        # Calculate impact scores
        result = []
        for event in upstream:
            # Sum confidence of all edges from this event
            total_impact = sum(
                e.confidence for e in graph.edges
                if e.from_event_id == event.event_id
            )
            result.append((event, round(total_impact, 4)))
        
        return sorted(result, key=lambda x: x[1], reverse=True)
    
    def get_edge_explanations(
        self,
        graph: BeliefGraph
    ) -> List[Dict]:
        """
        Get explanations for all edges.
        
        Args:
            graph: Built graph
        
        Returns:
            List of edge explanation dictionaries
        """
        explanations = []
        
        for edge in sorted(graph.edges, key=lambda e: e.confidence, reverse=True):
            from_event = graph.event_nodes.get(edge.from_event_id)
            
            explanations.append({
                "edge_id": edge.edge_id,
                "from_event": from_event.action if from_event else edge.from_event_id,
                "to_event": edge.to_event_id,
                "mechanism": edge.mechanism_type,
                "direction": edge.direction,
                "confidence": edge.confidence,
                "explanation": edge.explanation
            })
        
        return explanations
