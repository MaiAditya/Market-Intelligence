"""
Evidence Scorer

Computes empirical evidence scores for belief graph edges.
NO LLM allowed in this step - only empirical/statistical methods.

Evidence components:
1. price_response: Normalized price change after event
2. volume_response: Abnormal volume following event
3. narrative_overlap: Entity/keyword overlap between events
4. historical_precedent: Similarity to past event patterns

All scores normalized to [0.0, 1.0].
"""

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from integrations.polymarket_client import PolymarketClient, get_polymarket_client
from pipeline.normalizer import DocumentNormalizer, NormalizedDocument
from belief_graph.models import EventNode, BeliefNode, EvidenceScores

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Get current UTC time as a naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class EvidenceScorer:
    """
    Computes empirical evidence scores for graph edges.
    
    Uses:
    - Polymarket API for price/volume data
    - Document entities for narrative overlap
    - Historical signal patterns for precedent
    
    NO LLM or ML models - purely empirical calculations.
    """
    
    def __init__(
        self,
        polymarket_client: Optional[PolymarketClient] = None,
        normalizer: Optional[DocumentNormalizer] = None,
        signals_dir: Optional[str] = None
    ):
        """
        Initialize evidence scorer.
        
        Args:
            polymarket_client: Client for Polymarket data
            normalizer: Document normalizer for entity access
            signals_dir: Directory containing signal files
        """
        logger.info("Initializing EvidenceScorer...")
        
        self.pm = polymarket_client or get_polymarket_client()
        self.normalizer = normalizer or DocumentNormalizer()
        
        if signals_dir is None:
            signals_dir = project_root / "data" / "signals"
        self.signals_dir = Path(signals_dir)
        
        # Cache for loaded documents
        self._doc_cache: Dict[str, NormalizedDocument] = {}
        
        # Historical patterns cache
        self._historical_patterns: Optional[Dict] = None
        
        logger.info("EvidenceScorer initialized")
    
    def score(
        self,
        event_a: EventNode,
        event_b_id: str,
        belief: BeliefNode,
        event_b: Optional[EventNode] = None
    ) -> EvidenceScores:
        """
        Compute evidence scores for an edge.
        
        Args:
            event_a: Source event
            event_b_id: Target event ID or belief ID
            belief: Target belief node
            event_b: Target event node (if event-to-event edge)
        
        Returns:
            EvidenceScores with all components
        """
        logger.debug(f"Scoring evidence for edge: {event_a.event_id} → {event_b_id}")
        
        # 1. Price Response (with event context)
        price_response = self._calculate_price_response(
            belief.polymarket_slug,
            event_a.timestamp,
            event=event_a
        )
        
        # 2. Volume Response (with event context)
        volume_response = self._calculate_volume_response(
            belief.polymarket_slug,
            event_a.timestamp,
            event=event_a
        )
        
        # 3. Narrative Overlap
        if event_b is not None:
            narrative_overlap = self._calculate_narrative_overlap(
                event_a,
                event_b
            )
        else:
            # For event → belief edges, check overlap with belief question
            narrative_overlap = self._calculate_belief_overlap(
                event_a,
                belief
            )
        
        # 4. Historical Precedent
        historical_precedent = self._calculate_historical_precedent(
            event_a.event_type,
            belief.event_id
        )
        
        return EvidenceScores(
            price_response=round(price_response, 4),
            volume_response=round(volume_response, 4),
            narrative_overlap=round(narrative_overlap, 4),
            historical_precedent=round(historical_precedent, 4)
        )
    
    def _calculate_price_response(
        self,
        polymarket_slug: str,
        event_timestamp: datetime,
        event: Optional[EventNode] = None,
        window_hours: int = 24
    ) -> float:
        """
        Estimate price response based on event characteristics.
        
        Since Polymarket API doesn't provide historical prices, we estimate
        based on:
        1. Event type impact potential
        2. Event recency
        3. Direction consistency with belief
        
        Args:
            polymarket_slug: Polymarket event slug
            event_timestamp: When the event occurred
            event: Source event node for context
            window_hours: Time window to measure response
        
        Returns:
            Normalized price response [0.0-1.0]
        """
        # Base impact by event type
        type_impact = {
            "policy": 0.60,  # Policy changes have high impact
            "legal": 0.65,   # Legal rulings are binding
            "economic": 0.50,
            "poll": 0.45,
            "signal": 0.55,
            "narrative": 0.35,
            "market": 0.50,
        }
        
        base_score = 0.40  # Default
        if event and event.event_type:
            base_score = type_impact.get(event.event_type, 0.40)
        
        # Recency adjustment
        now = _utc_now()
        try:
            time_since_event = (now - event_timestamp).total_seconds() / 3600
        except:
            time_since_event = 168  # Default to 1 week
        
        # Recent events have more price impact potential
        if time_since_event < 24:
            recency_factor = 1.2
        elif time_since_event < 72:
            recency_factor = 1.0
        elif time_since_event < 168:
            recency_factor = 0.8
        else:
            recency_factor = 0.6
        
        score = base_score * recency_factor
        
        # Certainty boost - higher certainty events have clearer price impact
        if event and event.certainty > 0.7:
            score *= 1.1
        
        return min(1.0, max(0.0, score))
    
    def _calculate_volume_response(
        self,
        polymarket_slug: str,
        event_timestamp: datetime,
        event: Optional[EventNode] = None,
        window_hours: int = 24
    ) -> float:
        """
        Estimate volume response based on event coverage and significance.
        
        Since we don't have historical volume data, we estimate based on:
        1. Source authority (more authoritative = more coverage)
        2. Actor significance (major actors = more volume)
        3. Event scope (global events = more volume)
        
        Args:
            polymarket_slug: Polymarket event slug
            event_timestamp: When the event occurred
            event: Source event node for context
            window_hours: Time window to measure response
        
        Returns:
            Normalized volume response [0.0-1.0]
        """
        base_score = 0.35  # Conservative default
        
        if not event:
            return base_score
        
        # Source authority boost
        source_lower = (event.source or "").lower()
        if any(x in source_lower for x in ["reuters", "bloomberg", "ap", "official"]):
            base_score += 0.20
        elif any(x in source_lower for x in ["bbc", "nytimes", "wsj", "ft.com"]):
            base_score += 0.15
        elif any(x in source_lower for x in [".gov", "europa.eu", ".edu"]):
            base_score += 0.15
        elif any(x in source_lower for x in ["techcrunch", "verge", "wired"]):
            base_score += 0.10
        
        # Actor significance boost
        major_actors = [
            "eu", "european", "congress", "senate", "parliament",
            "openai", "google", "microsoft", "anthropic", "meta",
            "biden", "trump", "commission", "court", "fda", "sec"
        ]
        if event.actors:
            actors_lower = [a.lower() for a in event.actors]
            actor_matches = sum(
                1 for actor in actors_lower
                for major in major_actors
                if major in actor
            )
            base_score += min(0.20, actor_matches * 0.05)
        
        # Scope boost
        scope_boost = {
            "global": 0.10,
            "national": 0.05,
            "local": 0.0
        }
        if event.scope:
            base_score += scope_boost.get(event.scope, 0.0)
        
        # Recency adjustment
        now = _utc_now()
        try:
            hours_since = (now - event_timestamp).total_seconds() / 3600
            if hours_since < 24:
                base_score *= 1.15
            elif hours_since > 168:
                base_score *= 0.85
        except:
            pass
        
        return min(1.0, max(0.0, base_score))
    
    def _calculate_narrative_overlap(
        self,
        event_a: EventNode,
        event_b: EventNode
    ) -> float:
        """
        Compute entity and keyword overlap between events.
        
        Uses Jaccard similarity on extracted entities.
        
        Args:
            event_a: Source event
            event_b: Target event
        
        Returns:
            Narrative overlap score [0.0-1.0]
        """
        # Get entities from source documents
        entities_a = self._get_entities_for_event(event_a)
        entities_b = self._get_entities_for_event(event_b)
        
        if not entities_a or not entities_b:
            # Fall back to actor overlap
            actors_a = set(a.lower() for a in event_a.actors)
            actors_b = set(a.lower() for a in event_b.actors)
            
            if not actors_a or not actors_b:
                return 0.2  # Low default
            
            intersection = actors_a & actors_b
            union = actors_a | actors_b
            
            return len(intersection) / len(union) if union else 0.0
        
        # Jaccard similarity on entities
        intersection = entities_a & entities_b
        union = entities_a | entities_b
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _calculate_belief_overlap(
        self,
        event: EventNode,
        belief: BeliefNode
    ) -> float:
        """
        Calculate overlap between event and belief question.
        
        Args:
            event: Source event
            belief: Target belief
        
        Returns:
            Overlap score [0.0-1.0]
        """
        # Get event entities
        event_entities = self._get_entities_for_event(event)
        
        # Extract keywords from belief question
        question_words = set(
            word.lower() for word in belief.question.split()
            if len(word) > 3  # Skip short words
        )
        
        # Remove common stop words
        stop_words = {
            "will", "the", "this", "that", "with", "from", "have", "been",
            "would", "could", "should", "what", "when", "where", "which",
            "there", "their", "about", "before", "after"
        }
        question_words -= stop_words
        
        if not event_entities and not event.actors:
            return 0.2
        
        # Combine entities and actors
        event_terms = event_entities | set(a.lower() for a in event.actors)
        
        # Add action words
        action_words = set(
            word.lower() for word in event.action.split()
            if len(word) > 3
        )
        event_terms |= action_words
        
        if not event_terms or not question_words:
            return 0.2
        
        # Calculate overlap
        intersection = event_terms & question_words
        
        # Use Dice coefficient (more lenient than Jaccard)
        dice = 2 * len(intersection) / (len(event_terms) + len(question_words))
        
        return min(1.0, dice * 1.5)  # Scale up slightly
    
    def _get_entities_for_event(self, event: EventNode) -> Set[str]:
        """
        Get normalized entity set for an event.
        
        Args:
            event: Event node
        
        Returns:
            Set of lowercase entity texts
        """
        if not event.source_doc_id:
            return set()
        
        # Check cache
        if event.source_doc_id in self._doc_cache:
            doc = self._doc_cache[event.source_doc_id]
        else:
            doc = self.normalizer.load(event.source_doc_id)
            if doc:
                self._doc_cache[event.source_doc_id] = doc
        
        if not doc:
            return set()
        
        return {
            e.get("text", "").lower()
            for e in doc.extracted_entities
            if e.get("text") and len(e.get("text", "")) > 2
        }
    
    def _calculate_historical_precedent(
        self,
        event_type: str,
        belief_event_id: str
    ) -> float:
        """
        Calculate historical precedent score.
        
        Based on past patterns of similar event types affecting outcomes.
        
        Args:
            event_type: Type of the source event
            belief_event_id: ID of the belief event
        
        Returns:
            Historical precedent score [0.0-1.0]
        """
        # Load historical patterns if not cached
        if self._historical_patterns is None:
            self._historical_patterns = self._load_historical_patterns()
        
        # Look up precedent for event type
        type_patterns = self._historical_patterns.get(event_type, {})
        
        if not type_patterns:
            # Return default based on event type reliability
            type_defaults = {
                "policy": 0.6,  # Policy events often have clear impact
                "legal": 0.7,   # Legal rulings are binding
                "economic": 0.5,
                "poll": 0.4,
                "narrative": 0.3,
                "market": 0.5,
                "signal": 0.4
            }
            return type_defaults.get(event_type, 0.4)
        
        # Calculate based on historical accuracy
        accuracy = type_patterns.get("accuracy", 0.5)
        sample_size = type_patterns.get("sample_size", 1)
        
        # Adjust confidence based on sample size
        confidence_factor = min(1.0, sample_size / 10)
        
        return accuracy * confidence_factor
    
    def _load_historical_patterns(self) -> Dict:
        """
        Load historical patterns from signals.
        
        Analyzes past signals to find patterns.
        
        Returns:
            Dictionary of event_type to pattern statistics
        """
        patterns: Dict[str, Dict] = {}
        
        if not self.signals_dir.exists():
            return patterns
        
        # Analyze signal files
        type_counts: Dict[str, int] = {}
        type_correct: Dict[str, int] = {}
        
        for signal_file in self.signals_dir.glob("*.json"):
            try:
                with open(signal_file, 'r', encoding='utf-8') as f:
                    signal = json.load(f)
                
                signal_type = signal.get("signal_type", "")
                direction = signal.get("direction", "")
                confidence = signal.get("confidence", 0)
                
                # Map signal types to event types for pattern learning
                # (This is a simplified mapping)
                if signal_type in ("official_confirmation", "executive_statement"):
                    event_type = "signal"
                elif signal_type == "regulation_update":
                    event_type = "policy"
                else:
                    event_type = "narrative"
                
                type_counts[event_type] = type_counts.get(event_type, 0) + 1
                
                # Assume high-confidence signals were "correct"
                if confidence > 0.6:
                    type_correct[event_type] = type_correct.get(event_type, 0) + 1
                    
            except Exception:
                continue
        
        # Calculate accuracy for each type
        for event_type, count in type_counts.items():
            correct = type_correct.get(event_type, 0)
            patterns[event_type] = {
                "accuracy": correct / count if count > 0 else 0.5,
                "sample_size": count
            }
        
        return patterns
    
    def score_batch(
        self,
        edges: List[Tuple[EventNode, str, BeliefNode, Optional[EventNode]]]
    ) -> List[EvidenceScores]:
        """
        Score evidence for multiple edges.
        
        Args:
            edges: List of (event_a, event_b_id, belief, event_b) tuples
        
        Returns:
            List of EvidenceScores
        """
        results = []
        
        for event_a, event_b_id, belief, event_b in edges:
            try:
                scores = self.score(event_a, event_b_id, belief, event_b)
                results.append(scores)
            except Exception as e:
                logger.warning(f"Error scoring edge {event_a.event_id} → {event_b_id}: {e}")
                # Return default scores
                results.append(EvidenceScores(
                    price_response=0.3,
                    volume_response=0.3,
                    narrative_overlap=0.2,
                    historical_precedent=0.3
                ))
        
        return results
