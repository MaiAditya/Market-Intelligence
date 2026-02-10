"""
Mechanism Classifier

Classifies the mechanism type by which one event could influence belief in another.
Uses sentence-transformers for zero-shot classification via semantic similarity.

This is the ONLY step where LLM-like inference is allowed, but strictly bounded:
- Outputs exactly one mechanism type from the predefined list
- Does NOT assert truth or strength
- Only classifies the TYPE of potential influence

Mechanism types:
- legal_constraint: Legal/regulatory requirements that constrain or enable outcomes
- economic_impact: Financial/economic effects that change incentives or resources
- signaling: Information signals revealing intentions, capabilities, or state
- expectation_shift: Changes in market expectations about future outcomes
- narrative_amplification: Media/social amplification affecting perception
- liquidity_reaction: Market liquidity changes affecting price discovery
- coordination_effect: Coordination among actors affecting collective outcomes
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.model_manager import get_model_manager
from models.semantic_relevance import SemanticRelevanceScorer
from belief_graph.models import (
    EventNode,
    BeliefNode,
    MechanismType,
    MECHANISM_DESCRIPTIONS
)

logger = logging.getLogger(__name__)


# Extended mechanism descriptions for better classification
MECHANISM_CLASSIFICATION_PROMPTS: Dict[MechanismType, str] = {
    "legal_constraint": (
        "Legal or regulatory constraint that limits or enables outcomes. "
        "Includes court rulings, regulatory requirements, compliance mandates, "
        "legal precedents, legislation, policy enforcement, and legal barriers. "
        "The influence comes from binding legal authority or regulatory power."
    ),
    "economic_impact": (
        "Economic or financial impact that changes incentives, resources, or costs. "
        "Includes funding changes, investment decisions, revenue effects, market valuations, "
        "budget constraints, pricing effects, and financial dependencies. "
        "The influence comes from economic/financial consequences."
    ),
    "signaling": (
        "Information signal that reveals intentions, capabilities, or current state. "
        "Includes official announcements, progress updates, capability demonstrations, "
        "leaked information, insider signals, and intention revelations. "
        "The influence comes from new information reducing uncertainty."
    ),
    "expectation_shift": (
        "Shift in market or public expectations about future outcomes. "
        "Includes prediction updates, timeline revisions, probability adjustments, "
        "forecast changes, and belief updates based on new evidence. "
        "The influence comes from changed expectations about what will happen."
    ),
    "narrative_amplification": (
        "Media or social amplification of narratives affecting public perception. "
        "Includes viral coverage, opinion pieces, social media trends, "
        "narrative framing, public discourse, and perception shaping. "
        "The influence comes from amplified attention and narrative momentum."
    ),
    "liquidity_reaction": (
        "Market liquidity or trading dynamics affecting price discovery. "
        "Includes volume spikes, liquidity changes, market maker behavior, "
        "price impact, order flow, and market structure effects. "
        "The influence comes from market microstructure and trading dynamics."
    ),
    "coordination_effect": (
        "Coordination among actors affecting collective outcomes. "
        "Includes collective action, coalition building, stakeholder alignment, "
        "industry consensus, coordinated responses, and group dynamics. "
        "The influence comes from multiple actors aligning behavior."
    )
}

# Event type to likely mechanism mapping for faster inference
EVENT_TYPE_MECHANISM_HINTS: Dict[str, List[MechanismType]] = {
    "policy": ["legal_constraint", "expectation_shift", "signaling"],
    "legal": ["legal_constraint", "economic_impact"],
    "economic": ["economic_impact", "expectation_shift", "liquidity_reaction"],
    "poll": ["expectation_shift", "narrative_amplification"],
    "narrative": ["narrative_amplification", "expectation_shift", "signaling"],
    "market": ["liquidity_reaction", "expectation_shift", "signaling"],
    "signal": ["signaling", "expectation_shift", "narrative_amplification"],
}


class MechanismClassifier:
    """
    Classifies mechanism type for belief influence edges.
    
    Uses zero-shot classification via semantic similarity to mechanism descriptions.
    Does NOT assert truth or strength - only classifies mechanism type.
    """
    
    def __init__(self, use_hints: bool = True):
        """
        Initialize mechanism classifier.
        
        Args:
            use_hints: Whether to use event type hints to narrow classification
        """
        logger.info("Initializing MechanismClassifier...")
        self.model_manager = get_model_manager()
        self.relevance_scorer = SemanticRelevanceScorer()
        self.use_hints = use_hints
        
        # Cache mechanism embeddings
        self._mechanism_embeddings: Optional[np.ndarray] = None
        self._mechanism_order: List[MechanismType] = list(MECHANISM_CLASSIFICATION_PROMPTS.keys())
        
        logger.info("MechanismClassifier initialized")
    
    def _get_mechanism_embeddings(self) -> Optional[np.ndarray]:
        """Get or compute mechanism description embeddings."""
        if self._mechanism_embeddings is not None:
            return self._mechanism_embeddings
        
        descriptions = [
            MECHANISM_CLASSIFICATION_PROMPTS[mech]
            for mech in self._mechanism_order
        ]
        
        self._mechanism_embeddings = self.relevance_scorer.encode(descriptions)
        return self._mechanism_embeddings
    
    def _construct_edge_description(
        self,
        event_a: EventNode,
        event_b_info: str
    ) -> str:
        """
        Construct a description of the edge for classification.
        
        Args:
            event_a: Source event
            event_b_info: Description of target (event or belief)
        
        Returns:
            Edge description text
        """
        actors_str = ", ".join(event_a.actors[:3]) if event_a.actors else "Unknown actors"
        
        description = (
            f"Event A: {event_a.action} by {actors_str}. "
            f"Type: {event_a.event_type}. "
            f"Target: {event_b_info}. "
            f"How does Event A influence belief about the target?"
        )
        
        return description
    
    def classify(
        self,
        event_a: EventNode,
        event_b: Optional[EventNode] = None,
        belief: Optional[BeliefNode] = None
    ) -> Tuple[MechanismType, float]:
        """
        Classify mechanism type for an edge.
        
        Returns exactly one mechanism type from the predefined list.
        Does NOT assert truth or strength.
        
        Args:
            event_a: Source event
            event_b: Target event (if event-to-event edge)
            belief: Target belief (if event-to-belief edge)
        
        Returns:
            Tuple of (mechanism_type, confidence)
        """
        # Construct target description
        if belief is not None:
            target_info = f"Belief: {belief.question}"
            target_type = "belief"
        elif event_b is not None:
            target_info = f"Event: {event_b.action} ({event_b.event_type})"
            target_type = event_b.event_type
        else:
            raise ValueError("Must provide either event_b or belief")
        
        # Get mechanism embeddings
        mech_embeddings = self._get_mechanism_embeddings()
        
        if mech_embeddings is None:
            # Fallback to hint-based classification
            return self._hint_based_classify(event_a.event_type, target_type)
        
        # Construct edge description
        edge_desc = self._construct_edge_description(event_a, target_info)
        
        # Encode edge description
        edge_embedding = self.relevance_scorer.encode([edge_desc])
        
        if edge_embedding is None:
            return self._hint_based_classify(event_a.event_type, target_type)
        
        edge_embedding = edge_embedding[0]
        
        # Compute similarity to each mechanism
        scores: Dict[MechanismType, float] = {}
        for i, mech_type in enumerate(self._mechanism_order):
            similarity = self.relevance_scorer.cosine_similarity(
                edge_embedding,
                mech_embeddings[i]
            )
            # Normalize to 0-1 range
            normalized = max(0.0, min(1.0, (similarity + 1) / 2))
            scores[mech_type] = normalized
        
        # Apply hints if enabled
        if self.use_hints:
            hints = EVENT_TYPE_MECHANISM_HINTS.get(event_a.event_type, [])
            for mech in hints:
                if mech in scores:
                    scores[mech] *= 1.2  # Boost hinted mechanisms
        
        # Find best mechanism
        best_mechanism = max(scores.items(), key=lambda x: x[1])
        
        logger.debug(
            f"Classified mechanism: {best_mechanism[0]} "
            f"(confidence={best_mechanism[1]:.3f}) "
            f"for {event_a.event_type} → {target_type}"
        )
        
        return best_mechanism[0], best_mechanism[1]
    
    def _hint_based_classify(
        self,
        from_type: str,
        to_type: str
    ) -> Tuple[MechanismType, float]:
        """
        Fallback classification using event type hints.
        
        Args:
            from_type: Source event type
            to_type: Target event/belief type
        
        Returns:
            Tuple of (mechanism_type, confidence)
        """
        hints = EVENT_TYPE_MECHANISM_HINTS.get(from_type, [])
        
        if hints:
            # Return first hint with moderate confidence
            return hints[0], 0.5
        
        # Default fallback
        return "signaling", 0.3
    
    def classify_batch(
        self,
        edges: List[Tuple[EventNode, Optional[EventNode], Optional[BeliefNode]]]
    ) -> List[Tuple[MechanismType, float]]:
        """
        Classify mechanism types for multiple edges.
        
        Args:
            edges: List of (event_a, event_b, belief) tuples
        
        Returns:
            List of (mechanism_type, confidence) tuples
        """
        results = []
        for event_a, event_b, belief in edges:
            try:
                mechanism, confidence = self.classify(event_a, event_b, belief)
                results.append((mechanism, confidence))
            except Exception as e:
                logger.warning(f"Classification failed: {e}")
                results.append(("signaling", 0.3))
        
        return results
    
    def get_all_mechanism_scores(
        self,
        event_a: EventNode,
        event_b: Optional[EventNode] = None,
        belief: Optional[BeliefNode] = None
    ) -> Dict[MechanismType, float]:
        """
        Get scores for all mechanism types.
        
        Useful for debugging and analysis.
        
        Args:
            event_a: Source event
            event_b: Target event
            belief: Target belief
        
        Returns:
            Dictionary of mechanism type to score
        """
        # Construct target description
        if belief is not None:
            target_info = f"Belief: {belief.question}"
        elif event_b is not None:
            target_info = f"Event: {event_b.action} ({event_b.event_type})"
        else:
            raise ValueError("Must provide either event_b or belief")
        
        # Get mechanism embeddings
        mech_embeddings = self._get_mechanism_embeddings()
        
        if mech_embeddings is None:
            # Return uniform low scores
            return {mech: 0.3 for mech in self._mechanism_order}
        
        # Construct and encode edge description
        edge_desc = self._construct_edge_description(event_a, target_info)
        edge_embedding = self.relevance_scorer.encode([edge_desc])
        
        if edge_embedding is None:
            return {mech: 0.3 for mech in self._mechanism_order}
        
        edge_embedding = edge_embedding[0]
        
        # Compute all scores
        scores: Dict[MechanismType, float] = {}
        for i, mech_type in enumerate(self._mechanism_order):
            similarity = self.relevance_scorer.cosine_similarity(
                edge_embedding,
                mech_embeddings[i]
            )
            normalized = max(0.0, min(1.0, (similarity + 1) / 2))
            scores[mech_type] = round(normalized, 4)
        
        return scores
