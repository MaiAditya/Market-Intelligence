"""
Confidence Calculator

Calculates edge confidence from evidence scores and applies threshold filtering.

The confidence formula is:
    confidence = 0.35 * price_response +
                 0.25 * volume_response +
                 0.20 * narrative_overlap +
                 0.20 * historical_precedent

Edges with confidence < 0.3 are discarded.
"""

import logging
from typing import List, Optional, Tuple

from belief_graph.models import BeliefEdge, EvidenceScores

logger = logging.getLogger(__name__)


# Default weights as specified in the plan
DEFAULT_WEIGHTS = {
    "price_response": 0.35,
    "volume_response": 0.25,
    "narrative_overlap": 0.20,
    "historical_precedent": 0.20
}

# Default confidence threshold
CONFIDENCE_THRESHOLD = 0.3


def calculate_confidence(
    evidence: EvidenceScores,
    weights: Optional[dict] = None
) -> float:
    """
    Calculate edge confidence from evidence scores.
    
    Formula:
        confidence = 0.35 * price_response +
                     0.25 * volume_response +
                     0.20 * narrative_overlap +
                     0.20 * historical_precedent
    
    Args:
        evidence: Evidence scores
        weights: Optional custom weights
    
    Returns:
        Confidence score [0.0-1.0]
    """
    w = weights or DEFAULT_WEIGHTS
    
    confidence = (
        w.get("price_response", 0.35) * evidence.price_response +
        w.get("volume_response", 0.25) * evidence.volume_response +
        w.get("narrative_overlap", 0.20) * evidence.narrative_overlap +
        w.get("historical_precedent", 0.20) * evidence.historical_precedent
    )
    
    # Ensure in valid range
    return min(1.0, max(0.0, confidence))


def passes_threshold(
    confidence: float,
    threshold: Optional[float] = None
) -> bool:
    """
    Check if confidence passes the threshold.
    
    Args:
        confidence: Calculated confidence score
        threshold: Optional custom threshold
    
    Returns:
        True if confidence >= threshold
    """
    threshold = threshold if threshold is not None else CONFIDENCE_THRESHOLD
    return confidence >= threshold


def filter_by_confidence(
    edges: List[BeliefEdge],
    threshold: Optional[float] = None
) -> List[BeliefEdge]:
    """
    Filter edges by confidence threshold.
    
    Removes edges with confidence < threshold.
    
    Args:
        edges: List of edges
        threshold: Optional custom threshold
    
    Returns:
        Filtered list of edges
    """
    threshold = threshold if threshold is not None else CONFIDENCE_THRESHOLD
    
    filtered = [e for e in edges if e.confidence >= threshold]
    
    removed = len(edges) - len(filtered)
    if removed > 0:
        logger.info(
            f"Filtered {removed} edges below confidence threshold {threshold} "
            f"({len(filtered)} remaining)"
        )
    
    return filtered


def calculate_and_filter(
    evidence_list: List[EvidenceScores],
    edge_data: List[dict],
    threshold: Optional[float] = None,
    weights: Optional[dict] = None
) -> List[Tuple[float, dict, bool]]:
    """
    Calculate confidence for multiple edges and filter.
    
    Args:
        evidence_list: List of evidence scores
        edge_data: List of edge metadata dicts
        threshold: Optional custom threshold
        weights: Optional custom weights
    
    Returns:
        List of (confidence, edge_data, passed) tuples
    """
    threshold = threshold if threshold is not None else CONFIDENCE_THRESHOLD
    
    results = []
    
    for evidence, data in zip(evidence_list, edge_data):
        confidence = calculate_confidence(evidence, weights)
        passed = confidence >= threshold
        results.append((confidence, data, passed))
    
    # Log statistics
    total = len(results)
    passed_count = sum(1 for _, _, passed in results if passed)
    
    if total > 0:
        logger.debug(
            f"Confidence calculation: {passed_count}/{total} edges passed "
            f"threshold {threshold} ({100*passed_count/total:.1f}%)"
        )
    
    return results


def get_confidence_stats(edges: List[BeliefEdge]) -> dict:
    """
    Get statistics about edge confidences.
    
    Args:
        edges: List of edges
    
    Returns:
        Statistics dictionary
    """
    if not edges:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "above_threshold": 0,
            "below_threshold": 0
        }
    
    confidences = [e.confidence for e in edges]
    sorted_conf = sorted(confidences)
    
    return {
        "count": len(edges),
        "min": min(confidences),
        "max": max(confidences),
        "mean": sum(confidences) / len(confidences),
        "median": sorted_conf[len(sorted_conf) // 2],
        "above_threshold": sum(1 for c in confidences if c >= CONFIDENCE_THRESHOLD),
        "below_threshold": sum(1 for c in confidences if c < CONFIDENCE_THRESHOLD)
    }


def adjust_weights_for_event_type(
    from_event_type: str,
    to_type: str
) -> dict:
    """
    Adjust weights based on event types.
    
    Different event type pairs may have different evidence importance.
    
    Args:
        from_event_type: Source event type
        to_type: Target event type or "belief"
    
    Returns:
        Adjusted weights dictionary
    """
    weights = DEFAULT_WEIGHTS.copy()
    
    # Policy and legal events have stronger precedent
    if from_event_type in ("policy", "legal"):
        weights["historical_precedent"] = 0.30
        weights["price_response"] = 0.30
    
    # Narrative events rely more on narrative overlap
    if from_event_type == "narrative":
        weights["narrative_overlap"] = 0.35
        weights["historical_precedent"] = 0.15
    
    # Market events emphasize price/volume
    if from_event_type == "market":
        weights["price_response"] = 0.45
        weights["volume_response"] = 0.30
        weights["narrative_overlap"] = 0.10
        weights["historical_precedent"] = 0.15
    
    # Normalize weights to sum to 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    
    return weights


class ConfidenceCalculator:
    """
    Stateful confidence calculator with configurable thresholds.
    """
    
    def __init__(
        self,
        threshold: float = CONFIDENCE_THRESHOLD,
        weights: Optional[dict] = None,
        adaptive_weights: bool = True
    ):
        """
        Initialize calculator.
        
        Args:
            threshold: Confidence threshold for filtering
            weights: Custom weights for evidence components
            adaptive_weights: Whether to adjust weights per event type
        """
        self.threshold = threshold
        self.base_weights = weights or DEFAULT_WEIGHTS.copy()
        self.adaptive_weights = adaptive_weights
    
    def calculate(
        self,
        evidence: EvidenceScores,
        from_event_type: Optional[str] = None,
        to_type: Optional[str] = None
    ) -> float:
        """
        Calculate confidence score.
        
        Args:
            evidence: Evidence scores
            from_event_type: Optional source event type for adaptive weights
            to_type: Optional target type for adaptive weights
        
        Returns:
            Confidence score
        """
        if self.adaptive_weights and from_event_type:
            weights = adjust_weights_for_event_type(
                from_event_type,
                to_type or "belief"
            )
        else:
            weights = self.base_weights
        
        return calculate_confidence(evidence, weights)
    
    def should_include(self, confidence: float) -> bool:
        """Check if edge should be included based on confidence."""
        return confidence >= self.threshold
    
    def filter_edges(self, edges: List[BeliefEdge]) -> List[BeliefEdge]:
        """Filter edges by threshold."""
        return filter_by_confidence(edges, self.threshold)
    
    def get_stats(self, edges: List[BeliefEdge]) -> dict:
        """Get confidence statistics."""
        stats = get_confidence_stats(edges)
        stats["threshold"] = self.threshold
        return stats
