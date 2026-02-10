"""
Candidate Edge Generator

Generates candidate edges for the belief update graph using ONLY rules.
NO ML or LLM allowed in this step.

Candidate edges (A → B) are generated if:
1. A.timestamp < B.timestamp
2. event_type(A) is allowed to precede event_type(B)
"""

import logging
from typing import Dict, List, Set, Tuple

from belief_graph.models import EventNode, BeliefNode, EventType

logger = logging.getLogger(__name__)


# Allowed predecessor rules as specified in the plan
# Maps event_type to list of event_types it can influence
ALLOWED_PREDECESSORS: Dict[EventType, List[str]] = {
    "policy": ["economic", "legal", "market", "belief"],
    "legal": ["policy", "market", "belief"],
    "poll": ["narrative", "market", "belief"],
    "narrative": ["market", "belief", "signal"],
    "economic": ["market", "belief"],
    "signal": ["narrative", "market", "belief"],
    "market": ["belief"],
}

# Reverse mapping for faster lookup
# Maps event_type to set of event_types that can precede it
ALLOWED_SUCCESSORS: Dict[str, Set[EventType]] = {}
for pred_type, successor_types in ALLOWED_PREDECESSORS.items():
    for succ_type in successor_types:
        if succ_type not in ALLOWED_SUCCESSORS:
            ALLOWED_SUCCESSORS[succ_type] = set()
        ALLOWED_SUCCESSORS[succ_type].add(pred_type)


def can_influence(from_type: EventType, to_type: str) -> bool:
    """
    Check if from_type is allowed to influence to_type.
    
    Uses the predefined predecessor rules - NO ML.
    
    Args:
        from_type: Source event type
        to_type: Target event type or "belief"
    
    Returns:
        True if from_type can influence to_type
    """
    allowed = ALLOWED_PREDECESSORS.get(from_type, [])
    return to_type in allowed


def generate_candidates(
    events: List[EventNode],
    belief: BeliefNode,
    max_candidates_per_event: int = 20,
    max_total_candidates: int = 500
) -> List[Tuple[str, str]]:
    """
    Generate candidate edges using ONLY rules.
    
    Rules:
    1. A.timestamp < B.timestamp
    2. event_type(A) is allowed to precede event_type(B)
    
    NO ML OR LLM HERE.
    
    Args:
        events: List of EventNodes
        belief: Target BeliefNode
        max_candidates_per_event: Maximum edges from each event
        max_total_candidates: Maximum total candidates
    
    Returns:
        List of (from_event_id, to_event_id) tuples
    """
    logger.info(
        f"Generating candidate edges from {len(events)} events "
        f"to belief {belief.belief_id}"
    )
    
    candidates: List[Tuple[str, str]] = []
    
    # Sort events by timestamp
    sorted_events = sorted(
        [e for e in events if e.timestamp is not None],
        key=lambda e: e.timestamp
    )
    
    logger.debug(f"Sorted {len(sorted_events)} events by timestamp")
    
    # Track candidates per event to avoid explosion
    candidates_per_event: Dict[str, int] = {}
    
    for i, event_a in enumerate(sorted_events):
        event_count = 0
        
        # Check against all later events
        for event_b in sorted_events[i + 1:]:
            # Rule 1: A.timestamp < B.timestamp (guaranteed by sort)
            
            # Rule 2: event_type(A) is allowed to precede event_type(B)
            if can_influence(event_a.event_type, event_b.event_type):
                candidates.append((event_a.event_id, event_b.event_id))
                event_count += 1
                
                if event_count >= max_candidates_per_event:
                    break
        
        # Check against belief node
        if belief.resolution_time and event_a.timestamp < belief.resolution_time:
            if can_influence(event_a.event_type, "belief"):
                candidates.append((event_a.event_id, belief.belief_id))
                event_count += 1
        
        candidates_per_event[event_a.event_id] = event_count
        
        # Early termination if we have enough candidates
        if len(candidates) >= max_total_candidates:
            logger.warning(
                f"Reached maximum candidate limit ({max_total_candidates}), "
                f"truncating"
            )
            break
    
    # Log statistics
    total_event_to_event = sum(
        1 for from_id, to_id in candidates 
        if to_id != belief.belief_id
    )
    total_to_belief = len(candidates) - total_event_to_event
    
    logger.info(
        f"Generated {len(candidates)} candidate edges: "
        f"{total_event_to_event} event→event, {total_to_belief} event→belief"
    )
    
    return candidates


def generate_candidates_with_metadata(
    events: List[EventNode],
    belief: BeliefNode,
    max_candidates_per_event: int = 20,
    max_total_candidates: int = 500
) -> List[Dict]:
    """
    Generate candidate edges with additional metadata.
    
    Similar to generate_candidates but includes type information.
    
    Args:
        events: List of EventNodes
        belief: Target BeliefNode
        max_candidates_per_event: Maximum edges from each event
        max_total_candidates: Maximum total candidates
    
    Returns:
        List of candidate edge dictionaries
    """
    # Build event lookup
    event_lookup: Dict[str, EventNode] = {e.event_id: e for e in events}
    
    # Get basic candidates
    raw_candidates = generate_candidates(
        events,
        belief,
        max_candidates_per_event,
        max_total_candidates
    )
    
    # Add metadata
    candidates_with_meta = []
    for from_id, to_id in raw_candidates:
        from_event = event_lookup.get(from_id)
        if not from_event:
            continue
        
        if to_id == belief.belief_id:
            to_type = "belief"
            time_diff = None
            if belief.resolution_time and from_event.timestamp:
                time_diff = (belief.resolution_time - from_event.timestamp).total_seconds()
        else:
            to_event = event_lookup.get(to_id)
            if not to_event:
                continue
            to_type = to_event.event_type
            time_diff = None
            if to_event.timestamp and from_event.timestamp:
                time_diff = (to_event.timestamp - from_event.timestamp).total_seconds()
        
        candidates_with_meta.append({
            "from_event_id": from_id,
            "to_event_id": to_id,
            "from_type": from_event.event_type,
            "to_type": to_type,
            "time_diff_seconds": time_diff,
            "from_certainty": from_event.certainty
        })
    
    return candidates_with_meta


def filter_candidates_by_time_window(
    candidates: List[Tuple[str, str]],
    events: Dict[str, EventNode],
    belief: BeliefNode,
    min_hours: float = 0.0,
    max_hours: float = 168.0  # 1 week
) -> List[Tuple[str, str]]:
    """
    Filter candidates by time window between events.
    
    Removes edges where time difference is outside the window.
    
    Args:
        candidates: List of candidate edges
        events: Dictionary of event_id to EventNode
        belief: Target BeliefNode
        min_hours: Minimum time difference in hours
        max_hours: Maximum time difference in hours
    
    Returns:
        Filtered list of candidates
    """
    min_seconds = min_hours * 3600
    max_seconds = max_hours * 3600
    
    filtered = []
    
    for from_id, to_id in candidates:
        from_event = events.get(from_id)
        if not from_event or not from_event.timestamp:
            continue
        
        if to_id == belief.belief_id:
            if not belief.resolution_time:
                filtered.append((from_id, to_id))
                continue
            diff = (belief.resolution_time - from_event.timestamp).total_seconds()
        else:
            to_event = events.get(to_id)
            if not to_event or not to_event.timestamp:
                continue
            diff = (to_event.timestamp - from_event.timestamp).total_seconds()
        
        if min_seconds <= diff <= max_seconds:
            filtered.append((from_id, to_id))
    
    logger.debug(
        f"Filtered candidates by time window [{min_hours}h, {max_hours}h]: "
        f"{len(candidates)} → {len(filtered)}"
    )
    
    return filtered


def get_predecessor_stats(events: List[EventNode]) -> Dict[str, Dict]:
    """
    Get statistics about event types and potential edges.
    
    Useful for debugging and understanding the graph structure.
    
    Args:
        events: List of EventNodes
    
    Returns:
        Statistics dictionary
    """
    type_counts: Dict[str, int] = {}
    for event in events:
        etype = event.event_type
        type_counts[etype] = type_counts.get(etype, 0) + 1
    
    # Calculate potential edges
    potential_edges: Dict[str, int] = {}
    for from_type, to_types in ALLOWED_PREDECESSORS.items():
        from_count = type_counts.get(from_type, 0)
        for to_type in to_types:
            to_count = type_counts.get(to_type, 0)
            edge_key = f"{from_type}→{to_type}"
            # Rough estimate: each from can connect to half of to (later in time)
            potential_edges[edge_key] = from_count * (to_count // 2)
    
    return {
        "type_counts": type_counts,
        "potential_edges": potential_edges,
        "total_events": len(events)
    }
