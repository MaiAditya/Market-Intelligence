"""
Temporal Decay Module

Applies time-weighted decay to edge confidence based on event type and time delta.
Uses exponential decay with event-type specific half-lives.
"""

import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import networkx as nx

from .logging_utils import setup_graph_logger, log_edge_update, log_summary


# Half-life values in hours for different event types
HALF_LIFE = {
    "policy": 168,      # 7 days
    "legal": 168,       # 7 days
    "economic": 120,    # 5 days
    "poll": 72,         # 3 days
    "narrative": 48,    # 2 days
    "default": 96       # 4 days
}


def calculate_time_weight(
    delta_t_hours: float,
    event_type: str
) -> float:
    """
    Calculate exponential time decay weight.
    
    Formula: time_weight = exp(-delta_t / tau)
    where tau is the half-life for the event type.
    
    Args:
        delta_t_hours: Time difference in hours
        event_type: Type of event (policy, legal, economic, poll, narrative)
    
    Returns:
        Time weight in range [0, 1]
    """
    tau = HALF_LIFE.get(event_type, HALF_LIFE["default"])
    
    if delta_t_hours < 0:
        # Future events get no decay
        return 1.0
    
    time_weight = math.exp(-delta_t_hours / tau)
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, time_weight))


def apply_temporal_decay(
    graph: nx.DiGraph,
    events: Dict[str, Dict],
    beliefs: Dict[str, Dict],
    output_dir: Optional[Path] = None
) -> Tuple[nx.DiGraph, Dict]:
    """
    Apply temporal decay to all edges in the graph.
    
    For each edge, calculates time difference between belief and event,
    applies exponential decay based on event type, and updates confidence.
    
    Args:
        graph: NetworkX directed graph with edges
        events: Dictionary mapping event_id to event data (must have 'timestamp', 'event_type')
        beliefs: Dictionary mapping belief_id to belief data (must have 'timestamp')
        output_dir: Optional directory for logging output
    
    Returns:
        Tuple of (updated_graph, summary_stats)
    """
    if output_dir is None:
        output_dir = Path("data/graph_logs/temporal_decay")
    else:
        output_dir = Path(output_dir)
    
    logger = setup_graph_logger("temporal_decay", output_dir)
    logger.info(f"Starting temporal decay on graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Prepare output files
    edge_log_file = output_dir / "edge_adjustments.jsonl"
    summary_file = output_dir / "summary.json"
    
    # Clear previous logs
    if edge_log_file.exists():
        edge_log_file.unlink()
    
    edges_processed = 0
    edges_updated = 0
    edges_pruned = 0
    total_confidence_before = 0.0
    total_confidence_after = 0.0
    
    edges_to_remove = []
    
    for source, target, edge_data in graph.edges(data=True):
        edges_processed += 1
        
        # Get event and belief data
        event_id = edge_data.get('from_event_id')
        belief_id = target  # Assuming target is belief node
        
        if not event_id or event_id not in events:
            logger.warning(f"Edge {source} -> {target}: Missing event data for {event_id}")
            continue
        
        if belief_id not in beliefs:
            logger.warning(f"Edge {source} -> {target}: Missing belief data for {belief_id}")
            continue
        
        event = events[event_id]
        belief = beliefs[belief_id]
        
        # Get timestamps
        event_timestamp = event.get('timestamp')
        belief_timestamp = belief.get('timestamp')
        
        if not event_timestamp or not belief_timestamp:
            logger.warning(f"Edge {source} -> {target}: Missing timestamps")
            continue
        
        # Parse timestamps if they're strings
        if isinstance(event_timestamp, str):
            event_timestamp = datetime.fromisoformat(event_timestamp.replace('Z', '+00:00'))
        if isinstance(belief_timestamp, str):
            belief_timestamp = datetime.fromisoformat(belief_timestamp.replace('Z', '+00:00'))
        
        # Calculate time delta in hours
        delta_t = belief_timestamp - event_timestamp
        delta_t_hours = delta_t.total_seconds() / 3600
        
        # Get event type
        event_type = event.get('event_type', 'default')
        
        # Calculate time weight
        time_weight = calculate_time_weight(delta_t_hours, event_type)
        
        # Get current confidence
        confidence_before = edge_data.get('confidence', 0.0)
        total_confidence_before += confidence_before
        
        # Apply decay
        confidence_after = confidence_before * time_weight
        
        # Clamp to [0, 1]
        confidence_after = max(0.0, min(1.0, confidence_after))
        total_confidence_after += confidence_after
        
        # Update edge
        if confidence_after != confidence_before:
            edges_updated += 1
            edge_data['confidence'] = confidence_after
            edge_data['temporal_decay_applied'] = True
            edge_data['time_weight'] = time_weight
        
        # Mark for pruning if below threshold
        if confidence_after < 0.3:
            edges_to_remove.append((source, target))
            edges_pruned += 1
        
        # Log edge update
        log_data = {
            "edge_id": f"{source}->{target}",
            "from_event_id": event_id,
            "to_belief_id": belief_id,
            "delta_t_hours": round(delta_t_hours, 2),
            "event_type": event_type,
            "tau": HALF_LIFE.get(event_type, HALF_LIFE["default"]),
            "time_weight": round(time_weight, 4),
            "confidence_before": round(confidence_before, 4),
            "confidence_after": round(confidence_after, 4),
            "pruned": confidence_after < 0.3
        }
        
        log_edge_update(edge_log_file, log_data, append=True)
    
    # Remove pruned edges
    for source, target in edges_to_remove:
        graph.remove_edge(source, target)
    
    logger.info(f"Temporal decay complete: {edges_updated}/{edges_processed} edges updated, {edges_pruned} pruned")
    
    # Summary statistics
    summary = {
        "module": "temporal_decay",
        "edges_processed": edges_processed,
        "edges_updated": edges_updated,
        "edges_pruned": edges_pruned,
        "edges_remaining": graph.number_of_edges(),
        "avg_confidence_before": round(total_confidence_before / edges_processed, 4) if edges_processed > 0 else 0.0,
        "avg_confidence_after": round(total_confidence_after / max(1, edges_processed - edges_pruned), 4) if (edges_processed - edges_pruned) > 0 else 0.0,
        "half_life_config": HALF_LIFE
    }
    
    log_summary(summary_file, summary)
    logger.info(f"Summary: {summary}")
    
    return graph, summary
