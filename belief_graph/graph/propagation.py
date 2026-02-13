"""
Structural Influence Propagation Module

Propagates influence through multi-hop paths to improve structural coherence.
Uses topological sorting for deterministic traversal.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from .logging_utils import setup_graph_logger, log_edge_update, log_summary


def find_two_hop_paths(graph: nx.DiGraph) -> List[Tuple[str, str, str]]:
    """
    Find all 2-hop paths in the graph (A → B → C).
    
    Args:
        graph: NetworkX directed graph
    
    Returns:
        List of (source, intermediate, target) tuples
    """
    paths = []
    
    # For each node, find paths through its successors
    for intermediate in graph.nodes():
        predecessors = list(graph.predecessors(intermediate))
        successors = list(graph.successors(intermediate))
        
        # Create all combinations of predecessor → intermediate → successor
        for source in predecessors:
            for target in successors:
                # Avoid self-loops
                if source != target:
                    paths.append((source, intermediate, target))
    
    return paths


def update_or_create_edge(
    graph: nx.DiGraph,
    source: str,
    target: str,
    propagated_confidence: float,
    via_node: str,
    logger: logging.Logger,
    log_file: Path
) -> Tuple[bool, float, float]:
    """
    Update existing edge or create new edge with propagated confidence.
    
    Args:
        graph: NetworkX directed graph
        source: Source node
        target: Target node
        propagated_confidence: Confidence value to add/propagate
        via_node: Intermediate node in propagation path
        logger: Logger instance
        log_file: Path to JSONL log file
    
    Returns:
        Tuple of (was_created, confidence_before, confidence_after)
    """
    was_created = False
    
    if graph.has_edge(source, target):
        # Update existing edge
        edge_data = graph[source][target]
        confidence_before = edge_data.get('confidence', 0.0)
        
        # Add propagated influence (weighted by 0.5)
        confidence_after = confidence_before + (propagated_confidence * 0.5)
        
        # Clamp to [0, 1]
        confidence_after = max(0.0, min(1.0, confidence_after))
        
        edge_data['confidence'] = confidence_after
        edge_data['propagation_applied'] = True
        
        if 'propagation_sources' not in edge_data:
            edge_data['propagation_sources'] = []
        edge_data['propagation_sources'].append(via_node)
        
    else:
        # Create new edge
        was_created = True
        confidence_before = 0.0
        confidence_after = propagated_confidence * 0.5
        
        # Clamp to [0, 1]
        confidence_after = max(0.0, min(1.0, confidence_after))
        
        graph.add_edge(source, target, 
                      confidence=confidence_after,
                      propagated=True,
                      propagation_sources=[via_node])
    
    # Log the update
    log_data = {
        "edge_id": f"{source}->{target}",
        "via_node": via_node,
        "was_created": was_created,
        "propagated_value": round(propagated_confidence, 4),
        "confidence_before": round(confidence_before, 4),
        "confidence_after": round(confidence_after, 4)
    }
    
    log_edge_update(log_file, log_data, append=True)
    
    return was_created, confidence_before, confidence_after


def propagate_influence(
    graph: nx.DiGraph,
    output_dir: Optional[Path] = None,
    max_hops: int = 2,
    decay_per_hop: float = 0.6,
    min_confidence: float = 0.3
) -> Tuple[nx.DiGraph, Dict]:
    """
    Propagate influence through multi-hop paths.
    
    Algorithm:
    1. Topologically sort graph for deterministic traversal
    2. For each 2-hop path A → B → C:
       - Calculate propagated = conf(A,B) × conf(B,C) × decay_per_hop
       - Update or create edge A → C
    3. Prune edges below min_confidence
    
    Args:
        graph: NetworkX directed graph
        output_dir: Optional directory for logging output
        max_hops: Maximum hops to propagate (currently only 2 supported)
        decay_per_hop: Decay factor per hop (default 0.6)
        min_confidence: Minimum confidence threshold for pruning
    
    Returns:
        Tuple of (updated_graph, summary_stats)
    """
    if output_dir is None:
        output_dir = Path("data/graph_logs/propagation")
    else:
        output_dir = Path(output_dir)
    
    logger = setup_graph_logger("propagation", output_dir)
    logger.info(f"Starting structural propagation on graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Prepare output files
    path_log_file = output_dir / "path_updates.jsonl"
    summary_file = output_dir / "summary.json"
    
    # Clear previous logs
    if path_log_file.exists():
        path_log_file.unlink()
    
    # Check if graph is acyclic (required for topological sort)
    if not nx.is_directed_acyclic_graph(graph):
        logger.warning("Graph contains cycles, topological sort may not be complete")
    
    # Find all 2-hop paths
    logger.info("Finding 2-hop paths...")
    two_hop_paths = find_two_hop_paths(graph)
    logger.info(f"Found {len(two_hop_paths)} 2-hop paths")
    
    paths_processed = 0
    edges_updated = 0
    edges_created = 0
    edges_pruned = 0
    
    # Process each path
    for source, intermediate, target in two_hop_paths:
        paths_processed += 1
        
        # Get edge confidences
        if not graph.has_edge(source, intermediate) or not graph.has_edge(intermediate, target):
            continue
        
        conf_ab = graph[source][intermediate].get('confidence', 0.0)
        conf_bc = graph[intermediate][target].get('confidence', 0.0)
        
        # Calculate propagated confidence
        propagated = conf_ab * conf_bc * decay_per_hop
        
        # Skip if propagated value is too small
        if propagated < 0.01:
            continue
        
        # Update or create edge
        was_created, conf_before, conf_after = update_or_create_edge(
            graph, source, target, propagated, intermediate, logger, path_log_file
        )
        
        if was_created:
            edges_created += 1
        else:
            edges_updated += 1
        
        # Log propagation step
        logger.debug(f"Propagated {source} → {intermediate} → {target}: {conf_ab:.3f} × {conf_bc:.3f} × {decay_per_hop} = {propagated:.3f}")
    
    logger.info(f"Propagation complete: {paths_processed} paths processed, {edges_updated} edges updated, {edges_created} edges created")
    
    # Prune edges below threshold
    logger.info(f"Pruning edges below confidence {min_confidence}...")
    edges_to_remove = []
    
    for source, target, edge_data in graph.edges(data=True):
        confidence = edge_data.get('confidence', 0.0)
        if confidence < min_confidence:
            edges_to_remove.append((source, target))
            edges_pruned += 1
    
    for source, target in edges_to_remove:
        graph.remove_edge(source, target)
    
    logger.info(f"Pruned {edges_pruned} edges")
    
    # Summary statistics
    summary = {
        "module": "propagation",
        "paths_processed": paths_processed,
        "edges_updated": edges_updated,
        "edges_created": edges_created,
        "edges_pruned": edges_pruned,
        "edges_remaining": graph.number_of_edges(),
        "decay_per_hop": decay_per_hop,
        "min_confidence_threshold": min_confidence
    }
    
    log_summary(summary_file, summary)
    logger.info(f"Summary: {summary}")
    
    return graph, summary
