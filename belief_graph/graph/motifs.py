"""
Causal Motif Detection Module

Detects frequent 3-node structural motifs and boosts edges participating in common patterns.
Motifs: chains, forks, colliders.
"""

import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import networkx as nx

from .logging_utils import setup_graph_logger, log_edge_update, log_summary


def detect_chain_motifs(graph: nx.DiGraph) -> List[Tuple[str, str, str]]:
    """
    Detect chain motifs: event → event → belief
    
    Args:
        graph: NetworkX directed graph
    
    Returns:
        List of (event1, event2, belief) tuples
    """
    chains = []
    
    for node in graph.nodes():
        node_data = graph.nodes[node]
        node_type = node_data.get('node_type', '')
        
        # Check if this is an intermediate event node
        if node_type == 'event':
            predecessors = list(graph.predecessors(node))
            successors = list(graph.successors(node))
            
            for pred in predecessors:
                pred_type = graph.nodes[pred].get('node_type', '')
                if pred_type == 'event':
                    for succ in successors:
                        succ_type = graph.nodes[succ].get('node_type', '')
                        if succ_type == 'belief':
                            chains.append((pred, node, succ))
    
    return chains


def detect_fork_motifs(graph: nx.DiGraph) -> List[Tuple[str, str, str]]:
    """
    Detect fork motifs: event → {belief, belief}
    
    Args:
        graph: NetworkX directed graph
    
    Returns:
        List of (event, belief1, belief2) tuples
    """
    forks = []
    
    for node in graph.nodes():
        node_data = graph.nodes[node]
        node_type = node_data.get('node_type', '')
        
        # Check if this is an event node
        if node_type == 'event':
            successors = list(graph.successors(node))
            
            # Filter for belief successors
            belief_successors = [
                s for s in successors
                if graph.nodes[s].get('node_type', '') == 'belief'
            ]
            
            # Create all pairs of beliefs
            for i, belief1 in enumerate(belief_successors):
                for belief2 in belief_successors[i+1:]:
                    forks.append((node, belief1, belief2))
    
    return forks


def detect_collider_motifs(graph: nx.DiGraph) -> List[Tuple[str, str, str]]:
    """
    Detect collider motifs: {event, event} → belief
    
    Args:
        graph: NetworkX directed graph
    
    Returns:
        List of (event1, event2, belief) tuples
    """
    colliders = []
    
    for node in graph.nodes():
        node_data = graph.nodes[node]
        node_type = node_data.get('node_type', '')
        
        # Check if this is a belief node
        if node_type == 'belief':
            predecessors = list(graph.predecessors(node))
            
            # Filter for event predecessors
            event_predecessors = [
                p for p in predecessors
                if graph.nodes[p].get('node_type', '') == 'event'
            ]
            
            # Create all pairs of events
            for i, event1 in enumerate(event_predecessors):
                for event2 in event_predecessors[i+1:]:
                    colliders.append((event1, event2, node))
    
    return colliders


def apply_motif_boost(
    graph: nx.DiGraph,
    participating_edges: Set[Tuple[str, str]],
    motif_count: int,
    motif_type: str,
    logger: logging.Logger,
    log_file: Path
) -> int:
    """
    Apply confidence boost to edges participating in motifs.
    
    Args:
        graph: NetworkX directed graph
        participating_edges: Set of (source, target) edge tuples
        motif_count: Number of motifs of this type detected
        motif_type: Type of motif (chain, fork, collider)
        logger: Logger instance
        log_file: Path to JSONL log file
    
    Returns:
        Number of edges boosted
    """
    # Calculate boost value
    motif_boost = min(0.15, math.log(1 + motif_count) / 10)
    
    edges_boosted = 0
    
    for source, target in participating_edges:
        if not graph.has_edge(source, target):
            continue
        
        edge_data = graph[source][target]
        confidence_before = edge_data.get('confidence', 0.0)
        
        # Apply boost
        confidence_after = confidence_before + motif_boost
        
        # Clamp to [0, 1]
        confidence_after = max(0.0, min(1.0, confidence_after))
        
        edge_data['confidence'] = confidence_after
        edge_data['motif_boost_applied'] = True
        
        if 'motif_types' not in edge_data:
            edge_data['motif_types'] = []
        edge_data['motif_types'].append(motif_type)
        
        edges_boosted += 1
        
        # Log boost
        log_data = {
            "edge_id": f"{source}->{target}",
            "motif_type": motif_type,
            "motif_count": motif_count,
            "boost_value": round(motif_boost, 4),
            "confidence_before": round(confidence_before, 4),
            "confidence_after": round(confidence_after, 4)
        }
        
        log_edge_update(log_file, log_data, append=True)
    
    return edges_boosted


def detect_and_boost_motifs(
    graph: nx.DiGraph,
    output_dir: Optional[Path] = None
) -> Tuple[nx.DiGraph, Dict]:
    """
    Detect causal motifs and boost participating edges.
    
    Detects three types of motifs:
    1. Chain: event → event → belief
    2. Fork: event → {belief, belief}
    3. Collider: {event, event} → belief
    
    For each motif type, calculates boost and applies to participating edges.
    
    Args:
        graph: NetworkX directed graph
        output_dir: Optional directory for logging output
    
    Returns:
        Tuple of (updated_graph, summary_stats)
    """
    if output_dir is None:
        output_dir = Path("data/graph_logs/motifs")
    else:
        output_dir = Path(output_dir)
    
    logger = setup_graph_logger("motifs", output_dir)
    logger.info(f"Starting motif detection on graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Prepare output files
    motif_log_file = output_dir / "detected_motifs.jsonl"
    boost_log_file = output_dir / "edge_boosts.jsonl"
    summary_file = output_dir / "summary.json"
    
    # Clear previous logs
    for log_file in [motif_log_file, boost_log_file]:
        if log_file.exists():
            log_file.unlink()
    
    motif_stats = {}
    total_edges_boosted = 0
    
    # Detect chain motifs
    logger.info("Detecting chain motifs (event → event → belief)...")
    chains = detect_chain_motifs(graph)
    logger.info(f"Found {len(chains)} chain motifs")
    
    # Track participating edges
    chain_edges = set()
    for event1, event2, belief in chains:
        chain_edges.add((event1, event2))
        chain_edges.add((event2, belief))
        
        # Log motif
        log_data = {
            "motif_type": "chain",
            "nodes": [event1, event2, belief],
            "edges": [f"{event1}->{event2}", f"{event2}->{belief}"]
        }
        log_edge_update(motif_log_file, log_data, append=True)
    
    # Apply boost
    if chains:
        edges_boosted = apply_motif_boost(
            graph, chain_edges, len(chains), "chain", logger, boost_log_file
        )
        total_edges_boosted += edges_boosted
        motif_stats['chain'] = {
            'count': len(chains),
            'edges_boosted': edges_boosted
        }
    
    # Detect fork motifs
    logger.info("Detecting fork motifs (event → {belief, belief})...")
    forks = detect_fork_motifs(graph)
    logger.info(f"Found {len(forks)} fork motifs")
    
    # Track participating edges
    fork_edges = set()
    for event, belief1, belief2 in forks:
        fork_edges.add((event, belief1))
        fork_edges.add((event, belief2))
        
        # Log motif
        log_data = {
            "motif_type": "fork",
            "nodes": [event, belief1, belief2],
            "edges": [f"{event}->{belief1}", f"{event}->{belief2}"]
        }
        log_edge_update(motif_log_file, log_data, append=True)
    
    # Apply boost
    if forks:
        edges_boosted = apply_motif_boost(
            graph, fork_edges, len(forks), "fork", logger, boost_log_file
        )
        total_edges_boosted += edges_boosted
        motif_stats['fork'] = {
            'count': len(forks),
            'edges_boosted': edges_boosted
        }
    
    # Detect collider motifs
    logger.info("Detecting collider motifs ({event, event} → belief)...")
    colliders = detect_collider_motifs(graph)
    logger.info(f"Found {len(colliders)} collider motifs")
    
    # Track participating edges
    collider_edges = set()
    for event1, event2, belief in colliders:
        collider_edges.add((event1, belief))
        collider_edges.add((event2, belief))
        
        # Log motif
        log_data = {
            "motif_type": "collider",
            "nodes": [event1, event2, belief],
            "edges": [f"{event1}->{belief}", f"{event2}->{belief}"]
        }
        log_edge_update(motif_log_file, log_data, append=True)
    
    # Apply boost
    if colliders:
        edges_boosted = apply_motif_boost(
            graph, collider_edges, len(colliders), "collider", logger, boost_log_file
        )
        total_edges_boosted += edges_boosted
        motif_stats['collider'] = {
            'count': len(colliders),
            'edges_boosted': edges_boosted
        }
    
    logger.info(f"Motif detection complete: {total_edges_boosted} edges boosted")
    
    # Summary statistics
    summary = {
        "module": "motifs",
        "total_motifs_detected": len(chains) + len(forks) + len(colliders),
        "motif_breakdown": motif_stats,
        "total_edges_boosted": total_edges_boosted,
        "edges_remaining": graph.number_of_edges()
    }
    
    log_summary(summary_file, summary)
    logger.info(f"Summary: {summary}")
    
    return graph, summary
