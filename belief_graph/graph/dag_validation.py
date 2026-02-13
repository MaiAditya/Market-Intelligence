"""
DAG Validation Module

Ensures graph is a directed acyclic graph (DAG) by detecting and removing cycles.
Removes lowest-confidence edge in each cycle.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .logging_utils import setup_graph_logger, log_edge_update, log_summary


def find_cycles(graph: nx.DiGraph) -> List[List[str]]:
    """
    Find all simple cycles in the graph.
    
    Args:
        graph: NetworkX directed graph
    
    Returns:
        List of cycles, where each cycle is a list of node IDs
    """
    try:
        cycles = list(nx.simple_cycles(graph))
        return cycles
    except:
        return []


def remove_weakest_edge_in_cycle(
    graph: nx.DiGraph,
    cycle: List[str],
    logger: logging.Logger,
    log_file: Path
) -> Tuple[str, str, float]:
    """
    Remove the edge with lowest confidence in a cycle.
    
    Args:
        graph: NetworkX directed graph
        cycle: List of node IDs forming a cycle
        logger: Logger instance
        log_file: Path to JSONL log file
    
    Returns:
        Tuple of (source, target, confidence) of removed edge
    """
    # Find all edges in the cycle
    cycle_edges = []
    for i in range(len(cycle)):
        source = cycle[i]
        target = cycle[(i + 1) % len(cycle)]
        
        if graph.has_edge(source, target):
            confidence = graph[source][target].get('confidence', 0.0)
            cycle_edges.append((source, target, confidence))
    
    if not cycle_edges:
        logger.warning(f"No edges found in cycle: {cycle}")
        return None, None, 0.0
    
    # Find edge with lowest confidence
    weakest_edge = min(cycle_edges, key=lambda x: x[2])
    source, target, confidence = weakest_edge
    
    # Remove the edge
    graph.remove_edge(source, target)
    
    # Log removal
    log_data = {
        "cycle": cycle,
        "removed_edge": f"{source}->{target}",
        "confidence": round(confidence, 4),
        "reason": "lowest_confidence_in_cycle"
    }
    
    log_edge_update(log_file, log_data, append=True)
    logger.info(f"Removed edge {source} -> {target} (confidence={confidence:.4f}) from cycle")
    
    return source, target, confidence


def ensure_dag(
    graph: nx.DiGraph,
    output_dir: Optional[Path] = None,
    max_iterations: int = 1000
) -> Tuple[nx.DiGraph, Dict]:
    """
    Ensure graph is a directed acyclic graph (DAG).
    
    Iteratively detects cycles and removes the lowest-confidence edge
    in each cycle until no cycles remain.
    
    Args:
        graph: NetworkX directed graph
        output_dir: Optional directory for logging output
        max_iterations: Maximum number of iterations to prevent infinite loops
    
    Returns:
        Tuple of (updated_graph, summary_stats)
    """
    if output_dir is None:
        output_dir = Path("data/graph_logs/dag_validation")
    else:
        output_dir = Path(output_dir)
    
    logger = setup_graph_logger("dag_validation", output_dir)
    logger.info(f"Starting DAG validation on graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Prepare output files
    cycle_log_file = output_dir / "cycles_removed.jsonl"
    summary_file = output_dir / "summary.json"
    
    # Clear previous logs
    if cycle_log_file.exists():
        cycle_log_file.unlink()
    
    # Check if already a DAG
    if nx.is_directed_acyclic_graph(graph):
        logger.info("Graph is already a DAG, no cycles to remove")
        summary = {
            "module": "dag_validation",
            "was_already_dag": True,
            "cycles_detected": 0,
            "edges_removed": 0,
            "iterations": 0,
            "final_edges": graph.number_of_edges()
        }
        log_summary(summary_file, summary)
        return graph, summary
    
    cycles_detected = 0
    edges_removed = 0
    iterations = 0
    removed_edges = []
    
    # Iteratively remove edges until DAG
    while not nx.is_directed_acyclic_graph(graph) and iterations < max_iterations:
        iterations += 1
        
        # Find a cycle
        cycles = find_cycles(graph)
        
        if not cycles:
            break
        
        # Take the first cycle
        cycle = cycles[0]
        cycles_detected += 1
        
        logger.info(f"Iteration {iterations}: Found cycle with {len(cycle)} nodes")
        
        # Remove weakest edge
        source, target, confidence = remove_weakest_edge_in_cycle(
            graph, cycle, logger, cycle_log_file
        )
        
        if source and target:
            edges_removed += 1
            removed_edges.append({
                "source": source,
                "target": target,
                "confidence": confidence
            })
    
    # Final check
    is_dag = nx.is_directed_acyclic_graph(graph)
    
    if not is_dag:
        logger.warning(f"Graph still contains cycles after {max_iterations} iterations")
    else:
        logger.info(f"DAG validation complete: removed {edges_removed} edges in {iterations} iterations")
    
    # Summary statistics
    summary = {
        "module": "dag_validation",
        "was_already_dag": False,
        "cycles_detected": cycles_detected,
        "edges_removed": edges_removed,
        "iterations": iterations,
        "is_dag": is_dag,
        "final_edges": graph.number_of_edges(),
        "removed_edges": removed_edges
    }
    
    log_summary(summary_file, summary)
    logger.info(f"Summary: {summary}")
    
    return graph, summary
