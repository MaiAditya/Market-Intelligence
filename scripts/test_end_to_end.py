#!/usr/bin/env python3
"""
End-to-end test script for all graph modules.

Tests the complete pipeline:
1. Temporal decay
2. Structural propagation
3. Motif detection
4. DAG validation
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx
from belief_graph.graph.temporal_decay import apply_temporal_decay
from belief_graph.graph.propagation import propagate_influence
from belief_graph.graph.motifs import detect_and_boost_motifs
from belief_graph.graph.dag_validation import ensure_dag


def create_comprehensive_graph():
    """Create a comprehensive graph for end-to-end testing."""
    G = nx.DiGraph()
    
    # Add event nodes
    for i in range(1, 10):
        G.add_node(f"event_{i}", node_type="event")
    
    # Add belief nodes
    for i in range(1, 7):
        G.add_node(f"belief_{i}", node_type="belief")
    
    # Create various patterns
    # Chain: event_1 -> event_2 -> belief_1
    G.add_edge("event_1", "event_2", confidence=0.8, from_event_id="event_1")
    G.add_edge("event_2", "belief_1", confidence=0.7, from_event_id="event_2")
    
    # Fork: event_3 -> {belief_2, belief_3}
    G.add_edge("event_3", "belief_2", confidence=0.6, from_event_id="event_3")
    G.add_edge("event_3", "belief_3", confidence=0.5, from_event_id="event_3")
    
    # Collider: {event_4, event_5} -> belief_4
    G.add_edge("event_4", "belief_4", confidence=0.7, from_event_id="event_4")
    G.add_edge("event_5", "belief_4", confidence=0.6, from_event_id="event_5")
    
    # 2-hop paths for propagation
    G.add_edge("event_6", "event_7", confidence=0.8, from_event_id="event_6")
    G.add_edge("event_7", "belief_5", confidence=0.9, from_event_id="event_7")
    
    # Cycle for DAG validation: event_8 -> event_9 -> event_8
    G.add_edge("event_8", "event_9", confidence=0.7, from_event_id="event_8")
    G.add_edge("event_9", "event_8", confidence=0.3, from_event_id="event_9")  # Weakest
    G.add_edge("event_8", "belief_6", confidence=0.6, from_event_id="event_8")
    
    return G


def create_comprehensive_data():
    """Create comprehensive event and belief data."""
    now = datetime.utcnow()
    
    events = {}
    for i in range(1, 10):
        # Vary timestamps and types
        hours_ago = i * 24  # 1 day, 2 days, etc.
        event_types = ["policy", "legal", "economic", "poll", "narrative"]
        event_type = event_types[i % len(event_types)]
        
        events[f"event_{i}"] = {
            "timestamp": now - timedelta(hours=hours_ago),
            "event_type": event_type
        }
    
    beliefs = {}
    for i in range(1, 7):
        beliefs[f"belief_{i}"] = {
            "timestamp": now
        }
    
    return events, beliefs


def print_graph_stats(graph, title):
    """Print graph statistics."""
    print(f"\n{title}")
    print(f"  Nodes: {graph.number_of_nodes()}")
    print(f"  Edges: {graph.number_of_edges()}")
    print(f"  Is DAG: {nx.is_directed_acyclic_graph(graph)}")
    
    if graph.number_of_edges() > 0:
        confidences = [data['confidence'] for _, _, data in graph.edges(data=True)]
        avg_conf = sum(confidences) / len(confidences)
        print(f"  Avg confidence: {avg_conf:.4f}")


def main():
    """Run end-to-end test."""
    print("=" * 60)
    print("END-TO-END GRAPH PIPELINE TEST")
    print("=" * 60)
    
    # Create graph and data
    print("\n1. Creating comprehensive test graph...")
    graph = create_comprehensive_graph()
    events, beliefs = create_comprehensive_data()
    print_graph_stats(graph, "Initial graph:")
    
    output_base = Path("data/graph_logs/test_end_to_end")
    
    # Step 1: Temporal Decay
    print("\n" + "=" * 60)
    print("STEP 1: TEMPORAL DECAY")
    print("=" * 60)
    graph, summary1 = apply_temporal_decay(
        graph, events, beliefs,
        output_dir=output_base / "temporal_decay"
    )
    print_graph_stats(graph, "After temporal decay:")
    print(f"  Edges updated: {summary1['edges_updated']}")
    print(f"  Edges pruned: {summary1['edges_pruned']}")
    
    # Step 2: Structural Propagation
    print("\n" + "=" * 60)
    print("STEP 2: STRUCTURAL PROPAGATION")
    print("=" * 60)
    graph, summary2 = propagate_influence(
        graph,
        output_dir=output_base / "propagation"
    )
    print_graph_stats(graph, "After propagation:")
    print(f"  Paths processed: {summary2['paths_processed']}")
    print(f"  Edges created: {summary2['edges_created']}")
    print(f"  Edges updated: {summary2['edges_updated']}")
    
    # Step 3: Motif Detection
    print("\n" + "=" * 60)
    print("STEP 3: MOTIF DETECTION")
    print("=" * 60)
    graph, summary3 = detect_and_boost_motifs(
        graph,
        output_dir=output_base / "motifs"
    )
    print_graph_stats(graph, "After motif detection:")
    print(f"  Total motifs: {summary3['total_motifs_detected']}")
    print(f"  Edges boosted: {summary3['total_edges_boosted']}")
    
    # Step 4: DAG Validation
    print("\n" + "=" * 60)
    print("STEP 4: DAG VALIDATION")
    print("=" * 60)
    graph, summary4 = ensure_dag(
        graph,
        output_dir=output_base / "dag_validation"
    )
    print_graph_stats(graph, "After DAG validation:")
    print(f"  Cycles detected: {summary4['cycles_detected']}")
    print(f"  Edges removed: {summary4['edges_removed']}")
    print(f"  Is DAG: {summary4['is_dag']}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Final nodes: {graph.number_of_nodes()}")
    print(f"  Final edges: {graph.number_of_edges()}")
    print(f"  Is DAG: {nx.is_directed_acyclic_graph(graph)}")
    
    print("\n  Log directories created:")
    print(f"    {output_base}/temporal_decay/")
    print(f"    {output_base}/propagation/")
    print(f"    {output_base}/motifs/")
    print(f"    {output_base}/dag_validation/")
    
    print("\n" + "=" * 60)
    print("END-TO-END TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
