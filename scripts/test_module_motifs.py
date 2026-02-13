#!/usr/bin/env python3
"""
Test script for Motif Detection module.

Creates a sample graph and detects motifs to demonstrate functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx
from belief_graph.graph.motifs import detect_and_boost_motifs


def create_sample_graph():
    """Create a sample graph with various motifs."""
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(1, 7):
        G.add_node(f"event_{i}", node_type="event")
    for i in range(1, 6):
        G.add_node(f"belief_{i}", node_type="belief")
    
    # Chain motif: event_1 -> event_2 -> belief_1
    G.add_edge("event_1", "event_2", confidence=0.7)
    G.add_edge("event_2", "belief_1", confidence=0.6)
    
    # Another chain: event_3 -> event_4 -> belief_2
    G.add_edge("event_3", "event_4", confidence=0.8)
    G.add_edge("event_4", "belief_2", confidence=0.7)
    
    # Fork motif: event_5 -> {belief_3, belief_4}
    G.add_edge("event_5", "belief_3", confidence=0.6)
    G.add_edge("event_5", "belief_4", confidence=0.5)
    
    # Collider motif: {event_1, event_6} -> belief_5
    G.add_edge("event_1", "belief_5", confidence=0.7)
    G.add_edge("event_6", "belief_5", confidence=0.6)
    
    return G


def main():
    """Run motif detection test."""
    print("=" * 60)
    print("MOTIF DETECTION MODULE TEST")
    print("=" * 60)
    
    # Create sample graph
    print("\n1. Creating sample graph...")
    graph = create_sample_graph()
    print(f"   Nodes: {graph.number_of_nodes()}")
    print(f"   Edges: {graph.number_of_edges()}")
    
    # Print initial edges
    print("\n2. Initial edges:")
    for source, target, data in graph.edges(data=True):
        print(f"   {source} -> {target}: {data['confidence']:.4f}")
    
    # Apply motif detection
    print("\n3. Detecting motifs and applying boosts...")
    output_dir = Path("data/graph_logs/test_motifs")
    graph, summary = detect_and_boost_motifs(graph, output_dir)
    
    # Print results
    print("\n4. Results after motif boosting:")
    for source, target, data in graph.edges(data=True):
        motif_types = data.get('motif_types', [])
        marker = f" [MOTIFS: {', '.join(motif_types)}]" if motif_types else ""
        print(f"   {source} -> {target}: {data['confidence']:.4f}{marker}")
    
    # Print summary
    print("\n5. Summary:")
    print(f"   Total motifs detected: {summary['total_motifs_detected']}")
    print(f"   Motif breakdown:")
    for motif_type, stats in summary['motif_breakdown'].items():
        print(f"     {motif_type}: {stats['count']} motifs, {stats['edges_boosted']} edges boosted")
    print(f"   Total edges boosted: {summary['total_edges_boosted']}")
    print(f"   Edges remaining: {summary['edges_remaining']}")
    
    # Print log locations
    print("\n6. Log files created:")
    print(f"   Detected motifs: {output_dir}/detected_motifs.jsonl")
    print(f"   Edge boosts: {output_dir}/edge_boosts.jsonl")
    print(f"   Summary: {output_dir}/summary.json")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
