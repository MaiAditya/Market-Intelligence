#!/usr/bin/env python3
"""
Test script for Structural Propagation module.

Creates a sample graph and applies structural propagation to demonstrate functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx
from belief_graph.graph.propagation import propagate_influence


def create_sample_graph():
    """Create a sample graph with 2-hop paths."""
    G = nx.DiGraph()
    
    # Add event nodes
    for i in range(1, 6):
        G.add_node(f"event_{i}", node_type="event")
    
    # Add belief nodes
    for i in range(1, 4):
        G.add_node(f"belief_{i}", node_type="belief")
    
    # Create 2-hop paths: event_1 -> event_2 -> belief_1
    G.add_edge("event_1", "event_2", confidence=0.8)
    G.add_edge("event_2", "belief_1", confidence=0.7)
    
    # Another path: event_3 -> event_4 -> belief_2
    G.add_edge("event_3", "event_4", confidence=0.6)
    G.add_edge("event_4", "belief_2", confidence=0.9)
    
    # Path with existing direct edge: event_1 -> event_5 -> belief_3
    G.add_edge("event_1", "event_5", confidence=0.5)
    G.add_edge("event_5", "belief_3", confidence=0.6)
    G.add_edge("event_1", "belief_3", confidence=0.3)  # Direct edge exists
    
    return G


def main():
    """Run structural propagation test."""
    print("=" * 60)
    print("STRUCTURAL PROPAGATION MODULE TEST")
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
    
    # Apply structural propagation
    print("\n3. Applying structural propagation...")
    output_dir = Path("data/graph_logs/test_propagation")
    graph, summary = propagate_influence(graph, output_dir)
    
    # Print results
    print("\n4. Results after propagation:")
    for source, target, data in graph.edges(data=True):
        propagated = data.get('propagated', False)
        marker = " [PROPAGATED]" if propagated else ""
        print(f"   {source} -> {target}: {data['confidence']:.4f}{marker}")
    
    # Print summary
    print("\n5. Summary:")
    print(f"   Paths processed: {summary['paths_processed']}")
    print(f"   Edges updated: {summary['edges_updated']}")
    print(f"   Edges created: {summary['edges_created']}")
    print(f"   Edges pruned: {summary['edges_pruned']}")
    print(f"   Edges remaining: {summary['edges_remaining']}")
    
    # Print log locations
    print("\n6. Log files created:")
    print(f"   Path updates: {output_dir}/path_updates.jsonl")
    print(f"   Summary: {output_dir}/summary.json")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
