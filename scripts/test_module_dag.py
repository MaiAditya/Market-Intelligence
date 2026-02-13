#!/usr/bin/env python3
"""
Test script for DAG Validation module.

Creates a sample graph with cycles and validates it as a DAG.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx
from belief_graph.graph.dag_validation import ensure_dag


def create_sample_graph_with_cycles():
    """Create a sample graph with cycles."""
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(1, 6):
        G.add_node(f"event_{i}", node_type="event")
    for i in range(1, 4):
        G.add_node(f"belief_{i}", node_type="belief")
    
    # Create a cycle: event_1 -> event_2 -> event_3 -> event_1
    G.add_edge("event_1", "event_2", confidence=0.8)
    G.add_edge("event_2", "event_3", confidence=0.7)
    G.add_edge("event_3", "event_1", confidence=0.3)  # Weakest edge in cycle
    
    # Another cycle: event_4 -> event_5 -> event_4
    G.add_edge("event_4", "event_5", confidence=0.6)
    G.add_edge("event_5", "event_4", confidence=0.4)  # Weakest edge in cycle
    
    # Some non-cyclic edges
    G.add_edge("event_1", "belief_1", confidence=0.9)
    G.add_edge("event_2", "belief_2", confidence=0.8)
    G.add_edge("event_5", "belief_3", confidence=0.7)
    
    return G


def main():
    """Run DAG validation test."""
    print("=" * 60)
    print("DAG VALIDATION MODULE TEST")
    print("=" * 60)
    
    # Create sample graph with cycles
    print("\n1. Creating sample graph with cycles...")
    graph = create_sample_graph_with_cycles()
    print(f"   Nodes: {graph.number_of_nodes()}")
    print(f"   Edges: {graph.number_of_edges()}")
    print(f"   Is DAG: {nx.is_directed_acyclic_graph(graph)}")
    
    # Print initial edges
    print("\n2. Initial edges:")
    for source, target, data in graph.edges(data=True):
        print(f"   {source} -> {target}: {data['confidence']:.4f}")
    
    # Apply DAG validation
    print("\n3. Ensuring graph is a DAG...")
    output_dir = Path("data/graph_logs/test_dag_validation")
    graph, summary = ensure_dag(graph, output_dir)
    
    # Print results
    print("\n4. Results after DAG validation:")
    print(f"   Is DAG: {nx.is_directed_acyclic_graph(graph)}")
    print(f"   Remaining edges:")
    for source, target, data in graph.edges(data=True):
        print(f"     {source} -> {target}: {data['confidence']:.4f}")
    
    # Print summary
    print("\n5. Summary:")
    print(f"   Was already DAG: {summary['was_already_dag']}")
    print(f"   Cycles detected: {summary['cycles_detected']}")
    print(f"   Edges removed: {summary['edges_removed']}")
    print(f"   Iterations: {summary['iterations']}")
    print(f"   Final is DAG: {summary['is_dag']}")
    print(f"   Final edges: {summary['final_edges']}")
    
    if summary['removed_edges']:
        print("\n6. Removed edges:")
        for edge in summary['removed_edges']:
            print(f"   {edge['source']} -> {edge['target']}: {edge['confidence']:.4f}")
    
    # Print log locations
    print("\n7. Log files created:")
    print(f"   Cycles removed: {output_dir}/cycles_removed.jsonl")
    print(f"   Summary: {output_dir}/summary.json")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
