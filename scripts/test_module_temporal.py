#!/usr/bin/env python3
"""
Test script for Temporal Decay module.

Creates a sample graph and applies temporal decay to demonstrate functionality.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx
from belief_graph.graph.temporal_decay import apply_temporal_decay


def create_sample_graph():
    """Create a sample graph with events and beliefs."""
    G = nx.DiGraph()
    
    # Add event nodes
    G.add_node("event_1", node_type="event")
    G.add_node("event_2", node_type="event")
    G.add_node("event_3", node_type="event")
    
    # Add belief nodes
    G.add_node("belief_1", node_type="belief")
    G.add_node("belief_2", node_type="belief")
    G.add_node("belief_3", node_type="belief")
    
    # Add edges with initial confidence
    G.add_edge("event_1", "belief_1", confidence=0.8, from_event_id="event_1")
    G.add_edge("event_2", "belief_2", confidence=0.7, from_event_id="event_2")
    G.add_edge("event_3", "belief_3", confidence=0.6, from_event_id="event_3")
    G.add_edge("event_1", "belief_2", confidence=0.5, from_event_id="event_1")
    
    return G


def create_sample_data():
    """Create sample event and belief data with timestamps."""
    now = datetime.utcnow()
    
    events = {
        "event_1": {
            "timestamp": now - timedelta(hours=24),  # 1 day ago
            "event_type": "policy"
        },
        "event_2": {
            "timestamp": now - timedelta(hours=72),  # 3 days ago
            "event_type": "poll"
        },
        "event_3": {
            "timestamp": now - timedelta(hours=168),  # 7 days ago
            "event_type": "narrative"
        }
    }
    
    beliefs = {
        "belief_1": {
            "timestamp": now
        },
        "belief_2": {
            "timestamp": now
        },
        "belief_3": {
            "timestamp": now
        }
    }
    
    return events, beliefs


def main():
    """Run temporal decay test."""
    print("=" * 60)
    print("TEMPORAL DECAY MODULE TEST")
    print("=" * 60)
    
    # Create sample graph
    print("\n1. Creating sample graph...")
    graph = create_sample_graph()
    print(f"   Nodes: {graph.number_of_nodes()}")
    print(f"   Edges: {graph.number_of_edges()}")
    
    # Create sample data
    print("\n2. Creating sample event/belief data...")
    events, beliefs = create_sample_data()
    print(f"   Events: {len(events)}")
    print(f"   Beliefs: {len(beliefs)}")
    
    # Print initial edge confidences
    print("\n3. Initial edge confidences:")
    for source, target, data in graph.edges(data=True):
        print(f"   {source} -> {target}: {data['confidence']:.4f}")
    
    # Apply temporal decay
    print("\n4. Applying temporal decay...")
    output_dir = Path("data/graph_logs/test_temporal_decay")
    graph, summary = apply_temporal_decay(graph, events, beliefs, output_dir)
    
    # Print results
    print("\n5. Results after temporal decay:")
    for source, target, data in graph.edges(data=True):
        time_weight = data.get('time_weight', 1.0)
        print(f"   {source} -> {target}: {data['confidence']:.4f} (time_weight={time_weight:.4f})")
    
    # Print summary
    print("\n6. Summary:")
    print(f"   Edges processed: {summary['edges_processed']}")
    print(f"   Edges updated: {summary['edges_updated']}")
    print(f"   Edges pruned: {summary['edges_pruned']}")
    print(f"   Edges remaining: {summary['edges_remaining']}")
    print(f"   Avg confidence before: {summary['avg_confidence_before']:.4f}")
    print(f"   Avg confidence after: {summary['avg_confidence_after']:.4f}")
    
    # Print log locations
    print("\n7. Log files created:")
    print(f"   Edge adjustments: {output_dir}/edge_adjustments.jsonl")
    print(f"   Summary: {output_dir}/summary.json")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
