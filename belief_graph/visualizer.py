"""
Belief Graph Visualizer

Generates visual representations of belief update graphs.
Supports:
- Interactive HTML visualization (pyvis)
- Static PNG/SVG images (networkx + matplotlib)
- Console-based ASCII representation
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from belief_graph.models import BeliefGraph, EventNode, BeliefEdge
from belief_graph.storage import get_storage

logger = logging.getLogger(__name__)


# Color schemes for event types
EVENT_TYPE_COLORS = {
    "policy": "#e74c3c",      # Red
    "legal": "#9b59b6",       # Purple
    "economic": "#2ecc71",    # Green
    "poll": "#f39c12",        # Orange
    "narrative": "#3498db",   # Blue
    "market": "#1abc9c",      # Teal
    "signal": "#95a5a6",      # Gray
    "belief": "#e91e63",      # Pink (for belief node)
}

# Mechanism type colors for edges
MECHANISM_COLORS = {
    "legal_constraint": "#9b59b6",
    "economic_impact": "#2ecc71",
    "signaling": "#3498db",
    "expectation_shift": "#f39c12",
    "narrative_amplification": "#e74c3c",
    "liquidity_reaction": "#1abc9c",
    "coordination_effect": "#34495e",
}


def generate_html_visualization(
    graph: BeliefGraph,
    output_path: str,
    height: str = "800px",
    width: str = "100%",
    show_labels: bool = True,
    physics_enabled: bool = True
) -> str:
    """
    Generate interactive HTML visualization using pyvis.
    
    Args:
        graph: BeliefGraph to visualize
        output_path: Path for output HTML file
        height: Canvas height
        width: Canvas width
        show_labels: Whether to show edge labels
        physics_enabled: Enable physics simulation
    
    Returns:
        Path to generated HTML file
    """
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError(
            "pyvis is required for HTML visualization. "
            "Install with: pip install pyvis"
        )
    
    # Create network
    net = Network(
        height=height,
        width=width,
        directed=True,
        bgcolor="#ffffff",
        font_color="#000000"
    )
    
    # Configure physics
    if physics_enabled:
        net.barnes_hut(
            gravity=-3000,
            central_gravity=0.3,
            spring_length=200,
            spring_strength=0.01
        )
    else:
        net.toggle_physics(False)
    
    # Add belief node (center, larger)
    belief = graph.belief_node
    net.add_node(
        belief.belief_id,
        label=f"BELIEF\n{belief.question[:30]}...",
        color=EVENT_TYPE_COLORS["belief"],
        size=40,
        shape="star",
        title=f"Target: {belief.question}\nPrice: {belief.current_price:.1%}\nLiquidity: ${belief.liquidity:,.0f}"
    )
    
    # Add event nodes
    for event_id, event in graph.event_nodes.items():
        color = EVENT_TYPE_COLORS.get(event.event_type, "#95a5a6")
        
        # Truncate action for label
        label = event.action[:20] + "..." if len(event.action) > 20 else event.action
        
        # Build tooltip
        actors_str = ", ".join(event.actors[:3]) if event.actors else "Unknown"
        tooltip = (
            f"Type: {event.event_type}\n"
            f"Action: {event.action}\n"
            f"Actors: {actors_str}\n"
            f"Certainty: {event.certainty:.2f}\n"
            f"Scope: {event.scope}"
        )
        
        net.add_node(
            event_id,
            label=label,
            color=color,
            size=20 + (event.certainty * 10),  # Size by certainty
            shape="dot",
            title=tooltip
        )
    
    # Add edges
    for edge in graph.edges:
        edge_color = MECHANISM_COLORS.get(edge.mechanism_type, "#7f8c8d")
        
        # Edge width by confidence
        width = 1 + (edge.confidence * 4)
        
        # Build tooltip
        tooltip = (
            f"Mechanism: {edge.mechanism_type}\n"
            f"Direction: {edge.direction}\n"
            f"Confidence: {edge.confidence:.3f}\n"
            f"Latency: {edge.latency}"
        )
        
        label = "" if not show_labels else f"{edge.confidence:.2f}"
        
        net.add_edge(
            edge.from_event_id,
            edge.to_event_id,
            color=edge_color,
            width=width,
            title=tooltip,
            label=label,
            arrows="to"
        )
    
    # Add legend
    legend_html = """
    <div style="position: absolute; top: 10px; left: 10px; background: white; padding: 10px; border: 1px solid #ccc; border-radius: 5px; font-family: sans-serif; font-size: 12px;">
        <strong>Event Types:</strong><br>
        <span style="color: #e74c3c;">● Policy</span><br>
        <span style="color: #9b59b6;">● Legal</span><br>
        <span style="color: #2ecc71;">● Economic</span><br>
        <span style="color: #f39c12;">● Poll</span><br>
        <span style="color: #3498db;">● Narrative</span><br>
        <span style="color: #1abc9c;">● Market</span><br>
        <span style="color: #95a5a6;">● Signal</span><br>
        <span style="color: #e91e63;">★ Belief (Target)</span>
    </div>
    """
    
    # Save
    output_path = Path(output_path)
    net.save_graph(str(output_path))
    
    # Inject legend into HTML
    with open(output_path, 'r') as f:
        html = f.read()
    
    html = html.replace('<body>', f'<body>\n{legend_html}')
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"Generated HTML visualization: {output_path}")
    return str(output_path)


def generate_matplotlib_visualization(
    graph: BeliefGraph,
    output_path: str,
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 150,
    layout: str = "spring"
) -> str:
    """
    Generate static image visualization using networkx + matplotlib.
    
    Args:
        graph: BeliefGraph to visualize
        output_path: Path for output image (PNG, SVG, PDF)
        figsize: Figure size in inches
        dpi: Resolution
        layout: Layout algorithm (spring, circular, kamada_kawai, shell)
    
    Returns:
        Path to generated image
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "networkx and matplotlib are required. "
            "Install with: pip install networkx matplotlib"
        )
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add belief node
    belief = graph.belief_node
    G.add_node(
        belief.belief_id,
        node_type="belief",
        label=f"BELIEF:\n{belief.question[:25]}..."
    )
    
    # Add event nodes
    for event_id, event in graph.event_nodes.items():
        label = event.action[:15] + "..." if len(event.action) > 15 else event.action
        G.add_node(
            event_id,
            node_type=event.event_type,
            label=label,
            certainty=event.certainty
        )
    
    # Add edges
    for edge in graph.edges:
        G.add_edge(
            edge.from_event_id,
            edge.to_event_id,
            mechanism=edge.mechanism_type,
            confidence=edge.confidence,
            direction=edge.direction
        )
    
    # Calculate layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "shell":
        # Put belief in center, others in shells by edge distance
        shells = [[belief.belief_id]]
        direct = [e.from_event_id for e in graph.edges if e.to_event_id == belief.belief_id]
        shells.append(list(set(direct)))
        others = [n for n in G.nodes() if n not in shells[0] and n not in shells[1]]
        if others:
            shells.append(others)
        pos = nx.shell_layout(G, shells)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Draw nodes by type
    for node_type, color in EVENT_TYPE_COLORS.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == node_type]
        if nodes:
            if node_type == "belief":
                nx.draw_networkx_nodes(
                    G, pos, nodelist=nodes, node_color=color,
                    node_size=1500, node_shape="*", ax=ax
                )
            else:
                sizes = [300 + G.nodes[n].get("certainty", 0.5) * 200 for n in nodes]
                nx.draw_networkx_nodes(
                    G, pos, nodelist=nodes, node_color=color,
                    node_size=sizes, ax=ax, alpha=0.8
                )
    
    # Draw edges with colors by mechanism
    for mechanism, color in MECHANISM_COLORS.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("mechanism") == mechanism]
        if edges:
            widths = [1 + G.edges[e].get("confidence", 0.5) * 3 for e in edges]
            nx.draw_networkx_edges(
                G, pos, edgelist=edges, edge_color=color,
                width=widths, alpha=0.6, arrows=True,
                arrowsize=15, ax=ax, connectionstyle="arc3,rad=0.1"
            )
    
    # Draw labels
    labels = {n: d.get("label", n[:10]) for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    
    # Add title
    ax.set_title(
        f"Belief Graph: {belief.question[:50]}...\n"
        f"Nodes: {len(graph.event_nodes)} | Edges: {len(graph.edges)}",
        fontsize=14
    )
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=etype.capitalize())
        for etype, color in EVENT_TYPE_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
    
    ax.axis("off")
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Generated image visualization: {output_path}")
    return str(output_path)


def print_ascii_graph(graph: BeliefGraph, max_nodes: int = 20) -> None:
    """
    Print ASCII representation of graph to console.
    
    Args:
        graph: BeliefGraph to display
        max_nodes: Maximum nodes to show
    """
    belief = graph.belief_node
    
    print(f"\n{'='*70}")
    print(f"BELIEF GRAPH: {belief.question[:50]}")
    print(f"{'='*70}")
    
    # Get edges to belief
    direct_edges = [e for e in graph.edges if e.to_event_id == belief.belief_id]
    direct_edges.sort(key=lambda e: e.confidence, reverse=True)
    
    print(f"\n★ TARGET: {belief.question}")
    print(f"   Price: {belief.current_price:.1%} | Liquidity: ${belief.liquidity:,.0f}")
    print(f"\n{'─'*70}")
    print(f"DIRECT INFLUENCES ({len(direct_edges)} edges):")
    print(f"{'─'*70}")
    
    for i, edge in enumerate(direct_edges[:max_nodes], 1):
        event = graph.event_nodes.get(edge.from_event_id)
        if event:
            direction_symbol = "↑" if edge.direction == "positive" else ("↓" if edge.direction == "negative" else "→")
            print(f"  {i:2}. [{event.event_type:10}] {event.action[:35]:<35}")
            print(f"      {direction_symbol} {edge.mechanism_type} (conf: {edge.confidence:.3f})")
    
    if len(direct_edges) > max_nodes:
        print(f"  ... and {len(direct_edges) - max_nodes} more")
    
    # Summary
    print(f"\n{'─'*70}")
    print(f"SUMMARY:")
    print(f"{'─'*70}")
    
    # Count by type
    type_counts = {}
    for event in graph.event_nodes.values():
        type_counts[event.event_type] = type_counts.get(event.event_type, 0) + 1
    
    for etype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {etype:15}: {count:3} events")
    
    # Mechanism breakdown
    mech_counts = {}
    for edge in graph.edges:
        mech_counts[edge.mechanism_type] = mech_counts.get(edge.mechanism_type, 0) + 1
    
    print(f"\nMECHANISMS:")
    for mech, count in sorted(mech_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {mech:25}: {count:3} edges")
    
    print(f"\n{'='*70}\n")


def visualize_graph(
    event_id: str,
    output_path: Optional[str] = None,
    format: str = "html",
    **kwargs
) -> str:
    """
    Main entry point for visualization.
    
    Args:
        event_id: Event ID to visualize
        output_path: Output file path (auto-generated if not provided)
        format: Output format (html, png, svg, pdf, ascii)
        **kwargs: Additional arguments for visualization functions
    
    Returns:
        Path to output file (or empty string for ascii)
    """
    storage = get_storage()
    graph = storage.load(event_id)
    
    if graph is None:
        raise ValueError(f"No graph found for event: {event_id}")
    
    # Auto-generate output path
    if output_path is None and format != "ascii":
        output_dir = project_root / "data" / "belief_graphs"
        output_path = output_dir / f"{event_id}_visualization.{format}"
    
    if format == "html":
        html_kwargs = {k: v for k, v in kwargs.items() if k in ("height", "width", "show_labels", "physics_enabled")}
        return generate_html_visualization(graph, str(output_path), **html_kwargs)
    elif format in ("png", "svg", "pdf"):
        img_kwargs = {k: v for k, v in kwargs.items() if k in ("figsize", "dpi", "layout")}
        return generate_matplotlib_visualization(graph, str(output_path), **img_kwargs)
    elif format == "ascii":
        ascii_kwargs = {k: v for k, v in kwargs.items() if k in ("max_nodes",)}
        print_ascii_graph(graph, **ascii_kwargs)
        return ""
    else:
        raise ValueError(f"Unsupported format: {format}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize belief graphs")
    parser.add_argument("event_id", help="Event ID to visualize")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument(
        "-f", "--format",
        choices=["html", "png", "svg", "pdf", "ascii"],
        default="html",
        help="Output format (default: html)"
    )
    parser.add_argument("--no-physics", action="store_true", help="Disable physics in HTML")
    parser.add_argument("--no-labels", action="store_true", help="Hide edge labels")
    
    args = parser.parse_args()
    
    try:
        result = visualize_graph(
            args.event_id,
            args.output,
            args.format,
            physics_enabled=not args.no_physics,
            show_labels=not args.no_labels
        )
        if result:
            print(f"Visualization saved to: {result}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
