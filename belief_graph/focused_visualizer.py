"""
Focused Belief Graph Visualizer

Creates a simplified N-1 and N-2 flow chart visualization showing:
- N-1: Direct influences on the belief (events that directly affect the target)
- N-2: Second-degree influences (events that influence the N-1 events)

This creates a clean, consumable hierarchical view focused on the most
impactful paths to the belief outcome.
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from belief_graph.models import BeliefGraph, EventNode, BeliefEdge
from belief_graph.storage import get_storage

logger = logging.getLogger(__name__)


# Color schemes
EVENT_TYPE_COLORS = {
    "policy": "#e74c3c",
    "legal": "#9b59b6",
    "economic": "#2ecc71",
    "poll": "#f39c12",
    "narrative": "#3498db",
    "market": "#1abc9c",
    "signal": "#95a5a6",
}

DIRECTION_ARROWS = {
    "positive": "▲",
    "negative": "▼",
    "ambiguous": "◆",
}

DIRECTION_COLORS = {
    "positive": "#27ae60",
    "negative": "#e74c3c",
    "ambiguous": "#95a5a6",
}


def extract_focused_subgraph(
    graph: BeliefGraph,
    top_n1: int = 10,
    top_n2_per_n1: int = 3,
    min_confidence: float = 0.3
) -> Dict:
    """
    Extract a focused subgraph showing only N-1 and N-2 relationships.
    
    Args:
        graph: Full belief graph
        top_n1: Number of top direct influences to show
        top_n2_per_n1: Number of second-degree influences per N-1 event
        min_confidence: Minimum confidence threshold
    
    Returns:
        Focused subgraph data structure
    """
    belief = graph.belief_node
    
    # Step 1: Get N-1 edges (direct to belief)
    n1_edges = [
        e for e in graph.edges 
        if e.to_event_id == belief.belief_id and e.confidence >= min_confidence
    ]
    n1_edges.sort(key=lambda e: e.confidence, reverse=True)
    n1_edges = n1_edges[:top_n1]
    
    # Get N-1 event IDs
    n1_event_ids = {e.from_event_id for e in n1_edges}
    
    # Step 2: Get N-2 edges (events that influence N-1 events)
    n2_edges = []
    n2_by_n1: Dict[str, List[BeliefEdge]] = defaultdict(list)
    
    for edge in graph.edges:
        if edge.to_event_id in n1_event_ids and edge.confidence >= min_confidence:
            # This edge goes TO an N-1 event
            if edge.from_event_id not in n1_event_ids:  # Avoid N-1 to N-1
                n2_by_n1[edge.to_event_id].append(edge)
    
    # Limit N-2 per N-1
    for n1_id, edges in n2_by_n1.items():
        edges.sort(key=lambda e: e.confidence, reverse=True)
        n2_edges.extend(edges[:top_n2_per_n1])
    
    n2_event_ids = {e.from_event_id for e in n2_edges}
    
    # Step 3: Build focused structure
    focused = {
        "belief": {
            "id": belief.belief_id,
            "question": belief.question,
            "price": belief.current_price,
            "liquidity": belief.liquidity,
        },
        "n1_events": [],
        "n2_events": [],
        "n1_edges": [],
        "n2_edges": [],
        "stats": {
            "total_n1": len(n1_edges),
            "total_n2": len(n2_edges),
            "original_nodes": len(graph.event_nodes),
            "original_edges": len(graph.edges),
        }
    }
    
    # Add N-1 events with aggregated impact
    for edge in n1_edges:
        event = graph.event_nodes.get(edge.from_event_id)
        if event:
            # Count how many N-2 feed into this
            n2_count = len(n2_by_n1.get(edge.from_event_id, []))
            n2_total_conf = sum(e.confidence for e in n2_by_n1.get(edge.from_event_id, []))
            
            # Get full title
            title = event.raw_title or f"{event.action} {event.object}"
            
            focused["n1_events"].append({
                "id": event.event_id,
                "type": event.event_type,
                "action": event.action,
                "title": title,  # Full title
                "actors": event.actors[:3],
                "certainty": event.certainty,
                "source": event.source,
                "url": event.url,  # Source URL
                "n2_feeders": n2_count,
                "n2_strength": round(n2_total_conf, 3),
            })
            
            focused["n1_edges"].append({
                "from": edge.from_event_id,
                "to": belief.belief_id,
                "mechanism": edge.mechanism_type,
                "direction": edge.direction,
                "confidence": edge.confidence,
                "explanation": edge.explanation,
            })
    
    # Add N-2 events
    for edge in n2_edges:
        event = graph.event_nodes.get(edge.from_event_id)
        if event:
            title = event.raw_title or f"{event.action} {event.object}"
            focused["n2_events"].append({
                "id": event.event_id,
                "type": event.event_type,
                "action": event.action,
                "title": title,
                "actors": event.actors[:2],
                "certainty": event.certainty,
                "feeds_into": edge.to_event_id,
                "url": event.url,
                "source": event.source,
            })
            
            focused["n2_edges"].append({
                "from": edge.from_event_id,
                "to": edge.to_event_id,
                "mechanism": edge.mechanism_type,
                "direction": edge.direction,
                "confidence": edge.confidence,
            })
    
    return focused


def generate_focused_html(
    graph: BeliefGraph,
    output_path: str,
    top_n1: int = 10,
    top_n2_per_n1: int = 3,
    min_confidence: float = 0.3
) -> str:
    """
    Generate a focused hierarchical HTML visualization.
    
    Shows N-1 and N-2 in a clean flowchart style.
    """
    focused = extract_focused_subgraph(graph, top_n1, top_n2_per_n1, min_confidence)
    
    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Belief Influence Flow: {focused['belief']['question'][:50]}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #eee;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 10px;
            font-size: 1.8em;
            color: #fff;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
            font-size: 0.9em;
        }}
        
        /* Belief Target */
        .belief-target {{
            background: linear-gradient(135deg, #e91e63 0%, #c2185b 100%);
            border-radius: 16px;
            padding: 25px;
            margin: 0 auto 40px;
            max-width: 600px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(233, 30, 99, 0.3);
        }}
        .belief-target h2 {{
            font-size: 1.4em;
            margin-bottom: 15px;
        }}
        .belief-stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
        }}
        .belief-stat {{
            text-align: center;
        }}
        .belief-stat-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .belief-stat-label {{
            font-size: 0.8em;
            opacity: 0.8;
        }}
        
        /* Flow Section */
        .flow-section {{
            margin-bottom: 40px;
        }}
        .section-header {{
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }}
        .section-badge {{
            background: #333;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            margin-right: 15px;
        }}
        .section-title {{
            font-size: 1.2em;
        }}
        .section-count {{
            margin-left: auto;
            color: #888;
        }}
        
        /* Event Cards */
        .events-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
        }}
        .event-card {{
            background: #1e1e30;
            border-radius: 12px;
            padding: 20px;
            border-left: 4px solid #666;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .event-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        .event-card.policy {{ border-left-color: #e74c3c; }}
        .event-card.legal {{ border-left-color: #9b59b6; }}
        .event-card.economic {{ border-left-color: #2ecc71; }}
        .event-card.poll {{ border-left-color: #f39c12; }}
        .event-card.narrative {{ border-left-color: #3498db; }}
        .event-card.market {{ border-left-color: #1abc9c; }}
        .event-card.signal {{ border-left-color: #95a5a6; }}
        
        .event-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }}
        .event-type {{
            font-size: 0.75em;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 3px 8px;
            border-radius: 4px;
            background: rgba(255,255,255,0.1);
        }}
        .event-confidence {{
            font-size: 1.1em;
            font-weight: bold;
        }}
        .event-confidence.high {{ color: #2ecc71; }}
        .event-confidence.medium {{ color: #f39c12; }}
        .event-confidence.low {{ color: #e74c3c; }}
        
        .event-action {{
            font-size: 1.1em;
            margin-bottom: 10px;
            line-height: 1.4;
        }}
        
        .event-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            font-size: 0.85em;
            color: #888;
        }}
        .event-meta-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .direction-indicator {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.85em;
        }}
        .direction-indicator.positive {{
            background: rgba(39, 174, 96, 0.2);
            color: #2ecc71;
        }}
        .direction-indicator.negative {{
            background: rgba(231, 76, 60, 0.2);
            color: #e74c3c;
        }}
        .direction-indicator.ambiguous {{
            background: rgba(149, 165, 166, 0.2);
            color: #bdc3c7;
        }}
        
        .mechanism-tag {{
            background: rgba(255,255,255,0.05);
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8em;
        }}
        
        .n2-feeders {{
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid #333;
            font-size: 0.85em;
            color: #888;
        }}
        .n2-feeders strong {{
            color: #3498db;
        }}
        
        /* Flow Arrow */
        .flow-arrow {{
            text-align: center;
            padding: 20px 0;
            font-size: 2em;
            color: #444;
        }}
        .flow-arrow::before {{
            content: "⬇";
        }}
        
        /* Legend */
        .legend {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 15px;
            margin-top: 40px;
            padding: 20px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85em;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}
        
        /* Summary Stats */
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 30px;
            padding: 20px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
        }}
        .summary-stat {{
            text-align: center;
        }}
        .summary-stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #3498db;
        }}
        .summary-stat-label {{
            font-size: 0.85em;
            color: #888;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Belief Influence Flow Chart</h1>
        <p class="subtitle">Showing top {top_n1} direct (N-1) and {top_n2_per_n1} second-degree (N-2) influences per event</p>
        
        <!-- Belief Target -->
        <div class="belief-target">
            <h2>🎯 {focused['belief']['question']}</h2>
            <div class="belief-stats">
                <div class="belief-stat">
                    <div class="belief-stat-value">{focused['belief']['price']:.0%}</div>
                    <div class="belief-stat-label">Current Price</div>
                </div>
                <div class="belief-stat">
                    <div class="belief-stat-value">${focused['belief']['liquidity']:,.0f}</div>
                    <div class="belief-stat-label">Liquidity</div>
                </div>
            </div>
        </div>
        
        <div class="flow-arrow"></div>
        
        <!-- N-1 Direct Influences -->
        <div class="flow-section">
            <div class="section-header">
                <span class="section-badge">N-1</span>
                <span class="section-title">Direct Influences on Belief</span>
                <span class="section-count">{len(focused['n1_events'])} events</span>
            </div>
            <div class="events-grid">
"""
    
    # Add N-1 event cards
    for i, (event, edge) in enumerate(zip(focused['n1_events'], focused['n1_edges'])):
        conf_class = "high" if edge['confidence'] >= 0.45 else ("medium" if edge['confidence'] >= 0.35 else "low")
        dir_symbol = DIRECTION_ARROWS.get(edge['direction'], "◆")
        
        # Get title and URL
        title = event.get('title', event['action'])
        url = event.get('url', '')
        source = event.get('source', '')[:30]
        
        # Escape HTML in title
        import html as html_lib
        title_safe = html_lib.escape(title[:120])
        if len(title) > 120:
            title_safe += "..."
        
        # Direction text for tooltip
        direction_text = {
            "positive": "Increases likelihood",
            "negative": "Decreases likelihood",
            "ambiguous": "Uncertain effect"
        }.get(edge['direction'], "")
        
        # Build source link
        if url:
            source_html = f'<a href="{url}" target="_blank" style="color: #3498db; text-decoration: none;">📰 {source}</a>'
        else:
            source_html = f'📍 {source}'
        
        html += f"""
                <div class="event-card {event['type']}" title="{direction_text}">
                    <div class="event-header">
                        <span class="event-type">{event['type']}</span>
                        <span class="event-confidence {conf_class}">{edge['confidence']:.0%}</span>
                    </div>
                    <div class="event-action">{title_safe}</div>
                    <div class="event-meta">
                        <span class="direction-indicator {edge['direction']}" title="{direction_text}">{dir_symbol} {edge['direction']}</span>
                        <span class="mechanism-tag" title="How this event influences the outcome">{edge['mechanism'].replace('_', ' ')}</span>
                    </div>
                    <div class="event-meta" style="margin-top: 8px;">
                        <span class="event-meta-item">{source_html}</span>
                        <span class="event-meta-item" title="Reliability of this event information">🎯 {event['certainty']:.0%} certainty</span>
                    </div>
                    {f'<div class="n2-feeders">⬆️ Fed by <strong>{event["n2_feeders"]}</strong> upstream events (strength: {event["n2_strength"]:.2f})</div>' if event['n2_feeders'] > 0 else ''}
                </div>
"""
    
    html += """
            </div>
        </div>
"""
    
    # Add N-2 section if there are any
    if focused['n2_events']:
        html += """
        <div class="flow-arrow"></div>
        
        <!-- N-2 Second-Degree Influences -->
        <div class="flow-section">
            <div class="section-header">
                <span class="section-badge">N-2</span>
                <span class="section-title">Second-Degree Influences</span>
                <span class="section-count">""" + str(len(focused['n2_events'])) + """ events</span>
            </div>
            <div class="events-grid">
"""
        
        for event, edge in zip(focused['n2_events'], focused['n2_edges']):
            # Find the N-1 event this feeds into
            n1_event = next((e for e in focused['n1_events'] if e['id'] == event['feeds_into']), None)
            n1_action = n1_event.get('title', n1_event['action'])[:40] if n1_event else event['feeds_into'][:20]
            
            # Get title and URL
            title = event.get('title', event['action'])
            url = event.get('url', '')
            source = event.get('source', '')[:25] if event.get('source') else ''
            
            import html as html_lib
            title_safe = html_lib.escape(title[:80])
            if len(title) > 80:
                title_safe += "..."
            
            # Build source link
            if url:
                source_html = f'<a href="{url}" target="_blank" style="color: #3498db; text-decoration: none; font-size: 0.8em;">📰 {source}</a>'
            else:
                source_html = f'<span style="font-size: 0.8em;">📍 {source}</span>' if source else ''
            
            html += f"""
                <div class="event-card {event['type']}" style="opacity: 0.85;">
                    <div class="event-header">
                        <span class="event-type">{event['type']}</span>
                        <span class="event-confidence">{edge['confidence']:.0%}</span>
                    </div>
                    <div class="event-action">{title_safe}</div>
                    <div class="event-meta">
                        <span class="direction-indicator {edge['direction']}">{DIRECTION_ARROWS.get(edge['direction'], '◆')} {edge['direction']}</span>
                        <span class="mechanism-tag">{edge['mechanism'].replace('_', ' ')}</span>
                    </div>
                    <div class="event-meta" style="margin-top: 5px;">
                        {source_html}
                    </div>
                    <div class="n2-feeders">➡️ Feeds into: <strong>{n1_action}...</strong></div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
    
    # Add legend and metrics explanation
    html += f"""
        <!-- Event Type Legend -->
        <div class="legend">
            <div class="legend-item"><div class="legend-color" style="background: #e74c3c;"></div> Policy</div>
            <div class="legend-item"><div class="legend-color" style="background: #9b59b6;"></div> Legal</div>
            <div class="legend-item"><div class="legend-color" style="background: #2ecc71;"></div> Economic</div>
            <div class="legend-item"><div class="legend-color" style="background: #f39c12;"></div> Poll</div>
            <div class="legend-item"><div class="legend-color" style="background: #3498db;"></div> Narrative</div>
            <div class="legend-item"><div class="legend-color" style="background: #1abc9c;"></div> Market</div>
            <div class="legend-item"><div class="legend-color" style="background: #95a5a6;"></div> Signal</div>
        </div>
        
        <!-- Metrics Explanation -->
        <div class="legend" style="margin-top: 15px; flex-direction: column; align-items: flex-start; gap: 10px;">
            <h4 style="color: #fff; margin-bottom: 5px;">Metrics Explained</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; width: 100%;">
                <div>
                    <strong style="color: #3498db;">Confidence:</strong>
                    <span style="color: #bbb;"> How strongly evidence supports this influence path (based on narrative overlap, source authority, event significance). Range: 30-100%</span>
                </div>
                <div>
                    <strong style="color: #e74c3c;">Certainty:</strong>
                    <span style="color: #bbb;"> How reliable is this event's information (based on source credibility, corroboration, content quality). Range: 40-95%</span>
                </div>
                <div>
                    <strong style="color: #2ecc71;">Direction:</strong>
                    <span style="color: #bbb;"> Positive (▲) = increases likelihood, Negative (▼) = decreases likelihood, Ambiguous (◆) = unclear effect</span>
                </div>
                <div>
                    <strong style="color: #f39c12;">Mechanism:</strong>
                    <span style="color: #bbb;"> How the event influences belief (legal constraint, economic impact, signaling, expectation shift, narrative amplification, etc.)</span>
                </div>
            </div>
        </div>
        
        <!-- Summary -->
        <div class="summary">
            <div class="summary-stat">
                <div class="summary-stat-value">{focused['stats']['total_n1']}</div>
                <div class="summary-stat-label">N-1 Influences Shown</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-value">{focused['stats']['total_n2']}</div>
                <div class="summary-stat-label">N-2 Influences Shown</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-value">{focused['stats']['original_nodes']}</div>
                <div class="summary-stat-label">Total Events in Graph</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-value">{focused['stats']['original_edges']}</div>
                <div class="summary-stat-label">Total Edges in Graph</div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    # Save
    output_path = Path(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    logger.info(f"Generated focused visualization: {output_path}")
    return str(output_path)


def generate_focused_text(
    graph: BeliefGraph,
    top_n1: int = 10,
    top_n2_per_n1: int = 3,
    min_confidence: float = 0.3
) -> str:
    """
    Generate a text-based focused summary.
    """
    focused = extract_focused_subgraph(graph, top_n1, top_n2_per_n1, min_confidence)
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"BELIEF INFLUENCE FLOW: {focused['belief']['question']}")
    lines.append(f"Price: {focused['belief']['price']:.0%} | Liquidity: ${focused['belief']['liquidity']:,.0f}")
    lines.append("=" * 80)
    
    lines.append(f"\n{'─' * 80}")
    lines.append(f"N-1: DIRECT INFLUENCES ({len(focused['n1_events'])} events)")
    lines.append(f"{'─' * 80}")
    
    for i, (event, edge) in enumerate(zip(focused['n1_events'], focused['n1_edges']), 1):
        dir_sym = DIRECTION_ARROWS.get(edge['direction'], '→')
        lines.append(f"\n{i:2}. [{event['type'].upper():10}] {event['action'][:50]}")
        lines.append(f"    {dir_sym} {edge['mechanism'].replace('_', ' ')} | Confidence: {edge['confidence']:.0%}")
        if event['n2_feeders'] > 0:
            lines.append(f"    ⬆ Fed by {event['n2_feeders']} upstream events")
    
    if focused['n2_events']:
        lines.append(f"\n{'─' * 80}")
        lines.append(f"N-2: SECOND-DEGREE INFLUENCES ({len(focused['n2_events'])} events)")
        lines.append(f"{'─' * 80}")
        
        for event, edge in zip(focused['n2_events'], focused['n2_edges']):
            n1_event = next((e for e in focused['n1_events'] if e['id'] == event['feeds_into']), None)
            n1_name = n1_event['action'][:25] if n1_event else "..."
            
            dir_sym = DIRECTION_ARROWS.get(edge['direction'], '→')
            lines.append(f"\n  [{event['type'].upper():10}] {event['action'][:45]}")
            lines.append(f"    {dir_sym} → {n1_name}... ({edge['confidence']:.0%})")
    
    lines.append(f"\n{'=' * 80}")
    lines.append(f"Summary: {focused['stats']['total_n1']} N-1 + {focused['stats']['total_n2']} N-2 shown")
    lines.append(f"Full graph: {focused['stats']['original_nodes']} nodes, {focused['stats']['original_edges']} edges")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def visualize_focused(
    event_id: str,
    output_path: Optional[str] = None,
    format: str = "html",
    top_n1: int = 10,
    top_n2_per_n1: int = 3,
    min_confidence: float = 0.3
) -> str:
    """
    Main entry point for focused visualization.
    """
    storage = get_storage()
    graph = storage.load(event_id)
    
    if graph is None:
        raise ValueError(f"No graph found for event: {event_id}")
    
    if format == "html":
        if output_path is None:
            output_path = project_root / "data" / "belief_graphs" / f"{event_id}_focused.html"
        return generate_focused_html(graph, str(output_path), top_n1, top_n2_per_n1, min_confidence)
    elif format == "text":
        text = generate_focused_text(graph, top_n1, top_n2_per_n1, min_confidence)
        if output_path:
            with open(output_path, 'w') as f:
                f.write(text)
        else:
            print(text)
        return output_path or ""
    else:
        raise ValueError(f"Unsupported format: {format}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate focused N-1/N-2 belief flow visualization")
    parser.add_argument("event_id", help="Event ID to visualize")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-f", "--format", choices=["html", "text"], default="html", help="Output format")
    parser.add_argument("--top-n1", type=int, default=10, help="Number of N-1 events to show (default: 10)")
    parser.add_argument("--top-n2", type=int, default=3, help="N-2 events per N-1 (default: 3)")
    parser.add_argument("--min-conf", type=float, default=0.3, help="Minimum confidence (default: 0.3)")
    
    args = parser.parse_args()
    
    try:
        result = visualize_focused(
            args.event_id,
            args.output,
            args.format,
            args.top_n1,
            args.top_n2,
            args.min_conf
        )
        if result:
            print(f"Visualization saved to: {result}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
