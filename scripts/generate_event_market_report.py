#!/usr/bin/env python3
"""
Generate a consolidated market report JSON for one event.

Output includes:
1) Price changes mapped to extracted graph events
2) N-1 / N-2 causal graph structure with edge mapping details
3) Related news references for each N-1 / N-2 node
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from belief_graph.focused_visualizer import extract_focused_subgraph
from belief_graph.storage import get_storage
from integrations.impact_analyzer import PolymarketImpactAnalyzer
from pipeline.event_registry import get_registry
from pipeline.normalizer import DocumentNormalizer
from utils.json_utils import dump_json

logger = logging.getLogger(__name__)


def _index_impacts_by_event(timeline: Dict) -> Dict[str, Dict]:
    impacts = timeline.get("impacts", []) if timeline else []
    return {i.get("event_id"): i for i in impacts if i.get("event_id")}


def _node_news(node: Dict, normalizer: DocumentNormalizer) -> List[Dict]:
    news = []
    doc_id = node.get("source_doc_id")

    if doc_id:
        doc = normalizer.load(doc_id)
        if doc:
            news.append(
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "url": doc.url,
                    "timestamp": doc.timestamp.isoformat() if doc.timestamp else None,
                    "source_type": doc.source_type,
                    "query_used": doc.query_used,
                    "query_type": doc.query_type,
                }
            )

    # Fallback from graph node fields
    if not news:
        news.append(
            {
                "doc_id": doc_id,
                "title": node.get("raw_title"),
                "url": node.get("url"),
                "timestamp": node.get("timestamp"),
                "source_type": node.get("source"),
            }
        )
    return news


def _enrich_focused(
    focused: Dict,
    node_lookup: Dict[str, Dict],
    impacts_by_event: Dict[str, Dict],
    normalizer: DocumentNormalizer,
) -> Dict:
    n1_nodes = []
    for n in focused.get("n1_events", []):
        node_id = n.get("id")
        base_node = node_lookup.get(node_id, {})
        n1_nodes.append(
            {
                **n,
                "graph_node": base_node,
                "price_impact": impacts_by_event.get(node_id),
                "related_news": _node_news(base_node, normalizer),
            }
        )

    n2_nodes = []
    for n in focused.get("n2_events", []):
        node_id = n.get("id")
        base_node = node_lookup.get(node_id, {})
        n2_nodes.append(
            {
                **n,
                "graph_node": base_node,
                "price_impact": impacts_by_event.get(node_id),
                "related_news": _node_news(base_node, normalizer),
            }
        )

    return {
        "belief": focused.get("belief", {}),
        "stats": focused.get("stats", {}),
        "n1_nodes": n1_nodes,
        "n2_nodes": n2_nodes,
        "n1_edges": focused.get("n1_edges", []),
        "n2_edges": focused.get("n2_edges", []),
    }


def generate_report(
    event_id: str,
    slug: Optional[str],
    graph_path: Optional[str],
    output_path: Optional[str],
    top_n1: int,
    top_n2: int,
    min_conf: float,
    impact_window: int,
    price_history_source: str = "clob",
    dome_bearer_token: Optional[str] = None,
    orders_max_pages: int = 30,
    orders_request_delay_sec: float = 0.35,
    orders_only: bool = False,
    dome_bearer_token_source: str = "none",
    skip_out_of_window_events: bool = False,
    explicit_token_candidates: Optional[List[Dict]] = None,
    orders_fetch_all: bool = False,
    orders_fetch_all_max_pages: int = 2000,
    price_mapping_mode: str = "all_events",
    burst_gap_minutes: int = 90,
    max_event_bursts: int = 1,
    burst_buffer_minutes: int = 120,
) -> Dict:
    registry = get_registry()
    event = registry.get_event(event_id)
    if event is None and not slug:
        raise ValueError(f"Event not found: {event_id}")

    market_slug = slug or (event.polymarket_slug if event else None)
    if not market_slug:
        raise ValueError(f"No market slug available for event: {event_id}")
    storage = get_storage()
    graph = storage.load(event_id)
    if graph is None:
        raise ValueError(f"No belief graph found for event: {event_id}")

    # Focused N-1/N-2 extraction
    focused = extract_focused_subgraph(
        graph,
        top_n1=top_n1,
        top_n2_per_n1=top_n2,
        min_confidence=min_conf,
    )

    # Optional price impact mapping
    timeline = {}
    try:
        prefer_orders = price_history_source in {"orders", "auto"}
        analyzer = PolymarketImpactAnalyzer(
            window_minutes=impact_window,
            prefer_orders_history=prefer_orders,
            orders_only=orders_only or price_history_source == "orders",
            dome_bearer_token=dome_bearer_token,
            orders_max_pages=orders_max_pages,
            orders_fetch_all=orders_fetch_all,
            orders_fetch_all_max_pages=orders_fetch_all_max_pages,
            orders_request_delay_sec=orders_request_delay_sec,
            price_mapping_mode=price_mapping_mode,
            burst_gap_minutes=burst_gap_minutes,
            max_event_bursts=max_event_bursts,
            burst_buffer_minutes=burst_buffer_minutes,
        )
        resolved_graph_path = graph_path or str(
            project_root / "data" / "belief_graphs" / f"{event_id}_graph.json"
        )
        impacts = analyzer.analyze_belief_graph(
            resolved_graph_path,
            market_slug,
            enforce_market_window=True,
            skip_out_of_window_events=skip_out_of_window_events,
            explicit_tokens=explicit_token_candidates,
        )
        timeline = analyzer.generate_timeline(impacts, market_slug, save=False)
    except Exception as e:
        logger.warning(f"Impact analysis failed, continuing without price impacts: {e}")

    impacts_by_event = _index_impacts_by_event(timeline)
    normalizer = DocumentNormalizer()
    node_lookup = {k: v.to_dict() for k, v in graph.event_nodes.items()}
    focused_enriched = _enrich_focused(focused, node_lookup, impacts_by_event, normalizer)

    report = {
        "event_id": event_id,
        "market_slug": market_slug,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": {
            "top_n1": top_n1,
            "top_n2": top_n2,
            "min_confidence": min_conf,
            "impact_window_minutes": impact_window,
            "price_history_source": price_history_source,
            "dome_bearer_token_source": dome_bearer_token_source,
            "orders_only": orders_only,
            "skip_out_of_window_events": skip_out_of_window_events,
            "explicit_token_candidates_count": len(explicit_token_candidates or []),
            "orders_fetch_all": orders_fetch_all,
            "orders_fetch_all_max_pages": orders_fetch_all_max_pages,
            "price_mapping_mode": price_mapping_mode,
            "burst_gap_minutes": burst_gap_minutes,
            "max_event_bursts": max_event_bursts,
            "burst_buffer_minutes": burst_buffer_minutes,
        },
        "summary": {
            "graph_nodes": len(graph.event_nodes),
            "graph_edges": len(graph.edges),
            "n1_count": len(focused_enriched["n1_nodes"]),
            "n2_count": len(focused_enriched["n2_nodes"]),
            "price_impacts_available": len(impacts_by_event),
        },
        "price_timeline": timeline.get("summary", {}),
        "price_history_runtime": getattr(analyzer, "_last_selected_token_meta", {}),
        "price_changes_by_event": [
            i
            for i in timeline.get("impacts", [])
            if i.get("delta") is not None
        ] if timeline else [],
        "focused_graph": focused_enriched,
    }

    out = output_path or str(project_root / "data" / "output" / f"{event_id}_market_report.json")
    with open(out, "w", encoding="utf-8") as f:
        dump_json(report, f)
    logger.info(f"Report written to {out}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate consolidated event market report")
    parser.add_argument("--event", "-e", required=True, help="Event ID")
    parser.add_argument("--slug", help="Polymarket slug (optional, from registry if omitted)")
    parser.add_argument("--graph", help="Path to graph json (optional)")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("--top-n1", type=int, default=15, help="Top N-1 nodes")
    parser.add_argument("--top-n2", type=int, default=5, help="Top N-2 per N-1")
    parser.add_argument("--min-conf", type=float, default=0.3, help="Minimum confidence for N-1/N-2")
    parser.add_argument("--impact-window", type=int, default=2, help="Impact window (minutes)")
    parser.add_argument(
        "--price-history-source",
        choices=["clob", "orders", "auto"],
        default="clob",
        help="Price history source for impact mapping",
    )
    parser.add_argument("--dome-bearer-token", default=None, help="Dome API bearer token (optional)")
    parser.add_argument(
        "--dome-bearer-token-source",
        choices=["arg", "env", "none"],
        default="none",
        help="How Dome token was supplied (for audit metadata)",
    )
    parser.add_argument("--orders-max-pages", type=int, default=30, help="Max orders API pages per token")
    parser.add_argument(
        "--orders-request-delay-sec",
        type=float,
        default=0.35,
        help="Delay between successful orders API page requests",
    )
    parser.add_argument(
        "--orders-only",
        action="store_true",
        help="Do not fallback to CLOB history if orders-based history fails",
    )
    parser.add_argument(
        "--orders-fetch-all",
        action="store_true",
        help="Fetch all available Dome orders pages (until has_more=false) for selected token",
    )
    parser.add_argument(
        "--orders-fetch-all-max-pages",
        type=int,
        default=2000,
        help="Safety cap for --orders-fetch-all mode",
    )
    parser.add_argument(
        "--skip-out-of-window-events",
        action="store_true",
        help="Drop graph events outside selected token history window instead of keeping as out_of_range",
    )
    parser.add_argument(
        "--price-mapping-mode",
        choices=["all_events", "burst_events"],
        default="all_events",
        help="all_events: map all graph events; burst_events: map only dense contiguous event windows",
    )
    parser.add_argument("--burst-gap-minutes", type=int, default=90, help="Max gap between contiguous events in burst mode")
    parser.add_argument("--max-event-bursts", type=int, default=1, help="How many top dense bursts to map in burst mode")
    parser.add_argument("--burst-buffer-minutes", type=int, default=120, help="Buffer around selected burst windows")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    generate_report(
        event_id=args.event,
        slug=args.slug,
        graph_path=args.graph,
        output_path=args.output,
        top_n1=args.top_n1,
        top_n2=args.top_n2,
        min_conf=args.min_conf,
        impact_window=args.impact_window,
        price_history_source=args.price_history_source,
        dome_bearer_token=args.dome_bearer_token,
        dome_bearer_token_source=args.dome_bearer_token_source,
        orders_max_pages=args.orders_max_pages,
        orders_request_delay_sec=args.orders_request_delay_sec,
        orders_only=args.orders_only,
        skip_out_of_window_events=args.skip_out_of_window_events,
        orders_fetch_all=args.orders_fetch_all,
        orders_fetch_all_max_pages=args.orders_fetch_all_max_pages,
        price_mapping_mode=args.price_mapping_mode,
        burst_gap_minutes=args.burst_gap_minutes,
        max_event_bursts=args.max_event_bursts,
        burst_buffer_minutes=args.burst_buffer_minutes,
    )


if __name__ == "__main__":
    main()
