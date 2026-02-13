#!/usr/bin/env python3
"""
Test script for Polymarket Event Impact Analysis.

Tests the full pipeline:
1. Fetch event tokens from Polymarket Gamma API
2. Fetch price history from CLOB API
3. Load belief graph event nodes
4. Calculate price delta around each event timestamp
5. Generate impact timeline
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from integrations.impact_analyzer import PolymarketImpactAnalyzer, analyze_market_impact


def test_token_fetching(slug: str):
    """Test Step 1: Fetch event tokens."""
    print("\n" + "=" * 60)
    print("STEP 1: FETCH EVENT TOKENS")
    print("=" * 60)
    
    analyzer = PolymarketImpactAnalyzer()
    tokens = analyzer.fetch_event_tokens(slug)
    
    if not tokens:
        print("  ❌ No tokens found!")
        return None
    
    print(f"\n  ✅ Found {len(tokens)} tokens:")
    for token in tokens:
        print(f"    • {token.outcome}: {token.current_price:.3f} (ID: {token.token_id[:30]}...)")
        print(f"      Question: {token.question}")
    
    return tokens


def test_price_history(analyzer, token_id: str):
    """Test Step 2: Fetch price history."""
    print("\n" + "=" * 60)
    print("STEP 2: FETCH PRICE HISTORY")
    print("=" * 60)
    
    history = analyzer.fetch_price_history(token_id)
    
    if not history:
        print("  ❌ No price history!")
        return None
    
    print(f"\n  ✅ Fetched {len(history)} price points")
    print(f"    Time range: {history[0].dt.isoformat()} → {history[-1].dt.isoformat()}")
    print(f"    Price range: {min(p.price for p in history):.3f} → {max(p.price for p in history):.3f}")
    print(f"\n    First 5 data points:")
    for p in history[:5]:
        print(f"      {p.dt.strftime('%Y-%m-%d %H:%M')} → {p.price:.3f}")
    print(f"    Last 5 data points:")
    for p in history[-5:]:
        print(f"      {p.dt.strftime('%Y-%m-%d %H:%M')} → {p.price:.3f}")
    
    return history


def test_impact_calculation(analyzer, history):
    """Test Step 3: Impact calculation with synthetic events."""
    print("\n" + "=" * 60)
    print("STEP 3: IMPACT CALCULATION (SYNTHETIC EVENTS)")
    print("=" * 60)
    
    if not history or len(history) < 10:
        print("  ❌ Not enough price data for testing")
        return
    
    # Test with timestamps from the middle of the price history
    mid_idx = len(history) // 2
    test_timestamps = [
        history[mid_idx].dt,                             # Exact match
        history[mid_idx].dt + timedelta(seconds=30),     # 30s after a data point
        history[0].dt - timedelta(hours=1),              # Before data range
    ]
    
    test_labels = ["Exact timestamp", "30s offset", "Before data range"]
    
    for label, ts in zip(test_labels, test_timestamps):
        result = analyzer.calculate_impact(history, ts)
        print(f"\n  Test: {label}")
        print(f"    Timestamp: {ts.isoformat()}")
        print(f"    Price before: {result['price_before']}")
        print(f"    Price after: {result['price_after']}")
        print(f"    Delta: {result['delta']}")
        print(f"    % Change: {result['pct_change']}")
        print(f"    Quality: {result['data_quality']}")
        print(f"    Before offset: {result['before_offset_seconds']}s")
        print(f"    After offset: {result['after_offset_seconds']}s")


def test_belief_graph_analysis(slug: str, graph_path: str = None):
    """Test Step 4: Full belief graph analysis."""
    print("\n" + "=" * 60)
    print("STEP 4: BELIEF GRAPH ANALYSIS")
    print("=" * 60)
    
    # Find a graph
    if graph_path is None:
        data_dir = Path("data/belief_graphs")
        graphs = list(data_dir.glob("*_graph.json"))
        if not graphs:
            print("  ❌ No belief graphs found in data/belief_graphs/")
            return None
        graph_path = str(graphs[0])
    
    print(f"\n  Using graph: {graph_path}")
    
    analyzer = PolymarketImpactAnalyzer(
        window_minutes=2,
        output_dir=Path("data/impact_analysis")
    )
    
    impacts = analyzer.analyze_belief_graph(graph_path, slug)
    
    if not impacts:
        print("  ❌ No impacts calculated!")
        return None
    
    print(f"\n  ✅ Calculated {len(impacts)} event impacts")
    
    # Generate timeline
    timeline = analyzer.generate_timeline(impacts, slug)
    
    return timeline


def main():
    parser = argparse.ArgumentParser(description="Test Polymarket Event Impact Analysis")
    parser.add_argument("--slug", default="gemini-3pt5-released-by-june-30",
                       help="Polymarket event slug")
    parser.add_argument("--graph", default=None,
                       help="Path to belief graph JSON")
    parser.add_argument("--step", type=int, default=0,
                       help="Run specific step (1-4) or 0 for all")
    parser.add_argument("--window", type=int, default=2,
                       help="Impact window in minutes (default: 2)")
    args = parser.parse_args()
    
    print("╔" + "═" * 58 + "╗")
    print("║   POLYMARKET EVENT IMPACT ANALYSIS — TEST SUITE          ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  Slug: {args.slug}")
    print(f"  Window: ±{args.window} minutes")
    
    analyzer = PolymarketImpactAnalyzer(
        window_minutes=args.window,
        output_dir=Path("data/impact_analysis")
    )
    
    if args.step == 0 or args.step == 1:
        tokens = test_token_fetching(args.slug)
    
    if args.step == 0 or args.step == 2:
        if args.step == 2:
            tokens = analyzer.fetch_event_tokens(args.slug)
        if tokens:
            # Use "Yes" token
            yes_token = None
            for t in tokens:
                if t.outcome.lower() == "yes":
                    yes_token = t
                    break
            target = yes_token or tokens[0]
            history = test_price_history(analyzer, target.token_id)
    
    if args.step == 0 or args.step == 3:
        if args.step == 3:
            tokens = analyzer.fetch_event_tokens(args.slug)
            target = tokens[0] if tokens else None
            history = analyzer.fetch_price_history(target.token_id) if target else None
        if history:
            test_impact_calculation(analyzer, history)
    
    if args.step == 0 or args.step == 4:
        timeline = test_belief_graph_analysis(args.slug, args.graph)
    
    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETE")
    print("  Output dir: data/impact_analysis/")
    print("=" * 60)


if __name__ == "__main__":
    main()
