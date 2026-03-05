#!/usr/bin/env python3
"""
Fetch Polymarket Price Deltas for Event Clusters

Takes a clustering diagnostics JSON, resolves the Polymarket market,
fetches CLOB price history, and calculates price movement around
each cluster's timestamp.

Usage:
    python scripts/fetch_cluster_prices.py [--slug SLUG] [--diagnostics PATH]

Output:
    data/diagnostics/clustering_<event_id>_with_prices.json
"""

import argparse
import json
import logging
import sys
import time
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
from pathlib import Path

import requests

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fetch_cluster_prices")

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"


def resolve_market(slug: str) -> dict:
    """Resolve a Polymarket event slug to market metadata + token_ids."""
    url = f"{GAMMA_URL}/events?slug={slug}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError(f"No market found for slug: {slug}")
    event = data[0] if isinstance(data, list) else data
    markets = event.get("markets", [])
    result = {
        "event_title": event.get("title", ""),
        "slug": slug,
        "sub_markets": [],
    }
    for m in markets:
        tokens = json.loads(m.get("clobTokenIds", "[]")) if isinstance(m.get("clobTokenIds"), str) else m.get("clobTokenIds", [])
        prices = json.loads(m.get("outcomePrices", "[]")) if isinstance(m.get("outcomePrices"), str) else m.get("outcomePrices", [])
        result["sub_markets"].append({
            "question": m.get("question", ""),
            "condition_id": m.get("conditionId", ""),
            "yes_token_id": tokens[0] if len(tokens) > 0 else None,
            "no_token_id": tokens[1] if len(tokens) > 1 else None,
            "yes_price": float(prices[0]) if len(prices) > 0 else None,
            "no_price": float(prices[1]) if len(prices) > 1 else None,
            "volume": float(m.get("volume", 0) or 0),
        })
    return result


def fetch_price_history(token_id: str, fidelity: int = 60) -> list:
    """Fetch CLOB price history for a token. Returns list of {t, p}."""
    url = f"{CLOB_URL}/prices-history"
    params = {"market": token_id, "interval": "all", "fidelity": fidelity}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("history", [])


def find_price_at(history: list, timestamps: list, target_epoch: float, window_seconds: int = 7200) -> dict:
    """
    Find the price at a specific time using binary search.

    Returns {price, offset_seconds, data_quality} or None.
    """
    if not timestamps:
        return None

    idx = bisect_left(timestamps, target_epoch)

    best = None
    best_offset = float("inf")

    for candidate in [idx - 1, idx]:
        if 0 <= candidate < len(timestamps):
            offset = abs(timestamps[candidate] - target_epoch)
            if offset < best_offset:
                best_offset = offset
                best = candidate

    if best is None or best_offset > window_seconds:
        return None

    quality = "exact" if best_offset < 120 else ("close" if best_offset < 3600 else "interpolated")
    return {
        "price": history[best]["p"],
        "offset_seconds": int(best_offset),
        "data_quality": quality,
    }


def compute_price_deltas(clusters: list, history: list, window_minutes: int = 120) -> list:
    """
    For each cluster with a timestamp, compute price_before and price_after.

    window_minutes: how far before/after the event timestamp to measure price.
    """
    if not history:
        return [
            {**c, "price_data": "no_history_available"}
            for c in clusters
        ]

    timestamps = [h["t"] for h in history]
    history_start = timestamps[0]
    history_end = timestamps[-1]
    window_sec = window_minutes * 60
    results = []

    for cluster in clusters:
        canon = cluster.get("canonical_event", {})
        ts_str = canon.get("timestamp")
        if not ts_str:
            results.append({**cluster, "price_data": "no_timestamp"})
            continue

        # Parse timestamp
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            event_epoch = dt.timestamp()
        except Exception:
            results.append({**cluster, "price_data": "invalid_timestamp"})
            continue

        # Check if event is within price history range
        if event_epoch < history_start - window_sec or event_epoch > history_end + window_sec:
            results.append({**cluster, "price_data": "out_of_range"})
            continue

        before = find_price_at(history, timestamps, event_epoch - window_sec)
        at_event = find_price_at(history, timestamps, event_epoch)
        after = find_price_at(history, timestamps, event_epoch + window_sec)

        price_info = {
            "price_before": before["price"] if before else None,
            "price_at_event": at_event["price"] if at_event else None,
            "price_after": after["price"] if after else None,
            "before_offset_seconds": before["offset_seconds"] if before else None,
            "after_offset_seconds": after["offset_seconds"] if after else None,
            "before_quality": before["data_quality"] if before else None,
            "after_quality": after["data_quality"] if after else None,
            "window_minutes": window_minutes,
        }

        # Calculate deltas
        if before and after:
            delta = after["price"] - before["price"]
            pct = (delta / before["price"] * 100) if before["price"] > 0 else 0
            price_info["delta"] = round(delta, 4)
            price_info["pct_change"] = round(pct, 2)
            price_info["price_data"] = "available"
        elif at_event:
            price_info["delta"] = None
            price_info["pct_change"] = None
            price_info["price_data"] = "partial"
        else:
            price_info["price_data"] = "no_data_near_event"

        results.append({**cluster, "price_impact": price_info})

    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch Polymarket price deltas for event clusters")
    parser.add_argument("--slug", default="gemini-3pt5-released-by-june-30",
                        help="Polymarket event slug")
    parser.add_argument("--diagnostics",
                        help="Path to clustering diagnostics JSON (default: auto-detect)")
    parser.add_argument("--sub-market", type=int, default=0,
                        help="Sub-market index to use (0=first, default)")
    parser.add_argument("--window", type=int, default=120,
                        help="Price window in minutes before/after event (default: 120)")
    parser.add_argument("--fidelity", type=int, default=60,
                        help="CLOB price history fidelity in seconds (default: 60)")
    args = parser.parse_args()

    # Find diagnostics file
    diag_dir = project_root / "data" / "diagnostics"
    if args.diagnostics:
        diag_path = Path(args.diagnostics)
    else:
        # Auto-detect: find the most recent clustering_*.json
        candidates = sorted(diag_dir.glob("clustering_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        candidates = [c for c in candidates if "_with_prices" not in c.name]
        if not candidates:
            print("No diagnostics JSON found. Run clustering_diagnostics.py first.")
            sys.exit(1)
        diag_path = candidates[0]

    print(f"Loading diagnostics: {diag_path}")
    with open(diag_path, "r") as f:
        diag = json.load(f)

    # Step 1: Resolve market
    print(f"\n{'='*60}")
    print(f"STEP 1: Resolving Polymarket market: {args.slug}")
    print(f"{'='*60}")
    market = resolve_market(args.slug)
    print(f"  Event: {market['event_title']}")
    print(f"  Sub-markets: {len(market['sub_markets'])}")
    for i, sm in enumerate(market["sub_markets"]):
        marker = " ← SELECTED" if i == args.sub_market else ""
        print(f"    [{i}] {sm['question']} (YES={sm['yes_price']}, vol=${sm['volume']:,.0f}){marker}")

    selected = market["sub_markets"][args.sub_market]
    token_id = selected["yes_token_id"]
    print(f"\n  Using YES token: {token_id[:20]}...")

    # Step 2: Fetch price history
    print(f"\n{'='*60}")
    print(f"STEP 2: Fetching CLOB price history (fidelity={args.fidelity}s)")
    print(f"{'='*60}")
    history = fetch_price_history(token_id, fidelity=args.fidelity)
    print(f"  Data points: {len(history)}")
    if history:
        t0 = datetime.fromtimestamp(history[0]["t"])
        t1 = datetime.fromtimestamp(history[-1]["t"])
        print(f"  Range: {t0.isoformat()} → {t1.isoformat()}")
        print(f"  Price: {history[0]['p']} → {history[-1]['p']}")

    # Step 3: Compute price deltas for each cluster
    print(f"\n{'='*60}")
    print(f"STEP 3: Computing price deltas (±{args.window}min window)")
    print(f"{'='*60}")
    clusters = diag.get("steps", {}).get("9_final_clusters", {}).get("clusters", [])
    enriched = compute_price_deltas(clusters, history, window_minutes=args.window)

    # Stats
    available = [c for c in enriched if c.get("price_impact", {}).get("price_data") == "available"]
    out_of_range = [c for c in enriched if c.get("price_data") == "out_of_range" or c.get("price_impact", {}).get("price_data") == "out_of_range"]
    print(f"  Clusters with price data: {len(available)}/{len(enriched)}")
    print(f"  Out of price range: {len(out_of_range)}")

    for c in enriched:
        pi = c.get("price_impact", {})
        canon = c.get("canonical_event", {})
        title = (canon.get("raw_title") or "?")[:50]
        if pi.get("price_data") == "available":
            print(f"    ✅ {title}")
            print(f"       Δ={pi['delta']:+.4f} ({pi['pct_change']:+.2f}%)  "
                  f"before={pi['price_before']:.3f} → after={pi['price_after']:.3f}")
        else:
            status = c.get("price_data", pi.get("price_data", "?"))
            print(f"    ⚠️  {title} [{status}]")

    # Step 4: Build output
    print(f"\n{'='*60}")
    print(f"STEP 4: Writing enriched JSON")
    print(f"{'='*60}")

    output = {
        "meta": {
            **diag.get("meta", {}),
            "polymarket": {
                "slug": args.slug,
                "event_title": market["event_title"],
                "sub_market": selected["question"],
                "token_id": token_id,
                "current_price": selected["yes_price"],
                "volume": selected["volume"],
            },
            "price_history": {
                "data_points": len(history),
                "date_range": {
                    "start": datetime.fromtimestamp(history[0]["t"]).isoformat() if history else None,
                    "end": datetime.fromtimestamp(history[-1]["t"]).isoformat() if history else None,
                },
                "price_range": {
                    "start": history[0]["p"] if history else None,
                    "end": history[-1]["p"] if history else None,
                },
            },
            "price_window_minutes": args.window,
        },
        "steps": diag.get("steps", {}),
        "price_history_raw": history,
        "clusters_with_prices": enriched,
    }

    out_name = diag_path.stem + "_with_prices.json"
    out_path = diag_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    size_kb = out_path.stat().st_size / 1024
    print(f"\n  Written: {out_path}")
    print(f"  Size: {size_kb:.0f} KB")
    print(f"\n  Open the cluster viewer and load this file to visualize!")


if __name__ == "__main__":
    main()
