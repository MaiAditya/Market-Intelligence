#!/usr/bin/env python3
"""
Fetch closed Polymarket events for backtest candidate selection.

Usage:
  python scripts/fetch_closed_events.py --limit 200 --min-volume 50000 -o data/backtests/closed_events.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _is_closed_event(event: Dict[str, Any], now: datetime) -> bool:
    if event.get("closed") is True:
        return True
    end_dt = _parse_dt(event.get("endDate"))
    return bool(end_dt and end_dt <= now)


def fetch_closed_events(
    limit: int,
    min_volume: float,
    max_pages: int,
    timeout: int,
) -> List[Dict[str, Any]]:
    gamma = "https://gamma-api.polymarket.com/events"
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "AIMarketIntelligence/1.0",
            "Accept": "application/json",
        }
    )

    now = datetime.now(timezone.utc)
    offset = 0
    page_size = min(100, max(10, limit))
    all_events: List[Dict[str, Any]] = []

    for _ in range(max_pages):
        params = {
            "closed": "true",
            "limit": page_size,
            "offset": offset,
            "order": "volume",
            "ascending": "false",
        }
        resp = session.get(gamma, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break

        all_events.extend(data)
        if len(all_events) >= limit:
            break
        offset += len(data)

    dedup: Dict[str, Dict[str, Any]] = {}
    for e in all_events:
        slug = e.get("slug")
        if not slug:
            continue
        if not _is_closed_event(e, now):
            continue
        vol = float(e.get("volume", 0) or 0)
        if vol < min_volume:
            continue
        dedup[slug] = {
            "slug": slug,
            "question": e.get("title"),
            "event_id": e.get("id"),
            "volume": vol,
            "liquidity": float(e.get("liquidity", 0) or 0),
            "closed": bool(e.get("closed", False)),
            "start_date": e.get("startDate") or e.get("createdAt") or e.get("creationDate"),
            "end_date": e.get("endDate"),
            "resolution_source": e.get("resolutionSource"),
            "category": e.get("category"),
        }

    ranked = sorted(dedup.values(), key=lambda x: x.get("volume", 0), reverse=True)
    return ranked[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch closed Polymarket events")
    parser.add_argument("--limit", type=int, default=100, help="Maximum events to output")
    parser.add_argument("--min-volume", type=float, default=10000, help="Minimum event volume filter")
    parser.add_argument("--max-pages", type=int, default=10, help="Maximum API pages to scan")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument(
        "-o",
        "--output",
        default="data/backtests/closed_events.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    events = fetch_closed_events(
        limit=args.limit,
        min_volume=args.min_volume,
        max_pages=args.max_pages,
        timeout=args.timeout,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(events),
        "params": {
            "limit": args.limit,
            "min_volume": args.min_volume,
            "max_pages": args.max_pages,
        },
        "events": events,
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(events)} closed events to {output}")
    for i, e in enumerate(events[:20], 1):
        print(
            f"{i:2d}. {e['slug']} | volume={e['volume']:.0f} | "
            f"end={e.get('end_date')}"
        )


if __name__ == "__main__":
    main()
