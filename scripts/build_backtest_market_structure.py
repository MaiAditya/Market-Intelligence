#!/usr/bin/env python3
"""
Build backtest market structure from closed Polymarket events.

What it does:
1) Fetches full event details for each parent slug
2) Detects single-binary vs multi-binary parent markets
3) Extracts child binary markets
4) Ranks child markets by volume and keeps top-K (default 5)
5) Checks token-level price-history availability for selected children

Output:
- Enriched parent/child structure JSON
- Flat backtest target list that can be consumed by a runner
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"


def _parse_json_field(value: Any, default: Any):
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return value


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except Exception:
        return default


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_event_detail(session: requests.Session, slug: str, timeout: int) -> Optional[Dict[str, Any]]:
    url = f"{GAMMA_URL}/events/slug/{slug}"
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def extract_child_markets(
    event_detail: Dict[str, Any],
    parent_slug: str,
    parent_start_date: Optional[str] = None,
    parent_end_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    markets = event_detail.get("markets", []) or []
    children: List[Dict[str, Any]] = []

    for m in markets:
        token_ids = _parse_json_field(m.get("clobTokenIds"), [])
        outcomes = _parse_json_field(m.get("outcomes"), [])
        prices = _parse_json_field(m.get("outcomePrices"), [])

        child = {
            "market_slug": m.get("slug") or parent_slug,
            "question": m.get("question") or event_detail.get("title") or "",
            "volume": _to_float(m.get("volume"), 0.0),
            "liquidity": _to_float(m.get("liquidity"), 0.0),
            "start_date": m.get("startDate") or m.get("createdAt") or event_detail.get("startDate") or parent_start_date,
            "end_date": m.get("endDate") or event_detail.get("endDate") or parent_end_date,
            "active": m.get("active"),
            "closed": m.get("closed"),
            "tokens": [],
        }

        for i, tid in enumerate(token_ids):
            outcome = outcomes[i] if i < len(outcomes) else f"outcome_{i}"
            price = _to_float(prices[i], 0.5) if i < len(prices) else 0.5
            child["tokens"].append(
                {
                    "token_id": str(tid),
                    "outcome": str(outcome),
                    "current_price": price,
                }
            )

        # keep only binary-like children for this workflow
        if len(child["tokens"]) >= 2:
            children.append(child)

    return children


def fetch_price_points_count(
    session: requests.Session,
    token_id: str,
    timeout: int,
    fidelity: int,
    interval: str = "max",
) -> int:
    url = f"{CLOB_URL}/prices-history"
    params = {
        "market": token_id,
        "interval": interval,
        "fidelity": fidelity,
    }
    try:
        resp = session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        return len(payload.get("history", []) or [])
    except Exception:
        return 0


def choose_best_token_with_history(
    session: requests.Session,
    tokens: List[Dict[str, Any]],
    timeout: int,
    fidelity: int,
) -> Dict[str, Any]:
    """
    Pick token with best available history.
    Preference order:
    1) Yes outcome with non-zero history
    2) Any token with max history points
    """
    token_checks: List[Dict[str, Any]] = []

    for t in tokens:
        points = fetch_price_points_count(
            session=session,
            token_id=t["token_id"],
            timeout=timeout,
            fidelity=fidelity,
        )
        token_checks.append(
            {
                "token_id": t["token_id"],
                "outcome": t["outcome"],
                "current_price": t["current_price"],
                "history_points": points,
                "has_price_history": points > 0,
            }
        )

    yes_with_history = [
        x for x in token_checks
        if str(x.get("outcome", "")).lower() == "yes" and x["has_price_history"]
    ]
    if yes_with_history:
        best = max(yes_with_history, key=lambda x: x["history_points"])
    else:
        best = max(token_checks, key=lambda x: x["history_points"]) if token_checks else {}

    return {
        "selected_token": best or None,
        "token_checks": token_checks,
        "any_history_available": any(x["has_price_history"] for x in token_checks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build enriched backtest structure for closed markets")
    parser.add_argument(
        "--closed-events",
        default="data/backtests/closed_events.json",
        help="Input closed-events JSON (from fetch_closed_events.py)",
    )
    parser.add_argument("--num-events", type=int, default=50, help="Number of parent events to process")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset in closed-events list")
    parser.add_argument("--top-child-markets", type=int, default=5, help="Top child markets by volume per parent")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    parser.add_argument("--fidelity", type=int, default=1, help="Price-history fidelity in minutes")
    parser.add_argument(
        "--require-history",
        action="store_true",
        help="Keep only child markets with at least one token having price history",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/backtests/closed_events_enriched.json",
        help="Output enriched structure JSON path",
    )
    args = parser.parse_args()

    with open(args.closed_events, "r", encoding="utf-8") as f:
        closed_payload = json.load(f)
    raw_events = closed_payload.get("events", []) or []
    selected_parents = raw_events[args.start_index: args.start_index + args.num_events]

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "AIMarketIntelligence/1.0",
            "Accept": "application/json",
        }
    )

    parents_out: List[Dict[str, Any]] = []
    backtest_targets: List[Dict[str, Any]] = []

    for idx, parent in enumerate(selected_parents, start=1):
        slug = parent.get("slug")
        if not slug:
            continue

        detail = fetch_event_detail(session, slug, args.timeout)
        if detail is None:
            parents_out.append(
                {
                    "parent_slug": slug,
                    "title": parent.get("question"),
                    "status": "fetch_failed",
                }
            )
            continue

        children = extract_child_markets(
            detail,
            slug,
            parent_start_date=parent.get("start_date"),
            parent_end_date=parent.get("end_date"),
        )
        child_count = len(children)
        market_type = "single_binary" if child_count <= 1 else "multi_binary_parent"

        children_sorted = sorted(children, key=lambda x: x.get("volume", 0.0), reverse=True)
        selected_children = children_sorted[: args.top_child_markets]

        selected_children_out: List[Dict[str, Any]] = []
        for child in selected_children:
            token_info = choose_best_token_with_history(
                session=session,
                tokens=child["tokens"],
                timeout=args.timeout,
                fidelity=args.fidelity,
            )
            child_out = {
                "market_slug": child["market_slug"],
                "question": child["question"],
                "volume": child["volume"],
                "liquidity": child["liquidity"],
                "start_date": child["start_date"],
                "end_date": child["end_date"],
                "selected_token": token_info["selected_token"],
                "any_history_available": token_info["any_history_available"],
                "token_checks": token_info["token_checks"],
            }
            selected_children_out.append(child_out)

        if args.require_history:
            selected_children_out = [c for c in selected_children_out if c["any_history_available"]]

        parent_out = {
            "parent_slug": slug,
            "title": detail.get("title") or parent.get("question"),
            "market_type": market_type,
            "markets_count": child_count,
            "selected_children_count": len(selected_children_out),
            "parent_volume": _to_float(parent.get("volume"), 0.0),
            "parent_liquidity": _to_float(parent.get("liquidity"), 0.0),
            "parent_start_date": parent.get("start_date"),
            "parent_end_date": parent.get("end_date"),
            "selected_children": selected_children_out,
            "status": "ok",
        }
        parents_out.append(parent_out)

        for c in selected_children_out:
            backtest_targets.append(
                {
                    "parent_slug": slug,
                    "market_type": market_type,
                    "child_market_slug": c["market_slug"],
                    "child_question": c["question"],
                    "child_volume": c["volume"],
                    "parent_start_date": parent.get("start_date"),
                    "parent_end_date": parent.get("end_date"),
                    "start_date": c.get("start_date"),
                    "end_date": c.get("end_date"),
                    "selected_token": c["selected_token"],
                    "token_checks": c.get("token_checks", []),
                    "any_history_available": c["any_history_available"],
                }
            )

        print(
            f"[{idx}/{len(selected_parents)}] {slug} "
            f"type={market_type} markets={child_count} selected={len(selected_children_out)}"
        )

    payload = {
        "generated_at": _utc_now_iso(),
        "params": {
            "num_events": args.num_events,
            "start_index": args.start_index,
            "top_child_markets": args.top_child_markets,
            "fidelity": args.fidelity,
            "require_history": args.require_history,
        },
        "parents_count": len(parents_out),
        "targets_count": len(backtest_targets),
        "parents": parents_out,
        "backtest_targets": backtest_targets,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved enriched structure to: {out_path}")
    print(f"Parent events processed: {len(parents_out)}")
    print(f"Backtest targets generated: {len(backtest_targets)}")


if __name__ == "__main__":
    main()
