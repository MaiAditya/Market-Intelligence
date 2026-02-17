#!/usr/bin/env python3
"""
Build offline token price-history series from saved Dome order-history JSON.

Input format (single file or directory of files):
{
  "orders": [...],
  "pagination": {...}
}

Output format includes:
1) CLOB-compatible close series for existing pipeline:
   {"history": [{"t": <unix_sec>, "p": <close_price>}, ...]}
2) Rich OHLCV bars for backtest diagnostics.
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import requests
import logging
import random
import time


logger = logging.getLogger("build_price_history_from_orders")


@dataclass
class Fill:
    timestamp: int
    price: float
    size: float
    side: str
    tx_hash: str
    token_id: str


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _iter_input_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.exists():
        return
    for p in sorted(path.glob("*.json")):
        if p.is_file():
            yield p


def _fetch_orders_pages(
    token_id: str,
    api_url: str,
    pages_dir: Path,
    limit: int,
    max_pages: int,
    timeout: int,
    pagination_key: str = "",
    api_key: str = "",
    api_key_header: str = "x-api-key",
    auth_bearer: str = "",
    extra_headers: Optional[List[str]] = None,
    max_retries: int = 6,
    backoff_base_sec: float = 1.5,
    backoff_max_sec: float = 60.0,
    request_delay_sec: float = 0.0,
) -> List[Path]:
    """
    Fetch paginated Dome orders for a token and persist pages as JSON files.
    """
    pages_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "AIMarketIntelligence/1.0"})
    if api_key:
        session.headers[api_key_header] = api_key
    if auth_bearer:
        session.headers["Authorization"] = f"Bearer {auth_bearer}"
    for header in (extra_headers or []):
        if ":" not in header:
            continue
        k, v = header.split(":", 1)
        session.headers[k.strip()] = v.strip()

    files: List[Path] = []
    next_key = pagination_key
    page = 1

    # Auto-resume: if pages already exist and no explicit pagination key provided,
    # continue from the last saved page.
    existing = sorted(pages_dir.glob(f"{token_id}_page_*.json"))
    if existing and not next_key:
        files.extend(existing)
        last_file = existing[-1]
        with open(last_file, "r", encoding="utf-8") as f:
            last_payload = json.load(f)
        last_pagination = last_payload.get("pagination", {}) or {}
        last_has_more = bool(last_pagination.get("has_more"))
        last_next_key = str(last_pagination.get("pagination_key", "") or "")
        if last_has_more and last_next_key:
            page = len(existing) + 1
            next_key = last_next_key
            logger.info(
                "Resuming from existing pages: count=%s next_page=%s",
                len(existing),
                page,
            )
        else:
            logger.info("Existing pages already complete; no additional fetch needed")
            return files

    logger.info(
        "Fetching orders pages from API: url=%s token_id=%s limit=%s max_pages=%s",
        api_url,
        token_id,
        limit,
        max_pages,
    )

    while page <= max_pages:
        params: Dict[str, Any] = {"token_id": token_id, "limit": limit}
        if next_key:
            params["pagination_key"] = next_key

        resp = None
        for attempt in range(1, max_retries + 1):
            resp = session.get(api_url, params=params, timeout=timeout)
            if resp.status_code == 403:
                raise SystemExit(
                    "403 Forbidden from orders API. Provide credentials via "
                    "--api-key/--api-key-header or --auth-bearer or --header 'Key: Value'."
                )
            if resp.status_code not in (429, 500, 502, 503, 504):
                break

            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    wait_s = float(retry_after)
                except Exception:
                    wait_s = 0.0
            else:
                wait_s = min(backoff_max_sec, backoff_base_sec * (2 ** (attempt - 1)))
                wait_s += random.uniform(0.0, 0.5)

            logger.warning(
                "Transient API error status=%s on page=%s attempt=%s/%s; sleeping %.2fs",
                resp.status_code,
                page,
                attempt,
                max_retries,
                wait_s,
            )
            time.sleep(wait_s)
        if resp is None:
            raise SystemExit("Failed to fetch orders: no HTTP response")
        resp.raise_for_status()
        payload = resp.json()

        out_file = pages_dir / f"{token_id}_page_{page:04d}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        files.append(out_file)

        pagination = payload.get("pagination", {}) or {}
        has_more = bool(pagination.get("has_more"))
        next_key = str(pagination.get("pagination_key", "") or "")

        logger.info(
            "Fetched page=%s orders=%s has_more=%s",
            page,
            len(payload.get("orders", []) or []),
            has_more,
        )

        if not has_more or not next_key:
            break
        if request_delay_sec > 0:
            time.sleep(request_delay_sec)
        page += 1

    return files


def _load_orders(files: Iterable[Path], token_id: str) -> List[Dict[str, Any]]:
    orders: List[Dict[str, Any]] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("orders", []) or []
        for row in rows:
            if str(row.get("token_id", "")) == token_id:
                orders.append(row)
    return orders


def _normalize_fill(order: Dict[str, Any], token_id: str) -> Optional[Fill]:
    ts = _as_int(order.get("timestamp"))
    price = _as_float(order.get("price"), -1.0)
    if ts <= 0 or price < 0.0:
        return None

    size = _as_float(order.get("shares_normalized"))
    if size <= 0:
        raw_shares = _as_float(order.get("shares"))
        size = raw_shares / 1_000_000.0 if raw_shares > 0 else 0.0
    if size <= 0:
        return None

    return Fill(
        timestamp=ts,
        price=price,
        size=size,
        side=str(order.get("side", "")).upper(),
        tx_hash=str(order.get("tx_hash", "")),
        token_id=token_id,
    )


def _dedupe_mirror_fills(fills: List[Fill]) -> List[Fill]:
    """
    Dome often returns mirrored BUY/SELL rows for one executed fill.
    Collapse mirror rows to one trade-sized fill to avoid double counting.
    """
    grouped: Dict[Tuple[str, int, str, float, float], List[Fill]] = defaultdict(list)
    for f in fills:
        key = (
            f.token_id,
            f.timestamp,
            f.tx_hash,
            round(f.price, 12),
            round(f.size, 8),
        )
        grouped[key].append(f)

    deduped: List[Fill] = []
    for rows in grouped.values():
        sides = {r.side for r in rows}
        representative = rows[0]
        if "BUY" in sides and "SELL" in sides:
            representative.side = "TRADE"
        deduped.append(representative)

    deduped.sort(key=lambda x: x.timestamp)
    return deduped


def _build_bars(fills: List[Fill], interval_sec: int) -> List[Dict[str, Any]]:
    buckets: Dict[int, List[Fill]] = defaultdict(list)
    for f in fills:
        bucket_ts = (f.timestamp // interval_sec) * interval_sec
        buckets[bucket_ts].append(f)

    bars: List[Dict[str, Any]] = []
    for bucket_ts in sorted(buckets):
        trades = sorted(buckets[bucket_ts], key=lambda x: x.timestamp)
        prices = [t.price for t in trades]
        vols = [t.size for t in trades]
        vol_sum = sum(vols)
        vwap = sum(t.price * t.size for t in trades) / vol_sum if vol_sum > 0 else trades[-1].price
        bars.append(
            {
                "t": bucket_ts,
                "o": prices[0],
                "h": max(prices),
                "l": min(prices),
                "c": prices[-1],
                "vwap": round(vwap, 8),
                "v": round(vol_sum, 8),
                "n": len(trades),
            }
        )
    return bars


def _validate_output_schema(out: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    if not isinstance(out, dict):
        return False, ["root is not an object"]

    history = out.get("history")
    bars = out.get("bars")
    meta = out.get("meta")

    if not isinstance(history, list):
        issues.append("history missing or not a list")
    else:
        for i, row in enumerate(history):
            if not isinstance(row, dict):
                issues.append(f"history[{i}] is not an object")
                continue
            if "t" not in row or "p" not in row:
                issues.append(f"history[{i}] missing required keys t/p")
                continue
            if not isinstance(row["t"], int):
                issues.append(f"history[{i}].t is not int")
            if not isinstance(row["p"], (int, float)):
                issues.append(f"history[{i}].p is not number")
            else:
                p = float(row["p"])
                if p < 0.0 or p > 1.0:
                    issues.append(f"history[{i}].p outside [0,1]: {p}")

    if not isinstance(bars, list):
        issues.append("bars missing or not a list")
    else:
        required_bar_keys = {"t", "o", "h", "l", "c", "vwap", "v", "n"}
        for i, row in enumerate(bars):
            if not isinstance(row, dict):
                issues.append(f"bars[{i}] is not an object")
                continue
            missing = required_bar_keys - set(row.keys())
            if missing:
                issues.append(f"bars[{i}] missing keys: {sorted(missing)}")
                continue

    if not isinstance(meta, dict):
        issues.append("meta missing or not an object")

    return len(issues) == 0, issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline price history from Dome orders JSON")
    parser.add_argument("--input", default="", help="Orders JSON file or directory of paginated JSON files")
    parser.add_argument("--token-id", required=True, help="Token ID to build history for")
    parser.add_argument("--interval-sec", type=int, default=60, help="Bar interval in seconds (default: 60)")
    parser.add_argument("--start-ts", type=int, default=0, help="Optional unix start timestamp filter")
    parser.add_argument("--end-ts", type=int, default=0, help="Optional unix end timestamp filter")
    parser.add_argument(
        "--fetch-from-api",
        action="store_true",
        help="Fetch paginated orders from API first, then build time-series from fetched pages",
    )
    parser.add_argument(
        "--api-url",
        default="https://api.domeapi.io/v1/polymarket/orders",
        help="Orders API URL",
    )
    parser.add_argument("--api-limit", type=int, default=100, help="API page size")
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages to fetch")
    parser.add_argument("--timeout-sec", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument("--pagination-key", default="", help="Optional pagination key to resume fetch")
    parser.add_argument("--api-key", default="", help="Optional API key value")
    parser.add_argument(
        "--api-key-header",
        default="x-api-key",
        help="Header name for --api-key (default: x-api-key)",
    )
    parser.add_argument("--auth-bearer", default="", help="Optional bearer token value")
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Extra request header, can be repeated. Format: 'Header-Name: value'",
    )
    parser.add_argument(
        "--pages-dir",
        default="data/backtests/orders_pages",
        help="Directory to store fetched API pages",
    )
    parser.add_argument("--max-retries", type=int, default=6, help="Max retries for 429/temporary errors")
    parser.add_argument("--backoff-base-sec", type=float, default=1.5, help="Backoff base seconds")
    parser.add_argument("--backoff-max-sec", type=float, default=60.0, help="Backoff max seconds")
    parser.add_argument(
        "--request-delay-sec",
        type=float,
        default=0.0,
        help="Fixed sleep between successful page requests",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path. Writes {history:[{t,p}], bars:[...], meta:{...}}",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--strict-schema",
        action="store_true",
        help="Fail if output schema validation reports any issues",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    files: List[Path] = []
    if args.fetch_from_api:
        fetched = _fetch_orders_pages(
            token_id=args.token_id,
            api_url=args.api_url,
            pages_dir=Path(args.pages_dir),
            limit=args.api_limit,
            max_pages=args.max_pages,
            timeout=args.timeout_sec,
            pagination_key=args.pagination_key,
            api_key=args.api_key,
            api_key_header=args.api_key_header,
            auth_bearer=args.auth_bearer,
            extra_headers=args.header,
            max_retries=args.max_retries,
            backoff_base_sec=args.backoff_base_sec,
            backoff_max_sec=args.backoff_max_sec,
            request_delay_sec=args.request_delay_sec,
        )
        files.extend(fetched)

    if args.input:
        files.extend(list(_iter_input_files(Path(args.input))))

    # Dedupe file list preserving order
    dedup_files: List[Path] = []
    seen = set()
    for f in files:
        s = str(f.resolve())
        if s not in seen:
            seen.add(s)
            dedup_files.append(f)
    files = dedup_files

    if not files:
        if args.fetch_from_api:
            raise SystemExit("No orders fetched from API and no local --input provided")
        raise SystemExit(f"No input JSON files found at: {args.input}")

    logger.info("Using %s input files", len(files))

    raw_orders = _load_orders(files, args.token_id)
    if not raw_orders:
        raise SystemExit(f"No orders found for token_id={args.token_id}")
    logger.info("Loaded raw orders: %s", len(raw_orders))

    fills: List[Fill] = []
    for row in raw_orders:
        f = _normalize_fill(row, args.token_id)
        if f is None:
            continue
        if args.start_ts and f.timestamp < args.start_ts:
            continue
        if args.end_ts and f.timestamp > args.end_ts:
            continue
        fills.append(f)

    if not fills:
        raise SystemExit("No usable fills after normalization/filtering")
    logger.info("Usable fills after normalization/filtering: %s", len(fills))

    deduped = _dedupe_mirror_fills(fills)
    bars = _build_bars(deduped, args.interval_sec)
    logger.info(
        "Built bars: bars=%s deduped_fills=%s dedupe_ratio=%.4f",
        len(bars),
        len(deduped),
        (len(deduped) / len(fills)) if fills else 0.0,
    )

    history = [{"t": b["t"], "p": b["c"]} for b in bars]
    out = {
        "history": history,
        "bars": bars,
        "meta": {
            "token_id": args.token_id,
            "input_files": [str(p) for p in files],
            "raw_orders": len(raw_orders),
            "usable_fills": len(fills),
            "deduped_fills": len(deduped),
            "bars": len(bars),
            "interval_sec": args.interval_sec,
            "start_ts": bars[0]["t"] if bars else None,
            "end_ts": bars[-1]["t"] if bars else None,
            "fetched_from_api": args.fetch_from_api,
            "api_url": args.api_url if args.fetch_from_api else None,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    ok, issues = _validate_output_schema(out)
    if ok:
        logger.info("Schema validation: PASS")
    else:
        logger.warning("Schema validation: FAIL issues=%s", len(issues))
        for item in issues[:20]:
            logger.warning("Schema issue: %s", item)
        if args.strict_schema:
            raise SystemExit("Schema validation failed in strict mode")

    print(f"Wrote: {output_path}")
    print(f"Raw orders: {len(raw_orders)}")
    print(f"Usable fills: {len(fills)}")
    print(f"Deduped fills: {len(deduped)}")
    print(f"Bars: {len(bars)}")


if __name__ == "__main__":
    main()
