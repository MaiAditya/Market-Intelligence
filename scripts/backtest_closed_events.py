#!/usr/bin/env python3
"""
Backtest runner for closed Polymarket events.

Reads closed events from JSON (from scripts/fetch_closed_events.py),
builds temporary registry entries, runs full pipeline + graph + report,
and writes per-event + aggregate timing/results.
"""

import argparse
import concurrent.futures
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.event_registry import EventRegistry
from pipeline.query_generator import QueryGenerator
from pipeline.ingestion import DataIngestor
from pipeline.normalizer import DocumentNormalizer
from pipeline.entity_extractor import EntityExtractor
from pipeline.event_mapper import EventMapper
from pipeline.signal_extractor import SignalExtractor
from pipeline.delta_engine import DeltaEngine
from belief_graph.graph_builder import GraphBuilder
from belief_graph.storage import get_storage
from scripts.generate_event_market_report import generate_report

logger = logging.getLogger(__name__)

DEFAULT_DEPENDENCIES = [
    "training",
    "compute",
    "safety",
    "regulation",
    "executive_statement",
    "public_narrative",
]

STOPWORDS = {
    "will", "the", "a", "an", "and", "or", "of", "to", "by", "in", "on",
    "for", "is", "are", "be", "with", "at", "from", "this", "that",
    "what", "when", "who", "which", "how", "after", "before", "than",
}

DOMAIN_HINTS = {
    "election": ["vote", "poll", "candidate", "result"],
    "president": ["vote", "poll", "candidate", "result"],
    "nba": ["playoffs", "finals", "season", "team"],
    "champion": ["playoffs", "finals", "season", "team"],
    "iphone": ["apple", "launch", "announcement", "release"],
    "apple": ["iphone", "launch", "announcement", "release"],
    "ai": ["model", "announcement", "release", "update"],
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _slug_to_event_id(slug: str, run_tag: str = "", isolate_run_ids: bool = True) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "-", slug).strip("-").lower()
    base = f"bt-{safe[:72]}"
    if isolate_run_ids and run_tag:
        return f"{base}-{run_tag[:12]}"
    return base


def _resolve_before_date(event: Any, explicit_before_date: str = "") -> str:
    """
    Resolve date string (YYYY-MM-DD) for query `before:` filter.
    Priority:
      1) explicit CLI value
      2) event.deadline
    """
    if explicit_before_date:
        return explicit_before_date.strip()

    deadline = getattr(event, "deadline", None)
    if not deadline:
        return ""
    try:
        dt = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
        return dt.date().isoformat()
    except Exception:
        return ""


def _resolve_after_date(event_meta: Dict[str, Any]) -> str:
    """
    Resolve date string (YYYY-MM-DD) for query `after:` filter.
    Uses backtest metadata start_date when available.
    """
    if not isinstance(event_meta, dict):
        return ""
    raw = event_meta.get("start_date")
    if not raw:
        return ""
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return dt.date().isoformat()
    except Exception:
        return ""


def _apply_time_filters(
    queries: List[Dict[str, str]],
    before_date: str,
    after_date: str,
) -> List[Dict[str, str]]:
    if not before_date and not after_date:
        return queries

    out: List[Dict[str, str]] = []
    suffix_parts: List[str] = []
    if after_date:
        suffix_parts.append(f"after:{after_date}")
    if before_date:
        suffix_parts.append(f"before:{before_date}")
    suffix = " " + " ".join(suffix_parts)

    for q in queries:
        text = q.get("query", "")
        has_before = "before:" in text
        has_after = "after:" in text
        if (before_date and has_before) or (after_date and has_after):
            out.append({"query": text, "query_type": q.get("query_type", "")})
            continue
        if suffix_parts:
            text = f"{text}{suffix}"
        out.append({"query": text, "query_type": q.get("query_type", "")})
    return out


def _parse_iso_datetime_utc_naive(value: str) -> datetime:
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _extract_entities(question: str, slug: str) -> Tuple[List[str], List[str], List[str]]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-\+\.]*", question or "")
    clean = [w for w in words if w.lower() not in STOPWORDS and len(w) > 2]
    slug_parts = [p.strip() for p in slug.replace("-", " ").split() if p.strip()]
    year_tokens = [p for p in slug_parts if p.isdigit() and len(p) == 4]

    prim: List[str] = []
    sec: List[str] = []

    # Prefer words with uppercase in question
    cap_words = [w for w in clean if any(c.isupper() for c in w)]
    for w in cap_words:
        if w not in prim:
            prim.append(w)
        if len(prim) >= 3:
            break

    # Add one phrase-like primary entity (helps substring matching)
    q_tokens = [t for t in re.findall(r"[A-Za-z]+", question or "") if t.lower() not in STOPWORDS]
    if len(q_tokens) >= 2:
        two_gram = f"{q_tokens[0]} {q_tokens[1]}"
        if two_gram not in prim:
            prim.insert(0, two_gram)

    # Backfill from slug tokens if needed
    if len(prim) < 2:
        for token in slug_parts:
            if len(token) < 3 or token.isdigit() or token.lower() in STOPWORDS:
                continue
            if token not in prim:
                prim.append(token)
            if len(prim) >= 3:
                break

    if not prim:
        prim = ["polymarket"]

    # Secondary entities from remaining title words
    for w in clean:
        if w not in prim and w not in sec:
            sec.append(w)
        if len(sec) >= 5:
            break

    # Add years from slug as secondary entities
    for y in year_tokens:
        if y not in sec:
            sec.append(y)
        if len(sec) >= 5:
            break

    # Add domain-specific hint terms based on slug/title tokens
    slug_lower = slug.lower()
    title_lower = (question or "").lower()
    for key, hints in DOMAIN_HINTS.items():
        if key in slug_lower or key in title_lower:
            for h in hints:
                if h not in sec and h not in prim:
                    sec.append(h)
                if len(sec) >= 5:
                    break
        if len(sec) >= 5:
            break

    # Ensure at least one secondary so hard gate isn't impossible
    if not sec:
        for token in slug_parts:
            if len(token) >= 2 and token not in prim and token.lower() not in STOPWORDS:
                sec.append(token)
                break

    # Rich alias set for robust alias matching
    aliases: List[str] = []
    lowered_q = (question or "").strip().lower()
    lowered_slug_space = slug.replace("-", " ").lower()
    if lowered_q:
        aliases.append(lowered_q)
    aliases.append(lowered_slug_space)
    aliases.append(slug.lower())
    if len(q_tokens) >= 2:
        aliases.append(f"{q_tokens[0].lower()} {q_tokens[1].lower()}")
    if len(q_tokens) >= 3:
        aliases.append(f"{q_tokens[0].lower()} {q_tokens[1].lower()} {q_tokens[2].lower()}")

    # Deduplicate aliases preserving order
    dedup_aliases: List[str] = []
    seen = set()
    for a in aliases:
        if a and a not in seen:
            seen.add(a)
            dedup_aliases.append(a)

    return prim[:4], sec[:5], dedup_aliases[:8]


def _build_temp_registry(
    selected_events: List[Dict[str, Any]],
    temp_config_path: Path,
    run_tag: str = "",
    isolate_run_ids: bool = True,
) -> Dict[str, Dict[str, Any]]:
    base_cfg_path = project_root / "config" / "events.json"
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    out_events = []
    event_lookup: Dict[str, Dict[str, Any]] = {}

    now = _utc_now()
    fallback_deadline = (now - timedelta(days=1)).isoformat()

    for item in selected_events:
        slug = item.get("slug")
        question = item.get("question") or slug
        if not slug:
            continue

        event_id = _slug_to_event_id(
            slug,
            run_tag=run_tag,
            isolate_run_ids=isolate_run_ids,
        )
        primary_entities, secondary_entities, aliases = _extract_entities(question, slug)
        deadline = item.get("end_date") or fallback_deadline
        start_date = item.get("start_date")

        event_obj = {
            "event_id": event_id,
            "event_type": "capability",
            "event_title": question,
            "event_description": f"{question}. Related market slug context: {slug.replace('-', ' ')}",
            "primary_entities": primary_entities,
            "secondary_entities": secondary_entities,
            "aliases": aliases,
            "deadline": deadline,
            "start_date": start_date,
            "dependencies": DEFAULT_DEPENDENCIES,
            "polymarket_slug": slug,
        }
        out_events.append(event_obj)
        event_lookup[event_id] = {
            "event_obj": event_obj,
            "backtest_meta": {
                "parent_slug": item.get("parent_slug"),
                "market_type": item.get("market_type"),
                "child_volume": item.get("child_volume"),
                "start_date": item.get("start_date"),
                "end_date": item.get("end_date"),
                "selected_token": item.get("selected_token"),
                "token_checks": item.get("token_checks", []),
                "any_history_available": item.get("any_history_available"),
            },
        }

    cfg = {
        "dependency_descriptions": base_cfg.get("dependency_descriptions", {}),
        "events": out_events,
    }

    temp_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    return event_lookup


def _run_single_event(
    event_id: str,
    temp_registry_path: str,
    run_dir: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    registry = EventRegistry(config_path=temp_registry_path)
    event = registry.get_event(event_id)
    if event is None:
        raise ValueError(f"Event not found in temp registry: {event_id}")

    query_generator = QueryGenerator()
    ingestor = DataIngestor()
    normalizer = DocumentNormalizer()
    entity_extractor = EntityExtractor(normalizer)
    mapper = EventMapper(registry, normalizer)
    signal_extractor = SignalExtractor(registry, normalizer, mapper)
    delta_engine = DeltaEngine(registry, signal_extractor, normalizer)
    graph_builder = GraphBuilder(registry=registry, normalizer=normalizer)
    storage = get_storage()

    event_dir = Path(run_dir) / event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    event_meta = (params.get("event_meta_by_id") or {}).get(event_id, {}) or {}

    steps: Dict[str, float] = {}
    diagnostics: Dict[str, Any] = {}
    event_start = time.perf_counter()

    # 1) queries
    t = time.perf_counter()
    query_set = query_generator.generate_queries_for_event(event)
    queries = [{"query": q.query, "query_type": q.query_type} for q in query_set.queries]
    before_date = ""
    after_date = _resolve_after_date(event_meta)
    if params.get("enable_before_date_filter", True):
        before_date = _resolve_before_date(event, params.get("search_before_date", ""))
    queries = _apply_time_filters(queries, before_date, after_date)
    steps["generate_queries_sec"] = round(time.perf_counter() - t, 3)
    diagnostics["query_count"] = len(queries)
    diagnostics["query_before_date"] = before_date or None
    diagnostics["query_after_date"] = after_date or None
    diagnostics["query_before_filter_enabled"] = bool(params.get("enable_before_date_filter", True))
    diagnostics["query_after_filter_enabled"] = bool(after_date)
    diagnostics["query_sample"] = queries[:3]

    # 2) ingestion
    t = time.perf_counter()
    if params["skip_ingestion"]:
        docs = ingestor.get_documents_for_event(event_id)
        diagnostics["ingestion_mode"] = "skip_ingestion"
    else:
        docs = ingestor.ingest_for_event(event_id, queries)
        diagnostics["ingestion_mode"] = "live_ingestion"
    steps["ingestion_sec"] = round(time.perf_counter() - t, 3)
    diagnostics["documents_raw"] = len(docs)

    # 3) normalize
    t = time.perf_counter()
    normalized_docs = []
    for doc in docs:
        normalized_docs.append(normalizer.normalize_and_save(doc.to_dict()))
    steps["normalize_sec"] = round(time.perf_counter() - t, 3)
    diagnostics["documents_normalized"] = len(normalized_docs)

    # 4) entities
    t = time.perf_counter()
    updated_docs = entity_extractor.process_event_documents(event_id)
    steps["entities_sec"] = round(time.perf_counter() - t, 3)
    diagnostics["documents_entity_updated"] = len(updated_docs)

    # 5) mapping
    t = time.perf_counter()
    mapping_summary = mapper.process_event(event_id)
    steps["mapping_sec"] = round(time.perf_counter() - t, 3)
    diagnostics["mapping_summary"] = mapping_summary

    # 6) signal + delta
    t = time.perf_counter()
    analysis = delta_engine.analyze_event(event_id)
    steps["delta_sec"] = round(time.perf_counter() - t, 3)
    diagnostics["suggested_delta"] = analysis.suggested_delta
    diagnostics["confidence"] = analysis.confidence
    diagnostics["signals_total"] = analysis.signal_summary.get("total_signals", 0)

    # 7) graph
    t = time.perf_counter()
    graph_window_start = None
    graph_window_end = None
    try:
        if after_date:
            graph_window_start = _parse_iso_datetime_utc_naive(f"{after_date}T00:00:00")
        if before_date:
            graph_window_end = _parse_iso_datetime_utc_naive(f"{before_date}T23:59:59")
    except Exception:
        graph_window_start = None
        graph_window_end = None

    graph = graph_builder.build(
        belief_event_id=event_id,
        max_events=params["max_events"],
        max_edges=params["max_edges"],
        market_window_only=True,
        window_start=graph_window_start,
        window_end=graph_window_end,
    )
    graph_path = storage.save(graph)
    steps["graph_sec"] = round(time.perf_counter() - t, 3)
    diagnostics["graph_nodes"] = len(graph.event_nodes)
    diagnostics["graph_edges"] = len(graph.edges)

    # 8) report
    t = time.perf_counter()
    report_path = event_dir / f"{event_id}_report.json"
    report = generate_report(
        event_id=event_id,
        slug=event.polymarket_slug,
        graph_path=str(graph_path),
        output_path=str(report_path),
        top_n1=params["top_n1"],
        top_n2=params["top_n2"],
        min_conf=params["min_conf"],
        impact_window=params["impact_window"],
        price_history_source=params["price_history_source"],
        dome_bearer_token=params.get("dome_bearer_token"),
        orders_max_pages=params["orders_max_pages"],
        orders_request_delay_sec=params["orders_request_delay_sec"],
        orders_only=params["orders_only"],
        dome_bearer_token_source=params.get("dome_bearer_token_source", "none"),
        skip_out_of_window_events=params["skip_out_of_window_events"],
        explicit_token_candidates=event_meta.get("token_checks", []),
        orders_fetch_all=params["orders_fetch_all"],
        orders_fetch_all_max_pages=params["orders_fetch_all_max_pages"],
        price_mapping_mode=params["price_mapping_mode"],
        burst_gap_minutes=params["burst_gap_minutes"],
        max_event_bursts=params["max_event_bursts"],
        burst_buffer_minutes=params["burst_buffer_minutes"],
    )
    steps["report_sec"] = round(time.perf_counter() - t, 3)
    diagnostics["report_summary"] = report.get("summary", {})

    event_total = round(time.perf_counter() - event_start, 3)
    steps["total_sec"] = event_total

    result = {
        "event_id": event_id,
        "slug": event.polymarket_slug,
        "title": event.event_title,
        "timings": steps,
        "diagnostics": diagnostics,
        "outputs": {
            "report_json": str(report_path),
            "graph_json": str(graph_path),
        },
    }

    with open(event_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def _select_events(closed_events_file: Path, num_events: int, start_index: int) -> List[Dict[str, Any]]:
    with open(closed_events_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Parent-level date fallbacks (useful for older enriched files that do not
    # include child start_date yet).
    parent_start_by_slug: Dict[str, str] = {}
    parent_end_by_slug: Dict[str, str] = {}
    for e in payload.get("events", []) or []:
        slug = e.get("slug")
        if not slug:
            continue
        if e.get("start_date"):
            parent_start_by_slug[str(slug)] = str(e.get("start_date"))
        if e.get("end_date"):
            parent_end_by_slug[str(slug)] = str(e.get("end_date"))

    if not parent_start_by_slug:
        fallback_closed = project_root / "data" / "backtests" / "closed_events.json"
        if fallback_closed.exists():
            try:
                with open(fallback_closed, "r", encoding="utf-8") as f:
                    closed_payload = json.load(f)
                for e in closed_payload.get("events", []) or []:
                    slug = e.get("slug")
                    if not slug:
                        continue
                    if e.get("start_date"):
                        parent_start_by_slug[str(slug)] = str(e.get("start_date"))
                    if e.get("end_date"):
                        parent_end_by_slug[str(slug)] = str(e.get("end_date"))
            except Exception:
                pass

    # Preferred mode for enriched structure:
    # run each child binary market independently from backtest_targets.
    targets = payload.get("backtest_targets", [])
    if isinstance(targets, list) and targets:
        # Build child market -> end_date map from parents/selected_children when available.
        end_date_by_child_slug: Dict[str, str] = {}
        start_date_by_child_slug: Dict[str, str] = {}
        token_checks_by_child_slug: Dict[str, List[Dict[str, Any]]] = {}
        selected_token_by_child_slug: Dict[str, Dict[str, Any]] = {}
        parents = payload.get("parents", [])
        if isinstance(parents, list):
            for p in parents:
                for c in p.get("selected_children", []) or []:
                    child_slug = c.get("market_slug")
                    child_end = c.get("end_date")
                    child_start = c.get("start_date")
                    if child_slug and child_end:
                        end_date_by_child_slug[str(child_slug)] = str(child_end)
                    if child_slug and child_start:
                        start_date_by_child_slug[str(child_slug)] = str(child_start)
                    if child_slug and isinstance(c.get("token_checks"), list):
                        token_checks_by_child_slug[str(child_slug)] = c.get("token_checks", [])
                    if child_slug and isinstance(c.get("selected_token"), dict):
                        selected_token_by_child_slug[str(child_slug)] = c.get("selected_token", {})

        events: List[Dict[str, Any]] = []
        for t in targets:
            child_slug = t.get("child_market_slug") or t.get("market_slug") or t.get("slug")
            if not child_slug:
                continue
            child_question = t.get("child_question") or t.get("question") or str(child_slug)
            child_end_date = (
                t.get("end_date")
                or t.get("parent_end_date")
                or parent_end_by_slug.get(str(t.get("parent_slug") or ""))
                or end_date_by_child_slug.get(str(child_slug))
            )
            child_start_date = (
                t.get("start_date")
                or t.get("parent_start_date")
                or parent_start_by_slug.get(str(t.get("parent_slug") or ""))
                or start_date_by_child_slug.get(str(child_slug))
            )
            events.append(
                {
                    "slug": str(child_slug),
                    "question": str(child_question),
                    "start_date": child_start_date,
                    "end_date": child_end_date,
                    "parent_slug": t.get("parent_slug"),
                    "market_type": t.get("market_type"),
                    "child_volume": t.get("child_volume"),
                    "selected_token": t.get("selected_token") or selected_token_by_child_slug.get(str(child_slug)),
                    "token_checks": t.get("token_checks") or token_checks_by_child_slug.get(str(child_slug), []),
                    "any_history_available": t.get("any_history_available"),
                }
            )

        if start_index > 0:
            events = events[start_index:]
        logger.info(
            f"Using enriched backtest_targets mode: {len(events[:num_events])} "
            f"child markets selected from {len(targets)} targets"
        )
        return events[:num_events]

    # Fallback mode: plain closed events list.
    events = payload.get("events", [])
    if start_index > 0:
        events = events[start_index:]

    logger.info(f"Using closed events mode: {len(events[:num_events])} parent markets selected")
    return events[:num_events]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest pipeline on closed Polymarket events")
    parser.add_argument(
        "--closed-events",
        default="data/backtests/closed_events.json",
        help="Path to closed events JSON from fetch_closed_events.py",
    )
    parser.add_argument("--num-events", type=int, default=2, help="Number of events to backtest")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset in closed events list")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip live ingestion and use cached documents if available")
    parser.add_argument("--max-events", type=int, default=120, help="Graph builder max events")
    parser.add_argument("--max-edges", type=int, default=250, help="Graph builder max edges")
    parser.add_argument("--top-n1", type=int, default=20, help="Focused graph top N-1 nodes")
    parser.add_argument("--top-n2", type=int, default=10, help="Focused graph top N-2 nodes per N-1")
    parser.add_argument("--min-conf", type=float, default=0.3, help="Focused graph minimum confidence")
    parser.add_argument("--impact-window", type=int, default=2, help="Impact window in minutes")
    parser.add_argument(
        "--search-before-date",
        default="",
        help="Apply query filter `before:YYYY-MM-DD` to all searches (default: event deadline date)",
    )
    parser.add_argument(
        "--disable-before-date-filter",
        action="store_true",
        help="Disable automatic query `before:` filter for backtests",
    )
    parser.add_argument(
        "--price-history-source",
        choices=["clob", "orders", "auto"],
        default="auto",
        help="Price history source for impact mapping in report",
    )
    parser.add_argument("--dome-bearer-token", default=None, help="Dome API bearer token (optional)")
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
        help="Do not fallback to CLOB if orders history fails",
    )
    parser.add_argument(
        "--orders-fetch-all",
        action="store_true",
        help="Fetch all available Dome orders pages for selected token during impact mapping",
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
        help="Drop graph events outside selected token history window",
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
    parser.add_argument(
        "--parallel-events",
        type=int,
        default=1,
        help="Number of events to process in parallel (recommended 2-3 max)",
    )
    parser.add_argument(
        "--no-isolate-run-ids",
        action="store_true",
        help="Disable run-specific event IDs (not recommended for backtest integrity)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/backtests",
        help="Output directory for backtest artifacts",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_id = _utc_now().strftime("closed-backtest-%Y%m%d-%H%M%S")
    run_dir = project_root / args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    selected = _select_events(Path(args.closed_events), args.num_events, args.start_index)
    if not selected:
        raise ValueError("No events selected. Check --closed-events file and filters.")

    logger.info(f"Selected {len(selected)} closed events for backtest")

    temp_registry = run_dir / "temp_events.json"
    isolate_run_ids = not args.no_isolate_run_ids
    short_run_tag = run_id.replace("closed-backtest-", "")
    event_lookup = _build_temp_registry(
        selected,
        temp_registry,
        run_tag=short_run_tag,
        isolate_run_ids=isolate_run_ids,
    )
    registry = EventRegistry(config_path=str(temp_registry))

    event_ids = [e.event_id for e in registry]
    aggregate_results: List[Dict[str, Any]] = []
    failed_results: List[Dict[str, Any]] = []

    run_start = time.perf_counter()
    resolved_dome_token = args.dome_bearer_token or os.getenv("DOME_BEARER_TOKEN")
    dome_token_source = "arg" if args.dome_bearer_token else ("env" if os.getenv("DOME_BEARER_TOKEN") else "none")
    worker_params = {
        "skip_ingestion": args.skip_ingestion,
        "enable_before_date_filter": not args.disable_before_date_filter,
        "search_before_date": args.search_before_date,
        "max_events": args.max_events,
        "max_edges": args.max_edges,
        "top_n1": args.top_n1,
        "top_n2": args.top_n2,
        "min_conf": args.min_conf,
        "impact_window": args.impact_window,
        "price_history_source": args.price_history_source,
        "dome_bearer_token": resolved_dome_token,
        "dome_bearer_token_source": dome_token_source,
        "orders_max_pages": args.orders_max_pages,
        "orders_request_delay_sec": args.orders_request_delay_sec,
        "orders_only": args.orders_only,
        "orders_fetch_all": args.orders_fetch_all,
        "orders_fetch_all_max_pages": args.orders_fetch_all_max_pages,
        "skip_out_of_window_events": args.skip_out_of_window_events,
        "price_mapping_mode": args.price_mapping_mode,
        "burst_gap_minutes": args.burst_gap_minutes,
        "max_event_bursts": args.max_event_bursts,
        "burst_buffer_minutes": args.burst_buffer_minutes,
        "event_meta_by_id": {
            eid: (meta.get("backtest_meta", {}) if isinstance(meta, dict) else {})
            for eid, meta in event_lookup.items()
        },
    }

    if args.parallel_events > 1:
        logger.info(
            f"Running with parallel event workers={args.parallel_events}, "
            f"isolate_run_ids={isolate_run_ids}"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_events) as executor:
            future_to_event = {
                executor.submit(
                    _run_single_event,
                    event_id,
                    str(temp_registry),
                    str(run_dir),
                    worker_params,
                ): event_id
                for event_id in event_ids
            }
            done_count = 0
            for future in concurrent.futures.as_completed(future_to_event):
                event_id = future_to_event[future]
                done_count += 1
                try:
                    result = future.result()
                    aggregate_results.append(result)
                    event_total = result["timings"]["total_sec"]
                    avg = sum(r["timings"]["total_sec"] for r in aggregate_results) / len(aggregate_results)
                    remaining = len(event_ids) - done_count
                    eta_sec = int(avg * remaining)
                    logger.info(
                        f"[{done_count}/{len(event_ids)}] done {event_id} in {event_total:.1f}s | "
                        f"avg={avg:.1f}s/event | ETA remaining ~{eta_sec}s"
                    )
                except Exception as e:
                    logger.error(f"Event failed: {event_id} -> {e}", exc_info=True)
                    failed_results.append({"event_id": event_id, "error": str(e)})
    else:
        for idx, event_id in enumerate(event_ids, start=1):
            event = registry.get_event(event_id)
            assert event is not None
            logger.info(f"[{idx}/{len(event_ids)}] Backtesting {event_id} ({event.polymarket_slug})")
            try:
                result = _run_single_event(
                    event_id=event_id,
                    temp_registry_path=str(temp_registry),
                    run_dir=str(run_dir),
                    params=worker_params,
                )
                aggregate_results.append(result)
                event_total = result["timings"]["total_sec"]
                avg = sum(r["timings"]["total_sec"] for r in aggregate_results) / len(aggregate_results)
                remaining = len(event_ids) - idx
                eta_sec = int(avg * remaining)
                logger.info(
                    f"[{idx}/{len(event_ids)}] done in {event_total:.1f}s | "
                    f"avg={avg:.1f}s/event | ETA remaining ~{eta_sec}s"
                )
            except Exception as e:
                logger.error(f"Event failed: {event_id} -> {e}", exc_info=True)
                failed_results.append({"event_id": event_id, "error": str(e)})

    total_runtime = round(time.perf_counter() - run_start, 3)
    avg_runtime = round(total_runtime / max(1, len(aggregate_results)), 3)

    params_summary = dict(vars(args))
    params_summary["dome_bearer_token"] = None
    params_summary["dome_bearer_token_present"] = bool(resolved_dome_token)
    params_summary["dome_bearer_token_source"] = dome_token_source

    summary = {
        "run_id": run_id,
        "generated_at": _utc_now().isoformat(),
        "params": params_summary,
        "events_run": len(aggregate_results),
        "events_failed": len(failed_results),
        "total_runtime_sec": total_runtime,
        "avg_runtime_sec": avg_runtime,
        "estimated_runtime_for_5_events_sec": round(avg_runtime * 5, 3),
        "estimated_runtime_for_20_events_sec": round(avg_runtime * 20, 3),
        "results": aggregate_results,
        "failures": failed_results,
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBacktest completed.")
    print(f"Run directory: {run_dir}")
    print(f"Summary: {summary_path}")
    print(f"Events run: {len(aggregate_results)}")
    print(f"Total runtime: {total_runtime:.1f}s")
    print(f"Average per event: {avg_runtime:.1f}s")
    print(f"Estimated for 5 events: {summary['estimated_runtime_for_5_events_sec']:.1f}s")
    print(f"Estimated for 20 events: {summary['estimated_runtime_for_20_events_sec']:.1f}s")


if __name__ == "__main__":
    main()
