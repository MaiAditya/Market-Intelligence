#!/usr/bin/env python3
"""
Scheduled Data Ingestion Pipeline

Runs hourly (via cron) to fetch fresh news data for all tracked events.

Flow:
  1. Load events from registry (config/events.json)
  2. Fetch current probabilities from Polymarket
  3. For each event: generate queries → ingest from web/Reddit/Twitter
  4. Deduplicate (URL + content hash)
  5. Push new docs to backend DB as feed items
  6. Log run summary

Usage:
    python scripts/scheduled_ingest.py                     # all events
    python scripts/scheduled_ingest.py --max-events 5      # first 5 events
    python scripts/scheduled_ingest.py --event gemini-5-release-2026  # one event
    python scripts/scheduled_ingest.py --verbose            # with detailed logging
    python scripts/scheduled_ingest.py --dry-run            # show what would run

Cron example (every hour):
    0 * * * * cd /path/to/ai-market-intelligence && conda run -n backend python scripts/scheduled_ingest.py --max-events 5 >> logs/scheduler.log 2>&1
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.event_registry import get_registry, Event
from pipeline.query_generator import QueryGenerator
from pipeline.ingestion.ingestor import DataIngestor
from integrations.polymarket_client import get_polymarket_client

logger = logging.getLogger("scheduled_ingest")


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")
    # Quiet noisy libraries (TLS, HTTP/2, cookies, etc.)
    for name in (
        "urllib3", "httpx", "httpcore", "praw",
        "rustls", "h2", "cookie_store", "primp",
        "hyper_util", "duckduckgo_search",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


# ── Run State ──────────────────────────────────────────────

class RunState:
    """Tracks stats for a single scheduler run."""

    def __init__(self):
        self.started_at = datetime.now(timezone.utc)
        self.events_processed = 0
        self.total_new_docs = 0
        self.total_skipped = 0
        self.by_event: Dict[str, Dict] = {}
        self.probabilities: Dict[str, Optional[float]] = {}
        self.errors: List[str] = []

    def add_event_result(self, event_id: str, new_docs: int, queries_run: int,
                         doc_details: list = None):
        self.by_event[event_id] = {
            "new_docs": new_docs,
            "queries_run": queries_run,
            "probability": self.probabilities.get(event_id),
            "documents": doc_details or [],
        }
        self.total_new_docs += new_docs
        self.events_processed += 1

    def add_error(self, msg: str):
        self.errors.append(msg)

    def summary(self) -> str:
        elapsed = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        lines = [
            "",
            "\033[1m" + "═" * 64 + "\033[0m",
            "  📊 Scheduled Ingestion Run Summary",
            f"  🕐 {self.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}  ({elapsed:.1f}s)",
            "\033[1m" + "═" * 64 + "\033[0m",
            f"  Events processed : {self.events_processed}",
            f"  New documents    : {self.total_new_docs}",
            "",
        ]
        for eid, stats in self.by_event.items():
            prob = stats["probability"]
            prob_str = f"{prob:.1%}" if prob is not None else "N/A"
            lines.append(f"  ┌─ {eid} (prob={prob_str})")
            lines.append(f"  │  {stats['new_docs']} new docs from {stats['queries_run']} queries")
            docs = stats.get("documents", [])
            if docs:
                # Group by source
                by_source = {}
                for d in docs:
                    by_source.setdefault(d['source'], []).append(d)
                for src, src_docs in by_source.items():
                    lines.append(f"  │  [{src}] {len(src_docs)} docs:")
                    for d in src_docs[:10]:
                        title = d['title'][:55] if d['title'] else '(no title)'
                        chars = d.get('chars', '?')
                        lines.append(f"  │    ✓ {title} ({chars} chars)")
                    if len(src_docs) > 10:
                        lines.append(f"  │    ... and {len(src_docs) - 10} more")
            lines.append(f"  └{'─' * 40}")
        if self.errors:
            lines.append("")
            lines.append(f"  ⚠️  Errors ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"    - {err}")
        lines.append("\033[1m" + "═" * 64 + "\033[0m")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at.isoformat(),
            "events_processed": self.events_processed,
            "total_new_docs": self.total_new_docs,
            "by_event": self.by_event,
            "errors": self.errors,
        }


# ── Core Logic ─────────────────────────────────────────────

def fetch_probabilities(events: List[Event]) -> Dict[str, Optional[float]]:
    """Fetch current Polymarket probabilities for all events."""
    client = get_polymarket_client()
    slugs = {e.event_id: e.polymarket_slug for e in events if e.polymarket_slug}

    logger.info(f"Fetching probabilities for {len(slugs)} events from Polymarket...")

    try:
        probs = client.get_probabilities_for_events(slugs)
        for eid, prob in probs.items():
            if prob is not None:
                logger.info(f"  {eid}: {prob:.1%}")
            else:
                logger.warning(f"  {eid}: probability not found")
        return probs
    except Exception as e:
        logger.error(f"Failed to fetch probabilities: {e}")
        return {}


def ingest_event(
    event: Event,
    query_generator: QueryGenerator,
    ingestor: DataIngestor,
    verbose: bool = False,
) -> tuple:
    """
    Run data ingestion for a single event.

    Returns:
        (num_docs, doc_details) tuple
    """
    print(f"\n{'─' * 50}")
    print(f"  📰 {event.event_title}")
    print(f"     Event ID: {event.event_id}")
    print(f"{'─' * 50}")

    # Generate search queries
    query_set = query_generator.generate_queries_for_event(event)
    queries = [{"query": q.query, "query_type": q.query_type} for q in query_set.queries]
    print(f"  🔍 Generated {len(queries)} search queries")

    if verbose:
        for q in queries[:5]:
            print(f"     [{q['query_type']}] {q['query']}")
        if len(queries) > 5:
            print(f"     ... and {len(queries) - 5} more")

    # Run ingestion
    docs = ingestor.ingest_for_event(event.event_id, queries)

    # Collect document details for summary
    doc_details = []
    for doc in docs:
        doc_details.append({
            "title": doc.title,
            "source": doc.source,
            "domain": doc.metadata.get("domain", doc.source),
            "chars": len(doc.raw_text) if doc.raw_text else 0,
            "url": doc.url[:80],
        })

    if docs:
        print(f"  ✅ {len(docs)} new documents ingested:")
        for d in doc_details:
            title = d['title'][:55] if d['title'] else '(no title)'
            print(f"     ✓ [{d['source']}] {title} ({d['chars']} chars)")
    else:
        print(f"  ⏭️  No new documents (all deduped or no results)")

    return len(docs), doc_details


def run_scheduled_ingestion(
    max_events: int = 5,
    event_id: Optional[str] = None,
    verbose: bool = False,
    dry_run: bool = False,
) -> RunState:
    """
    Main scheduler entry point.

    Args:
        max_events: Maximum number of events to process
        event_id: If set, only process this specific event
        verbose: Enable verbose logging
        dry_run: If True, show what would run without ingesting
    """
    state = RunState()
    registry = get_registry()

    # Select events
    if event_id:
        event = registry.get_event(event_id)
        if event is None:
            state.add_error(f"Event not found: {event_id}")
            return state
        events = [event]
    else:
        events = list(registry.get_all_events())[:max_events]

    logger.info(f"Processing {len(events)} events: {[e.event_id for e in events]}")

    # Fetch probabilities
    state.probabilities = fetch_probabilities(events)

    if dry_run:
        qg = QueryGenerator()
        print("\n[DRY RUN] Would process these events:\n")
        for event in events:
            qs = qg.generate_queries_for_event(event)
            prob = state.probabilities.get(event.event_id)
            prob_str = f"{prob:.1%}" if prob is not None else "N/A"
            print(f"  {event.event_id} (prob={prob_str})")
            print(f"    {len(qs.queries)} queries would be run")
            for q in qs.queries[:3]:
                print(f"      [{q.query_type}] {q.query}")
            if len(qs.queries) > 3:
                print(f"      ... and {len(qs.queries) - 3} more")
            print()
        return state

    # Initialize pipeline components
    query_generator = QueryGenerator()
    ingestor = DataIngestor()

    # Process each event
    for event in events:
        try:
            query_set = query_generator.generate_queries_for_event(event)
            num_queries = len(query_set.queries)

            new_docs, doc_details = ingest_event(event, query_generator, ingestor, verbose)
            state.add_event_result(event.event_id, new_docs, num_queries, doc_details)

        except Exception as e:
            err_msg = f"Failed to process {event.event_id}: {e}"
            logger.error(err_msg, exc_info=verbose)
            state.add_error(err_msg)
            state.add_event_result(event.event_id, 0, 0)

    # Persist content hashes at end of run
    ingestor._save_seen_hashes()

    return state


def save_run_log(state: RunState) -> Path:
    """Save run state to logs directory."""
    log_dir = project_root / "logs" / "scheduler"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = state.started_at.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{ts}.json"

    with open(log_path, "w") as f:
        json.dump(state.to_dict(), f, indent=2, default=str)

    return log_path


# ── CLI ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scheduled data ingestion pipeline"
    )
    parser.add_argument(
        "--max-events", type=int, default=5,
        help="Maximum number of events to process (default: 5)"
    )
    parser.add_argument(
        "--event", type=str, default=None,
        help="Process only this specific event ID"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would run without ingesting"
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Skip saving run log to disk"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger.info("Starting scheduled ingestion run")

    state = run_scheduled_ingestion(
        max_events=args.max_events,
        event_id=args.event,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )

    # Print summary
    print(state.summary())

    # Save run log
    if not args.no_log and not args.dry_run:
        log_path = save_run_log(state)
        logger.info(f"Run log saved to: {log_path}")

    # Exit with error code if there were errors
    if state.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
