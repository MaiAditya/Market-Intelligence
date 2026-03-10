"""
LLM Causal Graph Generator — CLI Script

Generates an LLM causal graph for one or more events and stores it to
the PostgreSQL DB. Designed to run at event registration time (once per
event), and is IDEMPOTENT — re-running for the same event is a no-op
unless --force is passed.

Usage:
    # Generate for a specific event
    python scripts/generate_llm_graph.py --event gemini-3pt5-release-2026

    # Generate AND enrich with evidence from ingested documents
    python scripts/generate_llm_graph.py --event gemini-3pt5-release-2026 --enrich

    # Only re-run evidence enrichment (graph already exists)
    python scripts/generate_llm_graph.py --event gemini-3pt5-release-2026 --enrich-only

    # Generate for ALL registered events
    python scripts/generate_llm_graph.py --all

    # Force regenerate (overwrites existing)
    python scripts/generate_llm_graph.py --event gemini-3pt5-release-2026 --force

    # Dry run (no DB write, just print the graph)
    python scripts/generate_llm_graph.py --event gemini-3pt5-release-2026 --dry-run

Environment variables required:
    GEMINI_API_KEY     — Google Gemini API key
    CAUSAL_DB_URL      — PostgreSQL URL (optional, defaults to local dev)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate LLM causal graphs for Polymarket events."
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--event", metavar="EVENT_ID", help="Single event ID to process")
    group.add_argument("--all", action="store_true", help="Process all registered events")
    p.add_argument("--force", action="store_true", help="Overwrite existing graphs")
    p.add_argument("--dry-run", action="store_true", help="Print graph without writing to DB")
    p.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    p.add_argument(
        "--enrich",
        action="store_true",
        help="After graph generation, map ingested docs to nodes and regenerate details with evidence",
    )
    p.add_argument(
        "--enrich-only",
        action="store_true",
        help="Skip graph generation — only run evidence matching + detail enrichment on existing graph",
    )
    return p


def print_graph(graph) -> None:
    """Pretty-print a generated graph to stdout."""
    print("\n" + "="*70)
    print(f"EVENT:  {graph.event_id}")
    print(f"BELIEF: {graph.belief_question}")
    print(f"MODEL:  {graph.model_used}")
    print(f"THESIS:\n  {graph.llm_thesis}")
    print(f"\nNODES ({len(graph.nodes)}):")
    for n in graph.nodes:
        tag = "[BELIEF]" if n.is_belief else f"[{n.event_type.upper()}]"
        print(f"  {tag} {n.node_id}")
        print(f"       label: {n.label}")
        print(f"       actors: {', '.join(n.actors)}")
        print(f"       direction: {n.direction}  probability: {n.probability:.0f}%")
    print(f"\nEDGES ({len(graph.edges)}):")
    for e in graph.edges:
        print(f"  {e.source_node_id} → {e.target_node_id}")
        print(f"       mechanism: {e.mechanism}  confidence: {e.confidence:.2f}  dir: {e.direction}")
        print(f"       {e.explanation}")
    print("="*70 + "\n")


def run_enrichment(event_id: str, graph_id: str, market_id: str, dry_run: bool = False) -> bool:
    """
    Phase 2 enrichment pipeline:
      1. NodeEvidenceMatcher: summarize docs + map to nodes
      2. NodeDetailGenerator: generate grounded detail per node
    """
    logger.info(f"=== Starting evidence enrichment for {event_id} ===")

    # Step A: Map evidence docs to nodes
    try:
        from belief_graph.node_evidence_matcher import NodeEvidenceMatcher
        matcher = NodeEvidenceMatcher.from_env()
        match_stats = matcher.match_event(
            event_id=event_id,
            graph_id=graph_id,
            market_id=market_id,
            sync_from_files=True,       # Import JSON-file docs into DB first
        )
        logger.info(
            f"Evidence matching: {match_stats['matched_docs']}/{match_stats['total_docs']} docs matched "
            f"across {len(match_stats['per_node_counts'])} nodes"
        )
    except Exception as e:
        logger.error(f"Evidence matching failed: {e}")
        return False

    # Step B: Generate grounded node details
    try:
        from belief_graph.node_detail_generator import NodeDetailGenerator
        detail_gen = NodeDetailGenerator.from_env()
        detail_stats = detail_gen.generate_all(
            event_id=event_id,
            graph_id=graph_id,
            dry_run=dry_run,
        )
        logger.info(
            f"Detail generation: success={detail_stats['success']} "
            f"(grounded) + {detail_stats['no_evidence']} (world-knowledge fallback), "
            f"failed={detail_stats['failed']}"
        )
    except Exception as e:
        logger.error(f"Node detail generation failed: {e}")
        return False

    logger.info(f"✓ Enrichment complete for {event_id}")
    return True


def process_event(event, args) -> bool:
    """
    Generate and optionally persist a causal graph for one event.

    Returns True on success, False on error.
    """
    from belief_graph.llm_causal_generator import LLMCausalGraphGenerator
    from belief_graph.causal_graph_db import CausalGraphDBWriter

    logger.info(f"Processing event: {event.event_id}")

    # ── Enrich-only mode: skip graph generation ─────────────────────────────
    if getattr(args, "enrich_only", False):
        writer = CausalGraphDBWriter.from_env()
        try:
            market_id = writer.get_market_id(event.event_id)
            graph_id = writer.get_existing_graph_id(event.event_id)
            if not graph_id:
                logger.error(f"No existing graph for {event.event_id} — run without --enrich-only first")
                return False
            return run_enrichment(event.event_id, graph_id, market_id, args.dry_run)
        finally:
            writer.close()

    # ── Step 1: Generate graph via LLM ───────────────────────────────────────
    try:
        gen = LLMCausalGraphGenerator.from_env(model=args.model)
        graph = gen.generate(event)
    except Exception as e:
        logger.error(f"Graph generation failed for {event.event_id}: {e}")
        return False

    # Step 2: Print (always)
    print_graph(graph)

    if args.dry_run:
        logger.info("DRY RUN — skipping DB write.")
        return True

    # ── Step 3: Persist to DB ────────────────────────────────────────────────
    try:
        writer = CausalGraphDBWriter.from_env()
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        return False

    graph_id = None
    try:
        market_id = writer.get_market_id(event.event_id)
        if not market_id:
            logger.warning(
                f"No market found for event_id={event.event_id}. "
                "The event must be registered in the markets table first. "
                "Skipping DB write."
            )
            return False

        graph_id, was_new = writer.write(graph, market_id=market_id, force=args.force)
        
        # Save to legacy local JSON GraphStorage for the report generator
        try:
            from belief_graph.storage import get_storage
            storage = get_storage()
            storage.save(graph.to_belief_graph(), overwrite=True)
            logger.info("✓ Saved generated graph to local JSON storage for legacy report generation.")
        except Exception as e:
            logger.error(f"Failed to save local JSON graph: {e}")

        if was_new:
            logger.info(f"✓ Saved new causal graph to DB: graph_id={graph_id}")
        else:
            logger.info(
                f"✓ Graph already exists (graph_id={graph_id}). "
                "Pass --force to regenerate."
            )

    except Exception as e:
        logger.error(f"DB write failed for {event.event_id}: {e}")
        return False
    finally:
        writer.close()

    # ── Step 4: Evidence enrichment (optional) ───────────────────────────────
    if getattr(args, "enrich", False) and graph_id:
        run_enrichment(event.event_id, graph_id, market_id, args.dry_run)

    return True


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Validate API key early
    if not os.environ.get("GEMINI_API_KEY"):
        logger.error(
            "GEMINI_API_KEY environment variable is not set.\n"
            "  export GEMINI_API_KEY=your_key_here"
        )
        sys.exit(1)

    # Load event registry
    from pipeline.event_registry import get_registry
    registry = get_registry()

    if args.all:
        events = registry.get_all_events()
        logger.info(f"Processing {len(events)} events...")
    else:
        event = registry.get_event(args.event)
        if event is None:
            logger.error(f"Event not found: {args.event!r}")
            logger.error(f"Available events: {registry.list_event_ids()}")
            sys.exit(1)
        events = [event]

    # Process each event
    results = {"success": 0, "failed": 0, "skipped": 0}
    for event in events:
        ok = process_event(event, args)
        if ok:
            results["success"] += 1
        else:
            results["failed"] += 1

    logger.info(
        f"Done. success={results['success']} "
        f"failed={results['failed']}"
    )
    if results["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
