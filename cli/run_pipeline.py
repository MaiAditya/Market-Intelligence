#!/usr/bin/env python3
"""
AI Market Intelligence CLI

Command-line interface for running the intelligence pipeline.

Usage:
    python cli/run_pipeline.py analyze-all
    python cli/run_pipeline.py analyze --event gemini-5-release-2026
    python cli/run_pipeline.py show --event gemini-5-release-2026
    python cli/run_pipeline.py update-probabilities
    python cli/run_pipeline.py list-events
    python cli/run_pipeline.py stats
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.json_utils import dump_json
from cli.commands import (
    PipelineOrchestrator,
    list_events,
    show_analysis,
    update_probabilities,
    show_stats
)
from belief_graph.graph_builder import GraphBuilder
from belief_graph.storage import get_storage

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    logger.info("Logging configured")


def cmd_analyze_all(args):
    """Run full pipeline for all events."""
    logger.info("Command: analyze-all")
    print("Running full pipeline for all events...")
    
    orchestrator = PipelineOrchestrator()
    results = orchestrator.run_full_pipeline(
        skip_ingestion=args.skip_ingestion,
        verbose=True
    )
    
    if args.output:
        output_data = [r.to_dict() for r in results]
        with open(args.output, 'w') as f:
            dump_json(output_data, f)
        print(f"\nResults saved to: {args.output}")
        logger.info(f"Results saved to: {args.output}")
    
    print(f"\nCompleted analysis for {len(results)} events")
    logger.info(f"Completed analysis for {len(results)} events")


def cmd_analyze(args):
    """Run pipeline for a specific event."""
    if not args.event:
        print("Error: --event is required")
        return
    
    logger.info(f"Command: analyze --event {args.event}")
    print(f"Analyzing event: {args.event}")
    
    orchestrator = PipelineOrchestrator()
    
    try:
        results = orchestrator.run_full_pipeline(
            event_id=args.event,
            skip_ingestion=args.skip_ingestion,
            verbose=True
        )
        
        if results and args.output:
            with open(args.output, 'w') as f:
                dump_json(results[0].to_dict(), f)
            print(f"\nResults saved to: {args.output}")
            logger.info(f"Results saved to: {args.output}")
            
    except ValueError as e:
        print(f"Error: {e}")
        logger.error(f"Error: {e}")


def cmd_show(args):
    """Show analysis for an event."""
    if not args.event:
        print("Error: --event is required")
        return
    
    logger.info(f"Command: show --event {args.event}")
    
    analysis = show_analysis(args.event, verbose=True)
    
    if analysis and args.output:
        with open(args.output, 'w') as f:
            dump_json(analysis.to_dict(), f)
        print(f"\nSaved to: {args.output}")


def cmd_ingest(args):
    """Run ingestion only."""
    logger.info("Command: ingest")
    print("Running data ingestion...")
    
    orchestrator = PipelineOrchestrator()
    stats = orchestrator.run_ingestion_only(
        event_id=args.event,
        verbose=True
    )
    print(f"\nTotal documents ingested: {stats['total_documents']}")


def cmd_extract_signals(args):
    """Extract signals from existing documents."""
    logger.info("Command: extract-signals")
    print("Extracting signals...")
    
    orchestrator = PipelineOrchestrator()
    results = orchestrator.run_signal_extraction(
        event_id=args.event,
        verbose=True
    )
    
    total = sum(r['signal_count'] for r in results.values())
    print(f"\nTotal signals extracted: {total}")


def cmd_list_events(args):
    """List all registered events."""
    logger.info("Command: list-events")
    list_events(verbose=True)


def cmd_update_probabilities(args):
    """Fetch current probabilities from Polymarket."""
    logger.info("Command: update-probabilities")
    update_probabilities(verbose=True)


def cmd_stats(args):
    """Show pipeline statistics."""
    logger.info("Command: stats")
    show_stats(verbose=True)


# ============================================================
# Belief Graph Commands
# ============================================================

def cmd_build_graph(args):
    """Build belief update graph for an event."""
    if not args.event:
        print("Error: --event is required")
        return
    
    logger.info(f"Command: build-graph --event {args.event}")
    print(f"Building belief graph for: {args.event}")
    
    try:
        builder = GraphBuilder()
        storage = get_storage()
        
        parsed_start = None
        parsed_end = None
        if args.window_start:
            parsed_start = datetime.fromisoformat(args.window_start.replace("Z", "+00:00")).replace(tzinfo=None)
        if args.window_end:
            parsed_end = datetime.fromisoformat(args.window_end.replace("Z", "+00:00")).replace(tzinfo=None)
        
        # Check if exists and not forcing rebuild
        if not args.rebuild and storage.exists(args.event):
            print(f"Graph already exists for {args.event}")
            if not args.force:
                print("Use --rebuild to force rebuild")
                return
        
        # Build graph
        print("  Building graph (this may take a moment)...")
        graph = builder.build(
            args.event,
            max_events=args.max_events,
            max_edges=args.max_edges,
            market_window_only=args.market_window_only,
            window_start=parsed_start,
            window_end=parsed_end,
        )
        
        # Save
        filepath = storage.save(graph)
        
        print(f"\n  Graph built successfully!")
        print(f"  {'─'*40}")
        print(f"  Nodes:     {len(graph.event_nodes)}")
        print(f"  Edges:     {len(graph.edges)}")
        print(f"  Saved to:  {filepath}")
        
        if args.output:
            with open(args.output, 'w') as f:
                dump_json(graph.to_dict(), f)
            print(f"  Output:    {args.output}")
        
        logger.info(f"Graph built: {len(graph.event_nodes)} nodes, {len(graph.edges)} edges")
        
    except ValueError as e:
        print(f"Error: {e}")
        logger.error(f"Error building graph: {e}")
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error building graph: {e}", exc_info=True)


def cmd_show_graph(args):
    """Show belief graph for an event."""
    if not args.event:
        print("Error: --event is required")
        return
    
    logger.info(f"Command: show-graph --event {args.event}")
    
    storage = get_storage()
    graph = storage.load(args.event)
    
    if graph is None:
        print(f"No belief graph found for: {args.event}")
        print("Build one with: build-graph --event <event_id>")
        return
    
    # Print summary
    print(f"\nBelief Graph: {args.event}")
    print(f"{'='*60}")
    print(f"\nTarget Belief:")
    print(f"  Question:    {graph.belief_node.question}")
    print(f"  Price:       {graph.belief_node.current_price:.2%}")
    print(f"  Liquidity:   ${graph.belief_node.liquidity:,.0f}")
    
    print(f"\nGraph Statistics:")
    print(f"  Nodes:       {len(graph.event_nodes)}")
    print(f"  Edges:       {len(graph.edges)}")
    print(f"  Generated:   {graph.generated_at}")
    
    # Show top upstream events
    print(f"\nTop Upstream Events (by impact):")
    print(f"  {'─'*55}")
    
    upstream = graph.get_upstream_events(limit=args.top)
    for i, event in enumerate(upstream, 1):
        impact = sum(e.confidence for e in graph.edges if e.from_event_id == event.event_id)
        print(f"  {i}. [{event.event_type:10}] {event.action[:40]:<40} (impact: {impact:.3f})")
    
    # Show top edges
    if args.edges:
        print(f"\nTop Edges (by confidence):")
        print(f"  {'─'*55}")
        
        sorted_edges = sorted(graph.edges, key=lambda e: e.confidence, reverse=True)
        for i, edge in enumerate(sorted_edges[:args.edges], 1):
            from_event = graph.event_nodes.get(edge.from_event_id)
            from_action = from_event.action[:30] if from_event else edge.from_event_id[:30]
            print(f"  {i}. {from_action} → {edge.to_event_id[:20]}")
            print(f"     Mechanism: {edge.mechanism_type}, Confidence: {edge.confidence:.3f}")
    
    # Save if output specified
    if args.output:
        with open(args.output, 'w') as f:
            dump_json(graph.to_dict(), f)
        print(f"\nSaved to: {args.output}")


def cmd_list_graphs(args):
    """List all stored belief graphs."""
    logger.info("Command: list-graphs")
    
    storage = get_storage()
    graphs = storage.list_graphs()
    
    if not graphs:
        print("No belief graphs found.")
        print("Build one with: build-graph --event <event_id>")
        return
    
    print(f"\nStored Belief Graphs ({len(graphs)}):")
    print(f"{'='*70}")
    print(f"{'Event ID':<30} {'Nodes':>7} {'Edges':>7} {'Saved At':<25}")
    print(f"{'─'*70}")
    
    for g in graphs:
        saved = g.get('saved_at', '')[:19] if g.get('saved_at') else 'unknown'
        print(f"{g['event_id']:<30} {g['node_count']:>7} {g['edge_count']:>7} {saved:<25}")
    
    print()


def cmd_graph_stats(args):
    """Show belief graph storage statistics."""
    logger.info("Command: graph-stats")
    
    storage = get_storage()
    stats = storage.get_stats()
    
    print(f"\nBelief Graph Storage Statistics:")
    print(f"{'='*40}")
    print(f"  Total graphs:   {stats['graph_count']}")
    print(f"  Total nodes:    {stats['total_nodes']}")
    print(f"  Total edges:    {stats['total_edges']}")
    print(f"  Storage size:   {stats['storage_size_mb']:.2f} MB")
    print(f"  Cache size:     {stats['cache_size']}/{stats['cache_max_size']}")
    print()


def cmd_build_all_graphs(args):
    """Build belief graphs for all events."""
    logger.info("Command: build-all-graphs")
    
    from pipeline.event_registry import get_registry
    registry = get_registry()
    
    events = list(registry)
    print(f"Building belief graphs for {len(events)} events...")
    
    builder = GraphBuilder()
    storage = get_storage()
    
    success = 0
    failed = 0
    skipped = 0
    
    for event in events:
        event_id = event.event_id
        
        # Check if exists
        if not args.rebuild and storage.exists(event_id):
            print(f"  [SKIP] {event_id} (already exists)")
            skipped += 1
            continue
        
        try:
            print(f"  [BUILD] {event_id}...")
            graph = builder.build(event_id, max_events=args.max_events)
            storage.save(graph)
            print(f"          → {len(graph.event_nodes)} nodes, {len(graph.edges)} edges")
            success += 1
        except Exception as e:
            print(f"  [FAIL] {event_id}: {e}")
            failed += 1
    
    print(f"\nCompleted: {success} built, {skipped} skipped, {failed} failed")


def main():
    parser = argparse.ArgumentParser(
        description="AI Market Intelligence Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli/run_pipeline.py analyze-all
  python cli/run_pipeline.py analyze --event gemini-5-release-2026
  python cli/run_pipeline.py show --event gpt-5-release-2026
  python cli/run_pipeline.py ingest --event gemini-5-release-2026
  python cli/run_pipeline.py update-probabilities
        """
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # analyze-all
    analyze_all_parser = subparsers.add_parser(
        "analyze-all",
        help="Run full pipeline for all events"
    )
    analyze_all_parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip data ingestion, use existing data"
    )
    analyze_all_parser.add_argument(
        "-o", "--output",
        help="Output file for results (JSON)"
    )
    analyze_all_parser.set_defaults(func=cmd_analyze_all)
    
    # analyze
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run pipeline for a specific event"
    )
    analyze_parser.add_argument(
        "--event", "-e",
        required=True,
        help="Event ID to analyze"
    )
    analyze_parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip data ingestion"
    )
    analyze_parser.add_argument(
        "-o", "--output",
        help="Output file (JSON)"
    )
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # show
    show_parser = subparsers.add_parser(
        "show",
        help="Show saved analysis for an event"
    )
    show_parser.add_argument(
        "--event", "-e",
        required=True,
        help="Event ID"
    )
    show_parser.add_argument(
        "-o", "--output",
        help="Output file (JSON)"
    )
    show_parser.set_defaults(func=cmd_show)
    
    # ingest
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Run data ingestion only"
    )
    ingest_parser.add_argument(
        "--event", "-e",
        help="Event ID (optional, all events if not specified)"
    )
    ingest_parser.set_defaults(func=cmd_ingest)
    
    # extract-signals
    signals_parser = subparsers.add_parser(
        "extract-signals",
        help="Extract signals from existing documents"
    )
    signals_parser.add_argument(
        "--event", "-e",
        help="Event ID (optional)"
    )
    signals_parser.set_defaults(func=cmd_extract_signals)
    
    # list-events
    list_parser = subparsers.add_parser(
        "list-events",
        help="List all registered events"
    )
    list_parser.set_defaults(func=cmd_list_events)
    
    # update-probabilities
    prob_parser = subparsers.add_parser(
        "update-probabilities",
        help="Fetch current probabilities from Polymarket"
    )
    prob_parser.set_defaults(func=cmd_update_probabilities)
    
    # stats
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show pipeline statistics"
    )
    stats_parser.set_defaults(func=cmd_stats)
    
    # ============================================================
    # Belief Graph Commands
    # ============================================================
    
    # build-graph
    build_graph_parser = subparsers.add_parser(
        "build-graph",
        help="Build belief update graph for an event"
    )
    build_graph_parser.add_argument(
        "--event", "-e",
        required=True,
        help="Event ID to build graph for"
    )
    build_graph_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild even if graph exists"
    )
    build_graph_parser.add_argument(
        "--force",
        action="store_true",
        help="Alias for --rebuild"
    )
    build_graph_parser.add_argument(
        "--max-events",
        type=int,
        default=100,
        help="Maximum events to include (default: 100)"
    )
    build_graph_parser.add_argument(
        "--max-edges",
        type=int,
        default=200,
        help="Maximum edges to include (default: 200)"
    )
    build_graph_parser.add_argument(
        "--market-window-only",
        action="store_true",
        help="Filter graph nodes to market active window (start/end)"
    )
    build_graph_parser.add_argument(
        "--window-start",
        help="Optional explicit window start (ISO datetime)"
    )
    build_graph_parser.add_argument(
        "--window-end",
        help="Optional explicit window end (ISO datetime)"
    )
    build_graph_parser.add_argument(
        "-o", "--output",
        help="Output file for graph (JSON)"
    )
    build_graph_parser.set_defaults(func=cmd_build_graph)
    
    # show-graph
    show_graph_parser = subparsers.add_parser(
        "show-graph",
        help="Show belief graph for an event"
    )
    show_graph_parser.add_argument(
        "--event", "-e",
        required=True,
        help="Event ID"
    )
    show_graph_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top upstream events to show (default: 10)"
    )
    show_graph_parser.add_argument(
        "--edges",
        type=int,
        default=0,
        help="Number of top edges to show (default: 0)"
    )
    show_graph_parser.add_argument(
        "-o", "--output",
        help="Output file (JSON)"
    )
    show_graph_parser.set_defaults(func=cmd_show_graph)
    
    # list-graphs
    list_graphs_parser = subparsers.add_parser(
        "list-graphs",
        help="List all stored belief graphs"
    )
    list_graphs_parser.set_defaults(func=cmd_list_graphs)
    
    # graph-stats
    graph_stats_parser = subparsers.add_parser(
        "graph-stats",
        help="Show belief graph storage statistics"
    )
    graph_stats_parser.set_defaults(func=cmd_graph_stats)
    
    # build-all-graphs
    build_all_parser = subparsers.add_parser(
        "build-all-graphs",
        help="Build belief graphs for all events"
    )
    build_all_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild existing graphs"
    )
    build_all_parser.add_argument(
        "--max-events",
        type=int,
        default=100,
        help="Maximum events per graph (default: 100)"
    )
    build_all_parser.set_defaults(func=cmd_build_all_graphs)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    setup_logging(args.verbose)
    
    # Run the command
    args.func(args)


if __name__ == "__main__":
    main()
