"""
CLI Command Handlers

Implementation of CLI commands for the pipeline.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.event_registry import get_registry, Event
from pipeline.query_generator import QueryGenerator, generate_queries_for_all_events
from pipeline.ingestion import DataIngestor
from pipeline.normalizer import DocumentNormalizer
from pipeline.entity_extractor import EntityExtractor
from pipeline.event_mapper import EventMapper
from pipeline.signal_extractor import SignalExtractor
from pipeline.delta_engine import DeltaEngine, EventAnalysis
from integrations.polymarket_client import get_polymarket_client

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates the full pipeline execution.
    
    Pipeline stages:
    1. Load events from registry
    2. Generate queries
    3. Ingest data (parallel by query)
    4. Normalize documents
    5. Extract entities
    6. Map to events (3-stage)
    7. Extract signals
    8. Calculate deltas
    9. Rank documents
    10. Output JSON
    """
    
    def __init__(self):
        """Initialize orchestrator with all pipeline components."""
        logger.info("Initializing PipelineOrchestrator...")
        self.registry = get_registry()
        logger.info(f"Loaded {len(self.registry)} events from registry")
        
        self.query_generator = QueryGenerator()
        self.ingestor = DataIngestor()
        self.normalizer = DocumentNormalizer()
        self.entity_extractor = EntityExtractor(self.normalizer)
        self.mapper = EventMapper(self.registry, self.normalizer)
        self.signal_extractor = SignalExtractor(self.registry, self.normalizer, self.mapper)
        self.delta_engine = DeltaEngine(
            self.registry,
            self.signal_extractor,
            self.normalizer
        )
        logger.info("PipelineOrchestrator initialized successfully")
    
    def run_full_pipeline(
        self,
        event_id: Optional[str] = None,
        skip_ingestion: bool = False,
        verbose: bool = True
    ) -> List[EventAnalysis]:
        """
        Run the full pipeline for one or all events.
        
        Args:
            event_id: Optional specific event ID
            skip_ingestion: Skip data ingestion (use existing data)
            verbose: Print progress messages
        
        Returns:
            List of EventAnalysis results
        """
        results = []
        
        # Get events to process
        if event_id:
            event = self.registry.get_event(event_id)
            if event is None:
                raise ValueError(f"Event not found: {event_id}")
            events = [event]
        else:
            events = list(self.registry)
        
        logger.info(f"Starting pipeline for {len(events)} event(s)")
        
        for event in events:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Processing: {event.event_title}")
                print(f"{'='*60}")
            
            try:
                analysis = self._process_event(event, skip_ingestion, verbose)
                results.append(analysis)
                
                if verbose:
                    self._print_analysis_summary(analysis)
                    
            except Exception as e:
                logger.error(f"Failed to process {event.event_id}: {e}", exc_info=True)
                if verbose:
                    print(f"ERROR: {e}")
        
        logger.info(f"Pipeline completed. Processed {len(results)} events successfully")
        return results
    
    def _process_event(
        self,
        event: Event,
        skip_ingestion: bool,
        verbose: bool
    ) -> EventAnalysis:
        """Process a single event through the pipeline."""
        logger.info(f"Processing event: {event.event_id}")
        
        # Step 1: Generate queries
        logger.info(f"[{event.event_id}] Step 1/6: Generating queries...")
        if verbose:
            print("  [1/6] Generating queries...")
        query_set = self.query_generator.generate_queries_for_event(event)
        queries = [{"query": q.query, "query_type": q.query_type} for q in query_set.queries]
        logger.info(f"[{event.event_id}] Generated {len(queries)} queries")
        if verbose:
            print(f"        Generated {len(queries)} queries")
        
        # Step 2: Ingest data
        if not skip_ingestion:
            logger.info(f"[{event.event_id}] Step 2/6: Ingesting data from sources...")
            if verbose:
                print("  [2/6] Ingesting data from sources...")
            docs = self.ingestor.ingest_for_event(event.event_id, queries)
            logger.info(f"[{event.event_id}] Ingested {len(docs)} documents")
            if verbose:
                print(f"        Ingested {len(docs)} documents")
        else:
            logger.info(f"[{event.event_id}] Step 2/6: Skipping ingestion (using existing data)")
            if verbose:
                print("  [2/6] Skipping ingestion (using existing data)")
            docs = self.ingestor.get_documents_for_event(event.event_id)
            logger.info(f"[{event.event_id}] Found {len(docs)} existing documents")
        
        # Step 3: Normalize documents
        logger.info(f"[{event.event_id}] Step 3/6: Normalizing documents...")
        if verbose:
            print("  [3/6] Normalizing documents...")
        normalized = []
        for doc in docs:
            norm = self.normalizer.normalize_and_save(doc.to_dict())
            normalized.append(norm)
        logger.info(f"[{event.event_id}] Normalized {len(normalized)} documents")
        if verbose:
            print(f"        Normalized {len(normalized)} documents")
        
        # Step 4: Extract entities
        logger.info(f"[{event.event_id}] Step 4/6: Extracting entities...")
        if verbose:
            print("  [4/6] Extracting entities...")
        updated_docs = self.entity_extractor.process_event_documents(event.event_id)
        logger.info(f"[{event.event_id}] Extracted entities from {len(updated_docs)} documents")
        
        # Step 5: Map documents to event
        logger.info(f"[{event.event_id}] Step 5/6: Mapping documents to event...")
        if verbose:
            print("  [5/6] Mapping documents to event...")
        mapping_summary = self.mapper.process_event(event.event_id)
        logger.info(f"[{event.event_id}] Mapping complete: {mapping_summary['relevant_documents']}/{mapping_summary['total_documents']} relevant")
        if verbose:
            print(f"        {mapping_summary['relevant_documents']}/{mapping_summary['total_documents']} relevant")
        
        # Step 6: Extract signals and calculate delta
        logger.info(f"[{event.event_id}] Step 6/6: Extracting signals and calculating delta...")
        if verbose:
            print("  [6/6] Extracting signals and calculating delta...")
        analysis = self.delta_engine.analyze_event(event.event_id)
        logger.info(f"[{event.event_id}] Analysis complete: delta={analysis.suggested_delta}, confidence={analysis.confidence}")
        
        return analysis
    
    def _print_analysis_summary(self, analysis: EventAnalysis) -> None:
        """Print analysis summary to console."""
        print(f"\n  RESULTS:")
        print(f"  {'─'*50}")
        print(f"  Current Probability: {analysis.current_probability:.1%}")
        print(f"  Suggested Delta:     {analysis.suggested_delta}")
        print(f"  Confidence:          {analysis.confidence:.0%}")
        print(f"  Days to Deadline:    {analysis.time_until_deadline_days}")
        print(f"  Total Signals:       {analysis.signal_summary.get('total_signals', 0)}")
        
        if analysis.top_documents:
            print(f"\n  Top Documents:")
            for i, doc in enumerate(analysis.top_documents[:3], 1):
                print(f"    {i}. {doc.title[:50]}...")
                print(f"       Score: {doc.rank_score:.2f} | Source: {doc.source_type}")
    
    def run_ingestion_only(
        self,
        event_id: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        """Run only the ingestion stage."""
        logger.info("Running ingestion-only mode")
        
        if event_id:
            event = self.registry.get_event(event_id)
            if event is None:
                raise ValueError(f"Event not found: {event_id}")
            events = [event]
        else:
            events = list(self.registry)
        
        stats = {"total_documents": 0, "by_event": {}}
        
        for event in events:
            logger.info(f"Ingesting data for event: {event.event_id}")
            if verbose:
                print(f"Ingesting data for: {event.event_title}")
            
            query_set = self.query_generator.generate_queries_for_event(event)
            queries = [{"query": q.query, "query_type": q.query_type} for q in query_set.queries]
            
            docs = self.ingestor.ingest_for_event(event.event_id, queries)
            
            stats["by_event"][event.event_id] = len(docs)
            stats["total_documents"] += len(docs)
            
            logger.info(f"Ingested {len(docs)} documents for {event.event_id}")
            if verbose:
                print(f"  → Ingested {len(docs)} documents")
        
        return stats
    
    def run_signal_extraction(
        self,
        event_id: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        """Run only signal extraction for existing documents."""
        logger.info("Running signal extraction mode")
        
        if event_id:
            events = [self.registry.get_event(event_id)]
        else:
            events = list(self.registry)
        
        results = {}
        
        for event in events:
            if event is None:
                continue
            
            logger.info(f"Extracting signals for event: {event.event_id}")
            if verbose:
                print(f"Extracting signals for: {event.event_title}")
            
            signals = self.signal_extractor.extract_signals_for_event(event.event_id)
            results[event.event_id] = {
                "signal_count": len(signals),
                "by_type": self.signal_extractor.get_signal_summary(signals).get("by_type", {})
            }
            
            logger.info(f"Extracted {len(signals)} signals for {event.event_id}")
            if verbose:
                print(f"  → Extracted {len(signals)} signals")
        
        return results


def list_events(verbose: bool = True) -> List[Dict]:
    """List all registered events."""
    logger.info("Listing all registered events")
    registry = get_registry()
    events = []
    
    for event in registry:
        event_info = {
            "event_id": event.event_id,
            "title": event.event_title,
            "type": event.event_type,
            "deadline": event.deadline.isoformat(),
            "days_remaining": event.days_until_deadline(),
            "polymarket_slug": event.polymarket_slug
        }
        events.append(event_info)
        
        if verbose:
            print(f"\n{event.event_id}")
            print(f"  Title: {event.event_title}")
            print(f"  Type: {event.event_type}")
            print(f"  Deadline: {event.deadline.date()} ({event.days_until_deadline()} days)")
    
    logger.info(f"Listed {len(events)} events")
    return events


def show_analysis(event_id: str, verbose: bool = True) -> Optional[EventAnalysis]:
    """Show saved analysis for an event."""
    logger.info(f"Loading analysis for event: {event_id}")
    delta_engine = DeltaEngine()
    analysis = delta_engine.load_analysis(event_id)
    
    if analysis is None:
        logger.warning(f"No analysis found for: {event_id}")
        if verbose:
            print(f"No analysis found for: {event_id}")
        return None
    
    logger.info(f"Loaded analysis for {event_id}: delta={analysis.suggested_delta}")
    
    if verbose:
        print(f"\nAnalysis for: {analysis.event_title}")
        print(f"{'='*60}")
        print(f"Current Probability: {analysis.current_probability:.1%}")
        print(f"Suggested Delta:     {analysis.suggested_delta}")
        print(f"Confidence:          {analysis.confidence:.0%}")
        print(f"Days to Deadline:    {analysis.time_until_deadline_days}")
        print(f"Analyzed At:         {analysis.analyzed_at}")
        
        print(f"\nSignal Summary:")
        summary = analysis.signal_summary
        print(f"  Total: {summary.get('total_signals', 0)}")
        for sig_type, count in summary.get('by_type', {}).items():
            print(f"  {sig_type}: {count}")
        
        print(f"\nTop Documents:")
        for i, doc in enumerate(analysis.top_documents, 1):
            print(f"  {i}. {doc.title}")
            print(f"     URL: {doc.url}")
            print(f"     Score: {doc.rank_score:.2f}")
    
    return analysis


def update_probabilities(verbose: bool = True) -> Dict[str, Optional[float]]:
    """Fetch current probabilities from Polymarket."""
    logger.info("Fetching probabilities from Polymarket")
    registry = get_registry()
    client = get_polymarket_client()
    
    slugs = {event.event_id: event.polymarket_slug for event in registry}
    probabilities = client.get_probabilities_for_events(slugs)
    
    logger.info(f"Fetched probabilities for {len(probabilities)} events")
    
    if verbose:
        print("\nCurrent Polymarket Probabilities:")
        print("="*50)
        for event_id, prob in probabilities.items():
            event = registry.get_event(event_id)
            title = event.event_title if event else event_id
            if prob is not None:
                print(f"  {title}: {prob:.1%}")
            else:
                print(f"  {title}: Unable to fetch")
    
    return probabilities


def show_stats(verbose: bool = True) -> Dict:
    """Show pipeline statistics."""
    logger.info("Gathering pipeline statistics")
    ingestor = DataIngestor()
    normalizer = DocumentNormalizer()
    
    ingestion_stats = ingestor.get_stats()
    
    # Count normalized documents
    normalized_count = len(list(normalizer.normalized_dir.glob("*.json")))
    
    stats = {
        "documents": ingestion_stats,
        "normalized_documents": normalized_count
    }
    
    logger.info(f"Stats: {ingestion_stats['total_documents']} raw docs, {normalized_count} normalized")
    
    if verbose:
        print("\nPipeline Statistics:")
        print("="*50)
        print(f"Total Raw Documents:        {ingestion_stats['total_documents']}")
        print(f"Normalized Documents:       {normalized_count}")
        print(f"\nBy Source:")
        for source, count in ingestion_stats.get('by_source', {}).items():
            print(f"  {source}: {count}")
        print(f"\nBy Event:")
        for event, count in ingestion_stats.get('by_event', {}).items():
            print(f"  {event}: {count}")
    
    return stats
