"""
API Endpoints

Route handlers for the FastAPI application.
"""

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now() -> datetime:
    """Get current UTC time as a naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from api.schemas import (
    EventInfo, EventListResponse,
    SignalResponse, SignalListResponse,
    DocumentResponse, DocumentListResponse,
    AnalysisResponse, TopDocumentResponse,
    ProbabilityResponse, StatsResponse, HealthResponse,
    RefreshEventRequest, AnalyzeAllRequest, ErrorResponse,
    RunFullPipelineRequest, PipelineStepResult, RunFullPipelineResponse
)
from belief_graph.graph_builder import GraphBuilder
from belief_graph.storage import get_storage
from pipeline.event_registry import get_registry
from pipeline.normalizer import DocumentNormalizer
from pipeline.signal_extractor import SignalExtractor
from pipeline.delta_engine import DeltaEngine
from integrations.polymarket_client import get_polymarket_client
from cli.commands import PipelineOrchestrator
from scripts.generate_event_market_report import generate_report

logger = logging.getLogger(__name__)

router = APIRouter()


def _analysis_to_response(analysis) -> AnalysisResponse:
    """Convert internal EventAnalysis object to API response model."""
    top_docs = [
        TopDocumentResponse(
            doc_id=d.doc_id,
            title=d.title,
            url=d.url,
            source_type=d.source_type,
            relevance_score=d.relevance_score,
            rank_score=d.rank_score,
            relevance_reason=d.relevance_reason
        )
        for d in analysis.top_documents
    ]

    return AnalysisResponse(
        event_id=analysis.event_id,
        event_title=analysis.event_title,
        current_probability=analysis.current_probability,
        suggested_delta=analysis.suggested_delta,
        delta_mid=round((analysis.delta_min + analysis.delta_max) / 2, 4),
        confidence=analysis.confidence,
        dominant_signal_types=analysis.dominant_signal_types,
        time_until_deadline_days=analysis.time_until_deadline_days,
        top_documents=top_docs,
        signal_summary=analysis.signal_summary,
        analyzed_at=analysis.analyzed_at
    )


# Health check
@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system health."""
    logger.debug("Health check requested")
    return HealthResponse(
        status="healthy",
        timestamp=_utc_now(),
        version="1.0.0"
    )


# Events endpoints
@router.get("/events", response_model=EventListResponse, tags=["Events"])
async def list_events():
    """List all tracked events."""
    logger.info("Listing all events")
    registry = get_registry()
    events = []
    
    for event in registry:
        events.append(EventInfo(
            event_id=event.event_id,
            event_title=event.event_title,
            event_type=event.event_type,
            deadline=event.deadline,
            days_until_deadline=event.days_until_deadline(),
            polymarket_slug=event.polymarket_slug,
            primary_entities=event.primary_entities,
            secondary_entities=event.secondary_entities
        ))
    
    logger.info(f"Found {len(events)} events")
    return EventListResponse(events=events, count=len(events))


@router.get("/events/{event_id}", response_model=AnalysisResponse, tags=["Events"])
async def get_event_analysis(event_id: str):
    """Get current analysis for an event."""
    logger.info(f"Getting analysis for event: {event_id}")
    registry = get_registry()
    event = registry.get_event(event_id)
    
    if event is None:
        logger.warning(f"Event not found: {event_id}")
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")
    
    delta_engine = DeltaEngine()
    analysis = delta_engine.load_analysis(event_id)
    
    if analysis is None:
        logger.warning(f"No analysis found for event: {event_id}")
        raise HTTPException(
            status_code=404, 
            detail=f"No analysis found for {event_id}. Run refresh first."
        )
    
    logger.info(f"Returning analysis for {event_id}: delta={analysis.suggested_delta}")
    return _analysis_to_response(analysis)


@router.post("/events/{event_id}/refresh", response_model=AnalysisResponse, tags=["Events"])
async def refresh_event_analysis(
    event_id: str,
    request: RefreshEventRequest,
    background_tasks: BackgroundTasks
):
    """Trigger pipeline run for a specific event."""
    registry = get_registry()
    event = registry.get_event(event_id)
    
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")
    
    logger.info(f"Refreshing analysis for: {event_id}")
    
    # Run pipeline
    orchestrator = PipelineOrchestrator()
    try:
        results = orchestrator.run_full_pipeline(
            event_id=event_id,
            skip_ingestion=request.skip_ingestion,
            verbose=False
        )
        
        if not results:
            raise HTTPException(status_code=500, detail="Pipeline produced no results")
        
        analysis = results[0]
        
        logger.info(f"Refresh complete for {event_id}: delta={analysis.suggested_delta}")
        return _analysis_to_response(analysis)
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/events/{event_id}/run-full",
    response_model=RunFullPipelineResponse,
    tags=["Pipeline"]
)
async def run_full_event_pipeline(event_id: str, request: RunFullPipelineRequest):
    """
    Run complete workflow for one event and return structured outputs.

    Includes:
    1) end-to-end analysis (ingestion->signals->delta),
    2) belief graph build/load,
    3) consolidated market report (price changes + focused N-1/N-2 graph + related news).
    """
    registry = get_registry()
    event = registry.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")

    started_at = _utc_now()

    def _run_sync() -> Dict[str, Any]:
        steps: List[PipelineStepResult] = []

        # Step 1: main pipeline
        t0 = time.perf_counter()
        orchestrator = PipelineOrchestrator()
        results = orchestrator.run_full_pipeline(
            event_id=event_id,
            skip_ingestion=request.skip_ingestion,
            verbose=False
        )
        if not results:
            raise RuntimeError("Pipeline produced no analysis results")
        analysis = results[0]
        steps.append(
            PipelineStepResult(
                name="analysis_pipeline",
                status="completed",
                duration_sec=round(time.perf_counter() - t0, 3),
                details={
                    "suggested_delta": analysis.suggested_delta,
                    "confidence": round(analysis.confidence, 4),
                    "signals_total": analysis.signal_summary.get("total_signals", 0),
                    "top_documents": len(analysis.top_documents),
                },
            )
        )

        # Step 2: graph
        t1 = time.perf_counter()
        storage = get_storage()
        graph_status = "loaded"
        graph = None
        if not request.rebuild_graph and storage.exists(event_id):
            graph = storage.load(event_id)
        if graph is None:
            builder = GraphBuilder()
            graph = builder.build(
                event_id=event_id,
                max_events=request.max_events,
                max_edges=request.max_edges,
                market_window_only=request.market_window_only,
                window_start=request.window_start,
                window_end=request.window_end,
            )
            storage.save(graph)
            graph_status = "built"

        steps.append(
            PipelineStepResult(
                name="belief_graph",
                status=graph_status,
                duration_sec=round(time.perf_counter() - t1, 3),
                details={
                    "nodes": len(graph.event_nodes),
                    "edges": len(graph.edges),
                    "market_window_only": request.market_window_only,
                },
            )
        )

        # Step 3: consolidated report
        t2 = time.perf_counter()
        report = generate_report(
            event_id=event_id,
            slug=request.market_slug,
            graph_path=None,
            output_path=request.report_output_path,
            top_n1=request.top_n1,
            top_n2=request.top_n2,
            min_conf=request.min_confidence,
            impact_window=request.impact_window_minutes,
        )
        steps.append(
            PipelineStepResult(
                name="market_report",
                status="completed",
                duration_sec=round(time.perf_counter() - t2, 3),
                details={
                    "n1_count": report.get("summary", {}).get("n1_count", 0),
                    "n2_count": report.get("summary", {}).get("n2_count", 0),
                    "price_impacts_available": report.get("summary", {}).get("price_impacts_available", 0),
                },
            )
        )

        return {
            "analysis": analysis,
            "graph": graph,
            "steps": steps,
            "report": report,
        }

    try:
        result = await run_in_threadpool(_run_sync)
    except Exception as e:
        logger.error(f"Full pipeline run failed for {event_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finished_at = _utc_now()
    total_duration_sec = round((finished_at - started_at).total_seconds(), 3)

    graph = result["graph"]
    response = RunFullPipelineResponse(
        status="completed",
        event_id=event_id,
        event_title=event.event_title,
        polymarket_slug=request.market_slug or event.polymarket_slug,
        started_at=started_at,
        finished_at=finished_at,
        total_duration_sec=total_duration_sec,
        steps=result["steps"],
        analysis=_analysis_to_response(result["analysis"]),
        graph_summary={
            "node_count": len(graph.event_nodes),
            "edge_count": len(graph.edges),
            "generated_at": graph.generated_at.isoformat() if graph.generated_at else None,
        },
        report=result["report"],
    )

    return response


# Signals endpoints
@router.get("/events/{event_id}/signals", response_model=SignalListResponse, tags=["Signals"])
async def get_event_signals(event_id: str, limit: int = 50):
    """Get recent signals for an event."""
    logger.info(f"Getting signals for event: {event_id}")
    registry = get_registry()
    event = registry.get_event(event_id)
    
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")
    
    signal_extractor = SignalExtractor()
    signals = signal_extractor.load_signals_for_event(event_id)
    
    signal_responses = [
        SignalResponse(
            signal_id=s.signal_id,
            event_id=s.event_id,
            doc_id=s.doc_id,
            signal_type=s.signal_type,
            direction=s.direction,
            origin=s.origin,
            magnitude=s.magnitude,
            confidence=s.confidence,
            extracted_at=s.extracted_at
        )
        for s in signals[:limit]
    ]
    
    logger.info(f"Found {len(signal_responses)} signals for {event_id}")
    
    return SignalListResponse(
        event_id=event_id,
        signals=signal_responses,
        count=len(signal_responses)
    )


# Documents endpoints
@router.get("/events/{event_id}/documents", response_model=DocumentListResponse, tags=["Documents"])
async def get_event_documents(event_id: str, limit: int = 50):
    """Get top documents for an event."""
    logger.info(f"Getting documents for event: {event_id}")
    registry = get_registry()
    event = registry.get_event(event_id)
    
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")
    
    normalizer = DocumentNormalizer()
    documents = normalizer.load_all_for_event(event_id)
    
    doc_responses = [
        DocumentResponse(
            doc_id=d.doc_id,
            title=d.title,
            url=d.url,
            source_type=d.source_type,
            author_type=d.author_type,
            timestamp=d.timestamp,
            query_used=d.query_used,
            query_type=d.query_type
        )
        for d in documents[:limit]
    ]
    
    logger.info(f"Found {len(doc_responses)} documents for {event_id}")
    
    return DocumentListResponse(
        event_id=event_id,
        documents=doc_responses,
        count=len(doc_responses)
    )


# Analyze all
@router.post("/analyze-all", tags=["Analysis"])
async def analyze_all_events(
    request: AnalyzeAllRequest,
    background_tasks: BackgroundTasks
):
    """Trigger full pipeline for all events."""
    logger.info("Starting full pipeline for all events")
    
    # Run in background for long-running operations
    def run_pipeline():
        orchestrator = PipelineOrchestrator()
        orchestrator.run_full_pipeline(
            skip_ingestion=request.skip_ingestion,
            verbose=False
        )
    
    background_tasks.add_task(run_pipeline)
    
    return {
        "status": "started",
        "message": "Pipeline started in background. Check individual event endpoints for results.",
        "started_at": _utc_now().isoformat()
    }


# Probabilities
@router.get("/probabilities", tags=["Market"])
async def get_all_probabilities():
    """Get current probabilities for all events."""
    logger.info("Fetching all probabilities from Polymarket")
    registry = get_registry()
    client = get_polymarket_client()
    
    results = []
    for event in registry:
        prob = client.get_probability(event.polymarket_slug)
        results.append(ProbabilityResponse(
            event_id=event.event_id,
            polymarket_slug=event.polymarket_slug,
            probability=prob,
            fetched_at=_utc_now()
        ))
    
    logger.info(f"Fetched probabilities for {len(results)} events")
    return {"probabilities": [r.dict() for r in results]}


@router.get("/events/{event_id}/probability", response_model=ProbabilityResponse, tags=["Market"])
async def get_event_probability(event_id: str):
    """Get current probability for a specific event."""
    logger.info(f"Fetching probability for event: {event_id}")
    registry = get_registry()
    event = registry.get_event(event_id)
    
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")
    
    client = get_polymarket_client()
    prob = client.get_probability(event.polymarket_slug)
    
    logger.info(f"Probability for {event_id}: {prob}")
    
    return ProbabilityResponse(
        event_id=event_id,
        polymarket_slug=event.polymarket_slug,
        probability=prob,
        fetched_at=_utc_now()
    )


# Stats
@router.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Get pipeline statistics."""
    logger.info("Getting pipeline statistics")
    from pipeline.ingestion import DataIngestor
    
    ingestor = DataIngestor()
    normalizer = DocumentNormalizer()
    
    ingestion_stats = ingestor.get_stats()
    normalized_count = 0
    
    # Query database for all normalized document artifacts
    try:
        from utils.db_storage import list_artifacts
        
        registry = get_registry()
        for event in registry:
            # list_artifacts returns a list of PipelineArtifact models
            # but load_artifact returns the actual data dict/list
            from utils.db_storage import load_artifact
            data = load_artifact(event.event_id, "normalized_documents")
            if data:
                normalized_count += len(data)
    except Exception as e:
        logger.warning(f"Error counting normalized documents in db: {e}")
    
    logger.info(f"Stats: {ingestion_stats.get('total_documents', 0)} raw, {normalized_count} normalized")
    
    return StatsResponse(
        total_documents=ingestion_stats.get("total_documents", 0),
        normalized_documents=normalized_count,
        by_source=ingestion_stats.get("by_source", {}),
        by_event=ingestion_stats.get("by_event", {})
    )
