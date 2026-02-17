"""
API Schemas

Pydantic models for API request/response validation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# Request Models

class RefreshEventRequest(BaseModel):
    """Request to refresh analysis for an event."""
    skip_ingestion: bool = Field(
        default=False,
        description="Skip data ingestion, use existing data"
    )


class AnalyzeAllRequest(BaseModel):
    """Request to analyze all events."""
    skip_ingestion: bool = Field(
        default=False,
        description="Skip data ingestion"
    )


class RunFullPipelineRequest(BaseModel):
    """Request to run end-to-end processing for one event."""
    skip_ingestion: bool = Field(
        default=False,
        description="Skip ingestion and use existing stored documents"
    )
    rebuild_graph: bool = Field(
        default=False,
        description="Force belief graph rebuild even if graph exists"
    )
    max_events: int = Field(
        default=120,
        description="Maximum events for graph builder"
    )
    max_edges: int = Field(
        default=250,
        description="Maximum edges for graph builder"
    )
    market_window_only: bool = Field(
        default=True,
        description="Restrict graph nodes to the market active time window"
    )
    window_start: Optional[datetime] = Field(
        default=None,
        description="Optional explicit graph lower time bound (UTC)"
    )
    window_end: Optional[datetime] = Field(
        default=None,
        description="Optional explicit graph upper time bound (UTC)"
    )
    market_slug: Optional[str] = Field(
        default=None,
        description="Override Polymarket slug for impact mapping"
    )
    top_n1: int = Field(
        default=15,
        description="Top N-1 nodes in focused graph"
    )
    top_n2: int = Field(
        default=5,
        description="Top N-2 nodes per N-1 node in focused graph"
    )
    min_confidence: float = Field(
        default=0.3,
        description="Minimum edge confidence for focused graph extraction"
    )
    impact_window_minutes: int = Field(
        default=2,
        description="Price-impact matching window in minutes"
    )
    report_output_path: Optional[str] = Field(
        default=None,
        description="Optional output path for consolidated report JSON"
    )


class PipelineStepResult(BaseModel):
    """One pipeline step execution summary."""
    name: str
    status: str
    duration_sec: float
    details: Dict[str, Any] = Field(default_factory=dict)


class RunFullPipelineResponse(BaseModel):
    """End-to-end pipeline response for one event."""
    status: str
    event_id: str
    event_title: str
    polymarket_slug: Optional[str] = None
    started_at: datetime
    finished_at: datetime
    total_duration_sec: float
    steps: List[PipelineStepResult]
    analysis: "AnalysisResponse"
    graph_summary: Dict[str, Any]
    report: Dict[str, Any]


# Response Models

class EventInfo(BaseModel):
    """Basic event information."""
    event_id: str
    event_title: str
    event_type: str
    deadline: datetime
    days_until_deadline: int
    polymarket_slug: str
    primary_entities: List[str]
    secondary_entities: List[str]


class SignalResponse(BaseModel):
    """Signal information."""
    signal_id: str
    event_id: str
    doc_id: str
    signal_type: str
    direction: str
    origin: str
    magnitude: float
    confidence: float
    extracted_at: datetime
    doc_title: Optional[str] = None
    doc_url: Optional[str] = None


class DocumentResponse(BaseModel):
    """Document information."""
    doc_id: str
    title: str
    url: str
    source_type: str
    author_type: str
    timestamp: Optional[datetime]
    query_used: str
    query_type: str
    relevance_score: Optional[float] = None
    rank_score: Optional[float] = None


class TopDocumentResponse(BaseModel):
    """Top ranked document for analysis output."""
    doc_id: str
    title: str
    url: str
    source_type: str
    timestamp: Optional[datetime] = None
    relevance_score: float
    rank_score: float
    relevance_reason: str
    key_signals: Optional[List[str]] = None


class SignalSummary(BaseModel):
    """Summary of signals for an event."""
    total_signals: int
    by_type: Dict[str, int]
    by_direction: Dict[str, int]
    by_origin: Dict[str, int]
    avg_magnitude: float
    avg_confidence: float


class AnalysisResponse(BaseModel):
    """Complete analysis for an event."""
    event_id: str
    event_title: str
    current_probability: float
    polymarket_slug: Optional[str] = None
    suggested_delta: str
    delta_mid: Optional[float] = None
    confidence: float
    dominant_signal_types: List[str]
    time_until_deadline_days: int
    top_documents: List[TopDocumentResponse]
    signal_summary: Dict
    analyzed_at: datetime


class EventListResponse(BaseModel):
    """Response for listing events."""
    events: List[EventInfo]
    count: int


class SignalListResponse(BaseModel):
    """Response for listing signals."""
    event_id: str
    signals: List[SignalResponse]
    count: int


class DocumentListResponse(BaseModel):
    """Response for listing documents."""
    event_id: str
    documents: List[DocumentResponse]
    count: int


class ProbabilityResponse(BaseModel):
    """Current probability from Polymarket."""
    event_id: str
    polymarket_slug: str
    probability: Optional[float]
    fetched_at: datetime


class StatsResponse(BaseModel):
    """Pipeline statistics."""
    total_documents: int
    normalized_documents: int
    by_source: Dict[str, int]
    by_event: Dict[str, int]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None


RunFullPipelineResponse.update_forward_refs()
