"""
API Schemas

Pydantic models for API request/response validation.
"""

from datetime import datetime
from typing import Dict, List, Optional
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
