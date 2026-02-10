"""
Belief Graph API Endpoints

REST API endpoints for belief update graph queries.

Endpoints:
- GET /belief-graph/{event_id} - Get full belief DAG for event
- GET /belief-graph/{event_id}/upstream - Top N upstream events by impact
- GET /belief-graph/{event_id}/edges - All edges with explanations
- POST /belief-graph/{event_id}/build - Build/rebuild graph for event
- GET /belief-graph/list - List all stored graphs
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.event_registry import get_registry
from belief_graph.graph_builder import GraphBuilder
from belief_graph.storage import get_storage, GraphStorage

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Get current UTC time as a naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# Pydantic Response Models

class EvidenceScoresResponse(BaseModel):
    """Evidence scores for an edge."""
    price_response: float
    volume_response: float
    narrative_overlap: float
    historical_precedent: float


class EventNodeResponse(BaseModel):
    """Event node in the graph."""
    event_id: str
    event_type: str
    timestamp: Optional[datetime]
    actors: List[str]
    action: str
    object: str
    certainty: float
    source: str
    scope: str
    source_doc_id: Optional[str] = None
    raw_title: Optional[str] = None
    url: Optional[str] = None


class BeliefNodeResponse(BaseModel):
    """Belief node (target market) in the graph."""
    belief_id: str
    question: str
    resolution_time: Optional[datetime]
    current_price: float
    liquidity: float
    event_id: str
    polymarket_slug: str


class BeliefEdgeResponse(BaseModel):
    """Edge in the belief graph."""
    edge_id: str
    from_event_id: str
    to_event_id: str
    mechanism_type: str
    direction: str
    latency: str
    confidence: float
    evidence: EvidenceScoresResponse
    explanation: str


class RankedEventResponse(BaseModel):
    """Upstream event ranked by impact."""
    event_id: str
    event_type: str
    action: str
    actors: List[str]
    certainty: float
    impact_score: float
    timestamp: Optional[datetime]
    source: str


class GraphMetadataResponse(BaseModel):
    """Graph metadata."""
    node_count: int
    edge_count: int
    depth: int
    generated_at: Optional[datetime]


class BeliefGraphResponse(BaseModel):
    """Complete belief graph response."""
    belief: BeliefNodeResponse
    nodes: List[EventNodeResponse]
    edges: List[BeliefEdgeResponse]
    ranked_upstream: List[RankedEventResponse]
    metadata: GraphMetadataResponse


class UpstreamEventsResponse(BaseModel):
    """Response for upstream events query."""
    event_id: str
    belief_question: str
    upstream_events: List[RankedEventResponse]
    count: int


class EdgeExplanationResponse(BaseModel):
    """Edge with explanation."""
    edge_id: str
    from_event: str
    to_event: str
    mechanism: str
    direction: str
    confidence: float
    explanation: str


class EdgesResponse(BaseModel):
    """Response for edges query."""
    event_id: str
    edges: List[EdgeExplanationResponse]
    count: int


class GraphListItem(BaseModel):
    """Summary of stored graph."""
    event_id: str
    question: str
    node_count: int
    edge_count: int
    saved_at: Optional[str]
    generated_at: Optional[str]


class GraphListResponse(BaseModel):
    """Response for listing graphs."""
    graphs: List[GraphListItem]
    count: int


class BuildGraphRequest(BaseModel):
    """Request to build/rebuild a graph."""
    rebuild: bool = Field(
        default=False,
        description="Force rebuild even if graph exists"
    )
    max_events: int = Field(
        default=100,
        description="Maximum events to include"
    )
    max_edges: int = Field(
        default=200,
        description="Maximum edges to include"
    )


class BuildGraphResponse(BaseModel):
    """Response after building graph."""
    event_id: str
    status: str
    node_count: int
    edge_count: int
    generated_at: datetime


class GraphStatsResponse(BaseModel):
    """Storage statistics."""
    graph_count: int
    total_nodes: int
    total_edges: int
    storage_size_mb: float


# Router

router = APIRouter(prefix="/belief-graph", tags=["Belief Graph"])


@router.get("/{event_id}", response_model=BeliefGraphResponse)
async def get_belief_graph(event_id: str):
    """
    Get complete belief update graph for an event.
    
    Returns the full DAG with nodes, edges, and ranked upstream events.
    """
    logger.info(f"Getting belief graph for event: {event_id}")
    
    storage = get_storage()
    graph = storage.load(event_id)
    
    if graph is None:
        logger.warning(f"Belief graph not found: {event_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Belief graph not found for {event_id}. Build it first using POST."
        )
    
    # Convert to response models
    belief = BeliefNodeResponse(
        belief_id=graph.belief_node.belief_id,
        question=graph.belief_node.question,
        resolution_time=graph.belief_node.resolution_time,
        current_price=graph.belief_node.current_price,
        liquidity=graph.belief_node.liquidity,
        event_id=graph.belief_node.event_id,
        polymarket_slug=graph.belief_node.polymarket_slug
    )
    
    nodes = [
        EventNodeResponse(
            event_id=n.event_id,
            event_type=n.event_type,
            timestamp=n.timestamp,
            actors=n.actors,
            action=n.action,
            object=n.object,
            certainty=n.certainty,
            source=n.source,
            scope=n.scope,
            source_doc_id=n.source_doc_id,
            raw_title=n.raw_title,
            url=n.url
        )
        for n in graph.event_nodes.values()
    ]
    
    edges = [
        BeliefEdgeResponse(
            edge_id=e.edge_id,
            from_event_id=e.from_event_id,
            to_event_id=e.to_event_id,
            mechanism_type=e.mechanism_type,
            direction=e.direction,
            latency=e.latency,
            confidence=e.confidence,
            evidence=EvidenceScoresResponse(
                price_response=e.evidence.price_response,
                volume_response=e.evidence.volume_response,
                narrative_overlap=e.evidence.narrative_overlap,
                historical_precedent=e.evidence.historical_precedent
            ),
            explanation=e.explanation
        )
        for e in graph.edges
    ]
    
    # Get ranked upstream events
    upstream = graph.get_upstream_events(limit=10)
    ranked = []
    for event in upstream:
        impact = sum(
            e.confidence for e in graph.edges
            if e.from_event_id == event.event_id
        )
        ranked.append(RankedEventResponse(
            event_id=event.event_id,
            event_type=event.event_type,
            action=event.action,
            actors=event.actors,
            certainty=event.certainty,
            impact_score=round(impact, 4),
            timestamp=event.timestamp,
            source=event.source
        ))
    
    ranked.sort(key=lambda x: x.impact_score, reverse=True)
    
    metadata = GraphMetadataResponse(
        node_count=len(graph.event_nodes),
        edge_count=len(graph.edges),
        depth=graph.depth,
        generated_at=graph.generated_at
    )
    
    logger.info(
        f"Returning graph for {event_id}: "
        f"{len(nodes)} nodes, {len(edges)} edges"
    )
    
    return BeliefGraphResponse(
        belief=belief,
        nodes=nodes,
        edges=edges,
        ranked_upstream=ranked,
        metadata=metadata
    )


@router.get("/{event_id}/upstream", response_model=UpstreamEventsResponse)
async def get_upstream_events(
    event_id: str,
    limit: int = Query(default=10, ge=1, le=50)
):
    """
    Get top N upstream events ranked by belief impact.
    
    Impact is calculated as sum of edge confidences.
    """
    logger.info(f"Getting upstream events for: {event_id}")
    
    storage = get_storage()
    graph = storage.load(event_id)
    
    if graph is None:
        raise HTTPException(
            status_code=404,
            detail=f"Belief graph not found for {event_id}"
        )
    
    upstream = graph.get_upstream_events(limit=limit)
    
    ranked = []
    for event in upstream:
        impact = sum(
            e.confidence for e in graph.edges
            if e.from_event_id == event.event_id
        )
        ranked.append(RankedEventResponse(
            event_id=event.event_id,
            event_type=event.event_type,
            action=event.action,
            actors=event.actors,
            certainty=event.certainty,
            impact_score=round(impact, 4),
            timestamp=event.timestamp,
            source=event.source
        ))
    
    ranked.sort(key=lambda x: x.impact_score, reverse=True)
    
    logger.info(f"Found {len(ranked)} upstream events for {event_id}")
    
    return UpstreamEventsResponse(
        event_id=event_id,
        belief_question=graph.belief_node.question,
        upstream_events=ranked,
        count=len(ranked)
    )


@router.get("/{event_id}/edges", response_model=EdgesResponse)
async def get_graph_edges(
    event_id: str,
    limit: int = Query(default=50, ge=1, le=200)
):
    """
    Get all edges with explanations.
    
    Edges are sorted by confidence (highest first).
    """
    logger.info(f"Getting edges for graph: {event_id}")
    
    storage = get_storage()
    graph = storage.load(event_id)
    
    if graph is None:
        raise HTTPException(
            status_code=404,
            detail=f"Belief graph not found for {event_id}"
        )
    
    # Sort by confidence and convert
    sorted_edges = sorted(graph.edges, key=lambda e: e.confidence, reverse=True)
    
    edges = []
    for edge in sorted_edges[:limit]:
        from_event = graph.event_nodes.get(edge.from_event_id)
        edges.append(EdgeExplanationResponse(
            edge_id=edge.edge_id,
            from_event=from_event.action if from_event else edge.from_event_id,
            to_event=edge.to_event_id,
            mechanism=edge.mechanism_type,
            direction=edge.direction,
            confidence=edge.confidence,
            explanation=edge.explanation
        ))
    
    logger.info(f"Returning {len(edges)} edges for {event_id}")
    
    return EdgesResponse(
        event_id=event_id,
        edges=edges,
        count=len(edges)
    )


@router.post("/{event_id}/build", response_model=BuildGraphResponse)
async def build_belief_graph(
    event_id: str,
    request: BuildGraphRequest,
    background_tasks: BackgroundTasks
):
    """
    Build or rebuild a belief graph for an event.
    
    This may take several seconds depending on the amount of data.
    """
    logger.info(f"Building belief graph for: {event_id}")
    
    # Verify event exists
    registry = get_registry()
    event = registry.get_event(event_id)
    
    if event is None:
        raise HTTPException(
            status_code=404,
            detail=f"Event not found: {event_id}"
        )
    
    storage = get_storage()
    
    # Check if already exists
    if not request.rebuild and storage.exists(event_id):
        graph = storage.load(event_id)
        if graph:
            logger.info(f"Graph already exists for {event_id}, returning cached")
            return BuildGraphResponse(
                event_id=event_id,
                status="exists",
                node_count=len(graph.event_nodes),
                edge_count=len(graph.edges),
                generated_at=graph.generated_at
            )
    
    # Build graph
    try:
        builder = GraphBuilder()
        graph = builder.build(
            event_id,
            max_events=request.max_events,
            max_edges=request.max_edges
        )
        
        # Save to storage
        storage.save(graph)
        
        logger.info(
            f"Built and saved graph for {event_id}: "
            f"{len(graph.event_nodes)} nodes, {len(graph.edges)} edges"
        )
        
        return BuildGraphResponse(
            event_id=event_id,
            status="built",
            node_count=len(graph.event_nodes),
            edge_count=len(graph.edges),
            generated_at=graph.generated_at
        )
        
    except Exception as e:
        logger.error(f"Error building graph for {event_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error building graph: {str(e)}"
        )


@router.get("/", response_model=GraphListResponse)
async def list_graphs():
    """
    List all stored belief graphs.
    """
    logger.info("Listing all belief graphs")
    
    storage = get_storage()
    graph_list = storage.list_graphs()
    
    items = [
        GraphListItem(
            event_id=g["event_id"],
            question=g["question"],
            node_count=g["node_count"],
            edge_count=g["edge_count"],
            saved_at=g.get("saved_at"),
            generated_at=g.get("generated_at")
        )
        for g in graph_list
    ]
    
    logger.info(f"Found {len(items)} stored graphs")
    
    return GraphListResponse(
        graphs=items,
        count=len(items)
    )


@router.get("/stats", response_model=GraphStatsResponse)
async def get_graph_stats():
    """
    Get statistics about stored graphs.
    """
    logger.info("Getting graph storage stats")
    
    storage = get_storage()
    stats = storage.get_stats()
    
    return GraphStatsResponse(
        graph_count=stats["graph_count"],
        total_nodes=stats["total_nodes"],
        total_edges=stats["total_edges"],
        storage_size_mb=stats["storage_size_mb"]
    )


@router.delete("/{event_id}")
async def delete_graph(event_id: str):
    """
    Delete a stored belief graph.
    """
    logger.info(f"Deleting belief graph: {event_id}")
    
    storage = get_storage()
    deleted = storage.delete(event_id)
    
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Graph not found: {event_id}"
        )
    
    return {"status": "deleted", "event_id": event_id}
