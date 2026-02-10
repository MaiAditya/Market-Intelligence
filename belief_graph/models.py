"""
Data Models for Belief Update Graph

Defines the core data structures for the causality-based event mapping system:
- EventNode: Real-world events normalized to a standard schema
- BeliefNode: Polymarket events (final target)
- BeliefEdge: Potential belief influence relationships
- EvidenceScores: Empirical evidence for edges
- BeliefGraph: Complete DAG structure
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Get current UTC time as a naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# Type definitions
EventType = Literal["policy", "legal", "economic", "poll", "narrative", "market", "signal"]
MechanismType = Literal[
    "legal_constraint",
    "economic_impact",
    "signaling",
    "expectation_shift",
    "narrative_amplification",
    "liquidity_reaction",
    "coordination_effect"
]
DirectionType = Literal["positive", "negative", "ambiguous"]
LatencyType = Literal["immediate", "delayed", "uncertain"]
ScopeType = Literal["local", "national", "global"]


@dataclass
class EventNode:
    """
    Represents a real-world event normalized to a standard schema.
    
    Events that cannot be represented in this schema are discarded.
    """
    event_id: str
    event_type: EventType
    timestamp: datetime
    actors: List[str]
    action: str
    object: str
    certainty: float  # 0.0-1.0
    source: str
    scope: ScopeType
    
    # Link back to source document/signal
    source_doc_id: Optional[str] = None
    source_signal_id: Optional[str] = None
    
    # Additional metadata
    raw_title: Optional[str] = None
    url: Optional[str] = None
    
    def __post_init__(self):
        """Validate fields after initialization."""
        if not 0.0 <= self.certainty <= 1.0:
            raise ValueError(f"certainty must be between 0.0 and 1.0, got {self.certainty}")
        
        valid_event_types = {"policy", "legal", "economic", "poll", "narrative", "market", "signal"}
        if self.event_type not in valid_event_types:
            raise ValueError(f"event_type must be one of {valid_event_types}, got {self.event_type}")
        
        valid_scopes = {"local", "national", "global"}
        if self.scope not in valid_scopes:
            raise ValueError(f"scope must be one of {valid_scopes}, got {self.scope}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "actors": self.actors,
            "action": self.action,
            "object": self.object,
            "certainty": round(self.certainty, 4),
            "source": self.source,
            "scope": self.scope,
            "source_doc_id": self.source_doc_id,
            "source_signal_id": self.source_signal_id,
            "raw_title": self.raw_title,
            "url": self.url
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EventNode":
        """Create from dictionary."""
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])
        
        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            timestamp=timestamp,
            actors=data.get("actors", []),
            action=data.get("action", ""),
            object=data.get("object", ""),
            certainty=data.get("certainty", 0.5),
            source=data.get("source", ""),
            scope=data.get("scope", "global"),
            source_doc_id=data.get("source_doc_id"),
            source_signal_id=data.get("source_signal_id"),
            raw_title=data.get("raw_title"),
            url=data.get("url")
        )


@dataclass
class BeliefNode:
    """
    Represents a Polymarket event (the final target of belief updates).
    """
    belief_id: str
    question: str
    resolution_time: datetime
    current_price: float  # 0.0-1.0 probability
    liquidity: float
    
    # Link to existing event registry
    event_id: str
    polymarket_slug: str
    
    def __post_init__(self):
        """Validate fields after initialization."""
        if not 0.0 <= self.current_price <= 1.0:
            raise ValueError(f"current_price must be between 0.0 and 1.0, got {self.current_price}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "belief_id": self.belief_id,
            "question": self.question,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None,
            "current_price": round(self.current_price, 4),
            "liquidity": round(self.liquidity, 2),
            "event_id": self.event_id,
            "polymarket_slug": self.polymarket_slug
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BeliefNode":
        """Create from dictionary."""
        resolution_time = None
        if data.get("resolution_time"):
            resolution_time = datetime.fromisoformat(data["resolution_time"])
        
        return cls(
            belief_id=data["belief_id"],
            question=data.get("question", ""),
            resolution_time=resolution_time,
            current_price=data.get("current_price", 0.5),
            liquidity=data.get("liquidity", 0.0),
            event_id=data["event_id"],
            polymarket_slug=data.get("polymarket_slug", "")
        )


@dataclass
class EvidenceScores:
    """
    Empirical evidence scores for an edge.
    
    All scores normalized to [0.0, 1.0].
    """
    price_response: float      # Normalized price change after event
    volume_response: float     # Abnormal volume following event
    narrative_overlap: float   # Entity/keyword overlap between events
    historical_precedent: float  # Similarity to past event patterns
    
    def __post_init__(self):
        """Validate all scores are in range."""
        for field_name in ["price_response", "volume_response", "narrative_overlap", "historical_precedent"]:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0, got {value}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "price_response": round(self.price_response, 4),
            "volume_response": round(self.volume_response, 4),
            "narrative_overlap": round(self.narrative_overlap, 4),
            "historical_precedent": round(self.historical_precedent, 4)
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EvidenceScores":
        """Create from dictionary."""
        return cls(
            price_response=data.get("price_response", 0.0),
            volume_response=data.get("volume_response", 0.0),
            narrative_overlap=data.get("narrative_overlap", 0.0),
            historical_precedent=data.get("historical_precedent", 0.0)
        )


@dataclass
class BeliefEdge:
    """
    Represents a potential belief influence relationship between events.
    
    Edges are directed: from_event_id → to_event_id
    The system does NOT assert true causality, only models belief influence.
    """
    edge_id: str
    from_event_id: str
    to_event_id: str
    mechanism_type: MechanismType
    direction: DirectionType
    latency: LatencyType
    confidence: float  # 0.0-1.0, calculated from evidence
    evidence: EvidenceScores
    explanation: str  # Auto-generated explanation string
    
    created_at: datetime = field(default_factory=_utc_now)
    
    def __post_init__(self):
        """Validate fields."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        valid_mechanisms = {
            "legal_constraint", "economic_impact", "signaling",
            "expectation_shift", "narrative_amplification",
            "liquidity_reaction", "coordination_effect"
        }
        if self.mechanism_type not in valid_mechanisms:
            raise ValueError(f"mechanism_type must be one of {valid_mechanisms}")
        
        valid_directions = {"positive", "negative", "ambiguous"}
        if self.direction not in valid_directions:
            raise ValueError(f"direction must be one of {valid_directions}")
        
        valid_latencies = {"immediate", "delayed", "uncertain"}
        if self.latency not in valid_latencies:
            raise ValueError(f"latency must be one of {valid_latencies}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "edge_id": self.edge_id,
            "from_event_id": self.from_event_id,
            "to_event_id": self.to_event_id,
            "mechanism_type": self.mechanism_type,
            "direction": self.direction,
            "latency": self.latency,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence.to_dict(),
            "explanation": self.explanation,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BeliefEdge":
        """Create from dictionary."""
        evidence = EvidenceScores.from_dict(data.get("evidence", {}))
        
        created_at = _utc_now()
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])
        
        return cls(
            edge_id=data["edge_id"],
            from_event_id=data["from_event_id"],
            to_event_id=data["to_event_id"],
            mechanism_type=data["mechanism_type"],
            direction=data.get("direction", "ambiguous"),
            latency=data.get("latency", "uncertain"),
            confidence=data.get("confidence", 0.0),
            evidence=evidence,
            explanation=data.get("explanation", ""),
            created_at=created_at
        )


@dataclass
class BeliefGraph:
    """
    Complete belief update DAG for a target market event.
    
    Contains:
    - belief_node: The target Polymarket event
    - event_nodes: All upstream events that may influence belief
    - edges: Directed edges representing potential influence
    """
    belief_node: BeliefNode
    event_nodes: Dict[str, EventNode]
    edges: List[BeliefEdge]
    
    # Metadata
    depth: int = 2
    generated_at: datetime = field(default_factory=_utc_now)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "belief": self.belief_node.to_dict(),
            "nodes": [node.to_dict() for node in self.event_nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": {
                "node_count": len(self.event_nodes),
                "edge_count": len(self.edges),
                "depth": self.depth,
                "generated_at": self.generated_at.isoformat() if self.generated_at else None
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BeliefGraph":
        """Create from dictionary."""
        belief_node = BeliefNode.from_dict(data["belief"])
        
        event_nodes = {}
        for node_data in data.get("nodes", []):
            node = EventNode.from_dict(node_data)
            event_nodes[node.event_id] = node
        
        edges = [BeliefEdge.from_dict(e) for e in data.get("edges", [])]
        
        metadata = data.get("metadata", {})
        generated_at = _utc_now()
        if metadata.get("generated_at"):
            generated_at = datetime.fromisoformat(metadata["generated_at"])
        
        return cls(
            belief_node=belief_node,
            event_nodes=event_nodes,
            edges=edges,
            depth=metadata.get("depth", 2),
            generated_at=generated_at
        )
    
    def get_upstream_events(self, limit: int = 10) -> List[EventNode]:
        """
        Get top N upstream events ranked by total impact on belief.
        
        Impact is calculated as sum of edge confidences leading to belief.
        """
        # Calculate impact scores
        impact_scores: Dict[str, float] = {}
        
        for edge in self.edges:
            if edge.to_event_id == self.belief_node.belief_id:
                # Direct edge to belief
                impact_scores[edge.from_event_id] = impact_scores.get(edge.from_event_id, 0) + edge.confidence
            else:
                # Indirect contribution (discounted)
                impact_scores[edge.from_event_id] = impact_scores.get(edge.from_event_id, 0) + edge.confidence * 0.5
        
        # Sort by impact
        sorted_events = sorted(
            [(eid, score) for eid, score in impact_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top N
        result = []
        for event_id, _ in sorted_events[:limit]:
            if event_id in self.event_nodes:
                result.append(self.event_nodes[event_id])
        
        return result
    
    def get_edges_for_event(self, event_id: str) -> List[BeliefEdge]:
        """Get all edges originating from a specific event."""
        return [e for e in self.edges if e.from_event_id == event_id]
    
    def get_edges_to_belief(self) -> List[BeliefEdge]:
        """Get all edges directly pointing to the belief node."""
        return [e for e in self.edges if e.to_event_id == self.belief_node.belief_id]


# Mechanism type descriptions for classification
MECHANISM_DESCRIPTIONS = {
    "legal_constraint": "Legal or regulatory requirements that constrain or enable outcomes",
    "economic_impact": "Financial or economic effects that change incentives or resources",
    "signaling": "Information signals that reveal intentions, capabilities, or state",
    "expectation_shift": "Changes in market expectations about future outcomes",
    "narrative_amplification": "Media or social amplification of narratives affecting perception",
    "liquidity_reaction": "Market liquidity changes affecting price discovery",
    "coordination_effect": "Coordination among actors affecting collective outcomes"
}

# Event type descriptions for classification
EVENT_TYPE_DESCRIPTIONS = {
    "policy": "Government or organizational policy decisions and announcements",
    "legal": "Legal rulings, regulations, compliance actions, or legal proceedings",
    "economic": "Economic data, financial reports, market movements, or funding events",
    "poll": "Surveys, polls, voting results, or public opinion measurements",
    "narrative": "Media coverage, opinion pieces, social discourse, or narrative shifts",
    "market": "Prediction market prices, trading activity, or market structure changes",
    "signal": "Signals about intentions, progress, delays, or internal developments"
}
