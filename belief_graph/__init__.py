"""
Belief Update Graph Module

Constructs directed, probabilistic belief-update graphs for prediction markets.
Models how upstream real-world events influence belief updates about market outcomes.

This module does NOT assert true causality - it models belief influence under uncertainty.
LLMs are only used for mechanism classification, not for truth or strength assertions.
All edges are evidence-scored using observable signals.

Key components:
- EventNode: Normalized real-world events
- BeliefNode: Target Polymarket events
- BeliefEdge: Directed influence relationships
- GraphBuilder: Main orchestrator for graph construction
- GraphStorage: Persistence layer

Pipeline steps:
1. Event Extraction (rule-based)
2. Candidate Generation (rule-based)
3. Mechanism Classification (LLM-bounded)
4. Evidence Scoring (empirical)
5. Confidence Calculation
6. DAG Assembly
"""

from .models import (
    EventNode,
    BeliefNode,
    BeliefEdge,
    EvidenceScores,
    BeliefGraph,
    EventType,
    MechanismType,
    DirectionType,
    LatencyType,
    ScopeType,
    MECHANISM_DESCRIPTIONS,
    EVENT_TYPE_DESCRIPTIONS,
)
from .event_extractor import EventExtractor
from .candidate_generator import generate_candidates, ALLOWED_PREDECESSORS
from .mechanism_classifier import MechanismClassifier
from .evidence_scorer import EvidenceScorer
from .confidence_calculator import (
    calculate_confidence,
    CONFIDENCE_THRESHOLD,
    ConfidenceCalculator,
)
from .graph_builder import GraphBuilder
from .storage import GraphStorage, get_storage

__all__ = [
    # Models
    "EventNode",
    "BeliefNode",
    "BeliefEdge",
    "EvidenceScores",
    "BeliefGraph",
    "EventType",
    "MechanismType",
    "DirectionType",
    "LatencyType",
    "ScopeType",
    "MECHANISM_DESCRIPTIONS",
    "EVENT_TYPE_DESCRIPTIONS",
    # Components
    "EventExtractor",
    "generate_candidates",
    "ALLOWED_PREDECESSORS",
    "MechanismClassifier",
    "EvidenceScorer",
    "calculate_confidence",
    "CONFIDENCE_THRESHOLD",
    "ConfidenceCalculator",
    "GraphBuilder",
    "GraphStorage",
    "get_storage",
]
