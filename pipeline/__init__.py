"""
Pipeline Module

Core pipeline components for the AI Market Intelligence system:
- Event Registry: Event schema and loading
- Query Generator: Template-based query generation
- Ingestion: Data collection from multiple sources
- Normalizer: Document normalization
- Entity Extractor: NER-based entity extraction
- Event Mapper: 3-stage document-event mapping
- Signal Extractor: Signal classification
- Time Extractor: Date/numeric extraction
- Delta Engine: Probability delta calculation
"""

from .event_registry import EventRegistry, Event, get_registry
from .query_generator import QueryGenerator, GeneratedQuery
from .normalizer import DocumentNormalizer, NormalizedDocument
from .entity_extractor import EntityExtractor
from .event_mapper import EventMapper, DocumentMapping
from .signal_extractor import SignalExtractor, Signal
from .time_extractor import TimeExtractor, NumericExtractor
from .delta_engine import DeltaEngine, EventAnalysis

__all__ = [
    "EventRegistry",
    "Event",
    "get_registry",
    "QueryGenerator",
    "GeneratedQuery",
    "DocumentNormalizer",
    "NormalizedDocument",
    "EntityExtractor",
    "EventMapper",
    "DocumentMapping",
    "SignalExtractor",
    "Signal",
    "TimeExtractor",
    "NumericExtractor",
    "DeltaEngine",
    "EventAnalysis"
]
