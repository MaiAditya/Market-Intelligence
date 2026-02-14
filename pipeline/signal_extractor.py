"""
Signal Extractor Pipeline

Extracts structured signals from relevant documents.
Each relevant document produces exactly one signal.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def _utc_now() -> datetime:
    """Get current UTC time as a naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.event_registry import EventRegistry, get_registry
from pipeline.normalizer import DocumentNormalizer, NormalizedDocument
from pipeline.event_mapper import EventMapper, DocumentMapping
from models.signal_classifier import SignalTypeClassifier, DirectionClassifier
from integrations.source_registry import (
    get_source_credibility,
    get_source_type_credibility,
    get_author_credibility
)

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """
    Extracted signal from a document.
    
    Each relevant document produces exactly one signal.
    """
    signal_id: str
    event_id: str
    doc_id: str
    
    signal_type: str  # training_progress, delay, rumor, official_confirmation, narrative_shift
    direction: str    # positive, negative, neutral
    origin: str       # official, journalist, public
    
    magnitude: float  # 0.0-1.0
    confidence: float # 0.0-1.0
    
    # Source info
    source_url: str
    source_credibility: float
    
    # Mapping info
    relevance_score: float
    top_dependencies: List[str]
    
    extracted_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "signal_id": self.signal_id,
            "event_id": self.event_id,
            "doc_id": self.doc_id,
            "signal_type": self.signal_type,
            "direction": self.direction,
            "origin": self.origin,
            "magnitude": round(self.magnitude, 4),
            "confidence": round(self.confidence, 4),
            "source_url": self.source_url,
            "source_credibility": round(self.source_credibility, 4),
            "relevance_score": round(self.relevance_score, 4),
            "top_dependencies": self.top_dependencies,
            "extracted_at": self.extracted_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Signal":
        """Create from dictionary."""
        extracted_at = _utc_now()
        if data.get("extracted_at"):
            extracted_at = datetime.fromisoformat(data["extracted_at"])
        
        return cls(
            signal_id=data["signal_id"],
            event_id=data["event_id"],
            doc_id=data["doc_id"],
            signal_type=data["signal_type"],
            direction=data["direction"],
            origin=data["origin"],
            magnitude=data["magnitude"],
            confidence=data["confidence"],
            source_url=data.get("source_url", ""),
            source_credibility=data.get("source_credibility", 0.5),
            relevance_score=data["relevance_score"],
            top_dependencies=data.get("top_dependencies", []),
            extracted_at=extracted_at
        )


class SignalExtractor:
    """
    Extract signals from relevant documents.
    
    Each document that passes the mapping stage produces one signal.
    Uses BERT-based classifiers for signal type and direction.
    """
    
    def __init__(
        self,
        registry: Optional[EventRegistry] = None,
        normalizer: Optional[DocumentNormalizer] = None,
        mapper: Optional[EventMapper] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize signal extractor.
        
        Args:
            registry: Event registry
            normalizer: Document normalizer
            mapper: Event-document mapper
            output_dir: Directory for signal storage
        """
        logger.info("Initializing SignalExtractor...")
        self.registry = registry or get_registry()
        self.normalizer = normalizer or DocumentNormalizer()
        self.mapper = mapper or EventMapper(self.registry, self.normalizer)
        
        if output_dir is None:
            project_root = Path(__file__).parent.parent
            output_dir = project_root / "data" / "signals"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize classifiers
        self.type_classifier = SignalTypeClassifier()
        self.direction_classifier = DirectionClassifier()
        logger.info("SignalExtractor initialized")
    
    def _determine_origin(
        self,
        doc: NormalizedDocument
    ) -> str:
        """
        Determine signal origin from document metadata.
        
        Maps source_type and author_type to origin.
        """
        source_type = doc.source_type.lower()
        author_type = doc.author_type.lower()
        
        # Official sources
        if source_type == "official" or author_type == "company":
            return "official"
        
        # Journalist sources
        if source_type in ["journalist", "research"] or author_type == "journalist":
            return "journalist"
        
        # Public sources
        return "public"
    
    def _calculate_magnitude(
        self,
        relevance_score: float,
        dependency_scores: Dict[str, float],
        source_credibility: float
    ) -> float:
        """
        Calculate signal magnitude.
        
        Combines relevance, dependency strength, and credibility.
        """
        # Average dependency score (top 3)
        dep_scores = sorted(dependency_scores.values(), reverse=True)[:3]
        avg_dep = sum(dep_scores) / len(dep_scores) if dep_scores else 0.5
        
        # Weighted combination
        magnitude = (
            0.4 * relevance_score +
            0.3 * avg_dep +
            0.3 * source_credibility
        )
        
        return min(1.0, max(0.0, magnitude))
    
    def _calculate_confidence(
        self,
        doc: NormalizedDocument,
        mapping: DocumentMapping,
        source_credibility: float,
        type_confidence: float,
        direction_confidence: float
    ) -> float:
        """
        Calculate signal confidence.
        
        Based on source reliability, mapping quality, and classifier confidence.
        """
        # Factors
        entity_match_score = (
            1.0 if mapping.entity_gate_passed else 0.5
        )
        
        # Combined confidence
        confidence = (
            0.25 * source_credibility +
            0.25 * mapping.relevance_score +
            0.20 * entity_match_score +
            0.15 * type_confidence +
            0.15 * direction_confidence
        )
        
        return min(1.0, max(0.0, confidence))
    
    def extract_signal(
        self,
        doc: NormalizedDocument,
        mapping: DocumentMapping
    ) -> Signal:
        """
        Extract a signal from a relevant document.
        
        Args:
            doc: Normalized document
            mapping: Document-event mapping
        
        Returns:
            Extracted Signal
        """
        logger.debug(f"Extracting signal from document: {doc.doc_id}")
        
        # Classify signal type
        signal_type, type_confidence = self.type_classifier.get_best_type(doc.raw_text)
        
        # Classify direction
        direction, direction_confidence = self.direction_classifier.get_direction(doc.raw_text)
        
        # Determine origin
        origin = self._determine_origin(doc)
        
        # Get source credibility
        source_credibility = get_source_credibility(doc.url)
        if source_credibility == 0.5:  # Unknown domain
            source_credibility = get_source_type_credibility(doc.source_type)
        
        # Calculate magnitude and confidence
        magnitude = self._calculate_magnitude(
            mapping.relevance_score,
            mapping.dependency_scores,
            source_credibility
        )
        
        confidence = self._calculate_confidence(
            doc,
            mapping,
            source_credibility,
            type_confidence,
            direction_confidence
        )
        
        # Generate signal ID
        signal_id = f"sig_{doc.doc_id[:12]}_{mapping.event_id[:12]}"
        
        signal = Signal(
            signal_id=signal_id,
            event_id=mapping.event_id,
            doc_id=doc.doc_id,
            signal_type=signal_type,
            direction=direction,
            origin=origin,
            magnitude=round(magnitude, 4),
            confidence=round(confidence, 4),
            source_url=doc.url,
            source_credibility=round(source_credibility, 4),
            relevance_score=mapping.relevance_score,
            top_dependencies=mapping.top_dependencies
        )
        
        logger.debug(
            f"Signal extracted: type={signal_type}, "
            f"direction={direction}, magnitude={magnitude:.3f}"
        )
        
        return signal
    
    def extract_signals_for_event(
        self,
        event_id: str,
        save_signals: bool = True
    ) -> List[Signal]:
        """
        Extract signals for all relevant documents of an event.
        
        Uses batch encoding to process all documents at once instead of
        encoding each document individually (2x per doc for type+direction).
        
        Args:
            event_id: Event ID
            save_signals: Whether to save signals to disk
        
        Returns:
            List of extracted signals
        """
        event = self.registry.get_event(event_id)
        if event is None:
            raise ValueError(f"Event not found: {event_id}")
        
        # Get relevant documents with mappings
        logger.info(f"Extracting signals for event: {event_id}")
        relevant_docs = self.mapper.get_relevant_documents(event)
        
        if not relevant_docs:
            logger.info(f"No relevant documents for event {event_id}")
            return []
        
        # --- Batch encode all document texts at once ---
        logger.info(f"Batch encoding {len(relevant_docs)} documents for signal extraction...")
        all_texts = [doc.raw_text[:2048] for doc, _ in relevant_docs]
        
        # Use the shared relevance scorer to encode all at once
        relevance_scorer = self.type_classifier.relevance_scorer
        all_embeddings = relevance_scorer.encode(all_texts)
        
        # Ensure type/direction reference embeddings are pre-computed
        self.type_classifier._get_type_embeddings()
        self.direction_classifier._get_direction_embeddings()
        
        signals = []
        for i, (doc, mapping) in enumerate(relevant_docs):
            # Use pre-computed embedding if available, else fall back
            if all_embeddings is not None:
                doc_embedding = all_embeddings[i]
                signal_type, type_confidence = self.type_classifier.get_best_type_from_embedding(doc_embedding)
                direction, direction_confidence = self.direction_classifier.get_direction_from_embedding(doc_embedding)
            else:
                # Fallback to per-doc encoding
                signal_type, type_confidence = self.type_classifier.get_best_type(doc.raw_text)
                direction, direction_confidence = self.direction_classifier.get_direction(doc.raw_text)
            
            # Determine origin
            origin = self._determine_origin(doc)
            
            # Get source credibility
            source_credibility = get_source_credibility(doc.url)
            if source_credibility == 0.5:
                source_credibility = get_source_type_credibility(doc.source_type)
            
            # Calculate magnitude and confidence
            magnitude = self._calculate_magnitude(
                mapping.relevance_score,
                mapping.dependency_scores,
                source_credibility
            )
            
            confidence = self._calculate_confidence(
                doc, mapping, source_credibility,
                type_confidence, direction_confidence
            )
            
            signal_id = f"sig_{doc.doc_id[:12]}_{mapping.event_id[:12]}"
            
            signal = Signal(
                signal_id=signal_id,
                event_id=mapping.event_id,
                doc_id=doc.doc_id,
                signal_type=signal_type,
                direction=direction,
                origin=origin,
                magnitude=round(magnitude, 4),
                confidence=round(confidence, 4),
                source_url=doc.url,
                source_credibility=round(source_credibility, 4),
                relevance_score=mapping.relevance_score,
                top_dependencies=mapping.top_dependencies
            )
            
            signals.append(signal)
            
            if save_signals:
                self.save_signal(signal)
        
        logger.info(f"Extracted {len(signals)} signals for event {event_id}")
        return signals
    
    def save_signal(self, signal: Signal) -> None:
        """Save a signal to disk."""
        from utils.json_utils import dump_json
        signal_path = self.output_dir / f"{signal.signal_id}.json"
        with open(signal_path, 'w', encoding='utf-8') as f:
            dump_json(signal.to_dict(), f)
    
    def load_signals_for_event(self, event_id: str) -> List[Signal]:
        """Load all signals for an event."""
        signals = []
        
        for signal_path in self.output_dir.glob("*.json"):
            try:
                with open(signal_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get("event_id") == event_id:
                        signals.append(Signal.from_dict(data))
            except Exception as e:
                logger.warning(f"Failed to load signal {signal_path}: {e}")
        
        return signals
    
    def get_signal_summary(
        self,
        signals: List[Signal]
    ) -> Dict:
        """
        Get summary statistics for signals.
        
        Args:
            signals: List of signals
        
        Returns:
            Summary dictionary
        """
        if not signals:
            return {
                "total_signals": 0,
                "by_type": {},
                "by_direction": {},
                "by_origin": {},
                "avg_magnitude": 0,
                "avg_confidence": 0
            }
        
        summary = {
            "total_signals": len(signals),
            "by_type": {},
            "by_direction": {},
            "by_origin": {},
            "avg_magnitude": sum(s.magnitude for s in signals) / len(signals),
            "avg_confidence": sum(s.confidence for s in signals) / len(signals)
        }
        
        for signal in signals:
            # By type
            summary["by_type"][signal.signal_type] = (
                summary["by_type"].get(signal.signal_type, 0) + 1
            )
            
            # By direction
            summary["by_direction"][signal.direction] = (
                summary["by_direction"].get(signal.direction, 0) + 1
            )
            
            # By origin
            summary["by_origin"][signal.origin] = (
                summary["by_origin"].get(signal.origin, 0) + 1
            )
        
        return summary
