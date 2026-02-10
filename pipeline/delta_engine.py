"""
Probability Delta Engine (Rule-Based)

Calculates probability delta ranges from aggregated signals.
Never outputs final probabilities - only delta ranges.
"""

import json
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _utc_now() -> datetime:
    """Get current UTC time as a naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.event_registry import Event, EventRegistry, get_registry
from pipeline.signal_extractor import Signal, SignalExtractor
from pipeline.normalizer import DocumentNormalizer, NormalizedDocument

logger = logging.getLogger(__name__)


@dataclass
class RankedDocument:
    """Document with ranking score for output."""
    doc_id: str
    title: str
    source_type: str
    url: str
    relevance_score: float
    source_credibility: float
    novelty_score: float
    dependency_strength: float
    rank_score: float
    relevance_reason: str
    
    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source_type": self.source_type,
            "url": self.url,
            "relevance_score": round(self.relevance_score, 4),
            "source_credibility": round(self.source_credibility, 4),
            "novelty_score": round(self.novelty_score, 4),
            "dependency_strength": round(self.dependency_strength, 4),
            "rank_score": round(self.rank_score, 4),
            "relevance_reason": self.relevance_reason
        }


@dataclass
class EventAnalysis:
    """
    Final analysis output for an event.
    
    Contains delta range, confidence, and top documents.
    """
    event_id: str
    event_title: str
    
    current_probability: float
    suggested_delta: str  # e.g., "+3% to +6%"
    delta_min: float
    delta_max: float
    confidence: float
    
    time_until_deadline_days: int
    dominant_reason: str
    dominant_signal_types: List[str]
    
    signal_summary: Dict
    top_documents: List[RankedDocument]
    
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_title": self.event_title,
            "current_probability": round(self.current_probability, 4),
            "suggested_delta": self.suggested_delta,
            "delta_min": round(self.delta_min, 4),
            "delta_max": round(self.delta_max, 4),
            "confidence": round(self.confidence, 4),
            "time_until_deadline_days": self.time_until_deadline_days,
            "dominant_reason": self.dominant_reason,
            "dominant_signal_types": self.dominant_signal_types,
            "signal_summary": self.signal_summary,
            "top_documents": [d.to_dict() for d in self.top_documents],
            "analyzed_at": self.analyzed_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EventAnalysis":
        analyzed_at = _utc_now()
        if data.get("analyzed_at"):
            analyzed_at = datetime.fromisoformat(data["analyzed_at"])
        
        top_docs = []
        for doc_data in data.get("top_documents", []):
            top_docs.append(RankedDocument(
                doc_id=doc_data["doc_id"],
                title=doc_data["title"],
                source_type=doc_data["source_type"],
                url=doc_data["url"],
                relevance_score=doc_data["relevance_score"],
                source_credibility=doc_data["source_credibility"],
                novelty_score=doc_data.get("novelty_score", 0.5),
                dependency_strength=doc_data.get("dependency_strength", 0.5),
                rank_score=doc_data["rank_score"],
                relevance_reason=doc_data.get("relevance_reason", "")
            ))
        
        return cls(
            event_id=data["event_id"],
            event_title=data.get("event_title", ""),
            current_probability=data["current_probability"],
            suggested_delta=data["suggested_delta"],
            delta_min=data.get("delta_min", 0),
            delta_max=data.get("delta_max", 0),
            confidence=data["confidence"],
            time_until_deadline_days=data.get("time_until_deadline_days", 0),
            dominant_reason=data["dominant_reason"],
            dominant_signal_types=data.get("dominant_signal_types", []),
            signal_summary=data.get("signal_summary", {}),
            top_documents=top_docs,
            analyzed_at=analyzed_at
        )


class DeltaEngine:
    """
    Rule-based probability delta calculation engine.
    
    Computes delta = Σ (signal_magnitude × confidence × source_weight × time_weight)
    
    Outputs delta ranges, never final probabilities.
    """
    
    # Source origin weights
    ORIGIN_WEIGHTS = {
        "official": 1.0,
        "journalist": 0.7,
        "public": 0.4
    }
    
    # Signal type weights (how much each type affects probability)
    SIGNAL_TYPE_WEIGHTS = {
        "official_confirmation": 1.0,
        "training_progress": 0.8,
        "delay": -0.7,
        "rumor": 0.3,
        "narrative_shift": 0.5
    }
    
    def __init__(
        self,
        registry: Optional[EventRegistry] = None,
        signal_extractor: Optional[SignalExtractor] = None,
        normalizer: Optional[DocumentNormalizer] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize delta engine.
        
        Args:
            registry: Event registry
            signal_extractor: Signal extractor
            normalizer: Document normalizer
            output_dir: Output directory
        """
        logger.info("Initializing DeltaEngine...")
        self.registry = registry or get_registry()
        
        if normalizer is None:
            normalizer = DocumentNormalizer()
        self.normalizer = normalizer
        
        if signal_extractor is None:
            signal_extractor = SignalExtractor(self.registry, self.normalizer)
        self.signal_extractor = signal_extractor
        
        if output_dir is None:
            project_root_path = Path(__file__).parent.parent
            output_dir = project_root_path / "data" / "output"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DeltaEngine initialized")
    
    def _calculate_time_weight(
        self,
        days_until_deadline: int
    ) -> float:
        """
        Calculate time-based weight.
        
        Signals closer to deadline have more impact.
        """
        if days_until_deadline <= 0:
            return 1.0
        
        # Exponential decay
        # Weight increases as deadline approaches
        weight = 1 - math.exp(-30 / max(1, days_until_deadline))
        return max(0.1, min(1.0, weight))
    
    def _aggregate_signals(
        self,
        signals: List[Signal],
        days_until_deadline: int
    ) -> Tuple[float, float, str]:
        """
        Aggregate signals into raw delta values.
        
        Returns: (weighted_delta, confidence, dominant_reason)
        """
        if not signals:
            logger.debug("No signals to aggregate")
            return 0.0, 0.0, "No signals"
        
        time_weight = self._calculate_time_weight(days_until_deadline)
        
        total_weighted_delta = 0.0
        total_weight = 0.0
        reason_scores = {}
        
        for signal in signals:
            # Get weights
            origin_weight = self.ORIGIN_WEIGHTS.get(signal.origin, 0.5)
            type_weight = self.SIGNAL_TYPE_WEIGHTS.get(signal.signal_type, 0.5)
            
            # Direction modifier
            direction_mod = {
                "positive": 1.0,
                "negative": -1.0,
                "neutral": 0.0
            }.get(signal.direction, 0.0)
            
            # Calculate signal contribution
            signal_contribution = (
                signal.magnitude *
                signal.confidence *
                origin_weight *
                abs(type_weight) *
                direction_mod *
                time_weight
            )
            
            total_weighted_delta += signal_contribution
            total_weight += signal.confidence * origin_weight
            
            # Track for dominant reason
            reason_key = f"{signal.signal_type}_{signal.direction}"
            reason_scores[reason_key] = (
                reason_scores.get(reason_key, 0) + 
                abs(signal_contribution)
            )
        
        # Average delta
        if total_weight > 0:
            avg_delta = total_weighted_delta / total_weight
        else:
            avg_delta = 0.0
        
        # Confidence based on signal consistency
        directions = [s.direction for s in signals]
        consistency = (
            max(directions.count(d) for d in set(directions)) / len(signals)
        )
        avg_signal_confidence = sum(s.confidence for s in signals) / len(signals)
        overall_confidence = (consistency + avg_signal_confidence) / 2
        
        # Dominant reason
        if reason_scores:
            dominant = max(reason_scores, key=reason_scores.get)
            signal_type, direction = dominant.rsplit("_", 1)
            dominant_reason = f"{signal_type} ({direction})"
        else:
            dominant_reason = "Mixed signals"
        
        logger.debug(
            f"Aggregated {len(signals)} signals: "
            f"delta={avg_delta:.4f}, confidence={overall_confidence:.3f}"
        )
        
        return avg_delta, overall_confidence, dominant_reason
    
    def _delta_to_range(
        self,
        delta: float,
        confidence: float
    ) -> Tuple[str, float, float]:
        """
        Convert raw delta to a percentage range.
        
        Higher confidence = tighter range.
        """
        # Scale delta to percentage
        base_delta_pct = delta * 10  # Scale factor
        
        # Range width based on confidence
        # Lower confidence = wider range
        range_width = (1 - confidence) * 5  # 0-5% additional range
        
        delta_min = base_delta_pct - range_width
        delta_max = base_delta_pct + range_width
        
        # Format as string
        if delta_min >= 0 and delta_max >= 0:
            delta_str = f"+{delta_min:.0f}% to +{delta_max:.0f}%"
        elif delta_min < 0 and delta_max < 0:
            delta_str = f"{delta_max:.0f}% to {delta_min:.0f}%"
        else:
            delta_str = f"{delta_min:.0f}% to +{delta_max:.0f}%"
        
        # Handle zero case
        if abs(delta_min) < 0.5 and abs(delta_max) < 0.5:
            delta_str = "0% (no change expected)"
        
        return delta_str, delta_min / 100, delta_max / 100
    
    def rank_documents(
        self,
        event: Event,
        signals: List[Signal],
        top_n: int = 5
    ) -> List[RankedDocument]:
        """
        Rank documents by importance for the event.
        
        rank_score = relevance_score × source_credibility × novelty × dependency_strength
        """
        logger.debug(f"Ranking {len(signals)} documents for event {event.event_id}")
        
        ranked = []
        seen_urls = set()
        
        for signal in signals:
            # Skip duplicates
            if signal.source_url in seen_urls:
                continue
            seen_urls.add(signal.source_url)
            
            # Load document
            doc = self.normalizer.load(signal.doc_id)
            if doc is None:
                continue
            
            # Calculate novelty (placeholder - would compare to historical)
            novelty = 1.0  # New documents have high novelty
            
            # Dependency strength
            dep_strength = (
                sum(signal.relevance_score for _ in signal.top_dependencies) /
                max(1, len(signal.top_dependencies))
                if signal.top_dependencies else 0.5
            )
            
            # Calculate rank score
            rank_score = (
                signal.relevance_score *
                signal.source_credibility *
                novelty *
                dep_strength
            )
            
            # Generate relevance reason
            deps = ", ".join(signal.top_dependencies[:2]) if signal.top_dependencies else "general"
            reason = f"Relevant to {deps} ({signal.signal_type})"
            
            ranked.append(RankedDocument(
                doc_id=doc.doc_id,
                title=doc.title,
                source_type=doc.source_type,
                url=doc.url,
                relevance_score=signal.relevance_score,
                source_credibility=signal.source_credibility,
                novelty_score=novelty,
                dependency_strength=dep_strength,
                rank_score=rank_score,
                relevance_reason=reason
            ))
        
        # Sort by rank score
        ranked.sort(key=lambda x: x.rank_score, reverse=True)
        
        return ranked[:top_n]
    
    def analyze_event(
        self,
        event_id: str,
        current_probability: Optional[float] = None,
        save_result: bool = True
    ) -> EventAnalysis:
        """
        Perform full delta analysis for an event.
        
        Args:
            event_id: Event to analyze
            current_probability: Current market probability
            save_result: Whether to save analysis
        
        Returns:
            EventAnalysis with delta range and top documents
        """
        event = self.registry.get_event(event_id)
        if event is None:
            raise ValueError(f"Event not found: {event_id}")
        
        logger.info(f"Analyzing event: {event_id}")
        
        # Get or fetch current probability
        if current_probability is None:
            # Try to fetch from Polymarket
            try:
                from integrations.polymarket_client import get_polymarket_client
                client = get_polymarket_client()
                current_probability = client.get_probability(event.polymarket_slug)
            except Exception as e:
                logger.warning(f"Could not fetch probability: {e}")
                current_probability = 0.5
        
        if current_probability is None:
            current_probability = 0.5
        
        # Extract signals
        signals = self.signal_extractor.extract_signals_for_event(event_id)
        
        # Get days until deadline
        days_until = event.days_until_deadline()
        
        # Aggregate signals
        raw_delta, confidence, dominant_reason = self._aggregate_signals(
            signals, days_until
        )
        
        # Convert to range
        delta_str, delta_min, delta_max = self._delta_to_range(raw_delta, confidence)
        
        # Get signal summary
        signal_summary = self.signal_extractor.get_signal_summary(signals)
        
        # Get dominant signal types
        dominant_types = sorted(
            signal_summary.get("by_type", {}).items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        dominant_signal_types = [t[0] for t in dominant_types]
        
        # Rank documents
        top_documents = self.rank_documents(event, signals)
        
        analysis = EventAnalysis(
            event_id=event_id,
            event_title=event.event_title,
            current_probability=current_probability,
            suggested_delta=delta_str,
            delta_min=delta_min,
            delta_max=delta_max,
            confidence=confidence,
            time_until_deadline_days=days_until,
            dominant_reason=dominant_reason,
            dominant_signal_types=dominant_signal_types,
            signal_summary=signal_summary,
            top_documents=top_documents
        )
        
        logger.info(
            f"Analysis complete for {event_id}: "
            f"delta={delta_str}, confidence={confidence:.2f}"
        )
        
        if save_result:
            self.save_analysis(analysis)
        
        return analysis
    
    def save_analysis(self, analysis: EventAnalysis) -> None:
        """Save analysis to disk."""
        from utils.json_utils import dump_json
        output_path = self.output_dir / f"{analysis.event_id}_analysis.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            dump_json(analysis.to_dict(), f)
        logger.info(f"Saved analysis to {output_path}")
    
    def load_analysis(self, event_id: str) -> Optional[EventAnalysis]:
        """Load saved analysis from disk."""
        output_path = self.output_dir / f"{event_id}_analysis.json"
        if not output_path.exists():
            return None
        
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return EventAnalysis.from_dict(data)
