"""
Tests for Phase 3: Enhanced Event Mapping

Tests soft entity matching, adaptive thresholds, and temporal context.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.event_mapper import (
    DEFAULT_THRESHOLDS,
    GLOBAL_DEFAULT_THRESHOLD,
    DocumentMapping,
    EntityGate,
    SoftEntityGate,
)


def _utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class TestAdaptiveThresholds:
    """Tests for per-event-type adaptive thresholds."""

    def test_default_thresholds_exist(self):
        """Default thresholds are defined for known event types."""
        assert "regulation" in DEFAULT_THRESHOLDS
        assert "model_release" in DEFAULT_THRESHOLDS
        assert "capability" in DEFAULT_THRESHOLDS

    def test_regulation_lower_than_model_release(self):
        """Regulation threshold is lower (broader language variance)."""
        assert DEFAULT_THRESHOLDS["regulation"] < DEFAULT_THRESHOLDS["model_release"]

    def test_global_default_reasonable(self):
        """Global default is within a reasonable range."""
        assert 0.3 <= GLOBAL_DEFAULT_THRESHOLD <= 0.6


class TestEntityGate:
    """Tests for the hard entity gate."""

    def test_primary_and_secondary_match(self):
        """Gate passes when primary AND secondary entities match."""
        gate = EntityGate()
        doc = MagicMock()
        doc.doc_id = "doc1"
        doc.raw_text = "The European Union passed new AI regulation."
        doc.extracted_entities = [{"text": "European Union"}, {"text": "AI"}]

        event = MagicMock()
        event.event_id = "evt1"
        event.primary_entities = ["European Union"]
        event.secondary_entities = ["AI"]
        event.aliases = []

        passed, primary, secondary, aliases = gate.check(doc, event)
        assert passed is True
        assert len(primary) > 0
        assert len(secondary) > 0

    def test_primary_only_fails(self):
        """Gate fails when only primary entities match."""
        gate = EntityGate()
        doc = MagicMock()
        doc.doc_id = "doc1"
        doc.raw_text = "The European Union announced something."
        doc.extracted_entities = [{"text": "European Union"}]

        event = MagicMock()
        event.event_id = "evt1"
        event.primary_entities = ["European Union"]
        event.secondary_entities = ["quantum computing"]
        event.aliases = []

        passed, _, _, _ = gate.check(doc, event)
        assert passed is False


class TestDocumentMappingSerialization:
    """Tests for DocumentMapping with new fields."""

    def test_roundtrip_with_new_fields(self):
        """DocumentMapping with soft gate and temporal fields serializes."""
        mapping = DocumentMapping(
            doc_id="doc1",
            event_id="evt1",
            entity_gate_passed=True,
            soft_gate_passed=True,
            primary_entities_matched=["EU"],
            secondary_entities_matched=["AI"],
            aliases_matched=[],
            soft_entity_score=0.72,
            relevance_score=0.65,
            relevance_passed=True,
            relevance_threshold_used=0.40,
            dependency_scores={"regulation": 0.8},
            top_dependencies=["regulation"],
            temporal_factor=1.15,
            adjusted_relevance_score=0.7475,
            is_relevant=True,
        )
        d = mapping.to_dict()
        restored = DocumentMapping.from_dict(d)
        assert restored.soft_gate_passed is True
        assert abs(restored.soft_entity_score - 0.72) < 0.01
        assert abs(restored.temporal_factor - 1.15) < 0.01
        assert abs(restored.relevance_threshold_used - 0.40) < 0.01

    def test_backward_compatible_from_dict(self):
        """Old-format dicts (without new fields) load with defaults."""
        old_data = {
            "doc_id": "doc1",
            "event_id": "evt1",
            "entity_gate_passed": True,
            "primary_entities_matched": ["EU"],
            "secondary_entities_matched": [],
            "aliases_matched": [],
            "relevance_score": 0.55,
            "relevance_passed": True,
            "dependency_scores": {},
            "top_dependencies": [],
            "is_relevant": True,
        }
        mapping = DocumentMapping.from_dict(old_data)
        assert mapping.soft_gate_passed is False
        assert mapping.soft_entity_score == 0.0
        assert mapping.temporal_factor == 1.0
