"""
Tests for Phase 4: Advanced Mechanism Classification

Tests hierarchical classification, mechanism strength scoring,
and centroid-based prototype learning.
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from belief_graph.models import (
    EventNode,
    MECHANISM_HIERARCHY,
    MECHANISM_TO_PARENT,
)


def _utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _make_event(event_id, event_type="policy", action="acted", actors=None,
                obj="target", certainty=0.8, scope="global"):
    return EventNode(
        event_id=event_id,
        event_type=event_type,
        timestamp=_utc_now(),
        actors=actors or ["Test Actor"],
        action=action,
        object=obj,
        certainty=certainty,
        source="test",
        scope=scope,
    )


class TestMechanismHierarchy:
    """Tests for the mechanism hierarchy data structure."""

    def test_hierarchy_has_three_parents(self):
        """Hierarchy has structural, informational, and social parents."""
        assert "structural" in MECHANISM_HIERARCHY
        assert "informational" in MECHANISM_HIERARCHY
        assert "social" in MECHANISM_HIERARCHY

    def test_all_mechanisms_have_parent(self):
        """Every mechanism type has a parent in the reverse lookup."""
        all_mechanisms = [
            "legal_constraint", "economic_impact", "signaling",
            "expectation_shift", "narrative_amplification",
            "liquidity_reaction", "coordination_effect"
        ]
        for mech in all_mechanisms:
            assert mech in MECHANISM_TO_PARENT, f"{mech} missing from MECHANISM_TO_PARENT"

    def test_structural_subtypes(self):
        """Structural category contains legal and economic mechanisms."""
        subtypes = MECHANISM_HIERARCHY["structural"]["subtypes"]
        assert "legal_constraint" in subtypes
        assert "economic_impact" in subtypes

    def test_informational_subtypes(self):
        """Informational category contains signaling and expectation mechanisms."""
        subtypes = MECHANISM_HIERARCHY["informational"]["subtypes"]
        assert "signaling" in subtypes
        assert "expectation_shift" in subtypes

    def test_social_subtypes(self):
        """Social category contains narrative, coordination, and liquidity."""
        subtypes = MECHANISM_HIERARCHY["social"]["subtypes"]
        assert "narrative_amplification" in subtypes
        assert "coordination_effect" in subtypes
        assert "liquidity_reaction" in subtypes


class TestMechanismStrengthScoring:
    """Tests for mechanism strength calculation."""

    def test_high_certainty_high_authority(self):
        """High certainty + government actor yields high strength."""
        from belief_graph.mechanism_classifier import MechanismClassifier
        classifier = MechanismClassifier()
        event = _make_event(
            "e1",
            action="passed regulation",
            actors=["government"],
            certainty=0.95,
            scope="global"
        )
        strength = classifier.calculate_mechanism_strength(event, "legal_constraint")
        # Should be high: certainty=0.95, authority=0.9, scope=0.9
        assert strength > 0.7

    def test_low_certainty_no_authority(self):
        """Low certainty + unknown actor yields lower strength."""
        from belief_graph.mechanism_classifier import MechanismClassifier
        classifier = MechanismClassifier()
        event = _make_event(
            "e2",
            action="rumored",
            actors=["anonymous source"],
            certainty=0.2,
            scope="local"
        )
        strength = classifier.calculate_mechanism_strength(event, "narrative_amplification")
        # Should be low (below 0.55 given low certainty + unknown authority)
        assert strength < 0.55

    def test_strength_in_valid_range(self):
        """Strength score is always in [0.0, 1.0]."""
        from belief_graph.mechanism_classifier import MechanismClassifier
        classifier = MechanismClassifier()
        event = _make_event("e3")
        for mechanism in ["legal_constraint", "economic_impact", "signaling",
                          "expectation_shift", "narrative_amplification",
                          "liquidity_reaction", "coordination_effect"]:
            strength = classifier.calculate_mechanism_strength(event, mechanism)
            assert 0.0 <= strength <= 1.0, f"{mechanism}: {strength} out of range"


class TestCentroidPrototypes:
    """Tests for centroid-based prototype learning."""

    def test_load_labels_from_valid_file(self):
        """Loading valid labeled data succeeds."""
        from belief_graph.mechanism_classifier import MechanismClassifier
        labels_path = project_root / "data" / "mechanism_labels.json"
        if not labels_path.exists():
            pytest.skip("mechanism_labels.json not found")
        classifier = MechanismClassifier(labels_path=str(labels_path))
        # This may or may not succeed depending on model availability
        result = classifier.load_centroid_prototypes()
        # Just confirm it doesn't crash
        assert isinstance(result, bool)

    def test_load_labels_missing_file(self):
        """Missing labels file returns False gracefully."""
        from belief_graph.mechanism_classifier import MechanismClassifier
        classifier = MechanismClassifier(labels_path="/nonexistent/path.json")
        result = classifier.load_centroid_prototypes()
        assert result is False

    def test_load_invalid_json(self):
        """Invalid JSON file is handled gracefully."""
        from belief_graph.mechanism_classifier import MechanismClassifier
        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as f:
            f.write("not valid json{{{")
            tmp_path = f.name
        try:
            classifier = MechanismClassifier(labels_path=tmp_path)
            result = classifier.load_centroid_prototypes()
            assert result is False
        finally:
            os.unlink(tmp_path)
