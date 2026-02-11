"""
Tests for Phase 1: Evidence Scoring Improvements

Tests semantic narrative overlap, Bayesian precedent database,
and integration with the EvidenceScorer.
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


class TestPrecedentDatabase:
    """Tests for belief_graph.precedent_database.PrecedentDatabase"""

    def setup_method(self):
        """Create a temp db file for each test."""
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()

    def teardown_method(self):
        os.unlink(self.tmp.name)

    def test_get_precedent_empty_db(self):
        """Empty DB returns Laplace-smoothed prior (0.5)."""
        from belief_graph.precedent_database import PrecedentDatabase
        db = PrecedentDatabase(self.tmp.name, laplace_alpha=1.0)
        score = db.get_precedent("policy")
        # (0 + 1) / (0 + 2) = 0.5
        assert abs(score - 0.5) < 0.01

    def test_record_and_get_precedent(self):
        """Recording outcomes shifts the Bayesian prior."""
        from belief_graph.precedent_database import PrecedentDatabase
        db = PrecedentDatabase(self.tmp.name)
        db.record_outcome("policy", True)
        db.record_outcome("policy", True)
        db.record_outcome("policy", False)
        # correct=2, total=3, alpha=1 → (2+1)/(3+2) = 0.6
        score = db.get_precedent("policy")
        assert abs(score - 0.6) < 0.01

    def test_mechanism_level_precedent(self):
        """Mechanism-level precedent is used when available."""
        from belief_graph.precedent_database import PrecedentDatabase
        db = PrecedentDatabase(self.tmp.name)
        db.record_outcome("policy", True, mechanism_type="legal_constraint")
        db.record_outcome("policy", True, mechanism_type="legal_constraint")
        # Mechanism has 2 observations → used
        score = db.get_precedent("policy", "legal_constraint")
        # correct=2, total=2, alpha=1 → (2+1)/(2+2) = 0.75
        assert abs(score - 0.75) < 0.01

    def test_persistence(self):
        """Database persists to disk and reloads."""
        from belief_graph.precedent_database import PrecedentDatabase
        db1 = PrecedentDatabase(self.tmp.name)
        db1.record_outcome("economic", True)
        db1.record_outcome("economic", True)
        # Reload from same file
        db2 = PrecedentDatabase(self.tmp.name)
        score = db2.get_precedent("economic")
        # correct=2, total=2, alpha=1 → 0.75
        assert abs(score - 0.75) < 0.01

    def test_reliability_stats(self):
        """get_reliability_stats returns correct structure."""
        from belief_graph.precedent_database import PrecedentDatabase
        db = PrecedentDatabase(self.tmp.name)
        db.record_outcome("legal", True)
        db.record_outcome("legal", False)
        stats = db.get_reliability_stats()
        assert "legal" in stats
        assert stats["legal"]["total"] == 2
        assert stats["legal"]["correct"] == 1

    def test_seed_from_defaults(self):
        """Seeding populates the database with initial priors."""
        from belief_graph.precedent_database import PrecedentDatabase
        db = PrecedentDatabase(self.tmp.name)
        db.seed_from_defaults()
        # After seeding, event types should have data
        stats = db.get_reliability_stats()
        assert len(stats) > 0

    def test_record_batch(self):
        """Batch recording works correctly."""
        from belief_graph.precedent_database import PrecedentDatabase
        db = PrecedentDatabase(self.tmp.name)
        outcomes = [
            {"event_type": "policy", "was_correct": True},
            {"event_type": "policy", "was_correct": False},
            {"event_type": "legal", "was_correct": True},
        ]
        db.record_batch(outcomes)
        stats = db.get_reliability_stats()
        assert stats["policy"]["total"] == 2
        assert stats["legal"]["total"] == 1
