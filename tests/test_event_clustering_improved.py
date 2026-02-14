"""
Tests for Improved Event Clustering

Tests all 4 phases:
- Phase 1: Temporal constraints
- Phase 2: HAC average-linkage (anti-chaining)
- Phase 3: Soft Entity Gate (tested in test_event_mapping_enhanced.py)
- Phase 4: Canonical event synthesis
"""

import math
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from belief_graph.event_clustering import ClusteredEvent, EventClusterer
from belief_graph.models import EventNode


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #
def _make_event(
    event_id: str,
    title: str,
    timestamp: datetime,
    actors: list | None = None,
    certainty: float = 0.7,
    source: str = "reuters.com",
) -> EventNode:
    """Factory for test EventNodes."""
    return EventNode(
        event_id=event_id,
        event_type="signal",
        timestamp=timestamp,
        actors=actors or [],
        action=title.split()[0] if title else "unknown",
        object=" ".join(title.split()[1:]) if title else "unknown",
        certainty=certainty,
        source=source,
        scope="global",
        raw_title=title,
        url=f"https://example.com/{event_id}",
    )


BASE_TIME = datetime(2025, 6, 15, 12, 0, 0)


# ------------------------------------------------------------------ #
#  Phase 1: Temporal Constraints                                      #
# ------------------------------------------------------------------ #
class TestTemporalMask:
    """Events outside the time window must NOT cluster together."""

    def test_distant_events_stay_separate(self):
        """Two identical-text events 3 days apart get separate clusters."""
        clusterer = EventClusterer(
            similarity_threshold=0.75, time_window_hours=48
        )

        events = [
            _make_event("a", "Layoffs at Google", BASE_TIME),
            _make_event("b", "Layoffs at Google", BASE_TIME + timedelta(days=3)),
        ]

        # Build a perfect-similarity matrix (sim=1.0 everywhere)
        sim_matrix = np.ones((2, 2))

        masked = clusterer._apply_temporal_mask(events, sim_matrix)

        # After masking, the off-diagonal should be 0 (too far apart)
        assert masked[0, 1] == 0.0
        assert masked[1, 0] == 0.0

    def test_close_events_keep_similarity(self):
        """Two events within the window keep their similarity."""
        clusterer = EventClusterer(
            similarity_threshold=0.75, time_window_hours=48
        )

        events = [
            _make_event("a", "Layoffs at Google", BASE_TIME),
            _make_event("b", "Layoffs at Google", BASE_TIME + timedelta(hours=12)),
        ]

        sim_matrix = np.array([[1.0, 0.95], [0.95, 1.0]])
        masked = clusterer._apply_temporal_mask(events, sim_matrix)

        assert masked[0, 1] == 0.95
        assert masked[1, 0] == 0.95

    def test_none_timestamp_is_conservative(self):
        """If one event has timestamp=None, similarity is kept (conservative)."""
        clusterer = EventClusterer(
            similarity_threshold=0.75, time_window_hours=48
        )

        events = [
            _make_event("a", "Layoffs at Google", BASE_TIME),
            _make_event("b", "Layoffs at Google", BASE_TIME),
        ]
        # Manually set one timestamp to None
        events[1].timestamp = None  # type: ignore[assignment]

        sim_matrix = np.array([[1.0, 0.9], [0.9, 1.0]])
        masked = clusterer._apply_temporal_mask(events, sim_matrix)

        # Kept intact because we can't judge
        assert masked[0, 1] == 0.9


# ------------------------------------------------------------------ #
#  Phase 2: HAC Average Linkage (anti-chaining)                       #
# ------------------------------------------------------------------ #
class TestHACClustering:
    """HAC average linkage should not chain A-B-C when A≠C."""

    def test_chaining_prevented(self):
        """A~B and B~C but A≁C → A and C should NOT be in the same cluster."""
        clusterer = EventClusterer(
            similarity_threshold=0.75, time_window_hours=48
        )

        events = [
            _make_event("a", "Gemini 1.5 Pro released", BASE_TIME),
            _make_event("b", "Gemini 1.5 Pro beats GPT-4", BASE_TIME + timedelta(hours=6)),
            _make_event("c", "GPT-4 Turbo update released", BASE_TIME + timedelta(hours=12)),
        ]

        # A-B: 0.80 (above threshold), B-C: 0.80, A-C: 0.30 (below)
        sim_matrix = np.array([
            [1.0, 0.80, 0.30],
            [0.80, 1.0, 0.80],
            [0.30, 0.80, 1.0],
        ])

        clusters = clusterer._cluster_by_similarity(events, sim_matrix)

        # Under average linkage the merge of {A,B} with {C} would need
        # avg(0.30, 0.80) = 0.55 which is < 0.75, so C stays separate.
        cluster_sets = [set(c) for c in clusters]
        assert not any(
            {0, 2}.issubset(s) for s in cluster_sets
        ), "A and C should NOT be in the same cluster"

    def test_all_similar_events_cluster(self):
        """Events that are all mutually similar should cluster together."""
        clusterer = EventClusterer(
            similarity_threshold=0.75, time_window_hours=48
        )

        events = [
            _make_event("a", "EU AI Act passes", BASE_TIME),
            _make_event("b", "EU AI Act passes", BASE_TIME + timedelta(hours=2)),
            _make_event("c", "EU AI Act passes", BASE_TIME + timedelta(hours=4)),
        ]

        sim_matrix = np.array([
            [1.0, 0.90, 0.88],
            [0.90, 1.0, 0.92],
            [0.88, 0.92, 1.0],
        ])

        clusters = clusterer._cluster_by_similarity(events, sim_matrix)
        assert len(clusters) == 1
        assert set(clusters[0]) == {0, 1, 2}

    def test_single_event(self):
        """Single event produces one cluster."""
        clusterer = EventClusterer(
            similarity_threshold=0.75, time_window_hours=48
        )
        events = [_make_event("a", "Test event", BASE_TIME)]
        sim_matrix = np.array([[1.0]])

        clusters = clusterer._cluster_by_similarity(events, sim_matrix)
        assert len(clusters) == 1
        assert clusters[0] == [0]


# ------------------------------------------------------------------ #
#  Phase 4: Canonical Event Synthesis                                 #
# ------------------------------------------------------------------ #
class TestCanonicalSynthesis:
    """Synthesis should produce consensus timestamp, merged actors, boosted certainty."""

    def test_consensus_timestamp(self):
        """Consensus picks the mode-day median timestamp."""
        clusterer = EventClusterer()

        events = [
            _make_event("a", "X", datetime(2025, 6, 15, 10, 0)),
            _make_event("b", "X", datetime(2025, 6, 15, 14, 0)),
            _make_event("c", "X long title with detail", datetime(2025, 6, 15, 18, 0)),
        ]

        result = clusterer._synthesize_canonical_event(events, [0, 1, 2])

        # Mode day = June 15.  Median of [10:00, 14:00, 18:00] = 14:00
        assert result.timestamp.date() == datetime(2025, 6, 15).date()
        assert result.timestamp.hour == 14

    def test_actors_merged(self):
        """All unique actors from all cluster members are merged."""
        clusterer = EventClusterer()

        events = [
            _make_event("a", "X", BASE_TIME, actors=["Google", "DeepMind"]),
            _make_event("b", "X", BASE_TIME + timedelta(hours=1), actors=["Google", "Anthropic"]),
        ]

        result = clusterer._synthesize_canonical_event(events, [0, 1])
        actor_set = set(result.actors)
        assert {"Google", "DeepMind", "Anthropic"}.issubset(actor_set)

    def test_certainty_boosted(self):
        """Certainty is boosted by log(num_sources) * 0.1."""
        clusterer = EventClusterer()

        events = [
            _make_event("a", "X", BASE_TIME, certainty=0.6),
            _make_event("b", "X", BASE_TIME + timedelta(hours=1), certainty=0.7),
            _make_event("c", "X", BASE_TIME + timedelta(hours=2), certainty=0.65),
        ]

        result = clusterer._synthesize_canonical_event(events, [0, 1, 2])

        avg_cert = (0.6 + 0.7 + 0.65) / 3
        expected = min(0.95, avg_cert + math.log(3) * 0.1)
        assert abs(result.certainty - round(expected, 4)) < 0.01

    def test_certainty_capped_at_095(self):
        """Certainty never exceeds 0.95 even with many sources."""
        clusterer = EventClusterer()

        events = [
            _make_event(f"e{i}", "X", BASE_TIME + timedelta(hours=i), certainty=0.9)
            for i in range(20)
        ]

        result = clusterer._synthesize_canonical_event(
            events, list(range(20))
        )
        assert result.certainty <= 0.95

    def test_longest_title_selected(self):
        """The longest raw_title in the cluster is used."""
        clusterer = EventClusterer()

        events = [
            _make_event("a", "Short", BASE_TIME),
            _make_event("b", "A much longer and more descriptive title about the event", BASE_TIME),
        ]

        result = clusterer._synthesize_canonical_event(events, [0, 1])
        assert "longer" in result.raw_title

    def test_singleton_returns_original(self):
        """Cluster of 1 returns the original event unchanged."""
        clusterer = EventClusterer()
        events = [_make_event("a", "Only event", BASE_TIME)]
        result = clusterer._synthesize_canonical_event(events, [0])
        assert result.event_id == "a"
        assert result.certainty == 0.7


# ------------------------------------------------------------------ #
#  Integration: Full cluster_events pipeline (mocked model)           #
# ------------------------------------------------------------------ #
class TestClusterEventsPipeline:
    """End-to-end test of cluster_events with mocked embeddings."""

    def _mock_embeddings(self, events):
        """Create fake embeddings where identical titles get identical vectors."""
        rng = np.random.RandomState(42)
        title_to_vec = {}
        vecs = []
        for e in events:
            title = e.raw_title or ""
            if title not in title_to_vec:
                title_to_vec[title] = rng.randn(384)
                title_to_vec[title] /= np.linalg.norm(title_to_vec[title])
            vecs.append(title_to_vec[title])
        return np.array(vecs)

    def test_temporal_split(self):
        """Same title but 3 days apart → 2 clusters."""
        clusterer = EventClusterer(
            similarity_threshold=0.75, time_window_hours=48
        )

        events = [
            _make_event("a", "Layoffs at Google", BASE_TIME),
            _make_event("b", "Layoffs at Google", BASE_TIME + timedelta(days=3)),
        ]

        with patch.object(
            clusterer, "_compute_embeddings", side_effect=lambda e: self._mock_embeddings(e)
        ):
            result = clusterer.cluster_events(events)

        assert len(result) == 2

    def test_close_identical_merge(self):
        """Same title and within window → 1 cluster."""
        clusterer = EventClusterer(
            similarity_threshold=0.75, time_window_hours=48
        )

        events = [
            _make_event("a", "Layoffs at Google", BASE_TIME),
            _make_event("b", "Layoffs at Google", BASE_TIME + timedelta(hours=6)),
        ]

        with patch.object(
            clusterer, "_compute_embeddings", side_effect=lambda e: self._mock_embeddings(e)
        ):
            result = clusterer.cluster_events(events)

        assert len(result) == 1
        assert result[0].num_sources == 2
