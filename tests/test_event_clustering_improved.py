"""
Tests for Improved Event Clustering

Tests all phases:
- Phase 1: Soft temporal decay (replaces hard mask)
- Phase 2: Entity overlap Jaccard boost
- Phase 3: HAC average-linkage (anti-chaining)
- Phase 4: Canonical event synthesis
- Phase 5: Market-context prefix embedding
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
    source_doc_id: str | None = None,
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
        source_doc_id=source_doc_id,
    )


BASE_TIME = datetime(2025, 6, 15, 12, 0, 0)


# ------------------------------------------------------------------ #
#  Phase 1: Soft Temporal Decay                                       #
# ------------------------------------------------------------------ #
class TestTemporalDecay:
    """Soft temporal decay replaces the hard mask."""

    def test_distant_events_heavily_decayed(self):
        """Events far beyond the window get near-zero similarity."""
        clusterer = EventClusterer(
            similarity_threshold=0.75,
            time_window_hours=48,
            temporal_decay_tau=12.0,
            entity_weight=0.0,  # disable entity boost for this test
        )

        events = [
            _make_event("a", "Layoffs at Google", BASE_TIME),
            _make_event("b", "Layoffs at Google", BASE_TIME + timedelta(days=3)),
        ]

        # Build a perfect-similarity matrix (sim=1.0 everywhere)
        sim_matrix = np.ones((2, 2))
        decayed = clusterer._apply_temporal_decay(events, sim_matrix)

        # 3 days = 72h, window = 48h, overshoot = 24h, tau = 12h
        # decay = exp(-24/12) = exp(-2) ≈ 0.135
        expected_decay = math.exp(-24 / 12)
        assert abs(decayed[0, 1] - expected_decay) < 0.01
        assert abs(decayed[1, 0] - expected_decay) < 0.01

    def test_within_window_keeps_full_similarity(self):
        """Events within the window keep their similarity unchanged."""
        clusterer = EventClusterer(
            similarity_threshold=0.75,
            time_window_hours=48,
            temporal_decay_tau=12.0,
            entity_weight=0.0,
        )

        events = [
            _make_event("a", "Layoffs at Google", BASE_TIME),
            _make_event("b", "Layoffs at Google", BASE_TIME + timedelta(hours=12)),
        ]

        sim_matrix = np.array([[1.0, 0.95], [0.95, 1.0]])
        decayed = clusterer._apply_temporal_decay(events, sim_matrix)

        assert decayed[0, 1] == 0.95
        assert decayed[1, 0] == 0.95

    def test_borderline_events_partially_decayed(self):
        """Events just outside the window get mild decay, not zero."""
        clusterer = EventClusterer(
            similarity_threshold=0.75,
            time_window_hours=48,
            temporal_decay_tau=12.0,
            entity_weight=0.0,
        )

        events = [
            _make_event("a", "Layoffs at Google", BASE_TIME),
            _make_event("b", "Layoffs at Google", BASE_TIME + timedelta(hours=49)),
        ]

        sim_matrix = np.array([[1.0, 0.95], [0.95, 1.0]])
        decayed = clusterer._apply_temporal_decay(events, sim_matrix)

        # 49h - 48h = 1h overshoot, tau=12h → decay = exp(-1/12) ≈ 0.92
        expected = 0.95 * math.exp(-1.0 / 12.0)
        assert abs(decayed[0, 1] - expected) < 0.01
        # Crucially, NOT zero as the old hard mask would give
        assert decayed[0, 1] > 0.8

    def test_none_timestamp_is_conservative(self):
        """If one event has timestamp=None, similarity is kept (conservative)."""
        clusterer = EventClusterer(
            similarity_threshold=0.75,
            time_window_hours=48,
            temporal_decay_tau=12.0,
            entity_weight=0.0,
        )

        events = [
            _make_event("a", "Layoffs at Google", BASE_TIME),
            _make_event("b", "Layoffs at Google", BASE_TIME),
        ]
        events[1].timestamp = None  # type: ignore[assignment]

        sim_matrix = np.array([[1.0, 0.9], [0.9, 1.0]])
        decayed = clusterer._apply_temporal_decay(events, sim_matrix)

        assert decayed[0, 1] == 0.9

    def test_96h_window_covers_plus_minus_2_days(self):
        """With 96h window, events 90h apart stay within the window."""
        clusterer = EventClusterer(
            similarity_threshold=0.75,
            time_window_hours=96,
            temporal_decay_tau=12.0,
            entity_weight=0.0,
        )

        events = [
            _make_event("a", "Event", BASE_TIME),
            _make_event("b", "Event", BASE_TIME + timedelta(hours=90)),
        ]

        sim_matrix = np.ones((2, 2))
        decayed = clusterer._apply_temporal_decay(events, sim_matrix)

        # 90h is within 96h window → no decay
        assert decayed[0, 1] == 1.0


# ------------------------------------------------------------------ #
#  Phase 2: Entity Overlap Jaccard Boost                              #
# ------------------------------------------------------------------ #
class TestEntityOverlap:
    """Entity overlap should boost clustering of entity-similar events."""

    def test_entity_jaccard_matrix(self):
        """Events sharing actors get positive Jaccard scores."""
        clusterer = EventClusterer(entity_weight=0.2)

        events = [
            _make_event("a", "X", BASE_TIME, actors=["Google", "DeepMind"]),
            _make_event("b", "Y", BASE_TIME, actors=["Google", "Anthropic"]),
            _make_event("c", "Z", BASE_TIME, actors=["OpenAI", "Microsoft"]),
        ]

        jaccard = clusterer._compute_entity_overlap_matrix(events)

        # a-b share "Google", Jaccard = 1/3
        assert abs(jaccard[0, 1] - 1 / 3) < 0.01
        # a-c share nothing, Jaccard = 0
        assert jaccard[0, 2] == 0.0
        # Diagonal = 1.0
        assert jaccard[0, 0] == 1.0

    def test_entity_boost_helps_clustering(self):
        """Entity-similar events with moderate semantic sim cluster together."""
        clusterer = EventClusterer(
            similarity_threshold=0.75,
            time_window_hours=96,
            temporal_decay_tau=12.0,
            entity_weight=0.2,  # 20% entity weight
        )

        events = [
            _make_event("a", "EU passes AI regulation", BASE_TIME,
                        actors=["EU", "European Commission", "AI Act"]),
            _make_event("b", "New rules for artificial intelligence in Europe", BASE_TIME,
                        actors=["EU", "European Commission", "AI Act"]),
        ]

        # Semantic sim = 0.70 (below 0.75 threshold)
        # Entity Jaccard = 1.0 (identical actors)
        # Fused = 0.8 * 0.70 + 0.2 * 1.0 = 0.76 (above threshold!)
        sim_matrix = np.array([[1.0, 0.70], [0.70, 1.0]])

        clusters = clusterer._cluster_by_similarity(events, sim_matrix)
        assert len(clusters) == 1, "Entity boost should help merge these events"


# ------------------------------------------------------------------ #
#  Phase 3: HAC Average Linkage (anti-chaining)                       #
# ------------------------------------------------------------------ #
class TestHACClustering:
    """HAC average linkage should not chain A-B-C when A≠C."""

    def test_chaining_prevented(self):
        """A~B and B~C but A≁C → A and C should NOT be in the same cluster."""
        clusterer = EventClusterer(
            similarity_threshold=0.75,
            time_window_hours=48,
            entity_weight=0.0,  # disable entity boost for this test
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

        cluster_sets = [set(c) for c in clusters]
        assert not any(
            {0, 2}.issubset(s) for s in cluster_sets
        ), "A and C should NOT be in the same cluster"

    def test_all_similar_events_cluster(self):
        """Events that are all mutually similar should cluster together."""
        clusterer = EventClusterer(
            similarity_threshold=0.75,
            time_window_hours=48,
            entity_weight=0.0,
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
            similarity_threshold=0.75,
            time_window_hours=48,
            entity_weight=0.0,
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
#  Phase 5: Market-Context Prefix                                     #
# ------------------------------------------------------------------ #
class TestMarketContext:
    """Market question prefix should appear in event text representation."""

    def test_market_question_prepended(self):
        """_get_event_text includes market question when provided."""
        clusterer = EventClusterer()
        event = _make_event("a", "New poll shows lead", BASE_TIME)

        text_without = clusterer._get_event_text(event)
        text_with = clusterer._get_event_text(
            event, market_question="Will Trump win 2024 election?"
        )

        assert "Market:" not in text_without
        assert "Market: Will Trump win 2024 election?" in text_with

    def test_no_market_question_no_prefix(self):
        """_get_event_text works normally when no market question provided."""
        clusterer = EventClusterer()
        event = _make_event("a", "EU AI Act passes", BASE_TIME, actors=["EU"])

        text = clusterer._get_event_text(event)
        assert "EU AI Act passes" in text
        assert "Actors: EU" in text
        assert "Market:" not in text


# ------------------------------------------------------------------ #
#  Integration: Full cluster_events pipeline (mocked model)           #
# ------------------------------------------------------------------ #
class TestClusterEventsPipeline:
    """End-to-end test of cluster_events with mocked embeddings."""

    def _mock_embeddings(self, events, market_question=None):
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

    def test_temporal_split_with_hard_decay(self):
        """Same title but 5 days apart with 48h window → 2 clusters.
        
        The soft decay for 5 days (120h) beyond 48h window:
        overshoot = 72h, tau = 12h → decay = exp(-6) ≈ 0.0025 → effectively 0.
        """
        clusterer = EventClusterer(
            similarity_threshold=0.75,
            time_window_hours=48,
            entity_weight=0.0,
        )

        events = [
            _make_event("a", "Layoffs at Google", BASE_TIME),
            _make_event("b", "Layoffs at Google", BASE_TIME + timedelta(days=5)),
        ]

        with patch.object(
            clusterer, "_compute_embeddings",
            side_effect=lambda e, market_question=None: self._mock_embeddings(e),
        ):
            result = clusterer.cluster_events(events)

        assert len(result) == 2

    def test_close_identical_merge(self):
        """Same title and within window → 1 cluster."""
        clusterer = EventClusterer(
            similarity_threshold=0.75,
            time_window_hours=48,
            entity_weight=0.0,
        )

        events = [
            _make_event("a", "Layoffs at Google", BASE_TIME),
            _make_event("b", "Layoffs at Google", BASE_TIME + timedelta(hours=6)),
        ]

        with patch.object(
            clusterer, "_compute_embeddings",
            side_effect=lambda e, market_question=None: self._mock_embeddings(e),
        ):
            result = clusterer.cluster_events(events)

        assert len(result) == 1
        assert result[0].num_sources == 2

    def test_market_question_passed_through(self):
        """cluster_events accepts and passes market_question."""
        clusterer = EventClusterer(
            similarity_threshold=0.75,
            time_window_hours=96,
            entity_weight=0.0,
        )

        events = [
            _make_event("a", "New poll data", BASE_TIME),
            _make_event("b", "New poll data", BASE_TIME + timedelta(hours=6)),
        ]

        captured_args = {}

        def mock_compute(e, market_question=None):
            captured_args["market_question"] = market_question
            return self._mock_embeddings(e)

        with patch.object(
            clusterer, "_compute_embeddings", side_effect=mock_compute,
        ):
            clusterer.cluster_events(
                events, market_question="Will Trump win 2024?"
            )

        assert captured_args["market_question"] == "Will Trump win 2024?"
