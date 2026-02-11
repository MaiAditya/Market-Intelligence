"""
Tests for Phase 2: Chain of Thought Reasoning

Tests template-based causal explanations, multi-hop graph traversal,
and counterfactual analysis.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from belief_graph.models import (
    BeliefEdge,
    BeliefGraph,
    BeliefNode,
    CausalExplanation,
    CounterfactualResult,
    EvidenceScores,
    EventNode,
    ReasoningChain,
)
from belief_graph.reasoning_engine import ReasoningEngine


def _utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _make_event(event_id, event_type="policy", action="acted", actors=None, obj="target"):
    return EventNode(
        event_id=event_id,
        event_type=event_type,
        timestamp=_utc_now(),
        actors=actors or ["Test Actor"],
        action=action,
        object=obj,
        certainty=0.8,
        source="test",
        scope="global",
    )


def _make_evidence(narrative=0.6, price=0.4, volume=0.3, historical=0.5):
    return EvidenceScores(
        price_response=price,
        volume_response=volume,
        narrative_overlap=narrative,
        historical_precedent=historical,
    )


def _make_edge(edge_id, from_id, to_id, mechanism="signaling", confidence=0.7, evidence=None):
    return BeliefEdge(
        edge_id=edge_id,
        from_event_id=from_id,
        to_event_id=to_id,
        mechanism_type=mechanism,
        direction="positive",
        latency="immediate",
        confidence=confidence,
        evidence=evidence or _make_evidence(),
        explanation="test explanation",
    )


def _make_belief(belief_id="belief-1"):
    return BeliefNode(
        belief_id=belief_id,
        question="Will AI regulation pass?",
        resolution_time=_utc_now(),
        current_price=0.65,
        liquidity=10000.0,
        event_id="event-reg",
        polymarket_slug="ai-regulation",
    )


class TestCausalExplanation:
    """Tests for template-based causal explanation generation."""

    def setup_method(self):
        self.engine = ReasoningEngine()

    def test_legal_constraint_explanation(self):
        """Legal constraint mechanism produces structured explanation."""
        event = _make_event("e1", action="passed AI regulation", actors=["EU Parliament"])
        edge = _make_edge("edge-1", "e1", "belief-1", mechanism="legal_constraint")
        explanation = self.engine.generate_explanation(edge, event)
        assert isinstance(explanation, CausalExplanation)
        assert "EU Parliament" in explanation.premise
        assert explanation.mechanism_detail  # Not empty
        assert explanation.conclusion
        assert explanation.reasoning_chain is not None
        assert len(explanation.reasoning_chain) == 5

    def test_economic_impact_explanation(self):
        """Economic impact mechanism produces correct template."""
        event = _make_event("e2", action="invested $10B", actors=["Microsoft"], obj="OpenAI")
        edge = _make_edge("edge-2", "e2", "belief-1", mechanism="economic_impact")
        explanation = self.engine.generate_explanation(edge, event)
        assert "economic" in explanation.mechanism_detail.lower()
        assert "Microsoft" in explanation.premise

    def test_uncommon_mechanism_produces_explanation(self):
        """Less common mechanism type still produces a valid explanation."""
        event = _make_event("e3", action="coordinated response", actors=["Industry leaders"])
        edge = _make_edge("edge-3", "e3", "belief-1", mechanism="coordination_effect")
        explanation = self.engine.generate_explanation(edge, event)
        assert isinstance(explanation, CausalExplanation)
        assert explanation.premise  # Not empty
        assert "Industry leaders" in explanation.premise

    def test_confidence_strength_words(self):
        """Confidence levels map to correct strength words."""
        event = _make_event("e4")
        # High confidence
        edge_high = _make_edge("edge-h", "e4", "b1", confidence=0.9)
        explanation = self.engine.generate_explanation(edge_high, event)
        assert "strongly" in explanation.conclusion
        # Low confidence
        edge_low = _make_edge("edge-l", "e4", "b1", confidence=0.15)
        explanation_low = self.engine.generate_explanation(edge_low, event)
        assert "marginally" in explanation_low.conclusion or "negligibly" in explanation_low.conclusion

    def test_batch_explanations(self):
        """Batch explanation generation works for all edges."""
        events = {
            "e1": _make_event("e1", action="passed law"),
            "e2": _make_event("e2", action="invested"),
        }
        belief = _make_belief()
        edges = [
            _make_edge("edge-1", "e1", "belief-1", mechanism="legal_constraint"),
            _make_edge("edge-2", "e2", "belief-1", mechanism="economic_impact"),
        ]
        graph = BeliefGraph(
            belief_node=belief,
            event_nodes=events,
            edges=edges,
        )
        explanations = self.engine.generate_explanations_batch(graph)
        assert len(explanations) == 2
        assert "edge-1" in explanations
        assert "edge-2" in explanations


class TestMultiHopReasoning:
    """Tests for multi-hop graph traversal."""

    def setup_method(self):
        self.engine = ReasoningEngine()

    def test_find_multi_hop_chain(self):
        """Graph with A→B→Belief finds a 2-hop chain."""
        events = {
            "e1": _make_event("e1"),
            "e2": _make_event("e2"),
        }
        belief = _make_belief()
        edges = [
            _make_edge("edge-1", "e1", "e2", confidence=0.8),
            _make_edge("edge-2", "e2", "belief-1", confidence=0.7),
        ]
        graph = BeliefGraph(
            belief_node=belief,
            event_nodes=events,
            edges=edges,
        )
        chains = self.engine.find_reasoning_chains(graph)
        # Should find at least one 2-hop chain: e1→e2→belief-1
        assert len(chains) >= 1
        chain = chains[0]
        assert chain.hop_count == 2
        assert chain.source_event_id == "e1"
        assert chain.target_event_id == "belief-1"

    def test_confidence_decay(self):
        """Combined confidence decays with each hop."""
        events = {"e1": _make_event("e1"), "e2": _make_event("e2")}
        belief = _make_belief()
        edges = [
            _make_edge("edge-1", "e1", "e2", confidence=0.8),
            _make_edge("edge-2", "e2", "belief-1", confidence=0.7),
        ]
        graph = BeliefGraph(belief_node=belief, event_nodes=events, edges=edges)
        chains = self.engine.find_reasoning_chains(graph, decay_factor=0.85)
        if chains:
            chain = chains[0]
            # Expected: 0.8 * 0.7 * 0.85^1 = 0.476
            assert chain.combined_confidence < 0.8 * 0.7

    def test_no_chains_for_single_hop(self):
        """Single-hop edges are not returned as multi-hop chains."""
        events = {"e1": _make_event("e1")}
        belief = _make_belief()
        edges = [_make_edge("edge-1", "e1", "belief-1")]
        graph = BeliefGraph(belief_node=belief, event_nodes=events, edges=edges)
        chains = self.engine.find_reasoning_chains(graph)
        # Single-hop should not be returned
        assert len(chains) == 0

    def test_max_hops_respected(self):
        """Chains longer than max_hops are not found."""
        events = {
            "e1": _make_event("e1"),
            "e2": _make_event("e2"),
            "e3": _make_event("e3"),
        }
        belief = _make_belief()
        edges = [
            _make_edge("edge-1", "e1", "e2", confidence=0.8),
            _make_edge("edge-2", "e2", "e3", confidence=0.7),
            _make_edge("edge-3", "e3", "belief-1", confidence=0.6),
        ]
        graph = BeliefGraph(belief_node=belief, event_nodes=events, edges=edges)
        # max_hops=2 should not find the 3-hop chain from e1
        chains = self.engine.find_reasoning_chains(graph, max_hops=2)
        long_chains = [c for c in chains if c.source_event_id == "e1" and c.hop_count > 2]
        assert len(long_chains) == 0


class TestCounterfactualAnalysis:
    """Tests for counterfactual edge removal analysis."""

    def setup_method(self):
        self.engine = ReasoningEngine()

    def test_removing_high_confidence_edge(self):
        """Removing a high-confidence edge changes overall confidence."""
        events = {"e1": _make_event("e1"), "e2": _make_event("e2")}
        belief = _make_belief()
        edges = [
            _make_edge("edge-high", "e1", "belief-1", confidence=0.9),
            _make_edge("edge-low", "e2", "belief-1", confidence=0.3),
        ]
        graph = BeliefGraph(belief_node=belief, event_nodes=events, edges=edges)
        result = self.engine.analyze_counterfactual(graph, "edge-high")
        assert isinstance(result, CounterfactualResult)
        # Removing high-confidence edge should decrease overall
        assert result.delta > 0  # positive = edge was helpful

    def test_missing_edge_returns_gracefully(self):
        """Analyzing a non-existent edge returns graceful result."""
        events = {"e1": _make_event("e1")}
        belief = _make_belief()
        edges = [_make_edge("edge-1", "e1", "belief-1")]
        graph = BeliefGraph(belief_node=belief, event_nodes=events, edges=edges)
        result = self.engine.analyze_counterfactual(graph, "nonexistent")
        assert result.delta == 0.0
        assert not result.is_critical

    def test_all_counterfactuals(self):
        """Analyze all counterfactuals returns sorted results."""
        events = {"e1": _make_event("e1"), "e2": _make_event("e2")}
        belief = _make_belief()
        edges = [
            _make_edge("edge-1", "e1", "belief-1", confidence=0.9),
            _make_edge("edge-2", "e2", "belief-1", confidence=0.1),
        ]
        graph = BeliefGraph(belief_node=belief, event_nodes=events, edges=edges)
        results = self.engine.analyze_all_counterfactuals(graph)
        assert len(results) == 2
        # Should be sorted by |delta| descending
        assert abs(results[0].delta) >= abs(results[1].delta)


class TestDataModels:
    """Tests for new data model serialization."""

    def test_causal_explanation_roundtrip(self):
        """CausalExplanation serializes and deserializes."""
        explanation = CausalExplanation(
            premise="Test premise",
            mechanism_detail="Test mechanism",
            evidence_summary="Test evidence",
            confidence_reasoning="Test confidence",
            conclusion="Test conclusion",
            reasoning_chain=["Step 1", "Step 2"],
        )
        d = explanation.to_dict()
        restored = CausalExplanation.from_dict(d)
        assert restored.premise == "Test premise"
        assert len(restored.reasoning_chain) == 2

    def test_reasoning_chain_roundtrip(self):
        """ReasoningChain serializes and deserializes."""
        chain = ReasoningChain(
            edge_ids=["e1", "e2"],
            combined_confidence=0.56,
            hop_count=2,
            chain_description="A → B → C",
            source_event_id="e1",
            target_event_id="e3",
        )
        d = chain.to_dict()
        restored = ReasoningChain.from_dict(d)
        assert restored.hop_count == 2
        assert len(restored.edge_ids) == 2

    def test_counterfactual_result_roundtrip(self):
        """CounterfactualResult serializes and deserializes."""
        result = CounterfactualResult(
            removed_edge_id="edge-1",
            original_confidence=0.7,
            new_confidence=0.5,
            delta=0.2,
            impact_assessment="Test impact",
            is_critical=True,
        )
        d = result.to_dict()
        restored = CounterfactualResult.from_dict(d)
        assert restored.is_critical is True
        assert abs(restored.delta - 0.2) < 0.01
