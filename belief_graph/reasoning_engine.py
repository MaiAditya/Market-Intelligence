"""
Reasoning Engine

Provides chain-of-thought reasoning capabilities for the belief graph:
1. Template-based causal explanations (deterministic, no LLM)
2. Multi-hop graph traversal for indirect influence paths
3. Counterfactual analysis ("what if this edge didn't exist?")

Design decision: We use template-based reasoning instead of LLM calls
because it is deterministic, fast, transparent, and avoids hallucination.
Each mechanism type has a parameterized template that interpolates
evidence scores and event details.
"""

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from belief_graph.models import (
    BeliefEdge,
    BeliefGraph,
    BeliefNode,
    CausalExplanation,
    CounterfactualResult,
    EventNode,
    MECHANISM_DESCRIPTIONS,
    MECHANISM_TO_PARENT,
    ReasoningChain,
)

logger = logging.getLogger(__name__)


# --- Template-Based Causal Explanation Templates ---

MECHANISM_TEMPLATES = {
    "legal_constraint": {
        "premise": "The {actors} {action} regarding {object}.",
        "mechanism": (
            "This creates a legal {direction_word} for the target belief "
            "because regulatory and legal actions directly constrain or enable outcomes."
        ),
        "evidence": (
            "Evidence: narrative overlap={narrative_overlap:.0%}, "
            "price response={price_response:.0%}, "
            "historical precedent={historical_precedent:.0%}."
        ),
        "conclusion": (
            "Therefore, this {direction} legal constraint {strength_word} "
            "influences the likelihood of the target outcome."
        ),
    },
    "economic_impact": {
        "premise": "The {actors} {action} affecting {object}.",
        "mechanism": (
            "This has economic implications because changes in funding, valuations, "
            "or market conditions alter the incentive structure."
        ),
        "evidence": (
            "Evidence: narrative overlap={narrative_overlap:.0%}, "
            "price response={price_response:.0%}, "
            "volume response={volume_response:.0%}."
        ),
        "conclusion": (
            "Therefore, this {direction} economic impact {strength_word} "
            "shifts the expected probability of the target."
        ),
    },
    "signaling": {
        "premise": "The {actors} {action} regarding {object}.",
        "mechanism": (
            "This serves as a signal because it reveals intentions, capabilities, "
            "or progress that the market uses to update expectations."
        ),
        "evidence": (
            "Evidence: narrative overlap={narrative_overlap:.0%}, "
            "historical precedent={historical_precedent:.0%}."
        ),
        "conclusion": (
            "Therefore, this {direction} signal {strength_word} "
            "updates market beliefs about the target outcome."
        ),
    },
    "expectation_shift": {
        "premise": "The {actors} {action} regarding {object}.",
        "mechanism": (
            "This shifts expectations because new information changes "
            "the consensus forecast about future outcomes."
        ),
        "evidence": (
            "Evidence: narrative overlap={narrative_overlap:.0%}, "
            "price response={price_response:.0%}."
        ),
        "conclusion": (
            "Therefore, this {direction} expectation shift {strength_word} "
            "revises the estimated probability of the target."
        ),
    },
    "narrative_amplification": {
        "premise": "The {actors} {action} regarding {object}.",
        "mechanism": (
            "This amplifies the narrative because media and social coverage "
            "increases salience and affects public perception of the outcome."
        ),
        "evidence": (
            "Evidence: narrative overlap={narrative_overlap:.0%}, "
            "volume response={volume_response:.0%}."
        ),
        "conclusion": (
            "Therefore, this {direction} narrative amplification {strength_word} "
            "influences perception of the target outcome."
        ),
    },
    "liquidity_reaction": {
        "premise": "The {actors} {action} regarding {object}.",
        "mechanism": (
            "This triggers a liquidity reaction because changes in trading "
            "activity and market depth affect price discovery accuracy."
        ),
        "evidence": (
            "Evidence: volume response={volume_response:.0%}, "
            "price response={price_response:.0%}."
        ),
        "conclusion": (
            "Therefore, this {direction} liquidity reaction {strength_word} "
            "impacts the market's pricing of the target."
        ),
    },
    "coordination_effect": {
        "premise": "The {actors} {action} regarding {object}.",
        "mechanism": (
            "This creates a coordination effect because collective behavior "
            "of multiple actors shifts the equilibrium outcome."
        ),
        "evidence": (
            "Evidence: narrative overlap={narrative_overlap:.0%}, "
            "historical precedent={historical_precedent:.0%}."
        ),
        "conclusion": (
            "Therefore, this {direction} coordination effect {strength_word} "
            "changes the collective trajectory toward the target."
        ),
    },
}


class ReasoningEngine:
    """
    Provides chain-of-thought reasoning for belief graph edges.
    
    Three capabilities:
    1. generate_explanation: Structured causal explanation for a single edge
    2. find_reasoning_chains: Multi-hop paths through the graph
    3. analyze_counterfactual: Impact of removing an edge
    """
    
    def __init__(self):
        """Initialize the reasoning engine."""
        logger.info("ReasoningEngine: initializing")
        logger.info("ReasoningEngine: ready with template-based reasoning")
    
    # ─── 2a: Causal Explanations ───────────────────────────────────
    
    def generate_explanation(
        self,
        edge: BeliefEdge,
        source_event: EventNode,
        target_event: Optional[EventNode] = None,
        target_belief: Optional[BeliefNode] = None
    ) -> CausalExplanation:
        """
        Generate a structured causal explanation for an edge.
        
        Uses template-based reasoning parameterized by:
        - Edge mechanism type (selects template)
        - Source event details (fills template variables)
        - Evidence scores (interpolated into evidence summary)
        - Confidence level (determines strength wording)
        
        Args:
            edge: The belief edge to explain
            source_event: The source event node
            target_event: The target event node (if event-to-event edge)
            target_belief: The target belief node (if event-to-belief edge)
        
        Returns:
            Structured CausalExplanation
        """
        logger.info(
            f"ReasoningEngine: generating explanation for edge {edge.edge_id} "
            f"({edge.mechanism_type}, confidence={edge.confidence:.4f})"
        )
        
        # Select template
        template = MECHANISM_TEMPLATES.get(
            edge.mechanism_type,
            MECHANISM_TEMPLATES.get("signaling")  # Fallback
        )
        
        if template is None:
            logger.warning(
                f"  No template found for mechanism '{edge.mechanism_type}', "
                f"using generic explanation"
            )
            return self._generic_explanation(edge, source_event)
        
        # Build template variables
        actors = ", ".join(source_event.actors[:3]) if source_event.actors else "Unknown actors"
        direction_word = {
            "positive": "facilitating force",
            "negative": "constraining force",
            "ambiguous": "uncertain influence"
        }.get(edge.direction, "influence")
        
        strength_word = self._confidence_to_strength(edge.confidence)
        
        template_vars = {
            "actors": actors,
            "action": source_event.action or "acted",
            "object": source_event.object or "the target",
            "direction": edge.direction,
            "direction_word": direction_word,
            "strength_word": strength_word,
            "narrative_overlap": edge.evidence.narrative_overlap,
            "price_response": edge.evidence.price_response,
            "volume_response": edge.evidence.volume_response,
            "historical_precedent": edge.evidence.historical_precedent,
        }
        
        # Fill templates
        premise = template["premise"].format(**template_vars)
        mechanism_detail = template["mechanism"].format(**template_vars)
        evidence_summary = template["evidence"].format(**template_vars)
        conclusion = template["conclusion"].format(**template_vars)
        
        # Build confidence reasoning
        confidence_reasoning = self._build_confidence_reasoning(edge)
        
        # Build reasoning chain (step-by-step)
        reasoning_chain = [
            f"1. OBSERVED: {premise}",
            f"2. MECHANISM: {mechanism_detail}",
            f"3. EVIDENCE: {evidence_summary}",
            f"4. CONFIDENCE: {confidence_reasoning}",
            f"5. CONCLUSION: {conclusion}",
        ]
        
        explanation = CausalExplanation(
            premise=premise,
            mechanism_detail=mechanism_detail,
            evidence_summary=evidence_summary,
            confidence_reasoning=confidence_reasoning,
            conclusion=conclusion,
            reasoning_chain=reasoning_chain
        )
        
        logger.info(
            f"  Generated explanation: premise='{premise[:60]}...', "
            f"conclusion='{conclusion[:60]}...'"
        )
        
        return explanation
    
    def _confidence_to_strength(self, confidence: float) -> str:
        """Convert confidence score to a human-readable strength word."""
        if confidence >= 0.8:
            return "strongly"
        elif confidence >= 0.6:
            return "moderately"
        elif confidence >= 0.4:
            return "weakly"
        elif confidence >= 0.2:
            return "marginally"
        else:
            return "negligibly"
    
    def _build_confidence_reasoning(self, edge: BeliefEdge) -> str:
        """Build a textual explanation of why we have this confidence level."""
        ev = edge.evidence
        parts = []
        
        if ev.narrative_overlap >= 0.6:
            parts.append("strong narrative connection between events")
        elif ev.narrative_overlap >= 0.3:
            parts.append("moderate narrative connection")
        else:
            parts.append("weak narrative connection")
        
        if ev.price_response >= 0.5:
            parts.append("significant price movement observed")
        
        if ev.volume_response >= 0.5:
            parts.append("elevated trading volume detected")
        
        if ev.historical_precedent >= 0.6:
            parts.append("strong historical precedent supports this pattern")
        elif ev.historical_precedent >= 0.4:
            parts.append("moderate historical support")
        
        reasoning = "; ".join(parts) if parts else "limited evidence available"
        return f"Confidence {edge.confidence:.0%} based on: {reasoning}."
    
    def _generic_explanation(
        self,
        edge: BeliefEdge,
        source_event: EventNode
    ) -> CausalExplanation:
        """Generate a generic explanation when no specific template matches."""
        actors = ", ".join(source_event.actors[:3]) if source_event.actors else "Unknown"
        
        return CausalExplanation(
            premise=f"{actors} {source_event.action} regarding {source_event.object}.",
            mechanism_detail=f"This event may influence the target through {edge.mechanism_type} mechanisms.",
            evidence_summary=f"Confidence: {edge.confidence:.0%}",
            confidence_reasoning=self._build_confidence_reasoning(edge),
            conclusion=f"This {edge.direction} influence has {self._confidence_to_strength(edge.confidence)} impact.",
            reasoning_chain=None
        )
    
    def generate_explanations_batch(
        self,
        graph: BeliefGraph
    ) -> Dict[str, CausalExplanation]:
        """
        Generate explanations for all edges in a graph.
        
        Args:
            graph: The complete belief graph
        
        Returns:
            Dict mapping edge_id to CausalExplanation
        """
        logger.info(
            f"ReasoningEngine: generating explanations for {len(graph.edges)} edges"
        )
        
        explanations = {}
        for edge in graph.edges:
            source_event = graph.event_nodes.get(edge.from_event_id)
            if not source_event:
                logger.warning(
                    f"  Skipping edge {edge.edge_id}: source event "
                    f"'{edge.from_event_id}' not found in graph"
                )
                continue
            
            target_event = graph.event_nodes.get(edge.to_event_id)
            target_belief = (
                graph.belief_node
                if edge.to_event_id == graph.belief_node.belief_id
                else None
            )
            
            explanation = self.generate_explanation(
                edge, source_event, target_event, target_belief
            )
            explanations[edge.edge_id] = explanation
        
        logger.info(
            f"ReasoningEngine: generated {len(explanations)} explanations"
        )
        return explanations
    
    # ─── 2b: Multi-Hop Graph Traversal ─────────────────────────────
    
    def find_reasoning_chains(
        self,
        graph: BeliefGraph,
        target_id: Optional[str] = None,
        max_hops: int = 3,
        min_confidence: float = 0.1,
        decay_factor: float = 0.85
    ) -> List[ReasoningChain]:
        """
        Find multi-hop reasoning chains through the belief graph.
        
        Uses BFS to discover paths from any event to the target (belief node).
        Combined confidence = product of edge confidences × decay^(hops-1).
        
        The decay factor (default 0.85) reflects that longer causal chains
        are inherently less reliable — each intermediate step introduces
        uncertainty.
        
        Args:
            graph: The belief graph to traverse
            target_id: Target node ID (defaults to belief node)
            max_hops: Maximum path length
            min_confidence: Minimum combined confidence to include
            decay_factor: Per-hop decay multiplier
        
        Returns:
            List of ReasoningChain objects, sorted by confidence
        """
        if target_id is None:
            target_id = graph.belief_node.belief_id
        
        logger.info(
            f"ReasoningEngine: finding reasoning chains to {target_id} "
            f"(max_hops={max_hops}, decay={decay_factor})"
        )
        
        # Build adjacency list
        adjacency: Dict[str, List[BeliefEdge]] = defaultdict(list)
        for edge in graph.edges:
            adjacency[edge.from_event_id].append(edge)
        
        logger.debug(f"  Built adjacency list with {len(adjacency)} source nodes")
        
        # BFS from each event node to find paths to target
        chains: List[ReasoningChain] = []
        
        for start_id in graph.event_nodes:
            if start_id == target_id:
                continue
            
            # BFS: (current_node, path_edges, path_confidence)
            queue: deque = deque()
            queue.append((start_id, [], 1.0))
            visited_in_path: set = set()
            
            while queue:
                current, path_edges, path_conf = queue.popleft()
                
                if current == target_id and len(path_edges) > 1:
                    # Found a multi-hop path (single-hop are trivial)
                    hop_count = len(path_edges)
                    decayed_conf = path_conf * (decay_factor ** (hop_count - 1))
                    
                    if decayed_conf >= min_confidence:
                        chain_desc = self._describe_chain(
                            path_edges, graph
                        )
                        chain = ReasoningChain(
                            edge_ids=[e.edge_id for e in path_edges],
                            combined_confidence=decayed_conf,
                            hop_count=hop_count,
                            chain_description=chain_desc,
                            source_event_id=start_id,
                            target_event_id=target_id
                        )
                        chains.append(chain)
                        logger.debug(
                            f"  Found chain: {start_id} → ... → {target_id} "
                            f"({hop_count} hops, conf={decayed_conf:.4f})"
                        )
                    continue
                
                if len(path_edges) >= max_hops:
                    continue
                
                for edge in adjacency.get(current, []):
                    next_node = edge.to_event_id
                    if next_node not in visited_in_path:
                        visited_in_path.add(next_node)
                        queue.append((
                            next_node,
                            path_edges + [edge],
                            path_conf * edge.confidence
                        ))
        
        # Sort by confidence descending
        chains.sort(key=lambda c: c.combined_confidence, reverse=True)
        
        logger.info(
            f"ReasoningEngine: found {len(chains)} multi-hop chains "
            f"(min_conf={min_confidence})"
        )
        
        return chains
    
    def _describe_chain(
        self,
        edges: List[BeliefEdge],
        graph: BeliefGraph
    ) -> str:
        """Build a human-readable description of a reasoning chain."""
        parts = []
        for i, edge in enumerate(edges):
            source = graph.event_nodes.get(edge.from_event_id)
            source_name = (
                source.raw_title or source.action
                if source else edge.from_event_id
            )
            
            if i == 0:
                parts.append(f"{source_name}")
            
            parts.append(f"→[{edge.mechanism_type}]→")
            
            if i == len(edges) - 1:
                target = graph.event_nodes.get(edge.to_event_id)
                if target:
                    parts.append(target.raw_title or target.action)
                elif edge.to_event_id == graph.belief_node.belief_id:
                    parts.append(f"Belief: {graph.belief_node.question[:40]}...")
                else:
                    parts.append(edge.to_event_id)
        
        return " ".join(parts)
    
    # ─── 2c: Counterfactual Analysis ───────────────────────────────
    
    def analyze_counterfactual(
        self,
        graph: BeliefGraph,
        edge_id: str,
        confidence_threshold: float = 0.05
    ) -> CounterfactualResult:
        """
        Analyze the impact of removing a specific edge.
        
        Computes the overall belief confidence with and without the edge,
        then determines if the edge is "critical" (delta > threshold).
        
        The overall confidence is calculated as the mean of all direct
        edges to the belief node, because that's how the graph's
        aggregate belief is determined.
        
        Args:
            graph: The belief graph
            edge_id: ID of the edge to remove
            confidence_threshold: Delta above which an edge is "critical"
        
        Returns:
            CounterfactualResult with impact assessment
        """
        logger.info(
            f"ReasoningEngine: analyzing counterfactual for edge {edge_id}"
        )
        
        # Find the edge
        target_edge = None
        for edge in graph.edges:
            if edge.edge_id == edge_id:
                target_edge = edge
                break
        
        if target_edge is None:
            logger.warning(f"  Edge {edge_id} not found in graph")
            return CounterfactualResult(
                removed_edge_id=edge_id,
                original_confidence=0.0,
                new_confidence=0.0,
                delta=0.0,
                impact_assessment="Edge not found in graph.",
                is_critical=False
            )
        
        # Calculate original aggregate confidence (mean of all edge confidences)
        all_confidences = [e.confidence for e in graph.edges]
        original_confidence = (
            sum(all_confidences) / len(all_confidences)
            if all_confidences else 0.0
        )
        
        # Calculate confidence without this edge
        remaining_confidences = [
            e.confidence for e in graph.edges
            if e.edge_id != edge_id
        ]
        new_confidence = (
            sum(remaining_confidences) / len(remaining_confidences)
            if remaining_confidences else 0.0
        )
        
        delta = original_confidence - new_confidence
        is_critical = abs(delta) > confidence_threshold
        
        # Build impact assessment
        if abs(delta) < 0.01:
            impact_assessment = (
                f"Removing edge {edge_id} ({target_edge.mechanism_type}) "
                f"has negligible impact (Δ={delta:+.4f}). "
                f"This edge is redundant with other evidence."
            )
        elif delta > 0:
            impact_assessment = (
                f"Removing edge {edge_id} ({target_edge.mechanism_type}) "
                f"DECREASES overall confidence by {abs(delta):.4f} "
                f"({original_confidence:.4f} → {new_confidence:.4f}). "
                f"This edge provides {'critical' if is_critical else 'supporting'} evidence."
            )
        else:
            impact_assessment = (
                f"Removing edge {edge_id} ({target_edge.mechanism_type}) "
                f"INCREASES overall confidence by {abs(delta):.4f} "
                f"({original_confidence:.4f} → {new_confidence:.4f}). "
                f"This edge was suppressing confidence (likely contradictory evidence)."
            )
        
        result = CounterfactualResult(
            removed_edge_id=edge_id,
            original_confidence=original_confidence,
            new_confidence=new_confidence,
            delta=delta,
            impact_assessment=impact_assessment,
            is_critical=is_critical
        )
        
        logger.info(
            f"  Counterfactual result: Δ={delta:+.4f}, "
            f"critical={is_critical}, edge={target_edge.mechanism_type}"
        )
        
        return result
    
    def analyze_all_counterfactuals(
        self,
        graph: BeliefGraph,
        confidence_threshold: float = 0.05
    ) -> List[CounterfactualResult]:
        """
        Analyze counterfactuals for ALL edges in the graph.
        
        Identifies which edges are most / least important.
        
        Args:
            graph: The belief graph
            confidence_threshold: Delta above which an edge is "critical"
        
        Returns:
            List of CounterfactualResult, sorted by |delta| descending
        """
        logger.info(
            f"ReasoningEngine: analyzing counterfactuals for {len(graph.edges)} edges"
        )
        
        results = []
        for edge in graph.edges:
            result = self.analyze_counterfactual(
                graph, edge.edge_id, confidence_threshold
            )
            results.append(result)
        
        results.sort(key=lambda r: abs(r.delta), reverse=True)
        
        critical_count = sum(1 for r in results if r.is_critical)
        logger.info(
            f"ReasoningEngine: {critical_count}/{len(results)} edges are critical"
        )
        
        return results
