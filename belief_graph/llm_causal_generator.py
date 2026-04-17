"""
LLM Causal Graph Generator

Generates a causal graph for a Polymarket belief event using an LLM.
Operates INDEPENDENTLY of document ingestion — the LLM reasons from
world knowledge about what events should causally influence the target belief.

Flow:
  1. Build prompt from Event metadata (belief question, deadline, entities)
  2. Call LLM → structured JSON {nodes, edges}
  3. Validate and return LLMCausalGraph dataclass

The graph is saved as a pipeline artifact and later ingested into causal_graphs tables by PipelineService.
"""

import hashlib
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from integrations.llm_client import LLMClient, LLMError
from pipeline.event_registry import Event
from belief_graph.models import BeliefGraph, BeliefNode, EventNode, BeliefEdge, EvidenceScores

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMCausalNode:
    """A node in the LLM-generated causal graph."""
    node_id: str                       # stable slug derived from label
    label: str                         # human-readable event title
    event_type: str                    # policy / economic / signal / narrative / legal / market
    actors: List[str]                  # key actors involved
    description: str                   # brief explanation of the event
    direction: str                     # positive / negative / ambiguous (relative to belief)
    probability: float                 # LLM's prior probability (0-100)
    is_belief: bool = False            # True only for the root belief node
    detail_data: dict = field(default_factory=dict)  # full node panel data (thesis, cases, etc.)

    def to_dict(self) -> dict:
        d = {
            "node_id": self.node_id,
            "label": self.label,
            "event_type": self.event_type,
            "actors": self.actors,
            "description": self.description,
            "direction": self.direction,
            "probability": self.probability,
            "is_belief": self.is_belief,
            "layer": self.detail_data.get("layer", 1),  # top-level for DB persistence
        }
        if self.detail_data:
            d["detail_data"] = self.detail_data
        return d


@dataclass
class LLMCausalEdge:
    """A directed causal edge in the LLM-generated graph."""
    source_node_id: str
    target_node_id: str
    mechanism: str          # signaling / economic_impact / legal_constraint / etc.
    direction: str          # positive / negative / ambiguous
    confidence: float       # 0.0 – 1.0
    explanation: str        # LLM's causal reasoning

    def to_dict(self) -> dict:
        return {
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "mechanism": self.mechanism,
            "direction": self.direction,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


@dataclass
class LLMCausalGraph:
    """Complete LLM-generated causal graph for a belief event."""
    event_id: str
    belief_question: str
    belief_node_id: str
    nodes: List[LLMCausalNode] = field(default_factory=list)
    edges: List[LLMCausalEdge] = field(default_factory=list)
    model_used: str = ""
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    llm_thesis: str = ""              # narrative summary from LLM

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "belief_question": self.belief_question,
            "belief_node_id": self.belief_node_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "model_used": self.model_used,
            "generated_at": self.generated_at.isoformat(),
            "llm_thesis": self.llm_thesis,
        }

    def to_belief_graph(self) -> BeliefGraph:
        """Convert to the legacy BeliefGraph format for downstream report generators."""
        # Try to load ingested documents for timestamp assignment, but don't fail if
        # sentence_transformers or other ML deps are unavailable (e.g., in scheduler container)
        docs = []
        best_doc_for_node = {}
        try:
            from belief_graph.node_evidence_matcher import _load_docs_from_artifacts, NodeEvidenceMatcher
            docs = _load_docs_from_artifacts(self.event_id)
            node_queries = [f"{n.label}: {n.description}" for n in self.nodes]
            doc_texts = [(d.summary or d.raw_text or "")[:1000] for d in docs]

            if docs:
                matcher = NodeEvidenceMatcher()
                matches = matcher.match(node_queries, doc_texts)
                best_sim_for_node = {}
                for doc_idx, node_idx, sim in matches:
                    if sim > best_sim_for_node.get(node_idx, 0):
                        best_sim_for_node[node_idx] = sim
                        best_doc_for_node[node_idx] = docs[doc_idx]
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Evidence matching unavailable for {self.event_id} (non-fatal): {e}"
            )

        belief_node = BeliefNode(
            belief_id=self.belief_node_id,
            question=self.belief_question,
            resolution_time=datetime.now(timezone.utc),
            current_price=0.5,
            liquidity=0.0,
            event_id=self.event_id,
            polymarket_slug="unknown"
        )

        event_nodes = {}
        for idx, n in enumerate(self.nodes):
            best_doc = best_doc_for_node.get(idx)
            
            ts = datetime.now(timezone.utc)
            source = "LLM"
            url = None
            doc_id = None
            raw_title = n.label  # use the short LLM-generated title by default

            if best_doc:
                if best_doc.published_at:
                    try:
                        ts = datetime.fromisoformat(best_doc.published_at)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                    except ValueError:
                        pass
                source = best_doc.source or "web"
                url = best_doc.url
                doc_id = best_doc.doc_id
                if best_doc.title:
                    raw_title = best_doc.title

            en = EventNode(
                event_id=n.node_id,
                event_type=n.event_type if n.event_type in {"policy", "legal", "economic", "poll", "narrative", "market", "signal"} else "narrative",
                timestamp=ts,
                actors=n.actors,
                action=n.label,
                object="",
                certainty=max(0.0, min(1.0, float(n.probability) / 100.0)),
                source=source,
                scope="global",
                raw_title=raw_title,
                url=url,
                source_doc_id=doc_id
            )
            event_nodes[en.event_id] = en

        edges = []
        for e in self.edges:
            be = BeliefEdge(
                edge_id=f"{e.source_node_id}_{e.target_node_id}",
                from_event_id=e.source_node_id,
                to_event_id=e.target_node_id,
                mechanism_type=e.mechanism if e.mechanism in {
                    "legal_constraint", "economic_impact", "signaling",
                    "expectation_shift", "narrative_amplification",
                    "liquidity_reaction", "coordination_effect"
                } else "expectation_shift",
                direction=e.direction if e.direction in {"positive", "negative", "ambiguous"} else "ambiguous",
                latency="uncertain",
                confidence=max(0.0, min(1.0, float(e.confidence))),
                evidence=EvidenceScores(0.0, 0.0, 0.0, 0.0),
                explanation=e.explanation
            )
            edges.append(be)

        return BeliefGraph(
            belief_node=belief_node,
            event_nodes=event_nodes,
            edges=edges,
            depth=3
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are an expert causal reasoning system that builds layered belief influence graphs for prediction markets.

Given a Polymarket prediction market question, you generate a LAYERED causal graph of REAL-WORLD events that meaningfully shift the probability of that outcome.


CRITICAL GRAPH STRUCTURE RULE — LAYERED HIERARCHY ONLY:
- Nodes must form a strict top-down hierarchy with 2-4 distinct layers.
- Layer 0: The belief node (the Polymarket question itself). Added automatically — do NOT include it.
- Layer 1: Direct causes — events that directly make the belief more or less likely.
  These are the ONLY nodes that may have edges pointing to "belief".
- Layer 2: Indirect causes — events that cause Layer 1 events to happen.
  These connect ONLY to Layer 1 nodes, NEVER directly to "belief".
- Layer 3 (optional): Root drivers — underlying forces that drive Layer 2 events.
  These connect ONLY to Layer 2 nodes.
- Edges may ONLY connect ADJACENT layers (n → n-1). Skip-level edges are FORBIDDEN.
  ❌ WRONG: Layer 2 node → belief
  ❌ WRONG: Layer 3 node → Layer 1 node
  ✅ CORRECT: Layer 2 node → Layer 1 node → belief
-One node of a lower hierarchy shouldn't connect to two nodes of a higher hierarchy 
    Example :
    Each Layer 2 node should be only connected to one layer 1 node and multiple layer 3 nodes.
    Each Layer 3 node should be only connected to one layer 2 node.
    Each Layer 1 node should be only connected with one Layer 0 node and multiple layer 2 nodes.
- Focus on events with HIGH causal impact (not just correlated).
- Each node is a distinct real-world event category, not a single news article.
- Edges represent causal influence: source event → affects → target.
- Confidence scores must reflect genuine uncertainty, not blind optimism.
- Return ONLY valid JSON matching the exact schema. No markdown, no extra keys.
"""

USER_PROMPT_TEMPLATE = """Generate a causal graph for this Polymarket prediction market:

BELIEF QUESTION: {belief_question}
EVENT ID: {event_id}
PRIMARY ENTITIES: {primary_entities}
SECONDARY ENTITIES: {secondary_entities}
DEADLINE: {deadline}
CONTEXT: {event_description}

Generate between 8 and 14 causal nodes organised into 2-3 layers.

Return this EXACT JSON (no markdown, no extra keys):

{{
  "thesis": "2-3 sentence summary of the key causal factors and overall graph logic.",
  "nodes": [
    {{
      "id": "short_snake_case_id",
      "label": "Short human-readable event title (max 8 words)",
      "layer": 1,
      "event_type": "policy|economic|signal|narrative|legal|market",
      "actors": ["Actor1", "Actor2"],
      "description": "A detailed explanation (~150-200 words) covering: (1) What this factor is about, (2) Why it matters for the market outcome, (3) What kind of news or developments would indicate change on this factor, (4) Key entities, organizations, or metrics involved, (5) Brief historical context. This description will be used for semantic matching against incoming news articles, so include specific terminology and keywords that relevant news would contain.",
      "direction": "positive|negative|ambiguous",
      "probability": 65
    }}
  ],
  "edges": [
    {{
      "source": "layer2_or_3_node_id",
      "target": "layer1_node_id_or_belief_for_layer1_only",
      "mechanism": "signaling|economic_impact|legal_constraint|expectation_shift|narrative_amplification|coordination_effect",
      "direction": "positive|negative|ambiguous",
      "confidence": 0.7,
      "explanation": "1 sentence explaining the causal mechanism."
    }}
  ]
}}

IMPORTANT EDGE CONSTRAINTS:
- Each node MUST declare its "layer" (integer: 1, 2, or 3).
- Layer 1 nodes: their edges target "belief" ONLY.
- Layer 2 nodes: their edges target Layer 1 node IDs ONLY — NEVER "belief".
- Layer 3 nodes: their edges target Layer 2 node IDs ONLY.
- An edge from Layer N to anything other than Layer N-1 is INVALID and will be rejected.
- Minimum: 2 Layer 1 nodes and 2 Layer 2 nodes.
- Maximum: 4 Layer 1 nodes, 6 Layer 2 nodes, 4 Layer 3 nodes.
- node ids must be snake_case slugs (e.g. "compute_investment", "safety_review_delay").
- probability is 0-100 (prior for how likely this event will occur before the deadline).
- confidence is 0.0-1.0 (how strongly this edge affects the belief if the source event occurs).
- impactOnMarket.ifResolvesYes is a positive integer (e.g. +18 = market goes up 18 pp).
- impactOnMarket.ifResolvesNo is a negative integer (e.g. -12 = market drops 12 pp).
- probabilityBreakdown items must sum to approximately the node's probability.
- Do NOT include the belief node in the nodes array — it is added automatically.
- sensitivityAnalysis: for each node, list 1-3 OTHER nodes that this node most affects.
-Make sure the nodes  on each hierarchy level are Mutually exclusive and collectively exhaustive.

EXAMPLE VALID STRUCTURE (3-layer graph):
  Layer 1: gemini_released_q2 (→ belief), budget_allocated (→ belief)
  Layer 2: safety_review_passed (→ gemini_released_q2), engineering_milestone_hit (→ gemini_released_q2), board_approved_budget (→ budget_allocated)
  Layer 3: regulatory_clearance (→ safety_review_passed), benchmark_exceeded (→ engineering_milestone_hit)
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Generator
# ─────────────────────────────────────────────────────────────────────────────

def _make_node_id(label: str) -> str:
    """Convert a label to a stable snake_case node_id."""
    import re
    slug = label.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")[:48]
    return slug


class LLMCausalGraphGenerator:
    """
    Generates a causal graph for a belief event using an LLM.

    Usage:
        gen = LLMCausalGraphGenerator.from_env()
        graph = gen.generate(event)
    """

    def __init__(self, client: LLMClient):
        self.client = client

    @classmethod
    def from_env(cls, **kwargs) -> "LLMCausalGraphGenerator":
        """Create generator using GEMINI_API_KEY from environment."""
        client = LLMClient.from_env(**kwargs)
        return cls(client=client)

    def generate(self, event: Event) -> LLMCausalGraph:
        """
        Generate a causal graph for the given event using two steps:

        Step 1 — Google Search grounding: Gather current real-world context
                  about this event (news, announcements, timelines, risks).
        Step 2 — JSON generation: Use that grounded context to build the
                  causal graph with accurate, up-to-date node/edge structure.

        Args:
            event: Event from the EventRegistry.

        Returns:
            LLMCausalGraph with nodes and edges.
        """
        logger.info(f"Generating LLM causal graph for event: {event.event_id}")

        # ── Step 1: Google Search grounding ──────────────────────────────────
        search_query = (
            f"{event.event_title} — key factors, recent news, risks, timeline "
            f"as of {event.deadline.strftime('%B %Y')}"
        )
        logger.info("Step 1: Performing search-grounded context gathering...")
        grounded_context = self.client.search_grounded_summary(search_query)

        if grounded_context:
            logger.info(f"  Got {len(grounded_context)} chars of grounded context")
            context_section = (
                "\n\nRECENT WEB CONTEXT (from Google Search — use this to improve accuracy):\n"
                "─────────────────────────────────────────────────────────────\n"
                f"{grounded_context[:4000]}\n"
                "─────────────────────────────────────────────────────────────\n"
            )
        else:
            context_section = ""
            logger.info("  No grounded context retrieved — using model knowledge only")

        # ── Step 2: JSON causal graph generation ─────────────────────────────
        logger.info("Step 2: Generating structured causal graph...")
        prompt = USER_PROMPT_TEMPLATE.format(
            belief_question=event.event_title,
            event_id=event.event_id,
            primary_entities=", ".join(event.primary_entities),
            secondary_entities=", ".join(event.secondary_entities),
            deadline=event.deadline.strftime("%Y-%m-%d"),
            event_description=event.event_description,
        ) + context_section

        try:
            raw = self.client.generate_json(
                prompt=prompt,
                system_instruction=SYSTEM_INSTRUCTION,
            )
        except LLMError as e:
            logger.error(f"LLM JSON generation failed for {event.event_id}: {e}")
            raise

        # Parse and return
        graph = self._parse_response(raw, event)

        # ── Step 3: Compute embeddings for node descriptions ──────────────
        # Pre-compute embeddings so pgvector can do fast cosine similarity
        # when mapping RSS articles to causal nodes.
        try:
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            for node in graph.nodes:
                if node.is_belief:
                    continue
                text_to_embed = f"{node.label}. {node.description}" if node.description else node.label
                embedding = embed_model.encode(text_to_embed, normalize_embeddings=True)
                node.detail_data["description_embedding"] = embedding.tolist()
            logger.info(f"Computed embeddings for {len(graph.nodes) - 1} causal nodes")
        except Exception as e:
            logger.warning(f"Could not compute node embeddings (non-fatal): {e}")

        return graph

    def _parse_response(self, raw: dict, event: Event) -> LLMCausalGraph:
        """Parse and validate the LLM response into a LLMCausalGraph."""
        belief_node_id = "belief"

        # Build the belief (root) node
        belief_node = LLMCausalNode(
            node_id=belief_node_id,
            label=event.event_title,
            event_type="belief",
            actors=event.primary_entities,
            description=event.event_description,
            direction="positive",
            probability=50.0,
            is_belief=True,
        )

        # Parse causal nodes
        nodes: List[LLMCausalNode] = [belief_node]
        seen_ids = {belief_node_id}

        for n in raw.get("nodes", []):
            raw_id = n.get("id", "").strip()
            if not raw_id:
                raw_id = _make_node_id(n.get("label", "unknown"))

            # Deduplicate
            node_id = raw_id
            if node_id in seen_ids:
                node_id = f"{raw_id}_{len(seen_ids)}"
            seen_ids.add(node_id)

            raw_detail = n.get("detail", {})
            nodes.append(LLMCausalNode(
                node_id=node_id,
                label=str(n.get("label", node_id))[:120],
                event_type=str(n.get("event_type", "signal")),
                actors=list(n.get("actors", [])),
                description=str(n.get("description", ""))[:500],
                direction=str(n.get("direction", "ambiguous")),
                probability=float(n.get("probability", 50)),
                is_belief=False,
                detail_data=raw_detail if isinstance(raw_detail, dict) else {},
            ))

        # Build node_id → layer map (belief = layer 0)
        node_layer: dict = {"belief": 0}
        for n in raw.get("nodes", []):
            raw_id = n.get("id", "").strip() or _make_node_id(n.get("label", "unknown"))
            layer = int(n.get("layer", 1))
            node_layer[raw_id] = layer

        # Parse edges — collect all valid structural edges
        # (Layer validation is done after BFS topology computation below)
        edges: List[LLMCausalEdge] = []
        all_node_ids = {n.node_id for n in nodes}

        for e in raw.get("edges", []):
            source = str(e.get("source", "")).strip()
            target = str(e.get("target", "")).strip()

            # Skip self-loops or missing endpoints
            if not source or not target or source == target:
                continue

            # Normalise "belief" literal
            if target == "belief":
                target = belief_node_id
            if source == "belief":
                source = belief_node_id

            # Only keep edges where both endpoints exist in our node set
            if source not in all_node_ids or target not in all_node_ids:
                logger.debug(f"Skipping edge {source}→{target}: unknown node(s)")
                continue

            confidence = float(e.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            edges.append(LLMCausalEdge(
                source_node_id=source,
                target_node_id=target,
                mechanism=str(e.get("mechanism", "signaling")),
                direction=str(e.get("direction", "ambiguous")),
                confidence=confidence,
                explanation=str(e.get("explanation", ""))[:500],
            ))

        # ── Topology-based layer assignment (BFS from belief root) ────────────
        # Computes layer depth from the actual edge topology, ignoring whatever
        # layer numbers the LLM assigned. This is always correct.
        #
        #   belief = layer 0
        #   nodes that have an edge → belief = layer 1
        #   nodes that have an edge → layer-1 node = layer 2
        #   etc.
        #
        # Uses reverse-BFS: start from belief, walk backwards along edges.

        # Build reverse adjacency (target → [sources])
        rev_adj: dict = {}
        for edge in edges:
            rev_adj.setdefault(edge.target_node_id, []).append(edge.source_node_id)

        topo_layer: dict = {belief_node_id: 0}
        queue = [belief_node_id]
        while queue:
            current = queue.pop(0)
            current_depth = topo_layer[current]
            for src in rev_adj.get(current, []):
                if src not in topo_layer:
                    topo_layer[src] = current_depth + 1
                    queue.append(src)

        # Nodes not reachable from belief (disconnected) default to layer 1
        for node in nodes:
            if node.node_id not in topo_layer:
                topo_layer[node.node_id] = 1
                logger.warning(f"Node {node.node_id} is disconnected — defaulting to layer 1")

        # Store topology-derived layer on each node's detail_data
        for node in nodes:
            node.detail_data["layer"] = topo_layer[node.node_id]

        logger.info(f"Topology layers: {topo_layer}")

        graph = LLMCausalGraph(
            event_id=event.event_id,
            belief_question=event.event_title,
            belief_node_id=belief_node_id,
            nodes=nodes,
            edges=edges,
            model_used=self.client.model_name,
            llm_thesis=str(raw.get("thesis", "")),
        )

        logger.info(
            f"Generated causal graph for {event.event_id}: "
            f"{len(nodes)} nodes, {len(edges)} edges"
        )
        return graph

if __name__ == "__main__":
    import argparse
    from pipeline.event_registry import get_registry
    
    parser = argparse.ArgumentParser(description="Generate LLM Causal Graph")
    parser.add_argument("--event", required=True, help="Event slug to process")
    args = parser.parse_args()

    # Load environment variables (e.g., GEMINI_API_KEY) from .env file
    import os
    from dotenv import load_dotenv
    env_path = "/home/admin_summonlm_com/vibetrading/Causal_Interface/backend/.env"
    load_dotenv(dotenv_path=env_path)

    # Configure logging for stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    registry = get_registry()
    event = registry.get_event(args.event)
    
    if not event:
        logger.error(f"Event not found: {args.event}")
        sys.exit(1)

    logger.info(f"Starting LLM Causal Graph generation for {args.event}...")
    
    try:
        from integrations.llm_client import LLMClient
        client = LLMClient.from_env()
        generator = LLMCausalGraphGenerator(client=client)

        # 1) Generate the raw graph
        graph = generator.generate(event)

        # 2) Save as pipeline artifact (read by PipelineService.ingest_from_files)
        from belief_graph.storage import get_storage
        storage = get_storage()
        storage.save(graph.to_belief_graph(), overwrite=True)

        logger.info(f"Successfully generated and saved graph for {args.event}")
    except Exception as e:
        logger.error(f"Failed to generate causal graph: {e}", exc_info=True)
        sys.exit(1)
