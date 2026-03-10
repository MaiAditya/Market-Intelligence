"""
Causal Graph Database Writer

Persists LLMCausalGraph to the PostgreSQL causal_interface database.
Writes to tables: causal_graphs, causal_nodes, causal_edges.

Key guarantee: IDEMPOTENT — if a system graph already exists for this
(market_id, event_id), it returns the existing graph_id and skips
generation. Pass force=True to regenerate.

This module uses psycopg2 (sync) intentionally — the AI pipeline is
synchronous and we do NOT want to import the async FastAPI session here.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class CausalGraphDBWriter:
    """
    Writes LLMCausalGraph to PostgreSQL.

    Table mapping:
      LLMCausalGraph  → causal_graphs  (is_system=True)
      LLMCausalNode   → causal_nodes
      LLMCausalEdge   → causal_edges

    Idempotency:
      Before writing, checks if a system graph already exists for the
      market_id. Returns existing graph_id if found (no rewrite), unless
      force=True.
    """

    def __init__(self, db_url: str):
        """
        Args:
            db_url: PostgreSQL sync URL.
                    e.g. postgresql://causal:causal@localhost:5432/causal_interface
        """
        import psycopg2
        import psycopg2.extras
        self._conn = psycopg2.connect(db_url)
        self._conn.autocommit = False
        self._cursor = self._conn.cursor()
        logger.info("CausalGraphDBWriter: connected to PostgreSQL")

    @classmethod
    def from_env(cls, env_var: str = "CAUSAL_DB_URL") -> "CausalGraphDBWriter":
        """Create from environment variable."""
        db_url = os.environ.get(env_var)
        if not db_url:
            # Fallback to well-known default for local dev
            db_url = "postgresql://causal:causal@localhost:5432/causal_interface"
            logger.warning(f"{env_var} not set, using default: {db_url}")
        return cls(db_url=db_url)

    # ─────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────

    def get_market_id(self, event_id: str, event_title: str = None) -> Optional[str]:
        """
        Look up the Market UUID for an event_id. If missing, auto-create a stub.

        Returns:
            UUID string or None if not found/created.
        """
        self._cursor.execute(
            "SELECT id FROM markets WHERE event_id = %s LIMIT 1",
            (event_id,),
        )
        row = self._cursor.fetchone()
        if row:
            return str(row[0])
            
        # Auto-create stub market so we can tie the graph to it
        try:
            market_id = str(uuid.uuid4())
            title = event_title or event_id.replace('-', ' ').title()
            self._cursor.execute(
                """
                INSERT INTO markets (id, event_id, title, probability, created_at)
                VALUES (%s, %s, %s, 0.5, %s)
                """,
                (market_id, event_id, title, datetime.now(timezone.utc))
            )
            self._conn.commit()
            logger.info(f"Auto-created stub market {market_id} for event {event_id}")
            return market_id
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Failed to auto-create stub market for {event_id}: {e}")
            return None

    def graph_exists(self, market_id: str) -> Optional[str]:
        """
        Check if a system causal graph already exists for this market.

        Returns:
            graph_id (UUID str) if exists, None otherwise.
        """
        self._cursor.execute(
            """
            SELECT id FROM causal_graphs
            WHERE market_id = %s AND is_system = TRUE
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (market_id,),
        )
        row = self._cursor.fetchone()
        return str(row[0]) if row else None

    def get_existing_graph_id(self, event_id: str) -> Optional[str]:
        """
        Look up the most recent system graph ID for an event, by event_id.
        Convenience wrapper: resolves market_id then graph_id in one call.

        Returns:
            graph_id (UUID str) or None if not found.
        """
        market_id = self.get_market_id(event_id)
        if not market_id:
            return None
        return self.graph_exists(market_id)

    def write(
        self,
        graph,  # LLMCausalGraph
        market_id: str,
        force: bool = False,
    ) -> Tuple[str, bool]:
        """
        Write a causal graph to the database.

        Args:
            graph: LLMCausalGraph instance.
            market_id: UUID of the associated Market row.
            force: If True, delete existing system graph and rewrite.

        Returns:
            (graph_id, was_new) — graph_id is UUID string,
            was_new is True if newly created, False if reused.
        """
        # Idempotency check
        existing_id = self.graph_exists(market_id)
        if existing_id and not force:
            logger.info(
                f"System graph already exists for market {market_id} "
                f"(graph_id={existing_id}). Use force=True to regenerate."
            )
            return existing_id, False

        # Delete old system graph (cascades to nodes + edges)
        if existing_id and force:
            self._cursor.execute(
                "DELETE FROM causal_graphs WHERE id = %s",
                (existing_id,),
            )
            logger.info(f"Deleted existing system graph {existing_id} (force=True)")

        try:
            graph_id = self._insert_graph(graph, market_id)
            self._insert_nodes(graph.nodes, graph_id)
            self._insert_edges(graph.edges, graph_id)
            self._conn.commit()
            logger.info(
                f"Wrote causal graph {graph_id} for event {graph.event_id}: "
                f"{len(graph.nodes)} nodes, {len(graph.edges)} edges"
            )
            return graph_id, True

        except Exception as e:
            self._conn.rollback()
            logger.error(f"Failed to write causal graph: {e}")
            raise

    # ─────────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────────

    def _insert_graph(self, graph, market_id: str) -> str:
        graph_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Store full graph as JSON blob for fast loading (graph_data_json)
        graph_blob = json.dumps(graph.to_dict())

        self._cursor.execute(
            """
            INSERT INTO causal_graphs
                (id, market_id, name, description, is_system,
                 visibility, graph_data_json, node_count, edge_count,
                 thesis, created_at, updated_at)
            VALUES
                (%s, %s, %s, %s, TRUE,
                 'public', %s, %s, %s,
                 %s, %s, %s)
            """,
            (
                graph_id,
                market_id,
                "LLM Causal Graph",
                f"Auto-generated by LLM ({graph.model_used}) on {now.strftime('%Y-%m-%d')}",
                graph_blob,
                len(graph.nodes),
                len(graph.edges),
                graph.llm_thesis,
                now,
                now,
            ),
        )
        return graph_id

    def _insert_nodes(self, nodes, graph_id: str) -> None:
        now = datetime.now(timezone.utc)
        for n in nodes:
            node_uuid = str(uuid.uuid4())
            # Map direction → state hint (UI uses state field)
            state = "waiting"

            # confidence_level: scale from 0-100 probability to 1-10 range
            cl = max(1, min(10, round(n.probability / 10)))

            actors_json = json.dumps(n.actors)

            self._cursor.execute(
                """
                INSERT INTO causal_nodes
                    (id, graph_id, node_id, label, event_type,
                     probability, state, certainty, confidence_level,
                     actors_json, raw_data_json, evidence_count)
                VALUES
                    (%s, %s, %s, %s, %s,
                     %s, %s, %s, %s,
                     %s, %s, 0)
                """,
                (
                    node_uuid,
                    graph_id,
                    n.node_id,
                    n.label,
                    n.event_type,
                    n.probability,
                    state,
                    n.probability / 100.0,   # certainty 0–1
                    cl,
                    actors_json,
                    json.dumps(n.to_dict()),
                ),
            )

    def _insert_edges(self, edges, graph_id: str) -> None:
        for e in edges:
            edge_uuid = str(uuid.uuid4())
            edge_id_label = f"{e.source_node_id}__{e.target_node_id}"

            # Map direction to bullish/bearish for UI
            direction_ui = (
                "bullish" if e.direction == "positive"
                else "bearish" if e.direction == "negative"
                else "neutral"
            )

            self._cursor.execute(
                """
                INSERT INTO causal_edges
                    (id, graph_id, edge_id,
                     source_node_id, target_node_id,
                     mechanism_type, direction,
                     confidence, strength, correlation_strength,
                     explanation)
                VALUES
                    (%s, %s, %s,
                     %s, %s,
                     %s, %s,
                     %s, %s, %s,
                     %s)
                """,
                (
                    edge_uuid,
                    graph_id,
                    edge_id_label[:128],
                    e.source_node_id,
                    e.target_node_id,
                    e.mechanism,
                    direction_ui,
                    e.confidence,
                    int(e.confidence * 100),
                    e.confidence,
                    e.explanation,
                ),
            )

    def close(self) -> None:
        """Close database connection."""
        try:
            self._cursor.close()
            self._conn.close()
        except Exception:
            pass
