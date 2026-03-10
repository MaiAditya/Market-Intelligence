"""
DB Migration: Phase 2 — Evidence Pipeline

Creates the `documents` table in the Causal Interface PostgreSQL database
(same DB as causal_graphs / causal_nodes / causal_edges).

The `documents` table stores:
  - Ingested articles / posts from the ai-market-intelligence pipeline
  - Their BART-generated summaries
  - Their mapping to causal graph nodes (set by NodeEvidenceMatcher)

Run:
    conda run -n backend python scripts/migrate_phase2.py
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    logger.error("psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

_DEFAULT_DB_URL = "postgresql://causal:causal@localhost:5432/causal_interface"

MIGRATION_SQL = """
-- ─── Documents table ─────────────────────────────────────────────────────────
-- Stores ingested articles/posts from the ai-market-intelligence pipeline,
-- their BART summaries, and their mapping to causal graph nodes.

CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id          VARCHAR(64) NOT NULL,           -- sha256 hash from ingestor
    event_id        VARCHAR(128) NOT NULL,           -- e.g. "gemini-3pt5-release-2026"
    market_id       UUID REFERENCES markets(id) ON DELETE SET NULL,

    -- Source metadata
    source          VARCHAR(32) NOT NULL,            -- reddit | twitter | web | news
    url             TEXT NOT NULL,
    title           TEXT,
    author          VARCHAR(256),
    published_at    TIMESTAMPTZ,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Content
    raw_text        TEXT,
    summary         TEXT,                            -- BART-generated summary

    -- Node mapping (set by NodeEvidenceMatcher)
    mapped_node_id  VARCHAR(128),                    -- causal_nodes.node_id
    similarity_score FLOAT,                          -- cosine similarity 0.0-1.0

    -- Source reliability signal
    source_metadata JSONB DEFAULT '{}',

    -- Dedup
    content_hash    VARCHAR(64),

    UNIQUE(doc_id),
    UNIQUE(url)
);

-- Indexes for common access patterns
CREATE INDEX IF NOT EXISTS idx_documents_event_id
    ON documents(event_id);

CREATE INDEX IF NOT EXISTS idx_documents_event_node
    ON documents(event_id, mapped_node_id);

CREATE INDEX IF NOT EXISTS idx_documents_mapped_node
    ON documents(mapped_node_id)
    WHERE mapped_node_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_documents_similarity
    ON documents(event_id, similarity_score DESC)
    WHERE similarity_score IS NOT NULL;
"""


def run_migration(db_url: str = None) -> None:
    db_url = db_url or os.getenv("CAUSAL_DB_URL", _DEFAULT_DB_URL)
    logger.info(f"Connecting to DB: {db_url.split('@')[-1]}")

    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute(MIGRATION_SQL)
        conn.commit()
        logger.info("✓ Migration complete — documents table ready")
    except Exception as e:
        conn.rollback()
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    db_url = sys.argv[1] if len(sys.argv) > 1 else None
    run_migration(db_url)
