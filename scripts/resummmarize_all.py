"""
One-off script: re-summarize all feed_items using BART via CTranslate2 INT8.

Usage (inside scheduler container):
    USE_BART=1 python -m scripts.resummmarize_all
"""

import os
import time
import logging
import psycopg2

os.environ["USE_BART"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DB_URL = os.getenv("DATABASE_URL_SYNC", "postgresql://causal:causal@localhost:5432/causal_interface")

BATCH_SIZE = 2  # matches inter_threads=2 for parallel CT2 inference


def main():
    from pipeline.ingestion.article_summarizer import (
        ArticleSummarizer, _heuristic_summary,
    )
    import pipeline.ingestion.article_summarizer as mod

    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False

    cur = conn.cursor()
    cur.execute("SELECT id, headline, raw_text FROM feed_items WHERE raw_text IS NOT NULL ORDER BY timestamp DESC")
    rows = cur.fetchall()
    total = len(rows)

    summarizer = ArticleSummarizer()
    summarizer._get_pipe()
    logger.info(f"Found {total} articles to re-summarize (backend={mod._backend}, batch={BATCH_SIZE})")

    if total == 0:
        return

    updated = 0
    t0 = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        ids = [r[0] for r in batch]
        titles = [r[1] for r in batch]
        texts = [r[2] for r in batch]

        try:
            summaries = summarizer.summarize_batch(texts, titles)
        except Exception as e:
            logger.warning(f"Batch {i} failed: {e}")
            summaries = [_heuristic_summary(summarizer._prepare_text(t, h)) for t, h in zip(texts, titles)]

        up = conn.cursor()
        for doc_id, summary in zip(ids, summaries):
            up.execute("UPDATE feed_items SET summary = %s WHERE id = %s", (summary, doc_id))
        conn.commit()
        updated += len(batch)

        if updated % 10 == 0:
            elapsed = time.time() - t0
            rate = updated / elapsed
            eta = (total - updated) / rate if rate > 0 else 0
            logger.info(f"Progress: {updated}/{total} ({rate:.2f}/sec, ETA {eta/60:.0f}min)")

    elapsed = time.time() - t0
    logger.info(f"Done. Updated {updated} articles in {elapsed:.0f}s ({updated/elapsed:.2f}/sec)")
    conn.close()


if __name__ == "__main__":
    main()
