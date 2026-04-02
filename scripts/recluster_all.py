"""
One-off script: re-cluster all feed_items using updated BART summaries.

Reads all feed_items ordered by timestamp, runs them through
IncrementalEventClusterer.add_article(), and upserts clusters to DB.

Usage (inside scheduler container):
    python -m scripts.recluster_all
"""

import asyncio
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DB_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://causal:causal@localhost:5432/causal_interface")


async def main():
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import text, select
    from sentence_transformers import SentenceTransformer

    from db.models import FeedItem, EventCluster
    from services.rss.event_clusterer import IncrementalEventClusterer

    engine = create_async_engine(DB_URL, pool_size=5)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        # Step 1: Wipe existing clusters and cluster references
        result = await db.execute(text("SELECT COUNT(*) FROM event_clusters"))
        old_count = result.scalar()
        logger.info(f"Wiping {old_count} existing clusters...")

        await db.execute(text("""
            UPDATE feed_items SET
                cluster_id = NULL,
                mapped_node_id = NULL,
                mapping_confidence = NULL
        """))
        await db.execute(text("DELETE FROM event_clusters"))
        await db.commit()

        # Step 2: Load all feed_items with summary, ordered chronologically
        result = await db.execute(
            select(FeedItem)
            .where(FeedItem.summary.isnot(None))
            .where(FeedItem.summary != "")
            .order_by(FeedItem.timestamp.asc())
        )
        items = list(result.scalars().all())
        total = len(items)
        logger.info(f"Re-clustering {total} feed_items...")

        # Step 3: Initialize clusterer with sentence model
        logger.info("Loading sentence-transformers model...")
        model = SentenceTransformer("all-mpnet-base-v2")
        clusterer = IncrementalEventClusterer(model=model)

        # Step 4: Process each feed_item through the clusterer
        t0 = time.time()
        for i, item in enumerate(items):
            cluster, is_new = clusterer.add_article(
                title=item.headline or "",
                published=item.timestamp,
                summary=item.summary,
                source=item.source or "",
                url=item.url or "",
            )

            # Update feed_item with new cluster_id
            item.cluster_id = cluster.cluster_id
            cluster.article_ids.append(str(item.id))

            # Upsert cluster to DB every article (keeps state consistent)
            await clusterer.upsert_cluster_to_db(cluster, db)

            if (i + 1) % 50 == 0:
                await db.commit()
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                logger.info(f"Progress: {i+1}/{total} ({rate:.1f}/sec, ETA {eta:.0f}s), {len(clusterer.clusters)} clusters")

        await db.commit()
        elapsed = time.time() - t0

        # Summary
        result = await db.execute(text("SELECT COUNT(*) FROM event_clusters"))
        new_count = result.scalar()
        result = await db.execute(text("SELECT COUNT(*) FROM feed_items WHERE cluster_id IS NOT NULL"))
        linked = result.scalar()
        logger.info(
            f"Done in {elapsed:.0f}s. "
            f"{new_count} clusters created, {linked} feed_items linked. "
            f"(was {old_count} clusters before)"
        )

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
