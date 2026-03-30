#!/usr/bin/env python3
"""
Standalone test: RSS Feed Ingestion → Article Clustering

Fetches articles from curated RSS feeds, normalizes them, and clusters
similar articles into events using sentence-transformer embeddings + HAC.

No database, no NER, no full pipeline — just RSS → normalize → cluster → print.

Usage:
  python scripts/test_rss_cluster.py
  python scripts/test_rss_cluster.py --categories politics,economics
  python scripts/test_rss_cluster.py --threshold 0.55
  python scripts/test_rss_cluster.py --max-articles 50
  python scripts/test_rss_cluster.py -v
"""

import argparse
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

# Allow HuggingFace model downloads
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import feedparser
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("rss_cluster")


# ═══════════════════════════════════════════════════════════════════════════════
# RSS Feed Configuration
# ═══════════════════════════════════════════════════════════════════════════════

RSS_FEEDS: Dict[str, List[Tuple[str, str]]] = {
    "politics": [
        ("Politico", "https://rss.politico.com/politics-news.xml"),
        ("The Hill", "https://thehill.com/feed/"),
        ("AP Politics", "https://rsshub.app/apnews/topics/politics"),
    ],
    "economics": [
        ("CNBC Economy", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
        ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories/"),
        ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    ],
    "geopolitics": [
        ("BBC World", "http://feeds.bbci.co.uk/news/world/rss.xml"),
        ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
        ("AP Top News", "https://rsshub.app/apnews/topics/apf-topnews"),
        ("Reuters via Google", "https://news.google.com/rss/search?q=site:reuters.com&hl=en-US&gl=US&ceid=US:en"),
    ],
    "sports": [
        ("ESPN", "https://www.espn.com/espn/rss/news"),
        ("BBC Sport", "http://feeds.bbci.co.uk/sport/rss.xml"),
        ("CBS Sports", "https://www.cbssports.com/rss/headlines/"),
    ],
    "general": [
        ("CNN Top", "http://rss.cnn.com/rss/edition.rss"),
        ("NPR News", "https://feeds.npr.org/1001/rss.xml"),
        ("Google News", "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class RSSArticle:
    title: str
    url: str
    published: Optional[datetime]
    source: str  # Feed name
    category: str  # Feed category
    description: str
    author: Optional[str] = None

    def text_for_embedding(self) -> str:
        """Concatenate title + description for semantic encoding."""
        parts = [self.title]
        if self.description:
            parts.append(self.description[:500])
        return " ".join(parts)


@dataclass
class ArticleCluster:
    canonical_title: str
    canonical_time: Optional[datetime]
    articles: List[RSSArticle]
    sources: List[str]
    num_articles: int
    avg_similarity: float


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RSS Fetching
# ═══════════════════════════════════════════════════════════════════════════════

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_html(text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    if not text:
        return ""
    text = unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _parse_published(entry) -> Optional[datetime]:
    """Extract published datetime from a feedparser entry."""
    for attr in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, attr, None)
        if parsed:
            try:
                return datetime(*parsed[:6])
            except (ValueError, TypeError):
                continue
    # Try ISO string fallback
    for attr in ("published", "updated"):
        val = getattr(entry, attr, None)
        if val:
            try:
                from dateutil.parser import parse as dateparse
                return dateparse(val).replace(tzinfo=None)
            except Exception:
                continue
    return None


def fetch_rss_feeds(
    categories: Optional[List[str]] = None,
    max_articles: Optional[int] = None,
    timeout: int = 15,
) -> List[RSSArticle]:
    """Fetch articles from all configured RSS feeds."""
    articles: List[RSSArticle] = []
    seen_urls: set = set()

    feeds_to_fetch = []
    for cat, feeds in RSS_FEEDS.items():
        if categories and cat not in categories:
            continue
        for name, url in feeds:
            feeds_to_fetch.append((cat, name, url))

    logger.info(f"Fetching {len(feeds_to_fetch)} RSS feeds...")

    for cat, name, url in feeds_to_fetch:
        try:
            logger.debug(f"  Fetching {name}: {url}")
            resp = requests.get(url, timeout=timeout, headers={
                "User-Agent": "Mozilla/5.0 (compatible; VibeTrading/1.0)"
            })
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)

            count = 0
            for entry in feed.entries:
                link = getattr(entry, "link", "") or ""
                if not link or link in seen_urls:
                    continue
                seen_urls.add(link)

                title = _clean_html(getattr(entry, "title", "") or "")
                if not title or len(title) < 10:
                    continue

                description = _clean_html(
                    getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
                )
                author = getattr(entry, "author", None)
                published = _parse_published(entry)

                articles.append(RSSArticle(
                    title=title,
                    url=link,
                    published=published,
                    source=name,
                    category=cat,
                    description=description,
                    author=author,
                ))
                count += 1

            logger.info(f"  ✓ {name}: {count} articles")

        except Exception as e:
            logger.warning(f"  ✗ {name}: {e}")
            continue

    # Deduplicate by title similarity (exact match after lowering)
    unique: List[RSSArticle] = []
    seen_titles: set = set()
    for a in articles:
        key = a.title.lower().strip()
        if key not in seen_titles:
            seen_titles.add(key)
            unique.append(a)

    logger.info(f"Fetched {len(unique)} unique articles ({len(articles) - len(unique)} title-dupes removed)")

    if max_articles and len(unique) > max_articles:
        unique = unique[:max_articles]
        logger.info(f"Capped to {max_articles} articles")

    return unique


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Clustering
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_articles(
    articles: List[RSSArticle],
    threshold: float = 0.65,
    time_window_hours: float = 48,
    temporal_decay_tau: float = 12.0,
) -> List[ArticleCluster]:
    """
    Cluster articles using sentence-transformer embeddings + HAC.

    Algorithm (mirrors belief_graph/event_clustering.py):
    1. Encode title + description with all-mpnet-base-v2
    2. Compute cosine similarity matrix
    3. Apply temporal decay for articles outside time window
    4. Hierarchical Agglomerative Clustering (average linkage)
    5. Build clusters with canonical title (longest) and time (earliest)
    """
    if len(articles) < 2:
        return [_single_cluster(a) for a in articles]

    logger.info(f"Loading sentence-transformer model...")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # Encode
    texts = [a.text_for_embedding() for a in articles]
    logger.info(f"Encoding {len(texts)} articles...")
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
    embeddings = np.array(embeddings)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    # Cosine similarity matrix
    sim_matrix = embeddings @ embeddings.T

    # Apply temporal decay
    if time_window_hours > 0:
        timestamps = []
        for a in articles:
            if a.published:
                timestamps.append(a.published)
            else:
                timestamps.append(_utc_now())  # No timestamp → assume recent

        for i in range(len(articles)):
            for j in range(i + 1, len(articles)):
                dt_seconds = abs((timestamps[i] - timestamps[j]).total_seconds())
                window_seconds = time_window_hours * 3600
                if dt_seconds > window_seconds:
                    overshoot = dt_seconds - window_seconds
                    tau_seconds = temporal_decay_tau * 3600
                    decay = math.exp(-overshoot / tau_seconds)
                    sim_matrix[i, j] *= decay
                    sim_matrix[j, i] *= decay

    # HAC clustering
    try:
        from sklearn.cluster import AgglomerativeClustering

        distance_matrix = 1.0 - sim_matrix
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = np.clip(distance_matrix, 0, 2)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - threshold,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(distance_matrix)
    except ImportError:
        logger.warning("sklearn not available, falling back to greedy clustering")
        labels = _greedy_cluster(sim_matrix, threshold)

    # Build clusters from labels
    cluster_map: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        cluster_map.setdefault(label, []).append(idx)

    clusters: List[ArticleCluster] = []
    for label, indices in sorted(cluster_map.items(), key=lambda x: -len(x[1])):
        members = [articles[i] for i in indices]

        # Canonical title: longest
        canonical_title = max(members, key=lambda a: len(a.title)).title

        # Canonical time: earliest
        times = [a.published for a in members if a.published]
        canonical_time = min(times) if times else None

        # Sources: unique
        sources = sorted(set(a.source for a in members))

        # Average pairwise similarity
        if len(indices) > 1:
            sims = []
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    sims.append(float(sim_matrix[indices[i], indices[j]]))
            avg_sim = sum(sims) / len(sims)
        else:
            avg_sim = 1.0

        clusters.append(ArticleCluster(
            canonical_title=canonical_title,
            canonical_time=canonical_time,
            articles=members,
            sources=sources,
            num_articles=len(members),
            avg_similarity=round(avg_sim, 3),
        ))

    # Sort by cluster size descending, then by time
    clusters.sort(key=lambda c: (-c.num_articles, c.canonical_time or _utc_now()))
    return clusters


def _single_cluster(article: RSSArticle) -> ArticleCluster:
    return ArticleCluster(
        canonical_title=article.title,
        canonical_time=article.published,
        articles=[article],
        sources=[article.source],
        num_articles=1,
        avg_similarity=1.0,
    )


def _greedy_cluster(sim_matrix: np.ndarray, threshold: float) -> List[int]:
    """Fallback greedy clustering if sklearn is unavailable."""
    n = sim_matrix.shape[0]
    labels = [-1] * n
    current_label = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        labels[i] = current_label
        for j in range(i + 1, n):
            if labels[j] == -1 and sim_matrix[i, j] >= threshold:
                labels[j] = current_label
        current_label += 1
    return labels


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Output
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(clusters: List[ArticleCluster], total_articles: int):
    now = _utc_now().strftime("%Y-%m-%d %H:%M UTC")
    num_feeds = len(set(
        a.source for c in clusters for a in c.articles
    ))

    print(f"\n{'=' * 80}")
    print(f"RSS CLUSTERING RESULTS — {now}")
    print(f"Fetched {total_articles} articles from {num_feeds} feeds")
    print(f"{'=' * 80}")

    multi_clusters = [c for c in clusters if c.num_articles > 1]
    singletons = [c for c in clusters if c.num_articles == 1]

    if multi_clusters:
        print(f"\n--- MULTI-ARTICLE CLUSTERS ({len(multi_clusters)}) ---")
        for i, cluster in enumerate(multi_clusters, 1):
            time_str = cluster.canonical_time.strftime("%Y-%m-%d %H:%M") if cluster.canonical_time else "unknown"
            print(f"\n{'─' * 60}")
            print(f"Cluster {i} ({cluster.num_articles} articles, {len(cluster.sources)} sources)")
            print(f"  Title:      {cluster.canonical_title}")
            print(f"  Time:       {time_str} (earliest)")
            print(f"  Sources:    {', '.join(cluster.sources)}")
            print(f"  Similarity: {cluster.avg_similarity:.2f}")
            print(f"  Articles:")
            for a in sorted(cluster.articles, key=lambda x: x.published or _utc_now()):
                t = a.published.strftime("%H:%M") if a.published else "??:??"
                print(f"    • {a.source:20s} — \"{a.title[:80]}\" ({t})")

    if singletons:
        print(f"\n--- SINGLETONS ({len(singletons)}) ---")
        for cluster in singletons[:20]:  # Show first 20 only
            a = cluster.articles[0]
            t = a.published.strftime("%H:%M") if a.published else "??:??"
            print(f"  [{a.source:20s}] \"{a.title[:80]}\" ({t})")
        if len(singletons) > 20:
            print(f"  ... and {len(singletons) - 20} more singletons")

    clustered_count = sum(c.num_articles for c in multi_clusters)
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: {total_articles} articles → {len(multi_clusters)} clusters "
          f"({clustered_count} articles clustered, {len(singletons)} singletons)")
    print(f"{'=' * 80}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Test RSS feed ingestion + article clustering"
    )
    parser.add_argument(
        "--categories", default=None,
        help="Comma-separated feed categories to fetch (politics,economics,geopolitics,sports,general)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65,
        help="Clustering similarity threshold (0-1, default: 0.65)"
    )
    parser.add_argument(
        "--max-articles", type=int, default=None,
        help="Max total articles to process"
    )
    parser.add_argument(
        "--time-window", type=float, default=48,
        help="Temporal window in hours for full similarity (default: 48)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]

    # Fetch
    articles = fetch_rss_feeds(
        categories=categories,
        max_articles=args.max_articles,
    )

    if not articles:
        logger.error("No articles fetched. Check network and feed URLs.")
        sys.exit(1)

    # Cluster
    clusters = cluster_articles(
        articles,
        threshold=args.threshold,
        time_window_hours=args.time_window,
    )

    # Print
    print_results(clusters, len(articles))


if __name__ == "__main__":
    main()
