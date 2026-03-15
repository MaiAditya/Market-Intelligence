"""
Database writer for the ingestion pipeline.

Pushes scraped documents directly to PostgreSQL (feed_items table)
with URL + content-hash dedup via ON CONFLICT DO NOTHING.
"""

import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_DOMAIN_RE = re.compile(
    r'^(https?://)?(www\.)?[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,6}(/)?$'
)


def _looks_like_domain(text: str) -> bool:
    """Return True if text looks like a bare domain name rather than a real headline."""
    if not text:
        return True
    t = text.strip()
    if len(t) > 80 or ' ' in t:
        return False
    if _DOMAIN_RE.match(t):
        return True
    if '.' in t:
        parts = t.rsplit('.', 1)
        if len(parts) == 2 and parts[1].lower() in (
            'com', 'org', 'net', 'io', 'co', 'gov', 'edu', 'news',
            'uk', 'us', 'eu', 'ca', 'au', 'de', 'fr', 'vn', 'my', 'ai', 'in',
        ):
            return True
    return False


def _fix_domain_title(title: str, text: str) -> str:
    """If title looks like a domain, extract the first sentence of text as the title."""
    if not _looks_like_domain(title):
        return title
    if not text:
        return title
    # Take first sentence (up to first period, question mark, or exclamation followed by space)
    first_line = text.strip().split('\n')[0].strip()
    # Try to find the first sentence
    m = re.match(r'^(.{10,200}?[.!?])(?:\s|$)', first_line)
    if m:
        return m.group(1).strip()
    # Otherwise use the first line (up to 200 chars)
    if len(first_line) >= 10:
        return first_line[:200].strip()
    return title


def _compute_content_hash(title: str, text: str) -> str:
    """SHA-256 of title + first 200 chars of text."""
    content = (title or "").strip().lower() + "|" + (text or "")[:200].strip().lower()
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()


def _infer_source_tier(source: str, domain: str) -> str:
    """Infer source credibility tier from source/domain."""
    official = {"reuters.com", "apnews.com", "bloomberg.com", "google.com", "apple.com"}
    verified = {"techcrunch.com", "theverge.com", "arstechnica.com", "wired.com", "bbc.com", "cnbc.com"}
    community = {"reddit.com", "twitter.com", "x.com", "news.ycombinator.com"}
    d = domain.lower().replace("www.", "")
    if d in official:
        return "official"
    if d in verified:
        return "verified"
    if d in community:
        return "community"
    return "unverified"


def _infer_confidence(tier: str) -> int:
    """Map source tier to confidence score (0-100)."""
    return {"official": 90, "verified": 75, "community": 50, "unverified": 30}.get(tier, 30)


class DBWriter:
    """
    Writes ingested documents directly to PostgreSQL feed_items table.

    Uses psycopg2 for sync writes (compatible with the sync scraping pipeline).
    Dedup is handled at DB level via UNIQUE(url) + ON CONFLICT DO NOTHING.
    """

    def __init__(self, db_url: str):
        """
        Args:
            db_url: PostgreSQL connection string (sync format).
                    e.g. postgresql://causal:causal@postgres:5432/causal_interface
        """
        import psycopg2
        self._conn = psycopg2.connect(db_url)
        self._conn.autocommit = True
        self._cursor = self._conn.cursor()
        self._inserted = 0
        self._skipped = 0
        logger.info(f"DBWriter connected to PostgreSQL")

    def insert_document(
        self,
        title: str,
        text: str,
        url: str,
        source: str,
        domain: str,
        event_id: str,
        doc_id: str,
        publish_date: Optional[datetime] = None,
    ) -> bool:
        """
        Insert a document into feed_items. Returns True if inserted, False if deduped.
        """
        title = _fix_domain_title(title, text)
        content_hash = _compute_content_hash(title, text)
        tier = _infer_source_tier(source, domain)
        confidence = _infer_confidence(tier)

        # Truncate summary to first 300 chars of text
        summary = (text or "")[:300].strip()
        if len(text or "") > 300:
            summary += "…"

        ts = publish_date or datetime.now(timezone.utc)

        try:
            self._cursor.execute(
                """
                INSERT INTO feed_items
                    (id, post_type, source, source_tier, headline, summary, url,
                     confidence_score, timestamp, pipeline_doc_id, event_id,
                     raw_text, content_hash, created_at)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s,
                     %s, %s, %s, %s,
                     %s, %s, %s)
                ON CONFLICT (url) DO NOTHING
                """,
                (
                    str(uuid.uuid4()),
                    "news",
                    source,
                    tier,
                    title,
                    summary,
                    url,
                    confidence,
                    ts,
                    doc_id,
                    event_id,
                    text,
                    content_hash,
                    datetime.now(timezone.utc),
                ),
            )
            if self._cursor.rowcount > 0:
                self._inserted += 1
                return True
            else:
                self._skipped += 1
                return False
        except Exception as e:
            logger.error(f"DB insert failed for {url}: {e}")
            return False

    def close(self):
        """Close DB connection."""
        try:
            self._cursor.close()
            self._conn.close()
        except Exception:
            pass

    @property
    def stats(self) -> dict:
        return {"inserted": self._inserted, "skipped_dedup": self._skipped}
