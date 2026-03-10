#!/usr/bin/env python3
"""
Feed Pusher

Pushes newly ingested documents to the backend PostgreSQL database as feed items.
This bridges the AI pipeline's raw document storage → backend's feed_items table.

Usage:
    python scripts/feed_pusher.py                          # push all unpushed docs
    python scripts/feed_pusher.py --event gemini-5-release-2026  # one event
    python scripts/feed_pusher.py --dry-run                # show what would push

The script tracks pushed doc IDs in data/.pushed_docs.json to avoid re-pushing.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
from uuid import uuid4

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.ingestion.ingestor import IngestedDocument, _should_skip_document

logger = logging.getLogger("feed_pusher")

# Backend path (relative to this repo)
BACKEND_ROOT = project_root.parent / "Causal_Interface" / "backend"


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class PushedTracker:
    """Tracks which doc_ids have been pushed to the backend."""

    def __init__(self, data_dir: Path):
        self._path = data_dir / ".pushed_docs.json"
        self._pushed: Set[str] = set()
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    self._pushed = set(json.load(f))
            except Exception:
                pass

    def save(self):
        try:
            with open(self._path, "w") as f:
                json.dump(sorted(self._pushed), f)
        except Exception as e:
            logger.warning(f"Failed to save pushed tracker: {e}")

    def is_pushed(self, doc_id: str) -> bool:
        return doc_id in self._pushed

    def mark_pushed(self, doc_id: str):
        self._pushed.add(doc_id)

    def count(self) -> int:
        return len(self._pushed)


def load_unpushed_docs(
    data_dir: Path,
    tracker: PushedTracker,
    event_id: Optional[str] = None,
) -> List[IngestedDocument]:
    """Load all documents that haven't been pushed yet."""
    docs = []
    for doc_file in data_dir.glob("*.json"):
        if doc_file.name.startswith("."):
            continue
        try:
            with open(doc_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            doc_id = data.get("doc_id", "")
            if tracker.is_pushed(doc_id):
                continue

            if event_id and data.get("event_id") != event_id:
                continue

            docs.append(IngestedDocument.from_dict(data))
        except Exception as e:
            logger.debug(f"Skipping {doc_file.name}: {e}")

    return docs


# ── Known brand name overrides ──────────────────────────────────────────────
_BRAND_OVERRIDES: dict = {
    "techcrunch": "TechCrunch",
    "youtube": "YouTube",
    "youtu": "YouTube",
    "businessinsider": "Business Insider",
    "theverge": "The Verge",
    "arstechnica": "Ars Technica",
    "9to5mac": "9to5Mac",
    "9to5google": "9to5Google",
    "macrumors": "MacRumors",
    "venturebeat": "VentureBeat",
    "fastcompany": "Fast Company",
    "nytimes": "New York Times",
    "washingtonpost": "Washington Post",
    "wsj": "Wall Street Journal",
    "reuters": "Reuters",
    "bloomberg": "Bloomberg",
    "wired": "Wired",
    "zdnet": "ZDNet",
    "cnet": "CNET",
    "engadget": "Engadget",
    "thenextweb": "TNW",
    "theguardian": "The Guardian",
    "bbc": "BBC",
    "cnbc": "CNBC",
    "apnews": "AP News",
    "reddit": "Reddit",
    "twitter": "Twitter",
    "linkedin": "LinkedIn",
    "medium": "Medium",
    "substack": "Substack",
    "github": "GitHub",
    "politico": "Politico",
    "axios": "Axios",
    "theatlantic": "The Atlantic",
    "fortune": "Fortune",
    "forbes": "Forbes",
    "ft": "Financial Times",
    "cnn": "CNN",
    "nbcnews": "NBC News",
    "msn": "MSN",
    "appleinsider": "AppleInsider",
    "marketwatch": "MarketWatch",
    "coindesk": "CoinDesk",
    "decrypt": "Decrypt",
    "hackernews": "Hacker News",
}


def _extract_site_name(url: str, domain: str) -> str:
    """
    Derive a human-readable site name from a URL or domain string.
    e.g. 'https://www.medium.com/foo'  → 'Medium'
         'geeky-gadgets.com'           → 'Geeky Gadgets'
    """
    # Try from full URL first
    try:
        from urllib.parse import urlparse
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lstrip("www.").lstrip("m.").lstrip("mobile.")
    except Exception:
        hostname = ""

    # Fall back to the metadata domain field
    if not hostname:
        hostname = (domain or "").split("/")[0].strip()

    if not hostname:
        return "Web"

    # Extract the main label (before first dot)
    main_label = hostname.split(".")[0].lower()

    # Apply brand override if known
    if main_label in _BRAND_OVERRIDES:
        return _BRAND_OVERRIDES[main_label]

    # Title-case with hyphen → space conversion
    return " ".join(w.capitalize() for w in main_label.split("-")) or "Web"


def _should_skip_feed_doc(doc: IngestedDocument) -> bool:
    """
    Return True if this document should NOT be pushed to the feed.
    Applies content filters: domain blocklist, non-English, and minimum content length.
    """
    # Use shared ingestor filter (YouTube, TikTok, language check)
    if _should_skip_document(doc.url, doc.raw_text):
        return True
    # Skip articles with very little text (likely scraped error pages)
    raw = doc.raw_text or ""
    if len(raw.split()) < 30:
        logger.debug(f"Skipping short content ({len(raw.split())} words): {doc.url}")
        return True
    return False


def map_doc_to_feed_item(doc: IngestedDocument) -> dict:
    """
    Map an IngestedDocument to a feed item dict matching the backend schema.

    Maps to FeedPostResponse fields:
      - post_type: "news"
      - source: human-readable site name (e.g. "Medium", "Reuters")
      - source_tier: inferred from domain
      - headline: doc.title
      - summary: first 300 chars of doc.raw_text
      - url: doc.url
      - timestamp: doc.timestamp or doc.ingested_at
    """
    domain = doc.metadata.get("domain", "")
    source = doc.source
    if source == "reddit":
        tier = "community"
        display_source = f"r/{doc.metadata.get('subreddit', 'unknown')}"
    elif source == "twitter":
        tier = "community"
        display_source = "Twitter/X"
    elif source == "web":
        tier = _infer_tier(domain)
        display_source = _extract_site_name(doc.url, domain)
    else:
        tier = "unverified"
        display_source = source

    # Truncate summary
    raw = doc.raw_text or ""
    summary = raw[:300].strip()
    if len(raw) > 300:
        summary += "..."

    ts = doc.timestamp or doc.ingested_at
    ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)

    return {
        "id": str(uuid4()),
        "post_type": "news",
        "source": display_source,
        "source_tier": tier,
        "headline": doc.title,
        "summary": summary,
        "url": doc.url,
        "confidence_score": _infer_confidence(tier),
        "timestamp": ts_str,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "event_id": doc.event_id,
        "doc_id": doc.doc_id,
    }


def _infer_tier(domain: str) -> str:
    """Infer source tier from domain."""
    official = {"congress.gov", "senate.gov", "whitehouse.gov", "sec.gov"}
    verified = {
        "reuters.com", "apnews.com", "bloomberg.com", "nytimes.com",
        "wsj.com", "ft.com", "bbc.com", "cnn.com", "politico.com",
        "techcrunch.com", "theverge.com", "arstechnica.com",
        "wired.com", "cnbc.com",
    }
    d = domain.lower().strip()
    if any(o in d for o in official):
        return "official"
    if any(v in d for v in verified):
        return "verified"
    return "unverified"


def _infer_confidence(tier: str) -> int:
    """Infer confidence score from tier."""
    return {"official": 95, "verified": 80, "community": 50, "unverified": 40}.get(tier, 40)


def push_to_backend_db(feed_items: List[dict], dry_run: bool = False) -> int:
    """
    Push feed items to the backend database.

    For now, saves as a JSON file that the backend can import.
    In production, this would use the backend's DB directly or API.
    """
    if dry_run:
        for item in feed_items[:5]:
            print(f"  [{item['source_tier']}] {item['source']}: {item['headline'][:60]}")
            print(f"    event_id={item['event_id']} | url={item['url'][:50]}")
        if len(feed_items) > 5:
            print(f"  ... and {len(feed_items) - 5} more")
        return 0

    # Save to a staging file the backend can import
    staging_dir = BACKEND_ROOT / "data" / "feed_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    staging_path = staging_dir / f"feed_batch_{ts}.json"

    with open(staging_path, "w", encoding="utf-8") as f:
        json.dump(feed_items, f, indent=2, default=str)

    logger.info(f"Saved {len(feed_items)} feed items to {staging_path}")
    return len(feed_items)


def main():
    parser = argparse.ArgumentParser(description="Push ingested docs to backend feed")
    parser.add_argument("--event", type=str, help="Only push docs for this event")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show what would push")
    args = parser.parse_args()

    setup_logging(args.verbose)

    data_dir = project_root / "data" / "documents"
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    tracker = PushedTracker(data_dir)
    logger.info(f"Already pushed: {tracker.count()} documents")

    docs = load_unpushed_docs(data_dir, tracker, event_id=args.event)
    logger.info(f"Found {len(docs)} unpushed documents")

    if not docs:
        print("No new documents to push.")
        return

    # Apply content filters (YouTube, non-English, too short, etc.)
    filtered_docs = [doc for doc in docs if not _should_skip_feed_doc(doc)]
    skipped = len(docs) - len(filtered_docs)
    if skipped:
        logger.info(f"Skipped {skipped} documents (video/non-English/too short)")

    if not filtered_docs:
        print("No valid documents to push after filtering.")
        return

    # Map to feed items
    feed_items = [map_doc_to_feed_item(doc) for doc in filtered_docs]
    logger.info(f"Mapped {len(feed_items)} feed items")

    # Push
    pushed = push_to_backend_db(feed_items, dry_run=args.dry_run)

    if not args.dry_run:
        # Mark as pushed
        for doc in docs:
            tracker.mark_pushed(doc.doc_id)
        tracker.save()
        print(f"\nPushed {pushed} feed items to backend staging.")
    else:
        print(f"\n[DRY RUN] Would push {len(feed_items)} feed items.")


if __name__ == "__main__":
    main()
