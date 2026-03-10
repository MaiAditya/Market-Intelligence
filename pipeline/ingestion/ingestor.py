"""
Unified Data Ingestor

Coordinates data ingestion from multiple sources:
- Reddit API
- Twitter/X (via Nitter fallback)
- Generic web scraping
- Search engines (DuckDuckGo, Google)

Implements the "ingest wide → filter strictly" principle.
"""

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse


def _utc_now() -> datetime:
    """Get current UTC time as a naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

from .reddit_api import RedditClient, RedditPost, AI_SUBREDDITS
from .twitter_scraper import TwitterScraperSync, Tweet
from .web_scraper import WebScraper, ScrapedPage

logger = logging.getLogger(__name__)

# ── Content filter constants ──────────────────────────────────────────────────
# Domains whose content is not useful for market intelligence (video/social noise)
_SKIP_DOMAINS: frozenset = frozenset({
    "youtube.com", "youtu.be",
    "tiktok.com",
    "vimeo.com",
    "instagram.com",
    "facebook.com", "fb.com", "fb.watch",
    "twitch.tv",
})

# Well-known brand name overrides for source display
_BRAND_OVERRIDES: dict = {
    "techcrunch": "TechCrunch",
    "businessinsider": "Business Insider",
    "theverge": "The Verge",
    "arstechnica": "Ars Technica",
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
    "coindesk": "CoinDesk",
    "marketwatch": "MarketWatch",
    "reddit": "Reddit",
}


def _domain_to_source_name(domain: str, url: str = "") -> str:
    """
    Convert a raw domain/URL into a human-readable source name.
    e.g. 'medium.com' → 'Medium', 'geeky-gadgets.com' → 'Geeky Gadgets'
    """
    # Try to extract hostname from full URL
    hostname = ""
    if url:
        try:
            hostname = urlparse(url).hostname or ""
            hostname = hostname.lstrip("www.").lstrip("m.").lstrip("mobile.")
        except Exception:
            pass
    if not hostname:
        hostname = (domain or "").split("/")[0].strip().lstrip("www.")
    if not hostname:
        return "Web"
    main_label = hostname.split(".")[0].lower()
    if main_label in _BRAND_OVERRIDES:
        return _BRAND_OVERRIDES[main_label]
    # Title-case with hyphen → space
    return " ".join(w.capitalize() for w in main_label.split("-")) or "Web"


def _should_skip_document(url: str, raw_text: str) -> bool:
    """
    Return True if a document should be skipped. Filters:
    1. Video/social-media domain blocklist (no useful text content)
    2. Non-English text (detected on first 500 chars)

    Falls back to False (accept) on any detection error to avoid dropping
    ambiguous or very short documents.
    """
    # 1. Domain blocklist
    try:
        host = urlparse(url).netloc.lower().lstrip("www.")
        # Match exact domain or any subdomain (e.g. m.youtube.com)
        if any(host == d or host.endswith("." + d) for d in _SKIP_DOMAINS):
            return True
    except Exception:
        pass

    # 2. Language detection (English only)
    sample = (raw_text or "")[:500].strip()
    if len(sample) >= 50:  # Only check if there's enough text
        try:
            from langdetect import detect, LangDetectException
            lang = detect(sample)
            if lang != "en":
                return True
        except Exception:
            pass  # Accept on detection failure

    return False


_summarizer = None  # Module-level lazy singleton

def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        try:
            from .article_summarizer import ArticleSummarizer
            _summarizer = ArticleSummarizer()
        except Exception as e:
            logger.warning(f"ArticleSummarizer unavailable: {e}")
            _summarizer = False  # Sentinel — don't retry
    return _summarizer if _summarizer else None


@dataclass
class IngestedDocument:
    """
    Raw ingested document before normalization.
    
    This is the intermediate format between ingestion and normalization.
    """
    doc_id: str
    source: str  # reddit | twitter | web | news
    url: str
    title: str
    raw_text: str
    raw_html: Optional[str]
    author: Optional[str]
    timestamp: Optional[datetime]
    query_used: str
    query_type: str
    event_id: str
    metadata: Dict = field(default_factory=dict)
    ingested_at: datetime = field(default_factory=_utc_now)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON storage."""
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "raw_text": self.raw_text,
            "author": self.author,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "query_used": self.query_used,
            "query_type": self.query_type,
            "event_id": self.event_id,
            "metadata": self.metadata,
            "ingested_at": self.ingested_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "IngestedDocument":
        """Create from dictionary."""
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])
        
        ingested_at = _utc_now()
        if data.get("ingested_at"):
            ingested_at = datetime.fromisoformat(data["ingested_at"])
        
        return cls(
            doc_id=data["doc_id"],
            source=data["source"],
            url=data["url"],
            title=data["title"],
            raw_text=data["raw_text"],
            raw_html=data.get("raw_html"),
            author=data.get("author"),
            timestamp=timestamp,
            query_used=data["query_used"],
            query_type=data["query_type"],
            event_id=data["event_id"],
            metadata=data.get("metadata", {}),
            ingested_at=ingested_at
        )


def generate_doc_id(url: str, timestamp: Optional[datetime] = None) -> str:
    """Generate a unique document ID from URL and timestamp."""
    content = url
    if timestamp:
        content += timestamp.isoformat()
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class DataIngestor:
    """
    Unified data ingestion coordinator.
    
    Implements the hybrid ingestion strategy:
    1. Reddit: Official API via PRAW
    2. Twitter: Nitter fallback
    3. News/Web: BeautifulSoup + newspaper3k
    4. Search: DuckDuckGo for URL discovery
    
    Follows "ingest wide → filter strictly" principle.
    """
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        max_results_per_query: int = 20,
        max_workers: int = 4,
        summarize_on_ingest: bool = False,
    ):
        """
        Initialize the data ingestor.
        
        Args:
            data_dir: Directory to store raw documents
            max_results_per_query: Max results per search query
            max_workers: Max parallel workers for scraping
            summarize_on_ingest: If True, auto-generate BART summary after each save
        """
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data" / "documents"
        
        self.data_dir = Path(data_dir)
        
        self.max_results = max_results_per_query
        self.max_workers = max_workers
        self.summarize_on_ingest = summarize_on_ingest
        
        # Initialize source clients
        self.reddit = RedditClient()
        self.twitter = TwitterScraperSync()
        self.web = WebScraper()
        self._url_lock = threading.Lock()
        
        # Track already ingested URLs to avoid duplicates
        self._ingested_urls: Set[str] = set()
        self._load_ingested_urls()
        
        # Track content hashes to catch same content from different URLs
        self._seen_hashes: Set[str] = set()
        self._hashes_path = self.data_dir / ".seen_hashes.json"
        self._load_seen_hashes()
    
    def _load_ingested_urls(self) -> None:
        pass
    
    def _load_seen_hashes(self) -> None:
        pass
    
    def _save_seen_hashes(self) -> None:
        pass
    
    @staticmethod
    def _compute_content_hash(title: str, text: str) -> str:
        """Compute SHA-256 hash of title + first 200 chars of text for dedup."""
        content = (title or "").strip().lower() + "|" + (text or "")[:200].strip().lower()
        return hashlib.sha256(content.encode('utf-8', errors='ignore')).hexdigest()
    
    def _save_document(self, doc: IngestedDocument) -> bool:
        """
        Save ingested document to disk if URL and content are new.

        Filtering → Two-layer dedup:
        0. Content filter — skip video domains, non-English text
        1. URL dedup — exact URL match
        2. Content hash dedup — SHA-256 of title + first 200 chars

        Returns:
            True if saved, False if skipped as duplicate or filtered.
        """
        # 0. Content filter gate
        if _should_skip_document(doc.url, doc.raw_text):
            logger.debug(f"Filtered out (domain/language): {doc.url}")
            return False

        content_hash = self._compute_content_hash(doc.title, doc.raw_text)

        with self._url_lock:
            if doc.url in self._ingested_urls:
                return False
            if content_hash in self._seen_hashes:
                logger.debug(f"Content hash duplicate: {doc.title[:60]}")
                return False
            self._ingested_urls.add(doc.url)
            self._seen_hashes.add(content_hash)

        try:
            # Optionally generate summary before saving
            if self.summarize_on_ingest and doc.raw_text and len(doc.raw_text.split()) >= 20:
                summarizer = _get_summarizer()
                if summarizer:
                    try:
                        doc.metadata["summary"] = summarizer.summarize(
                            doc.raw_text, title=doc.title
                        )
                    except Exception as se:
                        logger.debug(f"Summary generation skipped: {se}")

            # FILE SAVE REMOVED: Managed centrally in ingest_for_event
        except Exception:
            with self._url_lock:
                self._ingested_urls.discard(doc.url)
                self._seen_hashes.discard(content_hash)
            raise
        
        return True
    
    def _reddit_post_to_document(
        self,
        post: RedditPost,
        query: str,
        query_type: str,
        event_id: str
    ) -> IngestedDocument:
        """Convert Reddit post to IngestedDocument."""
        # Combine title and text for full content
        full_text = post.title
        if post.text:
            full_text += "\n\n" + post.text
        
        return IngestedDocument(
            doc_id=generate_doc_id(post.permalink, post.created_utc),
            source="reddit",
            url=post.permalink,
            title=post.title,
            raw_text=full_text,
            raw_html=None,
            author=post.author,
            timestamp=post.created_utc,
            query_used=query,
            query_type=query_type,
            event_id=event_id,
            metadata={
                "subreddit": post.subreddit,
                "score": post.score,
                "num_comments": post.num_comments,
                "is_self": post.is_self
            }
        )
    
    def _tweet_to_document(
        self,
        tweet: Tweet,
        query: str,
        query_type: str,
        event_id: str
    ) -> IngestedDocument:
        """Convert Tweet to IngestedDocument."""
        return IngestedDocument(
            doc_id=generate_doc_id(tweet.url, tweet.created_at),
            source="twitter",
            url=tweet.url,
            title=f"Tweet by @{tweet.author_handle}",
            raw_text=tweet.text,
            raw_html=None,
            author=tweet.author,
            timestamp=tweet.created_at,
            query_used=query,
            query_type=query_type,
            event_id=event_id,
            metadata={
                "likes": tweet.likes,
                "retweets": tweet.retweets,
                "is_retweet": tweet.is_retweet
            }
        )
    
    def _scraped_page_to_document(
        self,
        page: ScrapedPage,
        query: str,
        query_type: str,
        event_id: str
    ) -> IngestedDocument:
        """Convert ScrapedPage to IngestedDocument."""
        return IngestedDocument(
            doc_id=generate_doc_id(page.url, page.scraped_at),
            source=_domain_to_source_name(page.domain, page.url),
            url=page.url,
            title=page.title,
            raw_text=page.text,
            raw_html=page.html,
            author=page.author,
            timestamp=page.publish_date,
            query_used=query,
            query_type=query_type,
            event_id=event_id,
            metadata={
                "domain": page.domain,
                "meta_description": page.meta_description
            }
        )
    
    def ingest_from_reddit(
        self,
        query: str,
        query_type: str,
        event_id: str,
        limit: int = None,
        subreddits: Optional[List[str]] = None
    ) -> List[IngestedDocument]:
        """
        Ingest documents from Reddit.
        
        Args:
            query: Search query
            query_type: Type of query (official, journalist, etc.)
            event_id: Associated event ID
            limit: Max results
            subreddits: Specific subreddits to search
        
        Returns:
            List of ingested documents
        """
        limit = limit or self.max_results
        documents = []
        
        if not self.reddit.is_available():
            logger.warning("Reddit API not available")
            return documents
        
        try:
            if subreddits:
                posts = self.reddit.search_subreddits(
                    query=query,
                    subreddits=subreddits,
                    limit_per_subreddit=limit // len(subreddits)
                )
            else:
                # Search AI-relevant subreddits
                posts = self.reddit.search_subreddits(
                    query=query,
                    subreddits=AI_SUBREDDITS[:4],  # Top 4 subreddits
                    limit_per_subreddit=limit // 4
                )
            
            for post in posts:
                doc = self._reddit_post_to_document(
                    post, query, query_type, event_id
                )
                if self._save_document(doc):
                    documents.append(doc)
            
            logger.info(f"Reddit ingestion for '{query}': {len(documents)} new documents")
            
        except Exception as e:
            logger.error(f"Reddit ingestion failed: {e}")
        
        return documents
    
    def ingest_from_twitter(
        self,
        query: str,
        query_type: str,
        event_id: str,
        limit: int = None
    ) -> List[IngestedDocument]:
        """
        Ingest documents from Twitter/X via Nitter.
        
        Args:
            query: Search query
            query_type: Type of query
            event_id: Associated event ID
            limit: Max results
        
        Returns:
            List of ingested documents
        """
        limit = limit or self.max_results
        documents = []
        
        try:
            tweets = self.twitter.search(query, limit=limit)
            
            for tweet in tweets:
                doc = self._tweet_to_document(
                    tweet, query, query_type, event_id
                )
                if self._save_document(doc):
                    documents.append(doc)
            
            logger.info(f"Twitter ingestion for '{query}': {len(documents)} new documents")
            
        except Exception as e:
            logger.error(f"Twitter ingestion failed: {e}")
        
        return documents
    
    def ingest_from_web(
        self,
        query: str,
        query_type: str,
        event_id: str,
        limit: int = None
    ) -> List[IngestedDocument]:
        """
        Ingest documents from web search results.
        
        Args:
            query: Search query
            query_type: Type of query
            event_id: Associated event ID
            limit: Max results
        
        Returns:
            List of ingested documents
        """
        limit = limit or self.max_results
        documents = []
        
        try:
            # Use the combined search method which tries multiple engines
            urls = self.web.search(
                query, 
                num_results=limit,
                engines=['duckduckgo', 'bing']
            )
            
            logger.info(f"Search for '{query}' found {len(urls)} URLs")
            
            # Filter already ingested URLs
            with self._url_lock:
                urls = [u for u in urls if u not in self._ingested_urls]
            
            if not urls:
                logger.info(f"No new URLs to scrape for '{query}'")
                return documents
            
            logger.info(f"Scraping {len(urls)} new URLs for '{query}'")
            
            # Scrape pages in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_url = {
                    executor.submit(self.web.scrape, url): url
                    for url in urls
                }
                
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        page = future.result()
                        if page and page.text and len(page.text) > 100:
                            doc = self._scraped_page_to_document(
                                page, query, query_type, event_id
                            )
                            if self._save_document(doc):
                                documents.append(doc)
                                logger.debug(f"Scraped: {page.title[:50] if page.title else url}")
                    except Exception as e:
                        logger.debug(f"Failed to scrape {url}: {e}")
            
            logger.info(f"Web ingestion for '{query}': {len(documents)} new documents")
            
        except Exception as e:
            logger.error(f"Web ingestion failed: {e}", exc_info=True)
        
        return documents
    
    def ingest_for_query(
        self,
        query: str,
        query_type: str,
        event_id: str,
        sources: Optional[List[str]] = None
    ) -> List[IngestedDocument]:
        """
        Ingest from all sources for a single query.
        
        Args:
            query: Search query
            query_type: Type of query
            event_id: Associated event ID
            sources: List of sources to use (reddit, twitter, web)
        
        Returns:
            Combined list of ingested documents
        """
        if sources is None:
            # Determine sources based on query type
            if query_type == "public_opinion":
                sources = ["reddit", "twitter"]
            elif query_type == "official":
                sources = ["web"]
            else:
                sources = ["web", "reddit"]
        
        all_documents = []
        
        if "reddit" in sources:
            docs = self.ingest_from_reddit(query, query_type, event_id)
            all_documents.extend(docs)
        
        if "twitter" in sources:
            docs = self.ingest_from_twitter(query, query_type, event_id)
            all_documents.extend(docs)
        
        if "web" in sources:
            docs = self.ingest_from_web(query, query_type, event_id)
            all_documents.extend(docs)
        
        return all_documents
    
    def ingest_for_event(
        self,
        event_id: str,
        queries: List[Dict],
        sources: Optional[List[str]] = None
    ) -> List[IngestedDocument]:
        """
        Ingest all queries for a single event.
        
        Args:
            event_id: Event ID
            queries: List of query dicts with 'query' and 'query_type'
            sources: Optional list of sources to use
        
        Returns:
            All ingested documents for the event
        """
        all_documents = []
        max_workers = min(self.max_workers, max(1, len(queries)))
        
        if max_workers <= 1:
            for q in queries:
                docs = self.ingest_for_query(
                    query=q["query"],
                    query_type=q["query_type"],
                    event_id=event_id,
                    sources=sources
                )
                all_documents.extend(docs)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self.ingest_for_query,
                        q["query"],
                        q["query_type"],
                        event_id,
                        sources
                    )
                    for q in queries
                ]
                for future in as_completed(futures):
                    try:
                        docs = future.result()
                        all_documents.extend(docs)
                    except Exception as e:
                        logger.error(f"Query ingestion task failed: {e}")
        
        logger.info(
            f"Event {event_id} ingestion complete: "
            f"{len(all_documents)} total documents"
        )
        
        from utils.db_storage import save_artifact
        save_artifact(event_id, "raw_documents", [d.to_dict() for d in all_documents])
        
        return all_documents
    
    def get_document(self, doc_id: str) -> Optional[IngestedDocument]:
        """No longer supported in DB-only mode. Use get_documents_for_event."""
        return None
    
    def get_documents_for_event(self, event_id: str) -> List[IngestedDocument]:
        """Load all documents for an event."""
        from utils.db_storage import load_artifact
        data = load_artifact(event_id, "raw_documents")
        if not data:
            return []
        
        return [IngestedDocument.from_dict(d) for d in data]
    
    def get_stats(self) -> Dict:
        """Get ingestion statistics."""
        return {}
