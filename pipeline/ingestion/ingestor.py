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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed


def _utc_now() -> datetime:
    """Get current UTC time as a naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

from .reddit_api import RedditClient, RedditPost, AI_SUBREDDITS
from .twitter_scraper import TwitterScraperSync, Tweet
from .web_scraper import WebScraper, ScrapedPage

logger = logging.getLogger(__name__)


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
        max_workers: int = 4
    ):
        """
        Initialize the data ingestor.
        
        Args:
            data_dir: Directory to store raw documents
            max_results_per_query: Max results per search query
            max_workers: Max parallel workers for scraping
        """
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data" / "documents"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_results = max_results_per_query
        self.max_workers = max_workers
        
        # Initialize source clients
        self.reddit = RedditClient()
        self.twitter = TwitterScraperSync()
        self.web = WebScraper()
        
        # Track already ingested URLs to avoid duplicates
        self._ingested_urls: Set[str] = set()
        self._load_ingested_urls()
    
    def _load_ingested_urls(self) -> None:
        """Load set of already ingested URLs from existing documents."""
        for doc_file in self.data_dir.glob("*.json"):
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    if "url" in doc:
                        self._ingested_urls.add(doc["url"])
            except Exception:
                pass
    
    def _save_document(self, doc: IngestedDocument) -> None:
        """Save ingested document to disk."""
        from utils.json_utils import dump_json
        doc_path = self.data_dir / f"{doc.doc_id}.json"
        with open(doc_path, 'w', encoding='utf-8') as f:
            dump_json(doc.to_dict(), f)
        self._ingested_urls.add(doc.url)
    
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
            source="web",
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
                if post.permalink not in self._ingested_urls:
                    doc = self._reddit_post_to_document(
                        post, query, query_type, event_id
                    )
                    self._save_document(doc)
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
                if tweet.url not in self._ingested_urls:
                    doc = self._tweet_to_document(
                        tweet, query, query_type, event_id
                    )
                    self._save_document(doc)
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
                            self._save_document(doc)
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
        
        for q in queries:
            docs = self.ingest_for_query(
                query=q["query"],
                query_type=q["query_type"],
                event_id=event_id,
                sources=sources
            )
            all_documents.extend(docs)
        
        logger.info(
            f"Event {event_id} ingestion complete: "
            f"{len(all_documents)} total documents"
        )
        
        return all_documents
    
    def get_document(self, doc_id: str) -> Optional[IngestedDocument]:
        """Load a document by ID."""
        doc_path = self.data_dir / f"{doc_id}.json"
        if not doc_path.exists():
            return None
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return IngestedDocument.from_dict(data)
    
    def get_documents_for_event(self, event_id: str) -> List[IngestedDocument]:
        """Load all documents for an event."""
        documents = []
        
        for doc_file in self.data_dir.glob("*.json"):
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get("event_id") == event_id:
                        documents.append(IngestedDocument.from_dict(data))
            except Exception as e:
                logger.debug(f"Error loading {doc_file}: {e}")
        
        return documents
    
    def get_stats(self) -> Dict:
        """Get ingestion statistics."""
        stats = {
            "total_documents": len(list(self.data_dir.glob("*.json"))),
            "by_source": {},
            "by_event": {}
        }
        
        for doc_file in self.data_dir.glob("*.json"):
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    source = data.get("source", "unknown")
                    event = data.get("event_id", "unknown")
                    
                    stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
                    stats["by_event"][event] = stats["by_event"].get(event, 0) + 1
            except Exception:
                pass
        
        return stats
