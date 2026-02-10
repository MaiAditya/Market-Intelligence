"""
Document Normalizer

Normalizes ingested documents into a standardized format.
Handles source type detection, author type inference, and text cleaning.
"""

import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse


def _utc_now() -> datetime:
    """Get current UTC time as a naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


logger = logging.getLogger(__name__)


@dataclass
class NormalizedDocument:
    """
    Normalized document ready for entity extraction and mapping.
    
    Schema as specified in the plan.
    """
    doc_id: str
    title: str
    raw_text: str
    source_type: str  # official | journalist | social | forum | research
    author_type: str  # company | journalist | anonymous | researcher
    timestamp: Optional[datetime]
    url: str
    query_used: str
    query_type: str
    event_id: str
    extracted_entities: List[Dict] = field(default_factory=list)
    normalized_at: datetime = field(default_factory=_utc_now)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON storage."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "raw_text": self.raw_text,
            "source_type": self.source_type,
            "author_type": self.author_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "url": self.url,
            "query_used": self.query_used,
            "query_type": self.query_type,
            "event_id": self.event_id,
            "extracted_entities": self.extracted_entities,
            "normalized_at": self.normalized_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "NormalizedDocument":
        """Create from dictionary."""
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])
        
        normalized_at = _utc_now()
        if data.get("normalized_at"):
            normalized_at = datetime.fromisoformat(data["normalized_at"])
        
        return cls(
            doc_id=data["doc_id"],
            title=data["title"],
            raw_text=data["raw_text"],
            source_type=data["source_type"],
            author_type=data["author_type"],
            timestamp=timestamp,
            url=data["url"],
            query_used=data["query_used"],
            query_type=data["query_type"],
            event_id=data["event_id"],
            extracted_entities=data.get("extracted_entities", []),
            normalized_at=normalized_at,
            metadata=data.get("metadata", {})
        )


class SourceTypeDetector:
    """
    Rule-based source type detection from URL/domain.
    
    Source types:
    - official: Company domains (google.com, openai.com)
    - journalist: News domains (nytimes.com, techcrunch.com)
    - social: Social media (reddit.com, twitter.com)
    - forum: Forums and discussion sites (hackernews, lesswrong)
    - research: Academic sources (arxiv.org, papers.ssrn.com)
    """
    
    # Domain to source type mapping
    OFFICIAL_DOMAINS = {
        "google.com", "blog.google", "deepmind.com", "deepmind.google",
        "openai.com", "anthropic.com", "meta.com", "ai.meta.com",
        "microsoft.com", "blogs.microsoft.com", "azure.microsoft.com",
        "nvidia.com", "developer.nvidia.com",
        "huggingface.co", "aws.amazon.com",
        "europa.eu", "ec.europa.eu",  # EU official
        "gov.uk", "gov.eu"
    }
    
    JOURNALIST_DOMAINS = {
        "nytimes.com", "washingtonpost.com", "theguardian.com",
        "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com",
        "techcrunch.com", "theverge.com", "wired.com", "wired.co.uk",
        "arstechnica.com", "engadget.com", "cnet.com",
        "technologyreview.com", "spectrum.ieee.org",
        "fortune.com", "bloomberg.com", "ft.com",
        "theinformation.com", "semafor.com", "axios.com",
        "venturebeat.com", "zdnet.com", "techradar.com"
    }
    
    SOCIAL_DOMAINS = {
        "reddit.com", "old.reddit.com", "redd.it",
        "twitter.com", "x.com", "nitter.net",
        "facebook.com", "linkedin.com",
        "threads.net", "mastodon.social",
        "bsky.app"
    }
    
    FORUM_DOMAINS = {
        "news.ycombinator.com", "ycombinator.com",
        "lesswrong.com", "alignmentforum.org",
        "effectivealtruism.org", "forum.effectivealtruism.org",
        "slashdot.org", "lobste.rs",
        "stackexchange.com", "stackoverflow.com",
        "quora.com"
    }
    
    RESEARCH_DOMAINS = {
        "arxiv.org", "openreview.net",
        "papers.ssrn.com", "ssrn.com",
        "scholar.google.com", "semanticscholar.org",
        "acm.org", "dl.acm.org",
        "ieee.org", "ieeexplore.ieee.org",
        "nature.com", "science.org", "sciencemag.org",
        "pnas.org", "cell.com",
        "biorxiv.org", "medrxiv.org"
    }
    
    @classmethod
    def detect(cls, url: str, source_hint: str = "") -> str:
        """
        Detect source type from URL.
        
        Args:
            url: Full URL
            source_hint: Hint from ingestion (reddit, twitter, web)
        
        Returns:
            Source type string
        """
        # Quick check based on ingestion source
        if source_hint == "reddit":
            return "social"
        if source_hint == "twitter":
            return "social"
        
        # Parse domain from URL
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            
            # Check against domain lists
            for official in cls.OFFICIAL_DOMAINS:
                if domain == official or domain.endswith("." + official):
                    return "official"
            
            for journalist in cls.JOURNALIST_DOMAINS:
                if domain == journalist or domain.endswith("." + journalist):
                    return "journalist"
            
            for social in cls.SOCIAL_DOMAINS:
                if domain == social or domain.endswith("." + social):
                    return "social"
            
            for forum in cls.FORUM_DOMAINS:
                if domain == forum or domain.endswith("." + forum):
                    return "forum"
            
            for research in cls.RESEARCH_DOMAINS:
                if domain == research or domain.endswith("." + research):
                    return "research"
            
            # Default heuristics based on path/content
            path = parsed.path.lower()
            if "/blog" in path or "/news" in path:
                return "journalist"
            if "/paper" in path or "/publication" in path:
                return "research"
            
        except Exception:
            pass
        
        # Default to journalist for web sources
        return "journalist"


class AuthorTypeInferrer:
    """
    Infer author type from source type and metadata.
    
    Author types:
    - company: Official company communications
    - journalist: Professional journalists
    - anonymous: Anonymous or pseudonymous
    - researcher: Academic researchers
    """
    
    @classmethod
    def infer(
        cls,
        source_type: str,
        author: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Infer author type.
        
        Args:
            source_type: Detected source type
            author: Author name if available
            metadata: Additional metadata
        
        Returns:
            Author type string
        """
        metadata = metadata or {}
        
        # Direct mapping from source type
        source_to_author = {
            "official": "company",
            "journalist": "journalist",
            "research": "researcher",
            "forum": "anonymous",
            "social": "anonymous"
        }
        
        author_type = source_to_author.get(source_type, "anonymous")
        
        # Refine based on author name
        if author:
            author_lower = author.lower()
            
            # Check for deleted/anonymous
            if "[deleted]" in author_lower or "anonymous" in author_lower:
                return "anonymous"
            
            # Reddit authors are typically pseudonymous
            if source_type == "social" and metadata.get("subreddit"):
                return "anonymous"
        
        return author_type


class TextCleaner:
    """
    Clean and normalize text content.
    """
    
    # Patterns to remove
    NOISE_PATTERNS = [
        r'\[deleted\]',
        r'\[removed\]',
        r'http[s]?://\S+',  # URLs (optional - keep for context?)
        r'<[^>]+>',  # HTML tags
        r'\s+',  # Multiple whitespace
    ]
    
    @classmethod
    def clean(cls, text: str, remove_urls: bool = False) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text content
            remove_urls: Whether to remove URLs
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)
        
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        text = re.sub(r'&#\d+;', ' ', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Optionally remove URLs
        if remove_urls:
            text = re.sub(r'http[s]?://\S+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @classmethod
    def extract_title(cls, text: str, max_length: int = 200) -> str:
        """
        Extract a title from text if none provided.
        
        Args:
            text: Text content
            max_length: Maximum title length
        
        Returns:
            Extracted title
        """
        if not text:
            return "Untitled"
        
        # Take first line or first sentence
        lines = text.split('\n')
        first_line = lines[0].strip() if lines else ""
        
        # If first line is short enough, use it
        if first_line and len(first_line) <= max_length:
            return first_line
        
        # Otherwise, take first sentence
        sentences = re.split(r'[.!?]', text)
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) <= max_length:
                return first_sentence
            return first_sentence[:max_length] + "..."
        
        return text[:max_length] + "..."


class DocumentNormalizer:
    """
    Main document normalization pipeline.
    
    Converts ingested documents to normalized format:
    1. Clean and normalize text
    2. Detect source type from URL
    3. Infer author type
    4. Generate consistent doc_id
    5. Save to normalized storage
    """
    
    def __init__(self, normalized_dir: Optional[str] = None):
        """
        Initialize normalizer.
        
        Args:
            normalized_dir: Directory for normalized documents
        """
        if normalized_dir is None:
            project_root = Path(__file__).parent.parent
            normalized_dir = project_root / "data" / "normalized"
        
        self.normalized_dir = Path(normalized_dir)
        self.normalized_dir.mkdir(parents=True, exist_ok=True)
        
        self.source_detector = SourceTypeDetector()
        self.author_inferrer = AuthorTypeInferrer()
        self.text_cleaner = TextCleaner()
    
    def normalize(self, ingested_doc: dict) -> NormalizedDocument:
        """
        Normalize a single ingested document.
        
        Args:
            ingested_doc: Dictionary from IngestedDocument.to_dict()
        
        Returns:
            NormalizedDocument
        """
        # Extract fields from ingested document
        url = ingested_doc.get("url", "")
        source_hint = ingested_doc.get("source", "")
        raw_text = ingested_doc.get("raw_text", "")
        title = ingested_doc.get("title", "")
        author = ingested_doc.get("author")
        metadata = ingested_doc.get("metadata", {})
        
        # Clean text
        cleaned_text = self.text_cleaner.clean(raw_text)
        
        # Clean or extract title
        if title:
            title = self.text_cleaner.clean(title)
        else:
            title = self.text_cleaner.extract_title(cleaned_text)
        
        # Detect source type
        source_type = self.source_detector.detect(url, source_hint)
        
        # Infer author type
        author_type = self.author_inferrer.infer(source_type, author, metadata)
        
        # Parse timestamp
        timestamp = None
        if ingested_doc.get("timestamp"):
            try:
                timestamp = datetime.fromisoformat(ingested_doc["timestamp"])
            except (ValueError, TypeError):
                pass
        
        # Create normalized document
        normalized = NormalizedDocument(
            doc_id=ingested_doc.get("doc_id", self._generate_doc_id(url)),
            title=title,
            raw_text=cleaned_text,
            source_type=source_type,
            author_type=author_type,
            timestamp=timestamp,
            url=url,
            query_used=ingested_doc.get("query_used", ""),
            query_type=ingested_doc.get("query_type", ""),
            event_id=ingested_doc.get("event_id", ""),
            extracted_entities=[],  # Will be filled by entity extractor
            metadata={
                "original_source": source_hint,
                "author": author,
                **metadata
            }
        )
        
        return normalized
    
    def _generate_doc_id(self, url: str) -> str:
        """Generate document ID from URL."""
        return hashlib.sha256(url.encode()).hexdigest()[:16]
    
    def normalize_and_save(self, ingested_doc: dict) -> NormalizedDocument:
        """
        Normalize and save a document.
        
        Args:
            ingested_doc: Ingested document dictionary
        
        Returns:
            NormalizedDocument
        """
        normalized = self.normalize(ingested_doc)
        self.save(normalized)
        return normalized
    
    def save(self, doc: NormalizedDocument) -> None:
        """Save normalized document to disk."""
        from utils.json_utils import dump_json
        doc_path = self.normalized_dir / f"{doc.doc_id}.json"
        with open(doc_path, 'w', encoding='utf-8') as f:
            dump_json(doc.to_dict(), f)
    
    def load(self, doc_id: str) -> Optional[NormalizedDocument]:
        """Load a normalized document by ID."""
        doc_path = self.normalized_dir / f"{doc_id}.json"
        if not doc_path.exists():
            return None
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return NormalizedDocument.from_dict(data)
    
    def load_all_for_event(self, event_id: str) -> List[NormalizedDocument]:
        """Load all normalized documents for an event."""
        documents = []
        
        for doc_file in self.normalized_dir.glob("*.json"):
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get("event_id") == event_id:
                        documents.append(NormalizedDocument.from_dict(data))
            except Exception as e:
                logger.debug(f"Error loading {doc_file}: {e}")
        
        return documents
    
    def normalize_batch(
        self,
        ingested_docs: List[dict],
        save: bool = True
    ) -> List[NormalizedDocument]:
        """
        Normalize a batch of documents.
        
        Args:
            ingested_docs: List of ingested document dicts
            save: Whether to save to disk
        
        Returns:
            List of normalized documents
        """
        normalized = []
        
        for doc in ingested_docs:
            try:
                norm_doc = self.normalize(doc)
                if save:
                    self.save(norm_doc)
                normalized.append(norm_doc)
            except Exception as e:
                logger.error(f"Failed to normalize document: {e}")
        
        logger.info(f"Normalized {len(normalized)} documents")
        return normalized
    
    def update_entities(
        self,
        doc_id: str,
        entities: List[Dict]
    ) -> Optional[NormalizedDocument]:
        """
        Update extracted entities for a document.
        
        Args:
            doc_id: Document ID
            entities: List of entity dicts
        
        Returns:
            Updated document or None
        """
        doc = self.load(doc_id)
        if doc is None:
            return None
        
        doc.extracted_entities = entities
        self.save(doc)
        return doc
