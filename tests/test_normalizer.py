"""
Tests for Document Normalizer
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.normalizer import (
    DocumentNormalizer,
    NormalizedDocument,
    SourceTypeDetector,
    AuthorTypeInferrer,
    TextCleaner
)


class TestSourceTypeDetector:
    """Tests for SourceTypeDetector class."""
    
    def test_detect_official_sources(self):
        """Test detecting official company sources."""
        detector = SourceTypeDetector()
        
        assert detector.detect("https://blog.google/ai/gemini") == "official"
        assert detector.detect("https://openai.com/blog/gpt-5") == "official"
        assert detector.detect("https://deepmind.google/research") == "official"
        assert detector.detect("https://ec.europa.eu/ai-act") == "official"
    
    def test_detect_journalist_sources(self):
        """Test detecting journalist sources."""
        detector = SourceTypeDetector()
        
        assert detector.detect("https://techcrunch.com/article") == "journalist"
        assert detector.detect("https://www.nytimes.com/tech") == "journalist"
        assert detector.detect("https://www.theverge.com/ai") == "journalist"
        assert detector.detect("https://wired.com/story") == "journalist"
    
    def test_detect_social_sources(self):
        """Test detecting social media sources."""
        detector = SourceTypeDetector()
        
        assert detector.detect("https://reddit.com/r/MachineLearning", "reddit") == "social"
        assert detector.detect("https://twitter.com/user/status", "twitter") == "social"
        assert detector.detect("https://x.com/user/status") == "social"
    
    def test_detect_forum_sources(self):
        """Test detecting forum sources."""
        detector = SourceTypeDetector()
        
        assert detector.detect("https://news.ycombinator.com/item") == "forum"
        assert detector.detect("https://lesswrong.com/posts") == "forum"
    
    def test_detect_research_sources(self):
        """Test detecting research sources."""
        detector = SourceTypeDetector()
        
        assert detector.detect("https://arxiv.org/abs/1234") == "research"
        assert detector.detect("https://openreview.net/forum") == "research"
        assert detector.detect("https://nature.com/articles") == "research"
    
    def test_unknown_source_defaults(self):
        """Test that unknown sources default to journalist."""
        detector = SourceTypeDetector()
        
        # Unknown domain defaults to journalist
        assert detector.detect("https://unknownsite.com/article") == "journalist"


class TestAuthorTypeInferrer:
    """Tests for AuthorTypeInferrer class."""
    
    def test_infer_from_source_type(self):
        """Test inferring author type from source type."""
        inferrer = AuthorTypeInferrer()
        
        assert inferrer.infer("official") == "company"
        assert inferrer.infer("journalist") == "journalist"
        assert inferrer.infer("research") == "researcher"
        assert inferrer.infer("social") == "anonymous"
        assert inferrer.infer("forum") == "anonymous"
    
    def test_infer_deleted_author(self):
        """Test handling deleted authors."""
        inferrer = AuthorTypeInferrer()
        
        assert inferrer.infer("social", "[deleted]") == "anonymous"
        assert inferrer.infer("social", "anonymous_user") == "anonymous"


class TestTextCleaner:
    """Tests for TextCleaner class."""
    
    def test_clean_html(self):
        """Test removing HTML tags."""
        cleaner = TextCleaner()
        
        text = "<p>Hello <b>World</b></p>"
        cleaned = cleaner.clean(text)
        
        assert "<p>" not in cleaned
        assert "<b>" not in cleaned
        assert "Hello" in cleaned
        assert "World" in cleaned
    
    def test_clean_whitespace(self):
        """Test normalizing whitespace."""
        cleaner = TextCleaner()
        
        text = "Hello    World\n\n\n\nTest"
        cleaned = cleaner.clean(text)
        
        assert "    " not in cleaned
        assert "\n\n\n\n" not in cleaned
    
    def test_extract_title(self):
        """Test title extraction."""
        cleaner = TextCleaner()
        
        text = "This is the title\n\nThis is the body text that goes on longer."
        title = cleaner.extract_title(text)
        
        assert "This is the title" in title


class TestDocumentNormalizer:
    """Tests for DocumentNormalizer class."""
    
    def test_normalize_document(self):
        """Test normalizing an ingested document."""
        normalizer = DocumentNormalizer()
        
        ingested = {
            "doc_id": "test123",
            "url": "https://techcrunch.com/article/ai-news",
            "source": "web",
            "title": "AI News Article",
            "raw_text": "This is the article content about AI.",
            "author": "John Doe",
            "timestamp": "2026-02-01T10:00:00",
            "query_used": "AI news",
            "query_type": "journalist",
            "event_id": "test-event",
            "metadata": {}
        }
        
        normalized = normalizer.normalize(ingested)
        
        assert normalized.doc_id == "test123"
        assert normalized.source_type == "journalist"
        assert normalized.author_type == "journalist"
        assert normalized.title == "AI News Article"
        assert "AI" in normalized.raw_text
    
    def test_normalize_reddit_document(self):
        """Test normalizing a Reddit document."""
        normalizer = DocumentNormalizer()
        
        ingested = {
            "doc_id": "reddit123",
            "url": "https://reddit.com/r/MachineLearning/comments/123",
            "source": "reddit",
            "title": "Discussion about GPT-5",
            "raw_text": "What do you think about GPT-5?",
            "author": "reddit_user",
            "timestamp": "2026-02-01T10:00:00",
            "query_used": "GPT-5 discussion",
            "query_type": "public_opinion",
            "event_id": "gpt-5-release-2026",
            "metadata": {"subreddit": "MachineLearning"}
        }
        
        normalized = normalizer.normalize(ingested)
        
        assert normalized.source_type == "social"
        assert normalized.author_type == "anonymous"
    
    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        original = NormalizedDocument(
            doc_id="test123",
            title="Test Title",
            raw_text="Test content",
            source_type="journalist",
            author_type="journalist",
            timestamp=None,
            url="https://example.com",
            query_used="test query",
            query_type="official",
            event_id="test-event"
        )
        
        data = original.to_dict()
        restored = NormalizedDocument.from_dict(data)
        
        assert restored.doc_id == original.doc_id
        assert restored.title == original.title
        assert restored.source_type == original.source_type
