"""
Pytest Configuration and Fixtures
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def event_registry():
    """Provide event registry for tests."""
    from pipeline.event_registry import get_registry
    return get_registry(force_reload=True)


@pytest.fixture(scope="session")
def query_generator():
    """Provide query generator for tests."""
    from pipeline.query_generator import QueryGenerator
    return QueryGenerator()


@pytest.fixture(scope="session")
def normalizer(tmp_path_factory):
    """Provide document normalizer with temp directory."""
    from pipeline.normalizer import DocumentNormalizer
    temp_dir = tmp_path_factory.mktemp("normalized")
    return DocumentNormalizer(normalized_dir=str(temp_dir))


@pytest.fixture
def sample_ingested_document():
    """Provide a sample ingested document for testing."""
    return {
        "doc_id": "test_doc_123",
        "url": "https://techcrunch.com/2026/02/01/ai-news",
        "source": "web",
        "title": "AI Model Release Imminent",
        "raw_text": (
            "Google is expected to release Gemini 5 soon. "
            "The model has been in training for months and shows "
            "impressive results on MMLU benchmarks. "
            "CEO Sundar Pichai stated the release is on track."
        ),
        "author": "Tech Reporter",
        "timestamp": "2026-02-01T10:00:00",
        "query_used": "Gemini 5 release",
        "query_type": "journalist",
        "event_id": "gemini-5-release-2026",
        "metadata": {}
    }


@pytest.fixture
def sample_reddit_document():
    """Provide a sample Reddit document for testing."""
    return {
        "doc_id": "reddit_doc_456",
        "url": "https://reddit.com/r/MachineLearning/comments/abc123",
        "source": "reddit",
        "title": "Discussion: When will GPT-5 be released?",
        "raw_text": (
            "I think OpenAI will release GPT-5 by the end of 2026. "
            "There have been rumors of significant improvements in reasoning. "
            "Sam Altman hinted at major announcements coming soon."
        ),
        "author": "ai_enthusiast",
        "timestamp": "2026-02-01T12:00:00",
        "query_used": "GPT-5 release discussion",
        "query_type": "public_opinion",
        "event_id": "gpt-5-release-2026",
        "metadata": {"subreddit": "MachineLearning", "score": 150}
    }


@pytest.fixture
def sample_official_document():
    """Provide a sample official source document."""
    return {
        "doc_id": "official_doc_789",
        "url": "https://blog.google/ai/gemini-5-announcement",
        "source": "web",
        "title": "Introducing Gemini 5: Our Most Capable Model Yet",
        "raw_text": (
            "Today we are excited to announce Gemini 5, our most advanced "
            "AI model to date. Gemini 5 represents a significant leap forward "
            "in AI capabilities, with breakthrough performance on MMLU, "
            "HumanEval, and other benchmarks. The model will be available "
            "to developers in Q4 2026."
        ),
        "author": "Google AI Team",
        "timestamp": "2026-09-15T10:00:00",
        "query_used": "Gemini 5 official announcement",
        "query_type": "official",
        "event_id": "gemini-5-release-2026",
        "metadata": {}
    }
