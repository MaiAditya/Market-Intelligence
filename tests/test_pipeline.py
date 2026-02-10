"""
Integration Tests for Pipeline
"""

import pytest
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.event_registry import get_registry
from pipeline.query_generator import QueryGenerator
from pipeline.normalizer import DocumentNormalizer, NormalizedDocument
from pipeline.time_extractor import TimeExtractor, NumericExtractor
from integrations.source_registry import get_source_credibility, is_official_source


class TestTimeExtractor:
    """Tests for time/date extraction."""
    
    def test_extract_month_year(self):
        """Test extracting month year dates."""
        text = "The release is expected in December 2026."
        dates = TimeExtractor.extract(text)
        
        assert len(dates) > 0
        assert any("December 2026" in d.raw_text for d in dates)
    
    def test_extract_quarter(self):
        """Test extracting quarter dates."""
        text = "We expect the release in Q3 2026."
        dates = TimeExtractor.extract(text)
        
        assert len(dates) > 0
        assert any("Q3 2026" in d.raw_text for d in dates)
    
    def test_extract_end_of_year(self):
        """Test extracting relative year expressions."""
        text = "The model will be ready by end of 2026."
        dates = TimeExtractor.extract(text)
        
        assert len(dates) > 0
        assert any("end of 2026" in d.raw_text.lower() for d in dates)
    
    def test_extract_early_late(self):
        """Test extracting early/late year expressions."""
        text = "We're targeting late 2026 for the release."
        dates = TimeExtractor.extract(text)
        
        assert len(dates) > 0


class TestNumericExtractor:
    """Tests for numeric extraction."""
    
    def test_extract_parameters(self):
        """Test extracting model parameters."""
        text = "The new model has 500B parameters."
        numerics = NumericExtractor.extract(text)
        
        assert len(numerics) > 0
        param_nums = [n for n in numerics if n.context == "parameters"]
        assert len(param_nums) > 0
        assert param_nums[0].normalized_value == 500_000_000_000
    
    def test_extract_percentage(self):
        """Test extracting percentages."""
        text = "The model achieved 85% accuracy on MMLU."
        numerics = NumericExtractor.extract(text)
        
        assert len(numerics) > 0
    
    def test_extract_benchmark(self):
        """Test extracting benchmark scores."""
        text = "GPT-5 scored 92% on MMLU and 88% on HumanEval."
        numerics = NumericExtractor.extract(text)
        
        # Should find benchmark scores
        benchmark_nums = [n for n in numerics if "benchmark" in n.context or "percent" in n.context]
        assert len(benchmark_nums) > 0


class TestSourceRegistry:
    """Tests for source credibility registry."""
    
    def test_official_source_credibility(self):
        """Test official source has high credibility."""
        cred = get_source_credibility("https://openai.com/blog/gpt5")
        assert cred >= 0.9
        
        cred = get_source_credibility("https://blog.google/ai/gemini5")
        assert cred >= 0.9
    
    def test_news_source_credibility(self):
        """Test news source credibility."""
        cred = get_source_credibility("https://techcrunch.com/article")
        assert 0.7 <= cred <= 0.8
    
    def test_social_source_credibility(self):
        """Test social media has lower credibility."""
        cred = get_source_credibility("https://reddit.com/r/test")
        assert cred <= 0.6
        
        cred = get_source_credibility("https://twitter.com/user")
        assert cred <= 0.5
    
    def test_is_official_source(self):
        """Test is_official_source function."""
        assert is_official_source("https://openai.com/blog")
        assert is_official_source("https://blog.google/ai")
        assert not is_official_source("https://techcrunch.com/article")
        assert not is_official_source("https://reddit.com/r/ai")


class TestPipelineIntegration:
    """Integration tests for full pipeline."""
    
    def test_event_to_queries(self):
        """Test generating queries from event."""
        registry = get_registry()
        generator = QueryGenerator()
        
        for event in registry:
            query_set = generator.generate_queries_for_event(event)
            assert len(query_set.queries) > 0
            
            # Verify queries contain event-related terms
            all_queries = " ".join(q.query.lower() for q in query_set.queries)
            entity_found = any(
                e.lower() in all_queries 
                for e in event.primary_entities
            )
            assert entity_found, f"No primary entity found in queries for {event.event_id}"
    
    def test_document_normalization_flow(self):
        """Test document normalization flow."""
        normalizer = DocumentNormalizer()
        
        # Create mock ingested document
        ingested = {
            "doc_id": "integration_test_123",
            "url": "https://techcrunch.com/2026/02/01/ai-news",
            "source": "web",
            "title": "AI Model Release Imminent",
            "raw_text": "Google is expected to release Gemini 5 soon. The model has been in training for months.",
            "author": "Tech Reporter",
            "timestamp": "2026-02-01T10:00:00",
            "query_used": "Gemini 5 release",
            "query_type": "journalist",
            "event_id": "gemini-5-release-2026",
            "metadata": {}
        }
        
        normalized = normalizer.normalize(ingested)
        
        # Verify normalization
        assert normalized.doc_id == "integration_test_123"
        assert normalized.source_type == "journalist"
        assert "Gemini 5" in normalized.raw_text
        assert normalized.event_id == "gemini-5-release-2026"
    
    def test_all_events_have_polymarket_slug(self):
        """Verify all events have Polymarket slugs configured."""
        registry = get_registry()
        
        for event in registry:
            assert event.polymarket_slug, f"{event.event_id} missing polymarket_slug"
            assert len(event.polymarket_slug) > 0
    
    def test_event_dependencies_valid(self):
        """Verify all event dependencies are valid."""
        registry = get_registry()
        valid_deps = {
            "training", "compute", "safety", "regulation",
            "executive_statement", "public_narrative"
        }
        
        for event in registry:
            for dep in event.dependencies:
                assert dep in valid_deps, f"Invalid dependency '{dep}' in {event.event_id}"
    
    def test_entity_variant_generation(self):
        """Test entity variant generation for matching."""
        registry = get_registry()
        event = registry.get_event("gemini-5-release-2026")
        
        variants = event.get_all_entity_variants()
        
        # Should include various forms
        assert "gemini 5" in variants
        assert "google" in variants
        # Should include normalized forms (no spaces/hyphens)
        assert "gemini5" in variants or "gemini-5" in variants


class TestDataIntegrity:
    """Tests for data integrity and configuration."""
    
    def test_events_json_valid(self):
        """Test events.json is valid and loadable."""
        registry = get_registry(force_reload=True)
        assert len(registry) > 0
    
    def test_query_templates_valid(self):
        """Test query templates are valid and loadable."""
        generator = QueryGenerator()
        assert len(generator.templates) > 0
    
    def test_dependency_descriptions_present(self):
        """Test dependency descriptions are present."""
        registry = get_registry()
        
        for dep in ["training", "compute", "safety", "regulation"]:
            desc = registry.get_dependency_description(dep)
            assert len(desc) > 0, f"Missing description for {dep}"
