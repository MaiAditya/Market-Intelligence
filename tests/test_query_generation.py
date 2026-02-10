"""
Tests for Query Generation
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.query_generator import QueryGenerator, GeneratedQuery, generate_queries_for_all_events
from pipeline.event_registry import get_registry


class TestQueryGenerator:
    """Tests for QueryGenerator class."""
    
    def test_generate_queries_for_event(self):
        """Test generating queries for a single event."""
        registry = get_registry()
        generator = QueryGenerator()
        
        event = registry.get_event("gemini-5-release-2026")
        query_set = generator.generate_queries_for_event(event)
        
        assert query_set.event_id == "gemini-5-release-2026"
        assert len(query_set.queries) > 0
    
    def test_query_families_present(self):
        """Test that all query families are generated."""
        registry = get_registry()
        generator = QueryGenerator()
        
        event = registry.get_event("gpt-5-release-2026")
        query_set = generator.generate_queries_for_event(event)
        
        # Get unique query types
        query_types = set(q.query_type for q in query_set.queries)
        
        assert "official" in query_types
        assert "journalist" in query_types
        assert "public_opinion" in query_types
        assert "critical" in query_types
    
    def test_query_contains_entity(self):
        """Test that queries contain event entities."""
        registry = get_registry()
        generator = QueryGenerator()
        
        event = registry.get_event("gemini-5-release-2026")
        query_set = generator.generate_queries_for_event(event)
        
        # At least some queries should contain primary entity
        queries_with_entity = [
            q for q in query_set.queries
            if "gemini" in q.query.lower() or "google" in q.query.lower()
        ]
        
        assert len(queries_with_entity) > 0
    
    def test_expected_bias_assignment(self):
        """Test that expected_bias is assigned correctly."""
        registry = get_registry()
        generator = QueryGenerator()
        
        event = registry.get_event("gemini-5-release-2026")
        query_set = generator.generate_queries_for_event(event)
        
        # Official queries should have positive bias
        official_queries = [q for q in query_set.queries if q.query_type == "official"]
        for q in official_queries:
            assert q.expected_bias == "positive"
        
        # Critical queries should have negative bias
        critical_queries = [q for q in query_set.queries if q.query_type == "critical"]
        for q in critical_queries:
            assert q.expected_bias == "negative"
    
    def test_generate_all_queries(self):
        """Test generating queries for all events."""
        generator = QueryGenerator()
        
        all_queries = generator.generate_all_queries()
        
        assert "gemini-5-release-2026" in all_queries
        assert "gpt-5-release-2026" in all_queries
        assert "eu-ai-act-enforcement-2027" in all_queries
    
    def test_max_queries_per_family(self):
        """Test max_queries_per_family limit."""
        registry = get_registry()
        generator = QueryGenerator()
        
        event = registry.get_event("gemini-5-release-2026")
        query_set = generator.generate_queries_for_event(event, max_queries_per_family=2)
        
        # Count queries per type
        from collections import Counter
        type_counts = Counter(q.query_type for q in query_set.queries)
        
        # Each family should have at most 2 queries (plus type-specific extras)
        for query_type, count in type_counts.items():
            assert count <= 4  # 2 base + 2 type-specific max
    
    def test_query_set_to_dict(self):
        """Test EventQuerySet serialization."""
        registry = get_registry()
        generator = QueryGenerator()
        
        event = registry.get_event("gemini-5-release-2026")
        query_set = generator.generate_queries_for_event(event)
        
        data = query_set.to_dict()
        
        assert data["event_id"] == "gemini-5-release-2026"
        assert "queries" in data
        assert len(data["queries"]) > 0
        assert "query" in data["queries"][0]
        assert "query_type" in data["queries"][0]
    
    def test_queries_by_type(self):
        """Test getting queries by specific type."""
        registry = get_registry()
        generator = QueryGenerator()
        
        event = registry.get_event("gemini-5-release-2026")
        
        official_queries = generator.get_queries_by_type(event, "official")
        
        assert len(official_queries) > 0
        for q in official_queries:
            assert q.query_type == "official"


class TestGeneratedQuery:
    """Tests for GeneratedQuery dataclass."""
    
    def test_generated_query_creation(self):
        """Test creating a GeneratedQuery."""
        query = GeneratedQuery(
            event_id="test-event",
            query="Test query string",
            query_type="official",
            expected_bias="positive",
            template_used="{primary_entity} test",
            entity_used="TestEntity"
        )
        
        assert query.event_id == "test-event"
        assert query.query == "Test query string"
        assert query.query_type == "official"
        assert query.expected_bias == "positive"
