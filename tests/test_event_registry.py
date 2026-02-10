"""
Tests for Event Registry
"""

import pytest
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.event_registry import EventRegistry, Event, get_registry


class TestEventRegistry:
    """Tests for EventRegistry class."""
    
    def test_load_events(self):
        """Test loading events from configuration."""
        registry = get_registry(force_reload=True)
        
        # Should load 3 events
        assert len(registry) == 3
    
    def test_get_event_by_id(self):
        """Test retrieving event by ID."""
        registry = get_registry()
        
        event = registry.get_event("gemini-5-release-2026")
        assert event is not None
        assert event.event_id == "gemini-5-release-2026"
        assert event.event_type == "model_release"
        assert "Gemini 5" in event.primary_entities
    
    def test_get_nonexistent_event(self):
        """Test retrieving non-existent event returns None."""
        registry = get_registry()
        
        event = registry.get_event("nonexistent-event")
        assert event is None
    
    def test_event_entities(self):
        """Test event entity accessors."""
        registry = get_registry()
        event = registry.get_event("gpt-5-release-2026")
        
        assert "GPT-5" in event.primary_entities
        assert "OpenAI" in event.primary_entities
        assert len(event.secondary_entities) > 0
        assert len(event.aliases) > 0
    
    def test_event_dependencies(self):
        """Test event dependencies."""
        registry = get_registry()
        event = registry.get_event("gemini-5-release-2026")
        
        assert "training" in event.dependencies
        assert "compute" in event.dependencies
        assert "safety" in event.dependencies
    
    def test_days_until_deadline(self):
        """Test deadline calculation."""
        registry = get_registry()
        event = registry.get_event("gemini-5-release-2026")
        
        days = event.days_until_deadline()
        assert days >= 0
    
    def test_get_all_entity_variants(self):
        """Test entity variant generation."""
        registry = get_registry()
        event = registry.get_event("gemini-5-release-2026")
        
        variants = event.get_all_entity_variants()
        assert "gemini 5" in variants
        assert "google" in variants
        assert "gemini-5" in variants or "gemini5" in variants
    
    def test_list_event_ids(self):
        """Test listing event IDs."""
        registry = get_registry()
        
        ids = registry.list_event_ids()
        assert "gemini-5-release-2026" in ids
        assert "gpt-5-release-2026" in ids
        assert "eu-ai-act-enforcement-2027" in ids
    
    def test_get_events_by_type(self):
        """Test filtering events by type."""
        registry = get_registry()
        
        model_releases = registry.get_events_by_type("model_release")
        assert len(model_releases) == 2
        
        regulations = registry.get_events_by_type("regulation")
        assert len(regulations) == 1
    
    def test_dependency_descriptions(self):
        """Test loading dependency descriptions."""
        registry = get_registry()
        
        desc = registry.get_dependency_description("training")
        assert len(desc) > 0
        assert "training" in desc.lower()


class TestEvent:
    """Tests for Event dataclass."""
    
    def test_event_creation(self):
        """Test creating an Event directly."""
        event = Event(
            event_id="test-event",
            event_type="model_release",
            event_title="Test Event",
            event_description="A test event",
            primary_entities=["Entity1"],
            secondary_entities=["Entity2"],
            aliases=["Alias1"],
            deadline=datetime(2027, 12, 31),
            dependencies=["training"],
            polymarket_slug="test-event-slug"
        )
        
        assert event.event_id == "test-event"
        assert event.event_type == "model_release"
        assert "Entity1" in event.primary_entities
    
    def test_primary_entity_set(self):
        """Test getting primary entity set."""
        registry = get_registry()
        event = registry.get_event("gemini-5-release-2026")
        
        primary_set = event.get_primary_entity_set()
        assert "gemini 5" in primary_set
        assert "google" in primary_set
