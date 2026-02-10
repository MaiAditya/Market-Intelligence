"""
Event Registry Module

Handles loading, validation, and lookup of events from the registry.
Events are the primary objects in the system - all data flows from events.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from pathlib import Path


def _utc_now() -> datetime:
    """Get current UTC time as a naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Represents a tracked event in the system."""
    event_id: str
    event_type: str  # model_release | regulation | capability
    event_title: str
    event_description: str
    primary_entities: List[str]
    secondary_entities: List[str]
    aliases: List[str]
    deadline: datetime
    dependencies: List[str]
    polymarket_slug: str
    
    def get_all_entity_variants(self) -> Set[str]:
        """Get all entity variants including aliases for matching."""
        variants = set()
        for entity in self.primary_entities + self.secondary_entities + self.aliases:
            variants.add(entity.lower())
            # Also add without hyphens/spaces for flexible matching
            variants.add(entity.lower().replace("-", "").replace(" ", ""))
        return variants
    
    def get_primary_entity_set(self) -> Set[str]:
        """Get lowercase set of primary entities."""
        return {e.lower() for e in self.primary_entities}
    
    def get_secondary_entity_set(self) -> Set[str]:
        """Get lowercase set of secondary entities."""
        return {e.lower() for e in self.secondary_entities}
    
    def get_alias_set(self) -> Set[str]:
        """Get lowercase set of aliases."""
        return {a.lower() for a in self.aliases}
    
    def days_until_deadline(self) -> int:
        """Calculate days remaining until deadline."""
        now = _utc_now()
        delta = self.deadline - now
        return max(0, delta.days)


class EventRegistry:
    """
    Manages the event registry.
    
    Responsibilities:
    - Load events from JSON configuration
    - Validate event schema compliance
    - Provide event lookup by ID
    - No ML - pure data structure management
    """
    
    VALID_EVENT_TYPES = {"model_release", "regulation", "capability"}
    VALID_DEPENDENCIES = {
        "training", "compute", "safety", "regulation",
        "executive_statement", "public_narrative"
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the event registry.
        
        Args:
            config_path: Path to events.json. If None, uses default location.
        """
        if config_path is None:
            # Default to config/events.json relative to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "events.json"
        
        self.config_path = Path(config_path)
        self.events: Dict[str, Event] = {}
        self.dependency_descriptions: Dict[str, str] = {}
        
        self._load_events()
    
    def _load_events(self) -> None:
        """Load and validate events from configuration file."""
        logger.debug(f"Loading events from: {self.config_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Event configuration not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Load dependency descriptions
        self.dependency_descriptions = config.get("dependency_descriptions", {})
        
        # Load and validate each event
        for event_data in config.get("events", []):
            event = self._parse_event(event_data)
            self._validate_event(event)
            self.events[event.event_id] = event
            logger.debug(f"Loaded event: {event.event_id}")
        
        logger.info(f"Loaded {len(self.events)} events from registry")
    
    def _parse_event(self, data: dict) -> Event:
        """Parse raw event data into Event object."""
        # Parse deadline to datetime
        deadline_str = data.get("deadline", "")
        try:
            deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
            # Convert to naive UTC for consistency
            deadline = deadline.replace(tzinfo=None)
        except ValueError:
            raise ValueError(f"Invalid deadline format: {deadline_str}")
        
        return Event(
            event_id=data.get("event_id", ""),
            event_type=data.get("event_type", ""),
            event_title=data.get("event_title", ""),
            event_description=data.get("event_description", ""),
            primary_entities=data.get("primary_entities", []),
            secondary_entities=data.get("secondary_entities", []),
            aliases=data.get("aliases", []),
            deadline=deadline,
            dependencies=data.get("dependencies", []),
            polymarket_slug=data.get("polymarket_slug", "")
        )
    
    def _validate_event(self, event: Event) -> None:
        """Validate event schema compliance."""
        errors = []
        
        # Required fields
        if not event.event_id:
            errors.append("event_id is required")
        if not event.event_type:
            errors.append("event_type is required")
        elif event.event_type not in self.VALID_EVENT_TYPES:
            errors.append(f"Invalid event_type: {event.event_type}")
        
        if not event.primary_entities:
            errors.append("At least one primary_entity is required")
        
        if not event.dependencies:
            errors.append("At least one dependency is required")
        else:
            invalid_deps = set(event.dependencies) - self.VALID_DEPENDENCIES
            if invalid_deps:
                errors.append(f"Invalid dependencies: {invalid_deps}")
        
        if not event.polymarket_slug:
            errors.append("polymarket_slug is required")
        
        if errors:
            raise ValueError(f"Event '{event.event_id}' validation failed: {errors}")
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """Get event by ID."""
        return self.events.get(event_id)
    
    def get_all_events(self) -> List[Event]:
        """Get all registered events."""
        return list(self.events.values())
    
    def get_events_by_type(self, event_type: str) -> List[Event]:
        """Get events filtered by type."""
        return [e for e in self.events.values() if e.event_type == event_type]
    
    def get_dependency_description(self, dependency: str) -> str:
        """Get the description for a dependency category."""
        return self.dependency_descriptions.get(dependency, "")
    
    def get_all_dependency_descriptions(self) -> Dict[str, str]:
        """Get all dependency descriptions."""
        return self.dependency_descriptions.copy()
    
    def list_event_ids(self) -> List[str]:
        """Get list of all event IDs."""
        return list(self.events.keys())
    
    def __len__(self) -> int:
        return len(self.events)
    
    def __iter__(self):
        return iter(self.events.values())


# Module-level singleton for convenience
_registry: Optional[EventRegistry] = None


def get_registry(config_path: Optional[str] = None, force_reload: bool = False) -> EventRegistry:
    """
    Get the event registry singleton.
    
    Args:
        config_path: Optional path to events.json
        force_reload: If True, reload the registry even if already loaded
    
    Returns:
        EventRegistry instance
    """
    global _registry
    if _registry is None or force_reload:
        _registry = EventRegistry(config_path)
    return _registry
