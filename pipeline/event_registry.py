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

# Important: ensure late import to avoid circular dependencies if needed
from integrations.polymarket_client import get_polymarket_client


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
    polymarket_token_id: str = None  # YES-token for CLOB sparkline (clobTokenIds[0])
    
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
    
    VALID_EVENT_TYPES = {
        "model_release", "regulation", "capability",
        "sports", "politics", "technology", "macro_economics", "finance",
        "geopolitics", "legal", "other",
    }
    VALID_DEPENDENCIES = {
        # AI / tech
        "training", "compute", "safety", "regulation",
        "executive_statement", "public_narrative",
        # Sports
        "performance", "injuries", "transfers", "schedule", "form",
        "standings", "momentum", "playoffs",
        # Politics / macro
        "polling", "fundraising", "endorsements", "legislation",
        "economic_data", "market_reaction", "geopolitics",
        # Monetary / macro economics
        "us_monetary_policy", "inflation_data", "employment_data",
        "interest_rates", "central_bank", "fed_policy", "cpi", "gdp",
        # General
        "news", "sentiment", "social_media", "media_coverage",

    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the event registry.
        
        Args:
            config_path: Path to events.json. If None, uses default location.
        """
        if config_path is None:
            # Check env var first (for Docker deployments)
            config_dir = os.environ.get("AI_PIPELINE_CONFIG_DIR")
            if config_dir:
                config_path = Path(config_dir) / "config" / "events.json"
            else:
                # Default to config/events.json relative to project root
                project_root = Path(__file__).parent.parent
                config_path = project_root / "config" / "events.json"
        
        self.config_path = Path(config_path)
        self.events: Dict[str, Event] = {}
        self.dependency_descriptions: Dict[str, str] = {}
        
        self._load_events()
    
    # Fallback dependency descriptions used when events.json is absent.
    # These are the canonical dependency labels — they don't change with markets.
    _FALLBACK_DEPENDENCY_DESCRIPTIONS: Dict[str, str] = {
        "training": "Model training progress, dataset preparation, convergence milestones",
        "compute": "GPU/TPU availability, infrastructure scaling, chip supply",
        "safety": "Red-teaming, alignment research, jailbreak testing, safety evaluations",
        "regulation": "Laws, policy decisions, compliance requirements, government oversight",
        "executive_statement": "CEO/CTO announcements, press releases, official statements",
        "public_narrative": "Media coverage, public sentiment, hype cycles, narrative shifts",
        "performance": "Athletic or competitive performance metrics and results",
        "injuries": "Player or team injury reports and health status",
        "transfers": "Player transfers, team composition changes",
        "schedule": "Event scheduling, fixture lists, calendar changes",
        "form": "Recent performance trends and momentum",
        "standings": "League tables, rankings, tournament brackets",
        "momentum": "Winning/losing streaks, psychological momentum",
        "playoffs": "Playoff qualification, bracket outcomes",
        "polling": "Survey data, approval ratings, voter sentiment",
        "fundraising": "Campaign finance, donor activity, funding rounds",
        "endorsements": "Public endorsements from key figures",
        "legislation": "Bills, laws, regulatory changes under consideration",
        "economic_data": "GDP, trade balance, economic indicators",
        "market_reaction": "Financial market responses to events",
        "geopolitics": "International relations, diplomatic events, conflicts",
        "us_monetary_policy": "Federal Reserve decisions, monetary policy signals",
        "inflation_data": "CPI, PPI, inflation measurements",
        "employment_data": "Jobs reports, unemployment figures",
        "interest_rates": "Central bank rate decisions and expectations",
        "central_bank": "Central bank communications and policy actions",
        "fed_policy": "Federal Reserve policy stance and communications",
        "cpi": "Consumer Price Index reports",
        "gdp": "Gross Domestic Product reports and revisions",
        "news": "General news coverage and developments",
        "sentiment": "Public and market sentiment indicators",
        "social_media": "Social media discussions and viral trends",
        "media_coverage": "Traditional and digital media coverage intensity",
    }

    def _load_events(self) -> None:
        """Load and validate events from configuration file.

        If events.json is not found, the registry starts empty and relies
        entirely on dynamic synthesis from Polymarket for any event lookups.
        This allows the system to run without manual event configuration.
        """
        logger.debug(f"Loading events from: {self.config_path}")
        if not self.config_path.exists():
            logger.warning(
                f"events.json not found at {self.config_path} — "
                f"starting with empty registry (dynamic synthesis only). "
                f"All markets will be resolved dynamically from Polymarket."
            )
            self.events = {}
            self.dependency_descriptions = dict(self._FALLBACK_DEPENDENCY_DESCRIPTIONS)
            return

        with open(self.config_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            logger.warning(
                f"events.json at {self.config_path} is empty — "
                f"starting with empty registry (dynamic synthesis only)."
            )
            self.events = {}
            self.dependency_descriptions = dict(self._FALLBACK_DEPENDENCY_DESCRIPTIONS)
            return
        config = json.loads(content)

        # Load dependency descriptions (fall back to hardcoded if not in file)
        self.dependency_descriptions = config.get(
            "dependency_descriptions", dict(self._FALLBACK_DEPENDENCY_DESCRIPTIONS)
        )

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
            polymarket_slug=data.get("polymarket_slug", ""),
            polymarket_token_id=data.get("polymarket_token_id"),
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
        """Get event by ID. If not found, dynamically synthesizes it from Polymarket."""
        event = self.events.get(event_id)
        if event is not None:
            return event
            
        logger.info(f"Event {event_id} not in registry. Synthesizing dynamically...")
        return self._resolve_dynamic_event(event_id)
        
    def _resolve_dynamic_event(self, slug: str) -> Optional[Event]:
        """Synthesize a dynamic Event object by fetching market metadata from Polymarket."""
        client = get_polymarket_client()
        market = client.get_market_by_slug(slug)
        
        if not market:
            logger.warning(f"Could not synthesize event: Market {slug} not found on Polymarket.")
            return None
            
        # Extract title as question or slug fallback
        title = market.question if market.question else slug.replace('-', ' ').title()
        
        # Heuristic entity extraction: split title words, grab capitalized ones, avoiding common stop words
        # This is a very rough fallback. A real NER model is used later in the pipeline anyway.
        stopwords = {"The", "A", "An", "Is", "Will", "Are", "By", "In", "On", "Of", "For", "To", "And"}
        words = title.split()
        entities = [w.strip('?.!,:;()') for w in words if w.istitle() and w not in stopwords]
        
        # Fallback if no capitalized words found (e.g. all lowercase title)
        if not entities:
            # Just use chunks of the slug
            entities = [part.title() for part in slug.split('-')[:4]]
            
        # Ensure we have primary and secondary entities
        primary = entities[:2] if len(entities) >= 1 else ["Polymarket", "Event"]
        secondary = entities[2:] if len(entities) > 2 else []
        
        # Add basic geography or thematic terms as fallback secondaries
        for term in ["politics", "election", "market", "policy", "law", "news"]:
            if term not in secondary and term.title() not in secondary:
                secondary.append(term)
        
        # Determine aliases
        aliases = [title]
        # Add individual slug parts as potential aliases for very flexible matching
        for part in slug.split('-'):
            if part and len(part) > 3 and part not in aliases:
                aliases.append(part)
        
        # Deadline handling - must be timezone unaware
        deadline = market.end_date
        if not deadline:
            # Fallback to end of year if no deadline is specified
            now = _utc_now()
            deadline = datetime(now.year, 12, 31, 23, 59, 59)
            
        # Ensure it is naive
        if deadline.tzinfo is not None:
            deadline = deadline.replace(tzinfo=None)
            
        # Build synthetic event
        synthetic_event = Event(
            event_id=slug,
            event_type="dynamic",  # Custom generated event type
            event_title=title,
            event_description=f"Automated event mapped from Polymarket: {title}",
            primary_entities=primary,
            secondary_entities=secondary,
            aliases=aliases,
            deadline=deadline,
            dependencies=["news", "sentiment"],  # Generic dependencies
            polymarket_slug=slug,
            polymarket_token_id=None # We might not have token_id easily, that's fine for the AI pipeline
        )
        
        # Validate and store locally
        try:
            self.events[slug] = synthetic_event
            logger.info(f"Successfully synthesized and cached Event: {title} with entities {synthetic_event.primary_entities}")
            return synthetic_event
        except Exception as e:
            logger.error(f"Failed to validate dynamically synthesized event {slug}: {e}")
            return None
    
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
