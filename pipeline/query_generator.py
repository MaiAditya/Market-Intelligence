"""
Query Generator Module

Generates search queries for each event using rule-based template expansion.
No ML involved - pure deterministic string templating.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .event_registry import Event, EventRegistry, get_registry

logger = logging.getLogger(__name__)


@dataclass
class GeneratedQuery:
    """Represents a generated search query."""
    event_id: str
    query: str
    query_type: str  # official | journalist | public_opinion | critical
    expected_bias: str  # positive | neutral | negative
    template_used: str
    entity_used: str


@dataclass
class EventQuerySet:
    """Collection of queries for a single event."""
    event_id: str
    event_title: str
    queries: List[GeneratedQuery]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "event_title": self.event_title,
            "queries": [
                {
                    "query": q.query,
                    "query_type": q.query_type,
                    "expected_bias": q.expected_bias,
                    "template_used": q.template_used,
                    "entity_used": q.entity_used
                }
                for q in self.queries
            ]
        }


class QueryGenerator:
    """
    Generates search queries for events using template expansion.
    
    Query families:
    - official: Factual announcements from official sources
    - journalist: Insider reports and journalistic coverage
    - public_opinion: Social media and forum discussions
    - critical: Negative framing, delays, problems
    
    No ML here - pure deterministic template substitution.
    """
    
    def __init__(self, templates_path: Optional[str] = None):
        """
        Initialize the query generator.
        
        Args:
            templates_path: Path to query_templates.json. Uses default if None.
        """
        if templates_path is None:
            project_root = Path(__file__).parent.parent
            templates_path = project_root / "config" / "query_templates.json"
        
        self.templates_path = Path(templates_path)
        self.templates: Dict = {}
        self.event_type_templates: Dict = {}
        
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load query templates from configuration."""
        logger.debug(f"Loading query templates from: {self.templates_path}")
        if not self.templates_path.exists():
            raise FileNotFoundError(f"Query templates not found: {self.templates_path}")
        
        with open(self.templates_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.templates = config.get("query_families", {})
        self.event_type_templates = config.get("event_type_specific_templates", {})
        logger.info(f"Loaded {len(self.templates)} query families")
    
    def _expand_template(
        self,
        template: str,
        primary_entity: str,
        secondary_entity: str = ""
    ) -> str:
        """
        Expand a template with entity substitution.
        
        Args:
            template: Template string with {primary_entity} and {secondary_entity} placeholders
            primary_entity: Main entity to substitute
            secondary_entity: Supporting entity (optional)
        
        Returns:
            Expanded query string
        """
        result = template.replace("{primary_entity}", primary_entity)
        result = result.replace("{secondary_entity}", secondary_entity)
        # Clean up any double spaces from empty secondary entity
        result = " ".join(result.split())
        return result
    
    def generate_queries_for_event(
        self,
        event: Event,
        max_queries_per_family: int = 5
    ) -> EventQuerySet:
        """
        Generate all query families for a single event.
        
        Args:
            event: The event to generate queries for
            max_queries_per_family: Maximum number of queries per family
        
        Returns:
            EventQuerySet containing all generated queries
        """
        queries: List[GeneratedQuery] = []
        
        # Get primary and secondary entities for substitution
        primary_entities = event.primary_entities[:2]  # Use top 2 primary entities
        secondary_entities = event.secondary_entities[:2]  # Use top 2 secondary
        
        # Generate queries for each family
        for family_name, family_config in self.templates.items():
            family_queries = self._generate_family_queries(
                event=event,
                family_name=family_name,
                templates=family_config.get("templates", []),
                expected_bias=family_config.get("expected_bias", "neutral"),
                primary_entities=primary_entities,
                secondary_entities=secondary_entities,
                max_queries=max_queries_per_family
            )
            queries.extend(family_queries)
        
        # Add event-type-specific templates
        event_type_config = self.event_type_templates.get(event.event_type, {})
        for family_name, templates in event_type_config.items():
            # Get expected bias from main family config
            expected_bias = self.templates.get(family_name, {}).get("expected_bias", "neutral")
            
            type_queries = self._generate_family_queries(
                event=event,
                family_name=family_name,
                templates=templates,
                expected_bias=expected_bias,
                primary_entities=primary_entities,
                secondary_entities=secondary_entities,
                max_queries=2  # Fewer for type-specific
            )
            queries.extend(type_queries)
        
        logger.debug(f"Generated {len(queries)} queries for event {event.event_id}")
        return EventQuerySet(
            event_id=event.event_id,
            event_title=event.event_title,
            queries=queries
        )
    
    def _generate_family_queries(
        self,
        event: Event,
        family_name: str,
        templates: List[str],
        expected_bias: str,
        primary_entities: List[str],
        secondary_entities: List[str],
        max_queries: int
    ) -> List[GeneratedQuery]:
        """Generate queries for a single family."""
        queries = []
        seen_queries = set()  # Deduplicate
        
        for template in templates:
            if len(queries) >= max_queries:
                break
            
            # Try with different entity combinations
            for primary in primary_entities:
                if len(queries) >= max_queries:
                    break
                
                # First try without secondary entity
                query_text = self._expand_template(template, primary, "")
                if query_text.lower() not in seen_queries:
                    seen_queries.add(query_text.lower())
                    queries.append(GeneratedQuery(
                        event_id=event.event_id,
                        query=query_text,
                        query_type=family_name,
                        expected_bias=expected_bias,
                        template_used=template,
                        entity_used=primary
                    ))
                
                # Then try with secondary entities if template uses it
                if "{secondary_entity}" in template:
                    for secondary in secondary_entities:
                        if len(queries) >= max_queries:
                            break
                        
                        query_text = self._expand_template(template, primary, secondary)
                        if query_text.lower() not in seen_queries:
                            seen_queries.add(query_text.lower())
                            queries.append(GeneratedQuery(
                                event_id=event.event_id,
                                query=query_text,
                                query_type=family_name,
                                expected_bias=expected_bias,
                                template_used=template,
                                entity_used=f"{primary} + {secondary}"
                            ))
        
        return queries
    
    def generate_all_queries(
        self,
        registry: Optional[EventRegistry] = None,
        max_queries_per_family: int = 5
    ) -> Dict[str, EventQuerySet]:
        """
        Generate queries for all events in the registry.
        
        Args:
            registry: EventRegistry instance. Uses default if None.
            max_queries_per_family: Maximum queries per family per event
        
        Returns:
            Dictionary mapping event_id to EventQuerySet
        """
        if registry is None:
            registry = get_registry()
        
        result = {}
        for event in registry:
            query_set = self.generate_queries_for_event(event, max_queries_per_family)
            result[event.event_id] = query_set
        
        return result
    
    def get_queries_by_type(
        self,
        event: Event,
        query_type: str
    ) -> List[GeneratedQuery]:
        """
        Get queries of a specific type for an event.
        
        Args:
            event: The event
            query_type: One of: official, journalist, public_opinion, critical
        
        Returns:
            List of queries matching the type
        """
        query_set = self.generate_queries_for_event(event)
        return [q for q in query_set.queries if q.query_type == query_type]
    
    def to_json(self, queries: Dict[str, EventQuerySet]) -> str:
        """Convert query sets to JSON string."""
        return json.dumps(
            {event_id: qs.to_dict() for event_id, qs in queries.items()},
            indent=2
        )


# Module-level convenience function
def generate_queries_for_all_events(
    max_queries_per_family: int = 5
) -> Dict[str, EventQuerySet]:
    """
    Generate queries for all registered events.
    
    Args:
        max_queries_per_family: Max queries per family per event
    
    Returns:
        Dictionary of event_id -> EventQuerySet
    """
    generator = QueryGenerator()
    return generator.generate_all_queries(max_queries_per_family=max_queries_per_family)
