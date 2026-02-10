"""
Event Extractor

Converts NormalizedDocument + Signal to EventNode schema.
Events that cannot be represented in the schema are discarded.

This is a rule-based extraction - NO ML or LLM used here.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.normalizer import NormalizedDocument, DocumentNormalizer
from pipeline.signal_extractor import Signal, SignalExtractor
from belief_graph.models import EventNode, EventType, ScopeType

logger = logging.getLogger(__name__)


# Event type mapping from source_type and signal_type
# This is rule-based as per the spec
EVENT_TYPE_MAPPING: Dict[Tuple[str, str], EventType] = {
    # (source_type, signal_type) -> event_type
    ("official", "regulation_update"): "policy",
    ("official", "official_confirmation"): "policy",
    ("official", "executive_statement"): "signal",
    ("official", "training_progress"): "signal",
    ("official", "delay"): "signal",
    ("official", "rumor"): "signal",
    
    ("journalist", "regulation_update"): "policy",
    ("journalist", "official_confirmation"): "narrative",
    ("journalist", "executive_statement"): "narrative",
    ("journalist", "training_progress"): "narrative",
    ("journalist", "delay"): "narrative",
    ("journalist", "rumor"): "narrative",
    ("journalist", "narrative_shift"): "narrative",
    
    ("social", "rumor"): "signal",
    ("social", "narrative_shift"): "narrative",
    ("social", "training_progress"): "signal",
    
    ("forum", "rumor"): "signal",
    ("forum", "narrative_shift"): "narrative",
    ("forum", "training_progress"): "signal",
    
    ("research", "training_progress"): "signal",
    ("research", "official_confirmation"): "signal",
}

# Keywords for event type inference when mapping not found
EVENT_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "policy": [
        "regulation", "policy", "law", "legislation", "mandate", "directive",
        "ban", "enforce", "compliance", "requirement", "rule", "act", "bill",
        "government", "congress", "parliament", "eu ai act", "executive order"
    ],
    "legal": [
        "court", "lawsuit", "litigation", "ruling", "judge", "verdict",
        "settlement", "legal", "liability", "sue", "trial", "case", "attorney",
        "plaintiff", "defendant", "injunction", "appeal"
    ],
    "economic": [
        "funding", "investment", "revenue", "profit", "stock", "market",
        "valuation", "billion", "million", "ipo", "acquisition", "merger",
        "financial", "quarterly", "earnings", "growth", "budget", "cost"
    ],
    "poll": [
        "poll", "survey", "vote", "election", "approval", "rating",
        "public opinion", "sentiment", "percent support", "approval rating"
    ],
    "narrative": [
        "report", "analysis", "opinion", "perspective", "commentary",
        "article", "coverage", "discussion", "debate", "narrative"
    ],
    "market": [
        "polymarket", "prediction market", "odds", "probability", "betting",
        "trading", "position", "liquidity", "price"
    ],
    "signal": [
        "announcement", "release", "launch", "delay", "progress", "update",
        "development", "milestone", "achievement", "breakthrough", "rumor"
    ]
}

# Scope detection patterns
SCOPE_PATTERNS: Dict[ScopeType, List[str]] = {
    "global": [
        "global", "worldwide", "international", "world", "universal",
        "multinational", "cross-border"
    ],
    "national": [
        "national", "country", "federal", "domestic", "nationwide",
        "united states", "u.s.", "usa", "uk", "china", "india", "japan",
        "germany", "france", "canada", "australia", "brazil"
    ],
    "local": [
        "local", "regional", "state", "city", "municipal", "county",
        "provincial", "district"
    ]
}

# Domain to scope mapping
DOMAIN_SCOPE: Dict[str, ScopeType] = {
    # US government
    ".gov": "national",
    "whitehouse.gov": "national",
    "congress.gov": "national",
    
    # EU
    "europa.eu": "global",
    "ec.europa.eu": "global",
    
    # UK
    "gov.uk": "national",
    
    # Global organizations
    "un.org": "global",
    "who.int": "global",
    "worldbank.org": "global",
    "imf.org": "global",
    "oecd.org": "global",
}


class EventExtractor:
    """
    Extracts EventNodes from NormalizedDocuments and Signals.
    
    Uses rule-based classification for event type, scope, and other fields.
    Events that cannot fit the schema are discarded.
    """
    
    def __init__(
        self,
        normalizer: Optional[DocumentNormalizer] = None,
        signal_extractor: Optional[SignalExtractor] = None
    ):
        """
        Initialize event extractor.
        
        Args:
            normalizer: Document normalizer for loading documents
            signal_extractor: Signal extractor for loading signals
        """
        logger.info("Initializing EventExtractor...")
        self.normalizer = normalizer or DocumentNormalizer()
        self.signal_extractor = signal_extractor
        logger.info("EventExtractor initialized")
    
    def classify_event_type(
        self,
        doc: NormalizedDocument,
        signal: Optional[Signal] = None
    ) -> Optional[EventType]:
        """
        Classify event type from document and signal.
        
        Rule-based classification - NO ML or LLM.
        
        Args:
            doc: Normalized document
            signal: Optional associated signal
        
        Returns:
            Event type or None if cannot classify
        """
        source_type = doc.source_type.lower()
        
        # Try mapping from signal type
        if signal:
            signal_type = signal.signal_type.lower()
            key = (source_type, signal_type)
            
            if key in EVENT_TYPE_MAPPING:
                return EVENT_TYPE_MAPPING[key]
        
        # Check for legal entities in extracted entities
        has_legal_entities = any(
            e.get("type") in ("ORG", "LAW") and
            any(kw in e.get("text", "").lower() for kw in ["court", "law", "legal", "judge"])
            for e in doc.extracted_entities
        )
        if has_legal_entities:
            return "legal"
        
        # Check for economic entities
        has_economic_entities = any(
            e.get("type") == "MONEY" or
            (e.get("type") == "ORG" and 
             any(kw in e.get("text", "").lower() for kw in ["bank", "fund", "capital"]))
            for e in doc.extracted_entities
        )
        if has_economic_entities:
            return "economic"
        
        # Keyword-based classification on title and text
        text_lower = f"{doc.title} {doc.raw_text[:1000]}".lower()
        
        for event_type, keywords in EVENT_TYPE_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches >= 2:  # Require at least 2 keyword matches
                return event_type  # type: ignore
        
        # Default based on source type
        source_defaults: Dict[str, EventType] = {
            "official": "policy",
            "journalist": "narrative",
            "social": "signal",
            "forum": "signal",
            "research": "signal"
        }
        
        return source_defaults.get(source_type, "narrative")
    
    def parse_action_object(
        self,
        title: str,
        text: str = ""
    ) -> Tuple[str, str]:
        """
        Extract action and object from title.
        
        Uses pattern matching to extract verb and object.
        
        Args:
            title: Document title
            text: Optional document text for context
        
        Returns:
            Tuple of (action, object)
        """
        # Clean title
        title = title.strip()
        
        # Common action patterns
        action_patterns = [
            r"^(\w+(?:\s+\w+)?)\s+(?:to\s+)?(.+)$",  # "announces to launch X"
            r"^(.+?)\s+(launches?|releases?|announces?|confirms?|delays?)\s+(.+)$",
            r"^(.+?)\s+(will|may|could)\s+(.+)$",
        ]
        
        # Try to extract using patterns
        for pattern in action_patterns:
            match = re.match(pattern, title, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    return groups[0], groups[-1]
        
        # Fallback: use first few words as action, rest as object
        words = title.split()
        if len(words) >= 3:
            return " ".join(words[:2]), " ".join(words[2:])
        elif len(words) == 2:
            return words[0], words[1]
        elif len(words) == 1:
            return words[0], ""
        
        return title, ""
    
    def determine_scope(
        self,
        url: str,
        entities: List[Dict],
        text: str = ""
    ) -> ScopeType:
        """
        Determine geographic/jurisdictional scope of event.
        
        Args:
            url: Document URL
            entities: Extracted entities
            text: Document text
        
        Returns:
            Scope type
        """
        # Check URL domain first
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            for pattern, scope in DOMAIN_SCOPE.items():
                if pattern in domain:
                    return scope
        except Exception:
            pass
        
        # Check entities for geographic locations
        location_entities = [
            e.get("text", "").lower()
            for e in entities
            if e.get("type") in ("GPE", "LOC", "NORP")
        ]
        
        # Check for global indicators
        for loc in location_entities:
            for pattern in SCOPE_PATTERNS["global"]:
                if pattern in loc:
                    return "global"
        
        # Check for national indicators
        for loc in location_entities:
            for pattern in SCOPE_PATTERNS["national"]:
                if pattern in loc:
                    return "national"
        
        # Check text for scope keywords
        text_lower = text[:1000].lower()
        
        for scope, patterns in SCOPE_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return scope  # type: ignore
        
        # Default to global for AI-related events
        return "global"
    
    def calculate_certainty(
        self,
        doc: NormalizedDocument,
        signal: Optional[Signal] = None,
        cluster_size: int = 1
    ) -> float:
        """
        Calculate evidence-based certainty score for event.
        
        Components:
        1. Source credibility (0.4-0.8 base)
        2. Entity extraction quality (0-0.1 boost)
        3. Cluster corroboration (0-0.1 boost)
        4. Signal confidence (0-0.1 boost)
        5. Content quality penalty for low-quality text
        
        Args:
            doc: Normalized document
            signal: Optional signal
            cluster_size: Number of corroborating sources
        
        Returns:
            Certainty score 0.4-0.95
        """
        # Component 1: Base source credibility (40-80%)
        # More conservative than before
        source_base = {
            "official": 0.80,  # Government, official org sites
            "research": 0.75,  # Academic, research institutions
            "journalist": 0.65,  # News outlets
            "forum": 0.45,  # Forums like HackerNews
            "social": 0.40   # Twitter, Reddit
        }.get(doc.source_type.lower(), 0.50)
        
        # Component 2: Entity extraction quality boost (0-10%)
        # More entities = more verifiable claims = higher certainty
        entity_count = len(doc.extracted_entities)
        entity_boost = min(0.10, entity_count * 0.01)  # 1% per entity, max 10%
        
        # Component 3: Cluster corroboration boost (0-10%)
        # Multiple sources reporting same event = higher certainty
        if cluster_size > 1:
            corroboration_boost = min(0.10, (cluster_size - 1) * 0.03)  # 3% per extra source
        else:
            corroboration_boost = 0.0
        
        # Component 4: Signal confidence boost (0-10%)
        signal_boost = 0.0
        if signal:
            # Only boost for high-confidence signals
            if signal.confidence > 0.6:
                signal_boost = min(0.10, signal.confidence * 0.12)
            
            # Penalty for speculation/rumor type signals
            if signal.signal_type in ("rumor", "speculation"):
                signal_boost = -0.10  # Reduce certainty
        
        # Component 5: Content quality check
        content_penalty = 0.0
        if doc.raw_text:
            text_len = len(doc.raw_text)
            # Very short text = less reliable
            if text_len < 200:
                content_penalty = -0.10
            elif text_len < 500:
                content_penalty = -0.05
            
            # Check for uncertainty language
            uncertainty_words = ["rumor", "reportedly", "allegedly", "unconfirmed", "speculation"]
            text_lower = doc.raw_text[:500].lower()
            uncertainty_count = sum(1 for w in uncertainty_words if w in text_lower)
            content_penalty -= uncertainty_count * 0.03  # 3% penalty per uncertainty word
        
        # Calculate total certainty
        certainty = (
            source_base +
            entity_boost +
            corroboration_boost +
            signal_boost +
            content_penalty
        )
        
        # Clamp to valid range
        return max(0.40, min(0.95, certainty))
    
    def extract_actors(
        self,
        doc: NormalizedDocument
    ) -> List[str]:
        """
        Extract actor names from document entities.
        
        Args:
            doc: Normalized document
        
        Returns:
            List of actor names
        """
        actors = []
        seen = set()
        
        for entity in doc.extracted_entities:
            entity_type = entity.get("type", "")
            entity_text = entity.get("text", "").strip()
            
            # Only include organizations and persons
            if entity_type not in ("ORG", "PER", "PERSON"):
                continue
            
            # Skip duplicates (case-insensitive)
            if entity_text.lower() in seen:
                continue
            
            # Skip very short names
            if len(entity_text) < 2:
                continue
            
            actors.append(entity_text)
            seen.add(entity_text.lower())
        
        return actors[:10]  # Limit to top 10 actors
    
    def extract_event_node(
        self,
        doc: NormalizedDocument,
        signal: Optional[Signal] = None
    ) -> Optional[EventNode]:
        """
        Convert normalized document to EventNode schema.
        
        Discards documents that cannot fit the schema.
        
        Args:
            doc: Normalized document
            signal: Optional associated signal
        
        Returns:
            EventNode or None if cannot convert
        """
        logger.debug(f"Extracting EventNode from document: {doc.doc_id}")
        
        # Classify event type
        event_type = self.classify_event_type(doc, signal)
        if event_type is None:
            logger.debug(f"Could not classify event type for {doc.doc_id}, discarding")
            return None
        
        # Check for timestamp - required for graph ordering
        if doc.timestamp is None:
            logger.debug(f"No timestamp for {doc.doc_id}, discarding")
            return None
        
        # Extract actors
        actors = self.extract_actors(doc)
        
        # Parse action and object from title
        action, obj = self.parse_action_object(doc.title, doc.raw_text)
        
        # Determine scope
        scope = self.determine_scope(doc.url, doc.extracted_entities, doc.raw_text)
        
        # Calculate certainty
        certainty = self.calculate_certainty(doc, signal)
        
        # Generate event ID
        event_id = f"evt_{doc.doc_id[:16]}"
        
        # Determine source string
        try:
            parsed = urlparse(doc.url)
            source = parsed.netloc.replace("www.", "")
        except Exception:
            source = doc.source_type
        
        try:
            event_node = EventNode(
                event_id=event_id,
                event_type=event_type,
                timestamp=doc.timestamp,
                actors=actors,
                action=action[:200] if action else "Unknown action",
                object=obj[:200] if obj else "Unknown",
                certainty=round(certainty, 4),
                source=source,
                scope=scope,
                source_doc_id=doc.doc_id,
                source_signal_id=signal.signal_id if signal else None,
                raw_title=doc.title,
                url=doc.url
            )
            
            logger.debug(
                f"Created EventNode: {event_id}, type={event_type}, "
                f"certainty={certainty:.3f}, scope={scope}"
            )
            
            return event_node
            
        except Exception as e:
            logger.warning(f"Failed to create EventNode for {doc.doc_id}: {e}")
            return None
    
    def extract_events_for_belief(
        self,
        belief_event_id: str,
        max_events: int = 100
    ) -> List[EventNode]:
        """
        Extract all EventNodes relevant to a belief event.
        
        Args:
            belief_event_id: Event ID from the event registry
            max_events: Maximum number of events to extract
        
        Returns:
            List of EventNodes
        """
        logger.info(f"Extracting events for belief: {belief_event_id}")
        
        # Load all normalized documents for the event
        documents = self.normalizer.load_all_for_event(belief_event_id)
        logger.info(f"Found {len(documents)} documents for event {belief_event_id}")
        
        # Load signals if extractor available
        signals_by_doc: Dict[str, Signal] = {}
        if self.signal_extractor:
            signals = self.signal_extractor.load_signals_for_event(belief_event_id)
            signals_by_doc = {s.doc_id: s for s in signals}
            logger.info(f"Found {len(signals)} signals for event {belief_event_id}")
        
        # Extract EventNodes
        event_nodes: List[EventNode] = []
        for doc in documents:
            signal = signals_by_doc.get(doc.doc_id)
            event_node = self.extract_event_node(doc, signal)
            
            if event_node:
                event_nodes.append(event_node)
            
            if len(event_nodes) >= max_events:
                break
        
        logger.info(f"Extracted {len(event_nodes)} EventNodes for belief {belief_event_id}")
        
        return event_nodes
