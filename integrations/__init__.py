"""
Integrations Module

External service integrations:
- Polymarket API client
- Source credibility registry
"""

from .polymarket_client import PolymarketClient, get_polymarket_client, fetch_event_probability
from .source_registry import (
    get_source_credibility,
    get_source_type_credibility,
    get_author_credibility,
    is_official_source,
    is_trusted_source
)

__all__ = [
    "PolymarketClient",
    "get_polymarket_client",
    "fetch_event_probability",
    "get_source_credibility",
    "get_source_type_credibility",
    "get_author_credibility",
    "is_official_source",
    "is_trusted_source"
]
