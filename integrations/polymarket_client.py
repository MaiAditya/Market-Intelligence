"""
Polymarket Client

Fetches current probabilities from Polymarket API for tracked events.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional
import requests


def _utc_now() -> datetime:
    """Get current UTC time as a naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Polymarket market data."""
    slug: str
    question: str
    probability: float
    volume: float
    liquidity: float
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    outcome_prices: Dict[str, float]
    fetched_at: datetime
    
    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "question": self.question,
            "probability": self.probability,
            "volume": self.volume,
            "liquidity": self.liquidity,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "outcome_prices": self.outcome_prices,
            "fetched_at": self.fetched_at.isoformat()
        }


class PolymarketClient:
    """
    Client for fetching data from Polymarket.
    
    Uses the public Polymarket API (CLOB/Gamma API).
    """
    
    BASE_URL = "https://clob.polymarket.com"
    GAMMA_URL = "https://gamma-api.polymarket.com"
    
    def __init__(self, timeout: int = 30):
        """
        Initialize Polymarket client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "AIMarketIntelligence/1.0",
            "Accept": "application/json"
        })
    
    def get_market_by_slug(self, slug: str) -> Optional[MarketData]:
        """
        Get market data by slug.
        
        Args:
            slug: Polymarket event slug
        
        Returns:
            MarketData or None if not found
        """
        try:
            # Try gamma API for event data
            url = f"{self.GAMMA_URL}/events?slug={slug}"
            response = self._session.get(url, timeout=self.timeout)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch market {slug}: HTTP {response.status_code}")
                return None
            
            data = response.json()
            
            if not data:
                logger.warning(f"No market found for slug: {slug}")
                return None
            
            # Handle both single event and list response
            event = data[0] if isinstance(data, list) else data
            
            # Extract market info
            markets = event.get("markets", [])
            if not markets:
                # Single market event
                probability = self._extract_probability(event)
                return MarketData(
                    slug=slug,
                    question=event.get("title", ""),
                    probability=probability,
                    volume=float(event.get("volume", 0) or 0),
                    liquidity=float(event.get("liquidity", 0) or 0),
                    start_date=(
                        self._parse_date(event.get("startDate")) or
                        self._parse_date(event.get("createdAt")) or
                        self._parse_date(event.get("creationDate"))
                    ),
                    end_date=self._parse_date(event.get("endDate")),
                    outcome_prices={"yes": probability, "no": 1 - probability},
                    fetched_at=_utc_now()
                )
            
            # Multi-market event - get primary market
            primary_market = markets[0]
            probability = self._extract_probability(primary_market)
            
            return MarketData(
                slug=slug,
                question=event.get("title", primary_market.get("question", "")),
                probability=probability,
                volume=float(primary_market.get("volume", 0) or 0),
                liquidity=float(primary_market.get("liquidity", 0) or 0),
                start_date=(
                    self._parse_date(event.get("startDate")) or
                    self._parse_date(event.get("createdAt")) or
                    self._parse_date(event.get("creationDate"))
                ),
                end_date=self._parse_date(event.get("endDate")),
                outcome_prices={"yes": probability, "no": 1 - probability},
                fetched_at=_utc_now()
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching market {slug}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching market {slug}: {e}")
            return None
    
    def _extract_probability(self, market_data: dict) -> float:
        """Extract probability from market data."""
        # Try different fields
        if "outcomePrices" in market_data:
            prices = market_data["outcomePrices"]
            if isinstance(prices, list) and len(prices) > 0:
                return float(prices[0])
            elif isinstance(prices, dict):
                return float(prices.get("Yes", prices.get("yes", 0.5)))
        
        if "bestAsk" in market_data:
            return float(market_data["bestAsk"])
        
        if "price" in market_data:
            return float(market_data["price"])
        
        # Default
        return 0.5
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None
        
        try:
            # Try ISO format
            return datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except (ValueError, TypeError):
            return None
    
    def search_markets(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for markets by query.
        
        Args:
            query: Search query
            limit: Max results
        
        Returns:
            List of market dictionaries
        """
        try:
            url = f"{self.GAMMA_URL}/events?title_contains={query}&limit={limit}"
            response = self._session.get(url, timeout=self.timeout)
            
            if response.status_code != 200:
                return []
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def get_probability(self, slug: str) -> Optional[float]:
        """
        Get current probability for a single event.
        
        Args:
            slug: Polymarket event slug
        
        Returns:
            Probability as float or None if not found
        """
        market = self.get_market_by_slug(slug)
        return market.probability if market else None
    
    def get_probabilities_for_events(
        self,
        event_slugs: Dict[str, str]
    ) -> Dict[str, Optional[float]]:
        """
        Get current probabilities for multiple events.
        
        Args:
            event_slugs: Dict mapping event_id to polymarket_slug
        
        Returns:
            Dict mapping event_id to probability (or None if not found)
        """
        results = {}
        
        for event_id, slug in event_slugs.items():
            market = self.get_market_by_slug(slug)
            if market:
                results[event_id] = market.probability
            else:
                results[event_id] = None
        
        return results


# Module-level singleton
_client: Optional[PolymarketClient] = None


def get_polymarket_client() -> PolymarketClient:
    """Get Polymarket client singleton."""
    global _client
    if _client is None:
        _client = PolymarketClient()
    return _client


def fetch_event_probability(slug: str) -> Optional[float]:
    """Convenience function to fetch probability for a single event."""
    client = get_polymarket_client()
    market = client.get_market_by_slug(slug)
    return market.probability if market else None
