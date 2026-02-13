"""
Polymarket Event Impact Analyzer

Measures actual market price impact of identified news events on Polymarket
prediction markets. For each event node, fetches historical price data and
calculates the price delta in a configurable window around the event timestamp.

No ML/LLM. Deterministic. Fully logged.
"""

import json
import logging
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("impact_analyzer")


@dataclass
class PricePoint:
    """A single price point from Polymarket."""
    timestamp: int       # Unix timestamp
    price: float         # Token price [0, 1]
    
    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)


@dataclass
class MarketToken:
    """A market token with metadata."""
    token_id: str
    question: str
    outcome: str          # "Yes" or "No"
    current_price: float
    market_slug: str


@dataclass
class EventImpact:
    """Impact measurement for a single event on a market."""
    event_id: str
    event_type: str
    event_timestamp: str
    raw_title: str
    
    # Price data
    price_before: Optional[float] = None
    price_after: Optional[float] = None
    delta: Optional[float] = None
    pct_change: Optional[float] = None
    
    # Metadata
    window_minutes: int = 2
    data_quality: str = "missing"     # exact | interpolated | missing
    before_offset_seconds: Optional[int] = None   # How far before event the price point is
    after_offset_seconds: Optional[int] = None     # How far after event the price point is
    token_id: Optional[str] = None
    market_slug: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class PolymarketImpactAnalyzer:
    """
    Analyzes price impact of events on Polymarket prediction markets.
    
    Workflow:
    1. Fetch event data from Gamma API using slug
    2. Extract token IDs for each market
    3. Fetch price history from CLOB API
    4. For each event node timestamp, calculate price delta in ±window
    5. Generate timeline of impacts
    """
    
    GAMMA_URL = "https://gamma-api.polymarket.com"
    CLOB_URL = "https://clob.polymarket.com"
    
    def __init__(
        self,
        window_minutes: int = 2,
        fidelity: int = 1,
        output_dir: Optional[Path] = None,
        timeout: int = 30
    ):
        """
        Args:
            window_minutes: ±minutes around event to measure impact
            fidelity: Price history granularity in minutes
            output_dir: Directory for output files
            timeout: HTTP request timeout
        """
        self.window_minutes = window_minutes
        self.fidelity = fidelity
        self.output_dir = Path(output_dir or "data/impact_analysis")
        self.timeout = timeout
        
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "AIMarketIntelligence/1.0",
            "Accept": "application/json"
        })
        
        # Cache for price history
        self._price_cache: Dict[str, List[PricePoint]] = {}
        
        # Setup output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure structured logging."""
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(console)
        
        # File handler for structured logs
        log_file = self.output_dir / "impact_analysis.log"
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
    
    def _log_json(self, event_type: str, data: dict):
        """Log structured JSON data."""
        log_entry = {"log_type": event_type, **data}
        logger.info(json.dumps(log_entry))
    
    # ──────────────────────────────────────────────
    # API: Fetch Event & Token Data
    # ──────────────────────────────────────────────
    
    def fetch_event_tokens(self, slug: str) -> List[MarketToken]:
        """
        Fetch market tokens for a Polymarket event by slug.
        
        Args:
            slug: Polymarket event slug
        
        Returns:
            List of MarketToken objects
        """
        url = f"{self.GAMMA_URL}/events/slug/{slug}"
        logger.info(f"Fetching event data: {url}")
        
        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
            event_data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch event {slug}: {e}")
            return []
        
        self._log_json("event_fetched", {
            "slug": slug,
            "title": event_data.get("title", ""),
            "markets_count": len(event_data.get("markets", []))
        })
        
        tokens = []
        for market in event_data.get("markets", []):
            # Parse clobTokenIds - stored as JSON string
            token_ids_raw = market.get("clobTokenIds", "[]")
            if isinstance(token_ids_raw, str):
                token_ids = json.loads(token_ids_raw)
            else:
                token_ids = token_ids_raw
            
            # Parse outcomes
            outcomes_raw = market.get("outcomes", "[]")
            if isinstance(outcomes_raw, str):
                outcomes = json.loads(outcomes_raw)
            else:
                outcomes = outcomes_raw
            
            # Parse prices
            prices_raw = market.get("outcomePrices", "[]")
            if isinstance(prices_raw, str):
                prices = json.loads(prices_raw)
            else:
                prices = prices_raw
            
            question = market.get("question", "")
            market_slug = market.get("slug", slug)
            
            for i, token_id in enumerate(token_ids):
                outcome = outcomes[i] if i < len(outcomes) else f"outcome_{i}"
                price = float(prices[i]) if i < len(prices) else 0.5
                
                token = MarketToken(
                    token_id=token_id,
                    question=question,
                    outcome=outcome,
                    current_price=price,
                    market_slug=market_slug
                )
                tokens.append(token)
                
                self._log_json("token_found", {
                    "token_id": token_id[:20] + "...",
                    "question": question,
                    "outcome": outcome,
                    "current_price": price
                })
        
        logger.info(f"Found {len(tokens)} tokens for {slug}")
        return tokens
    
    # ──────────────────────────────────────────────
    # API: Fetch Price History
    # ──────────────────────────────────────────────
    
    def fetch_price_history(
        self,
        token_id: str,
        interval: str = "max",
        fidelity: Optional[int] = None
    ) -> List[PricePoint]:
        """
        Fetch historical price data for a token.
        
        Args:
            token_id: CLOB token ID
            interval: Time interval (max, 1d, 1w, 1m)
            fidelity: Granularity in minutes (1 = per minute)
        
        Returns:
            List of PricePoint sorted by timestamp
        """
        # Check cache
        cache_key = f"{token_id}_{interval}_{fidelity}"
        if cache_key in self._price_cache:
            logger.info(f"Using cached price history for token {token_id[:20]}...")
            return self._price_cache[cache_key]
        
        fidelity = fidelity or self.fidelity
        url = f"{self.CLOB_URL}/prices-history"
        params = {
            "market": token_id,
            "interval": interval,
            "fidelity": fidelity
        }
        
        logger.info(f"Fetching price history: token={token_id[:20]}..., interval={interval}, fidelity={fidelity}")
        
        try:
            response = self._session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch price history for {token_id[:20]}...: {e}")
            return []
        
        history = data.get("history", [])
        
        price_points = [
            PricePoint(timestamp=int(p["t"]), price=float(p["p"]))
            for p in history
        ]
        
        # Sort by timestamp (should already be sorted, but ensure)
        price_points.sort(key=lambda p: p.timestamp)
        
        # Cache it
        self._price_cache[cache_key] = price_points
        
        self._log_json("price_history_fetched", {
            "token_id": token_id[:20] + "...",
            "data_points": len(price_points),
            "time_range_start": price_points[0].dt.isoformat() if price_points else None,
            "time_range_end": price_points[-1].dt.isoformat() if price_points else None
        })
        
        logger.info(f"Fetched {len(price_points)} price points")
        return price_points
    
    # ──────────────────────────────────────────────
    # Core: Calculate Impact
    # ──────────────────────────────────────────────
    
    def calculate_impact(
        self,
        price_history: List[PricePoint],
        event_timestamp: datetime,
        window_minutes: Optional[int] = None
    ) -> dict:
        """
        Calculate price impact around an event timestamp.
        
        Finds the closest price points before and after the event within
        the ±window and calculates the delta.
        
        Args:
            price_history: Sorted list of PricePoints
            event_timestamp: Event timestamp (UTC)
            window_minutes: ±minutes window (default: self.window_minutes)
        
        Returns:
            Dict with price_before, price_after, delta, pct_change, data_quality
        """
        window = window_minutes or self.window_minutes
        
        if not price_history:
            return {
                "price_before": None,
                "price_after": None,
                "delta": None,
                "pct_change": None,
                "data_quality": "missing",
                "before_offset_seconds": None,
                "after_offset_seconds": None
            }
        
        # Convert event timestamp to unix
        if event_timestamp.tzinfo is None:
            event_ts = int(event_timestamp.replace(tzinfo=timezone.utc).timestamp())
        else:
            event_ts = int(event_timestamp.timestamp())
        
        window_seconds = window * 60
        
        # Extract timestamps for binary search
        timestamps = [p.timestamp for p in price_history]
        
        # Find closest point BEFORE event (within window)
        idx_before = bisect_right(timestamps, event_ts) - 1
        price_before = None
        before_offset = None
        
        if idx_before >= 0:
            offset = event_ts - timestamps[idx_before]
            if offset <= window_seconds:
                price_before = price_history[idx_before].price
                before_offset = offset
        
        # Find closest point AFTER event (within window)
        idx_after = bisect_left(timestamps, event_ts)
        price_after = None
        after_offset = None
        
        if idx_after < len(timestamps):
            offset = timestamps[idx_after] - event_ts
            if offset <= window_seconds:
                price_after = price_history[idx_after].price
                after_offset = offset
        
        # Calculate delta
        delta = None
        pct_change = None
        data_quality = "missing"
        
        if price_before is not None and price_after is not None:
            delta = round(price_after - price_before, 6)
            if price_before > 0:
                pct_change = round((delta / price_before) * 100, 4)
            
            # Determine quality
            if before_offset <= 60 and after_offset <= 60:
                data_quality = "exact"
            else:
                data_quality = "interpolated"
        elif price_before is not None or price_after is not None:
            data_quality = "partial"
        
        return {
            "price_before": price_before,
            "price_after": price_after,
            "delta": delta,
            "pct_change": pct_change,
            "data_quality": data_quality,
            "before_offset_seconds": before_offset,
            "after_offset_seconds": after_offset
        }
    
    # ──────────────────────────────────────────────
    # Pipeline: Analyze Belief Graph
    # ──────────────────────────────────────────────
    
    def analyze_belief_graph(
        self,
        graph_path: str,
        slug: str,
        token_outcome: str = "Yes"
    ) -> List[EventImpact]:
        """
        Analyze price impact of all events in a belief graph.
        
        Args:
            graph_path: Path to belief graph JSON
            slug: Polymarket event slug
            token_outcome: Which outcome token to analyze ("Yes" or "No")
        
        Returns:
            List of EventImpact measurements
        """
        logger.info(f"═══ Starting Impact Analysis ═══")
        logger.info(f"Graph: {graph_path}")
        logger.info(f"Slug: {slug}")
        
        # Step 1: Load belief graph
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)
        
        nodes = graph_data.get("nodes", [])
        logger.info(f"Step 1: Loaded {len(nodes)} event nodes from belief graph")
        
        self._log_json("graph_loaded", {
            "path": str(graph_path),
            "node_count": len(nodes),
            "edge_count": len(graph_data.get("edges", []))
        })
        
        # Step 2: Fetch tokens
        tokens = self.fetch_event_tokens(slug)
        if not tokens:
            logger.error("No tokens found for slug")
            return []
        
        # Pick the target outcome token
        target_token = None
        for token in tokens:
            if token.outcome.lower() == token_outcome.lower():
                target_token = token
                break
        
        if not target_token:
            # Fallback to first token
            target_token = tokens[0]
            logger.warning(f"Token for outcome '{token_outcome}' not found, using {target_token.outcome}")
        
        logger.info(f"Step 2: Using token for '{target_token.outcome}' (price: {target_token.current_price})")
        
        # Step 3: Fetch price history
        price_history = self.fetch_price_history(target_token.token_id)
        if not price_history:
            logger.error("No price history available")
            return []
        
        logger.info(f"Step 3: Fetched {len(price_history)} price points")
        
        # Step 4: Calculate impact for each event node
        logger.info(f"Step 4: Calculating impact for {len(nodes)} events...")
        impacts = []
        
        for node in nodes:
            event_id = node.get("event_id", "")
            event_type = node.get("event_type", "")
            timestamp_str = node.get("timestamp", "")
            raw_title = node.get("raw_title", node.get("action", "") + " " + node.get("object", ""))
            
            # Parse timestamp
            try:
                if timestamp_str:
                    event_dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    if event_dt.tzinfo is None:
                        event_dt = event_dt.replace(tzinfo=timezone.utc)
                else:
                    logger.warning(f"No timestamp for event {event_id}, skipping")
                    continue
            except ValueError as e:
                logger.warning(f"Invalid timestamp for {event_id}: {timestamp_str}, error: {e}")
                continue
            
            # Calculate impact
            impact_data = self.calculate_impact(price_history, event_dt)
            
            impact = EventImpact(
                event_id=event_id,
                event_type=event_type,
                event_timestamp=timestamp_str,
                raw_title=raw_title[:200],
                price_before=impact_data["price_before"],
                price_after=impact_data["price_after"],
                delta=impact_data["delta"],
                pct_change=impact_data["pct_change"],
                window_minutes=self.window_minutes,
                data_quality=impact_data["data_quality"],
                before_offset_seconds=impact_data["before_offset_seconds"],
                after_offset_seconds=impact_data["after_offset_seconds"],
                token_id=target_token.token_id,
                market_slug=slug
            )
            impacts.append(impact)
            
            # Log each impact
            self._log_json("event_impact", {
                "event_id": event_id,
                "event_type": event_type,
                "timestamp": timestamp_str,
                "title": raw_title[:80],
                "price_before": impact_data["price_before"],
                "price_after": impact_data["price_after"],
                "delta": impact_data["delta"],
                "pct_change": impact_data["pct_change"],
                "data_quality": impact_data["data_quality"]
            })
        
        # Sort by timestamp
        impacts.sort(key=lambda x: x.event_timestamp)
        
        logger.info(f"Step 4: Calculated impact for {len(impacts)} events")
        
        return impacts
    
    # ──────────────────────────────────────────────
    # Output: Generate Timeline
    # ──────────────────────────────────────────────
    
    def generate_timeline(
        self,
        impacts: List[EventImpact],
        slug: str,
        save: bool = True
    ) -> dict:
        """
        Generate a timeline of event impacts with summary statistics.
        
        Args:
            impacts: List of EventImpact measurements
            slug: Market slug (for naming)
            save: Whether to save to file
        
        Returns:
            Timeline dict with impacts and stats
        """
        # Filter to impacts with data
        measured = [i for i in impacts if i.data_quality != "missing"]
        positive = [i for i in measured if i.delta and i.delta > 0]
        negative = [i for i in measured if i.delta and i.delta < 0]
        neutral = [i for i in measured if i.delta is not None and i.delta == 0]
        
        # Calculate stats
        if measured:
            deltas = [i.delta for i in measured if i.delta is not None]
            avg_delta = sum(deltas) / len(deltas) if deltas else 0
            max_positive = max(deltas) if deltas else 0
            max_negative = min(deltas) if deltas else 0
            abs_deltas = [abs(d) for d in deltas]
            avg_abs_delta = sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0
        else:
            avg_delta = max_positive = max_negative = avg_abs_delta = 0
        
        timeline = {
            "market_slug": slug,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "window_minutes": self.window_minutes,
            "summary": {
                "total_events": len(impacts),
                "events_with_data": len(measured),
                "events_missing_data": len(impacts) - len(measured),
                "positive_impacts": len(positive),
                "negative_impacts": len(negative),
                "neutral_impacts": len(neutral),
                "avg_delta": round(avg_delta, 6),
                "avg_abs_delta": round(avg_abs_delta, 6),
                "max_positive_delta": round(max_positive, 6),
                "max_negative_delta": round(max_negative, 6)
            },
            "impacts": [i.to_dict() for i in impacts]
        }
        
        if save:
            output_file = self.output_dir / f"{slug}_timeline.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(timeline, f, indent=2, default=str)
            logger.info(f"Timeline saved to {output_file}")
        
        # Log summary
        self._log_json("timeline_summary", timeline["summary"])
        
        # Print readable timeline
        self._print_timeline(impacts, slug, timeline["summary"])
        
        return timeline
    
    def _print_timeline(self, impacts: List[EventImpact], slug: str, summary: dict):
        """Print a human-readable timeline."""
        print("\n" + "═" * 80)
        print(f"  POLYMARKET EVENT IMPACT TIMELINE: {slug}")
        print("═" * 80)
        
        print(f"\n  Window: ±{self.window_minutes} minutes")
        print(f"  Total events: {summary['total_events']}")
        print(f"  Events with data: {summary['events_with_data']}")
        print(f"  Positive impacts: {summary['positive_impacts']}")
        print(f"  Negative impacts: {summary['negative_impacts']}")
        print(f"  Avg |delta|: {summary['avg_abs_delta']:.4f}")
        print(f"  Max positive: +{summary['max_positive_delta']:.4f}")
        print(f"  Max negative: {summary['max_negative_delta']:.4f}")
        
        print("\n  " + "─" * 76)
        print(f"  {'Timestamp':<22} {'Type':<10} {'Δ':>8} {'%':>8} {'Quality':<12} Title")
        print("  " + "─" * 76)
        
        for impact in impacts:
            delta_str = f"{impact.delta:+.4f}" if impact.delta is not None else "   N/A"
            pct_str = f"{impact.pct_change:+.2f}%" if impact.pct_change is not None else "  N/A"
            ts = impact.event_timestamp[:19]
            title = impact.raw_title[:40]
            
            # Color indicator
            if impact.delta and impact.delta > 0.01:
                indicator = "▲"
            elif impact.delta and impact.delta < -0.01:
                indicator = "▼"
            elif impact.delta is not None:
                indicator = "─"
            else:
                indicator = "?"
            
            print(f"  {indicator} {ts:<20} {impact.event_type:<10} {delta_str:>8} {pct_str:>8} {impact.data_quality:<12} {title}")
        
        print("\n" + "═" * 80)


def analyze_market_impact(
    slug: str,
    graph_path: Optional[str] = None,
    window_minutes: int = 2,
    output_dir: Optional[str] = None
) -> dict:
    """
    Convenience function to run impact analysis.
    
    Args:
        slug: Polymarket event slug
        graph_path: Path to belief graph JSON (auto-detected if None)
        window_minutes: ±minutes window
        output_dir: Output directory
    
    Returns:
        Timeline dict
    """
    analyzer = PolymarketImpactAnalyzer(
        window_minutes=window_minutes,
        output_dir=Path(output_dir) if output_dir else None
    )
    
    # Auto-detect graph path
    if graph_path is None:
        data_dir = Path("data/belief_graphs")
        graphs = list(data_dir.glob("*_graph.json"))
        if graphs:
            graph_path = str(graphs[0])
            logger.info(f"Auto-detected graph: {graph_path}")
        else:
            logger.error("No belief graphs found in data/belief_graphs/")
            return {}
    
    impacts = analyzer.analyze_belief_graph(graph_path, slug)
    timeline = analyzer.generate_timeline(impacts, slug)
    
    return timeline
