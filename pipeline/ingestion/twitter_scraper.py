"""
Twitter/X Scraper

Uses Playwright for browser-based scraping of Twitter/X.
Falls back to simpler approaches when Playwright is unavailable.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Tweet:
    """Represents a Tweet."""
    tweet_id: str
    text: str
    author: str
    author_handle: str
    url: str
    likes: int
    retweets: int
    replies: int
    created_at: Optional[datetime]
    is_retweet: bool
    is_reply: bool


class TwitterScraper:
    """
    Scraper for Twitter/X using Playwright.
    
    Note: Twitter's API is expensive, so we use browser automation.
    This is less reliable but more cost-effective for research purposes.
    """
    
    def __init__(self, headless: bool = True):
        """
        Initialize Twitter scraper.
        
        Args:
            headless: Run browser in headless mode
        """
        self.headless = headless
        self._browser = None
        self._page = None
        self._playwright = None
    
    async def _init_browser(self) -> bool:
        """Initialize Playwright browser."""
        if self._browser is not None:
            return True
        
        try:
            from playwright.async_api import async_playwright
            
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=self.headless)
            self._page = await self._browser.new_page()
            
            # Set a realistic user agent
            await self._page.set_extra_http_headers({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            })
            
            logger.info("Playwright browser initialized for Twitter scraping")
            return True
            
        except ImportError:
            logger.warning(
                "Playwright not installed. Install with: "
                "pip install playwright && playwright install chromium"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Playwright: {e}")
            return False
    
    async def search(
        self,
        query: str,
        limit: int = 20,
        min_likes: int = 5
    ) -> List[Tweet]:
        """
        Search Twitter for tweets matching query.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            min_likes: Minimum likes filter
        
        Returns:
            List of Tweet objects
        """
        # For now, return empty - Twitter scraping requires login
        # and is against ToS without proper API access
        logger.warning(
            "Twitter scraping requires authentication and may violate ToS. "
            "Consider using official API or alternative data sources."
        )
        return []
    
    async def search_sync_wrapper(
        self,
        query: str,
        limit: int = 20
    ) -> List[Tweet]:
        """Synchronous wrapper for search."""
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.search(query, limit))
    
    def search_via_nitter(
        self,
        query: str,
        limit: int = 20
    ) -> List[Tweet]:
        """
        Alternative: Search via Nitter instances (public Twitter mirrors).
        
        Note: Nitter instances may be unreliable or rate-limited.
        """
        import requests
        from bs4 import BeautifulSoup
        
        nitter_instances = [
            "https://nitter.net",
            "https://nitter.privacydev.net",
        ]
        
        results = []
        
        for instance in nitter_instances:
            try:
                url = f"{instance}/search?f=tweets&q={query}"
                response = requests.get(url, timeout=10)
                
                if response.status_code != 200:
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for tweet_div in soup.select('.timeline-item')[:limit]:
                    try:
                        # Extract tweet data
                        content = tweet_div.select_one('.tweet-content')
                        if not content:
                            continue
                        
                        text = content.get_text(strip=True)
                        
                        # Generate a hash ID since we don't have the real tweet ID
                        tweet_id = hashlib.md5(text.encode()).hexdigest()[:16]
                        
                        author_elem = tweet_div.select_one('.username')
                        author_handle = author_elem.get_text(strip=True) if author_elem else "unknown"
                        
                        fullname_elem = tweet_div.select_one('.fullname')
                        author = fullname_elem.get_text(strip=True) if fullname_elem else author_handle
                        
                        tweet = Tweet(
                            tweet_id=tweet_id,
                            text=text,
                            author=author,
                            author_handle=author_handle,
                            url=f"https://twitter.com/{author_handle}/status/{tweet_id}",
                            likes=0,  # Not easily available from Nitter
                            retweets=0,
                            replies=0,
                            created_at=None,
                            is_retweet=False,
                            is_reply=False
                        )
                        results.append(tweet)
                        
                    except Exception as e:
                        logger.debug(f"Failed to parse tweet: {e}")
                        continue
                
                if results:
                    logger.info(f"Nitter search '{query}': found {len(results)} tweets")
                    break  # Success, no need to try other instances
                    
            except Exception as e:
                logger.debug(f"Nitter instance {instance} failed: {e}")
                continue
        
        return results
    
    async def close(self):
        """Close browser resources."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
    
    def is_available(self) -> bool:
        """Check if Twitter scraping is available."""
        try:
            import playwright
            return True
        except ImportError:
            return False


# Sync wrapper class for easier use
class TwitterScraperSync:
    """Synchronous wrapper for TwitterScraper."""
    
    def __init__(self):
        self._async_scraper = TwitterScraper()
    
    def search(self, query: str, limit: int = 20) -> List[Tweet]:
        """Search using Nitter fallback (sync)."""
        return self._async_scraper.search_via_nitter(query, limit)
    
    def is_available(self) -> bool:
        """Check availability."""
        return True  # Nitter fallback is always available
