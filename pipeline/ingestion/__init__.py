"""
Data Ingestion Module

Handles fetching and scraping data from various sources:
- Reddit API (official PRAW)
- Twitter/X (Playwright scraping)
- Generic web scraping (BeautifulSoup + requests)
"""

from .ingestor import DataIngestor, IngestedDocument
from .reddit_api import RedditClient
from .twitter_scraper import TwitterScraper
from .web_scraper import WebScraper

__all__ = [
    "DataIngestor",
    "IngestedDocument",
    "RedditClient",
    "TwitterScraper",
    "WebScraper"
]
