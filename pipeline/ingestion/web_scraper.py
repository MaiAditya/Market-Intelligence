"""
Generic Web Scraper

Uses BeautifulSoup + requests for general web scraping.
Also uses newspaper3k for article extraction when available.
"""

import hashlib
import logging
import time
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse, quote_plus
import re

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class ScrapedPage:
    """Represents a scraped web page."""
    url: str
    title: str
    text: str
    html: str
    author: Optional[str]
    publish_date: Optional[datetime]
    domain: str
    meta_description: str
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    scraped_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "author": self.author,
            "publish_date": self.publish_date.isoformat() if self.publish_date else None,
            "domain": self.domain,
            "meta_description": self.meta_description,
            "scraped_at": self.scraped_at.isoformat()
        }


class WebScraper:
    """
    Generic web scraper using requests and BeautifulSoup.
    
    Features:
    - Rate limiting
    - User agent rotation
    - Article extraction with newspaper3k
    - Fallback to basic HTML parsing
    - Multiple search engine support
    """
    
    USER_AGENTS = [
        (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    ]
    
    def __init__(
        self,
        rate_limit: float = 1.0,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize web scraper.
        
        Args:
            rate_limit: Minimum seconds between requests to same domain
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.max_retries = max_retries
        
        self._session = requests.Session()
        self._last_request_time: Dict[str, float] = {}
        self._user_agent_idx = 0
    
    def _get_user_agent(self) -> str:
        """Get next user agent from rotation."""
        ua = self.USER_AGENTS[self._user_agent_idx]
        self._user_agent_idx = (self._user_agent_idx + 1) % len(self.USER_AGENTS)
        return ua
    
    def _respect_rate_limit(self, domain: str) -> None:
        """Wait if needed to respect rate limit for domain."""
        last_time = self._last_request_time.get(domain, 0)
        elapsed = time.time() - last_time
        
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        
        self._last_request_time[domain] = time.time()
    
    def _extract_with_newspaper(self, url: str) -> Optional[ScrapedPage]:
        """Try to extract article using newspaper3k."""
        try:
            from newspaper import Article
            
            article = Article(url)
            article.download()
            article.parse()
            
            domain = urlparse(url).netloc
            
            return ScrapedPage(
                url=url,
                title=article.title or "",
                text=article.text or "",
                html=article.html or "",
                author=", ".join(article.authors) if article.authors else None,
                publish_date=article.publish_date,
                domain=domain,
                meta_description=article.meta_description or "",
                images=list(article.images)[:10]
            )
            
        except ImportError:
            logger.debug("newspaper3k not available, using fallback")
            return None
        except Exception as e:
            logger.debug(f"newspaper3k extraction failed: {e}")
            return None
    
    def _extract_with_beautifulsoup(self, url: str, html: str) -> ScrapedPage:
        """Extract content using BeautifulSoup."""
        soup = BeautifulSoup(html, 'html.parser')
        domain = urlparse(url).netloc
        
        # Extract title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
        else:
            h1_tag = soup.find('h1')
            if h1_tag:
                title = h1_tag.get_text(strip=True)
        
        # Extract meta description
        meta_desc = ""
        meta_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_tag:
            meta_desc = meta_tag.get('content', '')
        
        # Extract author
        author = None
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta:
            author = author_meta.get('content')
        
        # Extract publish date
        publish_date = None
        date_meta = soup.find('meta', attrs={'property': 'article:published_time'})
        if date_meta:
            try:
                date_str = date_meta.get('content', '')
                publish_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                publish_date = publish_date.replace(tzinfo=None)
            except (ValueError, TypeError):
                pass
        
        # Remove unwanted elements
        for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        # Try to find main content
        main_content = None
        for selector in ['article', 'main', '.article-content', '.post-content', '#content']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if main_content is None:
            main_content = soup.find('body') or soup
        
        # Extract text
        text = main_content.get_text(separator='\n', strip=True)
        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Extract links
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith(('http://', 'https://')):
                links.append(href)
            elif href.startswith('/'):
                links.append(urljoin(url, href))
        
        # Extract images
        images = []
        for img_tag in soup.find_all('img', src=True):
            src = img_tag['src']
            if src.startswith(('http://', 'https://')):
                images.append(src)
            elif src.startswith('/'):
                images.append(urljoin(url, src))
        
        return ScrapedPage(
            url=url,
            title=title,
            text=text,
            html=html,
            author=author,
            publish_date=publish_date,
            domain=domain,
            meta_description=meta_desc,
            links=links[:50],
            images=images[:10]
        )
    
    def scrape(self, url: str, use_newspaper: bool = True) -> Optional[ScrapedPage]:
        """
        Scrape a web page.
        
        Args:
            url: URL to scrape
            use_newspaper: Try newspaper3k first for better article extraction
        
        Returns:
            ScrapedPage object or None if failed
        """
        domain = urlparse(url).netloc
        self._respect_rate_limit(domain)
        
        # Try newspaper3k first for better article extraction
        if use_newspaper:
            result = self._extract_with_newspaper(url)
            if result and result.text:
                return result
        
        # Fallback to basic scraping
        for attempt in range(self.max_retries):
            try:
                headers = {
                    "User-Agent": self._get_user_agent(),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                }
                
                response = self._session.get(
                    url,
                    headers=headers,
                    timeout=self.timeout,
                    allow_redirects=True
                )
                
                if response.status_code == 200:
                    return self._extract_with_beautifulsoup(url, response.text)
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout scraping {url} (attempt {attempt + 1})")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error for {url}: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def scrape_multiple(
        self,
        urls: List[str],
        use_newspaper: bool = True
    ) -> List[ScrapedPage]:
        """
        Scrape multiple URLs.
        
        Args:
            urls: List of URLs to scrape
            use_newspaper: Use newspaper3k for extraction
        
        Returns:
            List of successfully scraped pages
        """
        results = []
        
        for url in urls:
            result = self.scrape(url, use_newspaper)
            if result:
                results.append(result)
            else:
                logger.warning(f"Failed to scrape: {url}")
        
        return results
    
    def search_duckduckgo(
        self,
        query: str,
        num_results: int = 10
    ) -> List[str]:
        """
        Search DuckDuckGo and return result URLs.
        
        Uses the HTML interface for more reliable results.
        
        Args:
            query: Search query
            num_results: Number of results to return
        
        Returns:
            List of result URLs
        """
        # Try the new ddgs package first (preferred)
        try:
            from ddgs import DDGS
            
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=num_results):
                    # The result dict has 'href' key for the URL
                    url = r.get('href') or r.get('link') or r.get('url')
                    if url:
                        results.append(url)
            
            if results:
                logger.info(f"DuckDuckGo search '{query}': found {len(results)} results")
                return results
                
        except ImportError:
            # Try the old package name as fallback
            try:
                from duckduckgo_search import DDGS
                
                results = []
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=num_results):
                        url = r.get('href') or r.get('link') or r.get('url')
                        if url:
                            results.append(url)
                
                if results:
                    logger.info(f"DuckDuckGo search '{query}': found {len(results)} results")
                    return results
            except ImportError:
                logger.debug("ddgs/duckduckgo-search not installed, trying HTML scraping")
            except Exception as e:
                logger.debug(f"DDGS search failed: {e}, trying HTML scraping")
        except Exception as e:
            logger.debug(f"DDGS search failed: {e}, trying HTML scraping")
        
        # Fallback: scrape DuckDuckGo HTML results directly
        return self._search_duckduckgo_html(query, num_results)
    
    def _extract_url_from_ddg_redirect(self, href: str) -> Optional[str]:
        """
        Extract actual URL from DuckDuckGo redirect URL.
        
        DuckDuckGo uses format: //duckduckgo.com/l/?uddg=https%3A%2F%2F...
        """
        if not href:
            return None
        
        # Handle DDG redirect URLs
        if 'uddg=' in href:
            try:
                from urllib.parse import parse_qs, urlparse, unquote
                # Parse the redirect URL
                if href.startswith('//'):
                    href = 'https:' + href
                parsed = urlparse(href)
                params = parse_qs(parsed.query)
                if 'uddg' in params:
                    return unquote(params['uddg'][0])
            except Exception as e:
                logger.debug(f"Failed to parse DDG redirect: {e}")
                return None
        
        # Direct URL
        if href.startswith(('http://', 'https://')):
            if 'duckduckgo.com' not in href:
                return href
        
        return None
    
    def _search_duckduckgo_html(
        self,
        query: str,
        num_results: int = 10
    ) -> List[str]:
        """
        Search DuckDuckGo by scraping HTML results.
        
        This is a fallback when the API doesn't work.
        """
        try:
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            
            headers = {
                "User-Agent": self._get_user_agent(),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
            
            response = self._session.get(url, headers=headers, timeout=self.timeout)
            
            if response.status_code != 200:
                logger.warning(f"DuckDuckGo HTML search returned {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            seen_urls = set()
            
            # DuckDuckGo HTML results are in .result__a
            for link in soup.select('.result__a'):
                href = link.get('href', '')
                actual_url = self._extract_url_from_ddg_redirect(href)
                
                if actual_url and actual_url not in seen_urls:
                    seen_urls.add(actual_url)
                    results.append(actual_url)
                    if len(results) >= num_results:
                        break
            
            logger.info(f"DuckDuckGo HTML search '{query}': found {len(results)} results")
            return results[:num_results]
            
        except Exception as e:
            logger.error(f"DuckDuckGo HTML search failed: {e}")
            return []
    
    def search_bing(
        self,
        query: str,
        num_results: int = 10
    ) -> List[str]:
        """
        Search Bing and return result URLs by scraping.
        
        Args:
            query: Search query
            num_results: Number of results to return
        
        Returns:
            List of result URLs
        """
        try:
            url = f"https://www.bing.com/search?q={quote_plus(query)}"
            
            headers = {
                "User-Agent": self._get_user_agent(),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
            
            response = self._session.get(url, headers=headers, timeout=self.timeout)
            
            if response.status_code != 200:
                logger.warning(f"Bing search returned {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            # Bing results are typically in <li class="b_algo"><h2><a href="...">
            for result in soup.select('li.b_algo h2 a'):
                href = result.get('href', '')
                if href and href.startswith(('http://', 'https://')):
                    # Skip Bing redirect URLs
                    if 'bing.com' not in href and 'microsoft.com' not in href:
                        results.append(href)
                        if len(results) >= num_results:
                            break
            
            logger.info(f"Bing search '{query}': found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Bing search failed: {e}")
            return []
    
    def search_google(
        self,
        query: str,
        num_results: int = 10
    ) -> List[str]:
        """
        Search Google and return result URLs.
        
        Note: This uses a simple approach that may be rate-limited.
        For production, consider using a proper search API.
        
        Args:
            query: Search query
            num_results: Number of results to return
        
        Returns:
            List of result URLs
        """
        try:
            from googlesearch import search as google_search
            
            results = list(google_search(query, num_results=num_results, stop=num_results))
            logger.info(f"Google search '{query}': found {len(results)} results")
            return results
            
        except ImportError:
            logger.debug("googlesearch-python not installed, trying scrape fallback")
            return self._search_google_scrape(query, num_results)
        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return self._search_google_scrape(query, num_results)
    
    def _search_google_scrape(
        self,
        query: str,
        num_results: int = 10
    ) -> List[str]:
        """
        Search Google by scraping (fallback).
        
        Note: Google is aggressive about blocking scrapers.
        """
        try:
            url = f"https://www.google.com/search?q={quote_plus(query)}&num={num_results}"
            
            headers = {
                "User-Agent": self._get_user_agent(),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
            
            response = self._session.get(url, headers=headers, timeout=self.timeout)
            
            if response.status_code != 200:
                logger.warning(f"Google scrape returned {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            # Google results are typically in <div class="g"><a href="...">
            for result in soup.select('div.g a'):
                href = result.get('href', '')
                if href and href.startswith(('http://', 'https://')):
                    # Skip Google redirect URLs
                    if 'google.com' not in href:
                        if href not in results:
                            results.append(href)
                            if len(results) >= num_results:
                                break
            
            logger.info(f"Google scrape '{query}': found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Google scrape failed: {e}")
            return []
    
    def search(
        self,
        query: str,
        num_results: int = 10,
        engines: Optional[List[str]] = None
    ) -> List[str]:
        """
        Search using multiple engines, returning combined unique results.
        
        Args:
            query: Search query
            num_results: Target number of results
            engines: List of engines to use (default: ['duckduckgo', 'bing'])
        
        Returns:
            List of unique result URLs
        """
        if engines is None:
            engines = ['duckduckgo', 'bing']
        
        all_results = []
        seen_urls = set()
        
        for engine in engines:
            if len(all_results) >= num_results:
                break
            
            try:
                if engine == 'duckduckgo':
                    results = self.search_duckduckgo(query, num_results)
                elif engine == 'bing':
                    results = self.search_bing(query, num_results)
                elif engine == 'google':
                    results = self.search_google(query, num_results)
                else:
                    logger.warning(f"Unknown search engine: {engine}")
                    continue
                
                for url in results:
                    if url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(url)
                        
            except Exception as e:
                logger.error(f"Search engine {engine} failed: {e}")
        
        logger.info(f"Combined search '{query}': {len(all_results)} unique results")
        return all_results[:num_results]
