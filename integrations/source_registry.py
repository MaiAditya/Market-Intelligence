"""
Source Registry

Contains source credibility weights and utilities.
"""

from typing import Dict, Optional
from urllib.parse import urlparse


# Source credibility weights by domain
# Higher = more credible (0.0 to 1.0)
SOURCE_CREDIBILITY: Dict[str, float] = {
    # Official company sources (highest credibility)
    "google.com": 1.0,
    "blog.google": 1.0,
    "deepmind.com": 1.0,
    "deepmind.google": 1.0,
    "openai.com": 1.0,
    "anthropic.com": 1.0,
    "meta.com": 1.0,
    "ai.meta.com": 1.0,
    "microsoft.com": 1.0,
    "blogs.microsoft.com": 1.0,
    "nvidia.com": 1.0,
    
    # Government/regulatory sources
    "europa.eu": 1.0,
    "ec.europa.eu": 1.0,
    "gov.uk": 0.95,
    "gov.eu": 0.95,
    
    # Major news outlets
    "nytimes.com": 0.85,
    "washingtonpost.com": 0.85,
    "theguardian.com": 0.85,
    "bbc.com": 0.85,
    "bbc.co.uk": 0.85,
    "reuters.com": 0.90,
    "apnews.com": 0.90,
    "bloomberg.com": 0.85,
    "ft.com": 0.85,
    
    # Tech journalism
    "techcrunch.com": 0.75,
    "theverge.com": 0.75,
    "wired.com": 0.75,
    "wired.co.uk": 0.75,
    "arstechnica.com": 0.75,
    "engadget.com": 0.70,
    "cnet.com": 0.70,
    "technologyreview.com": 0.80,
    "spectrum.ieee.org": 0.80,
    "venturebeat.com": 0.70,
    "zdnet.com": 0.70,
    "theinformation.com": 0.80,
    "semafor.com": 0.75,
    "axios.com": 0.75,
    
    # Research/academic sources
    "arxiv.org": 0.85,
    "openreview.net": 0.80,
    "papers.ssrn.com": 0.80,
    "semanticscholar.org": 0.80,
    "nature.com": 0.90,
    "science.org": 0.90,
    
    # AI community forums
    "huggingface.co": 0.70,
    "news.ycombinator.com": 0.55,
    "lesswrong.com": 0.60,
    "alignmentforum.org": 0.60,
    
    # Social media (lower credibility)
    "reddit.com": 0.50,
    "old.reddit.com": 0.50,
    "twitter.com": 0.40,
    "x.com": 0.40,
    "nitter.net": 0.40,
    "linkedin.com": 0.50,
    
    # Forums/Q&A
    "stackoverflow.com": 0.55,
    "quora.com": 0.40,
}

# Default credibility for unknown sources
DEFAULT_CREDIBILITY = 0.50

# Source type to credibility mapping
SOURCE_TYPE_CREDIBILITY: Dict[str, float] = {
    "official": 1.0,
    "journalist": 0.75,
    "research": 0.80,
    "forum": 0.50,
    "social": 0.45,
}


def get_source_credibility(url: str) -> float:
    """
    Get credibility score for a URL.
    
    Args:
        url: Full URL
    
    Returns:
        Credibility score (0.0 to 1.0)
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        
        # Direct match
        if domain in SOURCE_CREDIBILITY:
            return SOURCE_CREDIBILITY[domain]
        
        # Check subdomains
        for known_domain, score in SOURCE_CREDIBILITY.items():
            if domain.endswith("." + known_domain):
                return score
        
        return DEFAULT_CREDIBILITY
        
    except Exception:
        return DEFAULT_CREDIBILITY


def get_source_type_credibility(source_type: str) -> float:
    """
    Get credibility score for a source type.
    
    Args:
        source_type: Source type (official, journalist, social, forum, research)
    
    Returns:
        Credibility score (0.0 to 1.0)
    """
    return SOURCE_TYPE_CREDIBILITY.get(source_type, DEFAULT_CREDIBILITY)


def get_author_credibility(author_type: str) -> float:
    """
    Get credibility score for an author type.
    
    Args:
        author_type: Author type (company, journalist, anonymous, researcher)
    
    Returns:
        Credibility score (0.0 to 1.0)
    """
    author_credibility = {
        "company": 0.95,
        "journalist": 0.75,
        "researcher": 0.80,
        "anonymous": 0.40
    }
    return author_credibility.get(author_type, 0.50)


def is_official_source(url: str) -> bool:
    """Check if URL is from an official source."""
    cred = get_source_credibility(url)
    return cred >= 0.90


def is_trusted_source(url: str, min_credibility: float = 0.70) -> bool:
    """Check if URL is from a trusted source."""
    return get_source_credibility(url) >= min_credibility
