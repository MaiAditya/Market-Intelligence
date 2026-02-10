"""
Reddit API Client

Uses PRAW (Python Reddit API Wrapper) for accessing Reddit data.
Requires Reddit API credentials in environment variables.
"""

import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import time

logger = logging.getLogger(__name__)


@dataclass
class RedditPost:
    """Represents a Reddit post or comment."""
    post_id: str
    title: str
    text: str
    author: str
    subreddit: str
    url: str
    score: int
    num_comments: int
    created_utc: datetime
    is_self: bool
    permalink: str


class RedditClient:
    """
    Client for fetching Reddit data via PRAW.
    
    Requires environment variables:
    - REDDIT_CLIENT_ID
    - REDDIT_CLIENT_SECRET
    - REDDIT_USER_AGENT (optional, defaults to 'AIMarketIntelligence/1.0')
    """
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Initialize Reddit client.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string
        """
        self.client_id = client_id or os.environ.get("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or os.environ.get(
            "REDDIT_USER_AGENT",
            "AIMarketIntelligence/1.0"
        )
        
        self._reddit = None
        self._initialized = False
    
    def _init_client(self) -> bool:
        """Lazy initialization of PRAW client."""
        if self._initialized:
            return self._reddit is not None
        
        self._initialized = True
        
        if not self.client_id or not self.client_secret:
            logger.warning(
                "Reddit API credentials not found. "
                "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables."
            )
            return False
        
        try:
            import praw
            self._reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
            logger.info("Reddit client initialized successfully")
            return True
        except ImportError:
            logger.warning("PRAW not installed. Install with: pip install praw")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            return False
    
    def search(
        self,
        query: str,
        subreddit: str = "all",
        limit: int = 20,
        time_filter: str = "month",
        sort: str = "relevance"
    ) -> List[RedditPost]:
        """
        Search Reddit for posts matching query.
        
        Args:
            query: Search query string
            subreddit: Subreddit to search (default: "all")
            limit: Maximum number of results
            time_filter: Time filter (hour, day, week, month, year, all)
            sort: Sort order (relevance, hot, top, new, comments)
        
        Returns:
            List of RedditPost objects
        """
        if not self._init_client():
            logger.warning("Reddit client not available, returning empty results")
            return []
        
        results = []
        
        try:
            subreddit_obj = self._reddit.subreddit(subreddit)
            
            for submission in subreddit_obj.search(
                query,
                limit=limit,
                time_filter=time_filter,
                sort=sort
            ):
                post = RedditPost(
                    post_id=submission.id,
                    title=submission.title,
                    text=submission.selftext if submission.is_self else "",
                    author=str(submission.author) if submission.author else "[deleted]",
                    subreddit=submission.subreddit.display_name,
                    url=submission.url,
                    score=submission.score,
                    num_comments=submission.num_comments,
                    created_utc=datetime.utcfromtimestamp(submission.created_utc),
                    is_self=submission.is_self,
                    permalink=f"https://reddit.com{submission.permalink}"
                )
                results.append(post)
                
                # Rate limiting
                time.sleep(0.1)
            
            logger.info(f"Reddit search '{query}': found {len(results)} posts")
            
        except Exception as e:
            logger.error(f"Reddit search failed: {e}")
        
        return results
    
    def get_post_with_comments(
        self,
        post_id: str,
        comment_limit: int = 10
    ) -> tuple[Optional[RedditPost], List[str]]:
        """
        Get a post with its top comments.
        
        Args:
            post_id: Reddit post ID
            comment_limit: Maximum number of comments to fetch
        
        Returns:
            Tuple of (RedditPost, list of comment texts)
        """
        if not self._init_client():
            return None, []
        
        try:
            submission = self._reddit.submission(id=post_id)
            
            post = RedditPost(
                post_id=submission.id,
                title=submission.title,
                text=submission.selftext if submission.is_self else "",
                author=str(submission.author) if submission.author else "[deleted]",
                subreddit=submission.subreddit.display_name,
                url=submission.url,
                score=submission.score,
                num_comments=submission.num_comments,
                created_utc=datetime.utcfromtimestamp(submission.created_utc),
                is_self=submission.is_self,
                permalink=f"https://reddit.com{submission.permalink}"
            )
            
            # Get top comments
            submission.comments.replace_more(limit=0)
            comments = []
            for comment in submission.comments[:comment_limit]:
                if hasattr(comment, 'body'):
                    comments.append(comment.body)
            
            return post, comments
            
        except Exception as e:
            logger.error(f"Failed to get post {post_id}: {e}")
            return None, []
    
    def search_subreddits(
        self,
        query: str,
        subreddits: List[str],
        limit_per_subreddit: int = 10
    ) -> List[RedditPost]:
        """
        Search across multiple specific subreddits.
        
        Args:
            query: Search query
            subreddits: List of subreddit names
            limit_per_subreddit: Max results per subreddit
        
        Returns:
            Combined list of posts from all subreddits
        """
        all_results = []
        
        for subreddit in subreddits:
            results = self.search(
                query=query,
                subreddit=subreddit,
                limit=limit_per_subreddit
            )
            all_results.extend(results)
            time.sleep(0.5)  # Rate limiting between subreddits
        
        return all_results
    
    def is_available(self) -> bool:
        """Check if Reddit API is available."""
        return self._init_client()


# AI-relevant subreddits for searching
AI_SUBREDDITS = [
    "MachineLearning",
    "artificial",
    "OpenAI",
    "LocalLLaMA",
    "singularity",
    "technology",
    "Futurology",
    "ArtificialIntelligence"
]
