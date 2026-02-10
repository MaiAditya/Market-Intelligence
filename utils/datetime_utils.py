"""
Datetime utilities for consistent timezone-aware handling.

Replaces deprecated datetime.utcnow() with timezone-aware alternatives.
"""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """
    Get current UTC time as a naive datetime (for JSON serialization compatibility).
    
    This replaces the deprecated datetime.utcnow() while maintaining
    compatibility with existing code that expects naive datetimes.
    
    Returns:
        Current UTC time as a naive datetime object
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)
