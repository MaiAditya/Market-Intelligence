"""
Time and Numeric Extractor

Extracts and normalizes temporal and numeric information from text.
Uses NER + regex patterns - no generative reasoning.

Normalizes:
- "by end of 2026" → 2026-12-31
- "Q3 2026" → 2026-09-30
- "late 2026" → 2026-10-01 to 2026-12-31 (range)
- "500B parameters" → 500000000000
- "85% on MMLU" → 0.85
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class ExtractedDate:
    """Represents an extracted date or date range."""
    raw_text: str
    normalized_start: date
    normalized_end: Optional[date]  # For ranges
    is_range: bool
    confidence: float
    
    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "normalized_start": self.normalized_start.isoformat(),
            "normalized_end": self.normalized_end.isoformat() if self.normalized_end else None,
            "is_range": self.is_range,
            "confidence": self.confidence
        }


@dataclass
class ExtractedNumeric:
    """Represents an extracted numeric value."""
    raw_text: str
    normalized_value: float
    unit: str  # parameters, percent, hours, etc.
    context: str  # benchmark, model_size, compute, etc.
    confidence: float
    
    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "normalized_value": self.normalized_value,
            "unit": self.unit,
            "context": self.context,
            "confidence": self.confidence
        }


class DateNormalizer:
    """
    Normalizes date expressions to standard date objects.
    """
    
    MONTH_MAP = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12
    }
    
    QUARTER_END_MONTHS = {
        "1": 3, "q1": 3, "first": 3,
        "2": 6, "q2": 6, "second": 6,
        "3": 9, "q3": 9, "third": 9,
        "4": 12, "q4": 12, "fourth": 12
    }
    
    @classmethod
    def get_quarter_end(cls, quarter: str, year: int) -> date:
        """Get the end date of a quarter."""
        quarter_lower = quarter.lower().strip()
        
        # Extract quarter number
        month = None
        for key, m in cls.QUARTER_END_MONTHS.items():
            if key in quarter_lower:
                month = m
                break
        
        if month is None:
            month = 12  # Default to Q4
        
        # Get last day of the quarter
        if month == 3:
            return date(year, 3, 31)
        elif month == 6:
            return date(year, 6, 30)
        elif month == 9:
            return date(year, 9, 30)
        else:
            return date(year, 12, 31)
    
    @classmethod
    def normalize_month_year(cls, month_str: str, year: int) -> Tuple[date, date]:
        """Normalize 'Month Year' to date range."""
        month_lower = month_str.lower().strip()
        
        month = None
        for name, num in cls.MONTH_MAP.items():
            if month_lower.startswith(name):
                month = num
                break
        
        if month is None:
            # Default to December
            month = 12
        
        # Get last day of month
        if month in [1, 3, 5, 7, 8, 10, 12]:
            last_day = 31
        elif month in [4, 6, 9, 11]:
            last_day = 30
        else:  # February
            if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                last_day = 29
            else:
                last_day = 28
        
        start = date(year, month, 1)
        end = date(year, month, last_day)
        
        return start, end
    
    @classmethod
    def normalize_relative_year(cls, expression: str, year: int) -> Tuple[date, date]:
        """Normalize relative year expressions like 'end of 2026'."""
        expr_lower = expression.lower()
        
        if "end" in expr_lower or "late" in expr_lower:
            # Last quarter of the year
            return date(year, 10, 1), date(year, 12, 31)
        elif "early" in expr_lower or "beginning" in expr_lower:
            # First quarter
            return date(year, 1, 1), date(year, 3, 31)
        elif "mid" in expr_lower or "middle" in expr_lower:
            # Middle of the year
            return date(year, 5, 1), date(year, 8, 31)
        else:
            # Default to full year
            return date(year, 1, 1), date(year, 12, 31)


class TimeExtractor:
    """
    Extracts temporal information from text.
    """
    
    # Patterns for date extraction
    PATTERNS = [
        # Full dates
        (r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b', "mdy"),
        (r'\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b', "ymd"),
        
        # Month Year
        (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b', "month_year"),
        (r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(\d{4})\b', "month_year"),
        
        # Quarter Year
        (r'\b(Q[1-4])\s*(\d{4})\b', "quarter_year"),
        (r'\b(first|second|third|fourth)\s+quarter\s+(?:of\s+)?(\d{4})\b', "quarter_year"),
        
        # Relative year expressions
        (r'\b(by\s+end\s+of|by\s+the\s+end\s+of|end\s+of)\s+(\d{4})\b', "relative_year"),
        (r'\b(early|late|mid-?)\s*(\d{4})\b', "relative_year"),
        (r'\b(by)\s+(\d{4})\b', "by_year"),
    ]
    
    @classmethod
    def extract(cls, text: str) -> List[ExtractedDate]:
        """
        Extract dates from text.
        
        Args:
            text: Text to extract from
        
        Returns:
            List of extracted dates
        """
        extracted = []
        
        for pattern, pattern_type in cls.PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    if pattern_type == "mdy":
                        month, day, year = match.groups()
                        d = date(int(year), int(month), int(day))
                        extracted.append(ExtractedDate(
                            raw_text=match.group(0),
                            normalized_start=d,
                            normalized_end=None,
                            is_range=False,
                            confidence=0.95
                        ))
                    
                    elif pattern_type == "ymd":
                        year, month, day = match.groups()
                        d = date(int(year), int(month), int(day))
                        extracted.append(ExtractedDate(
                            raw_text=match.group(0),
                            normalized_start=d,
                            normalized_end=None,
                            is_range=False,
                            confidence=0.95
                        ))
                    
                    elif pattern_type == "month_year":
                        month_str, year_str = match.groups()
                        year = int(year_str)
                        start, end = DateNormalizer.normalize_month_year(month_str, year)
                        extracted.append(ExtractedDate(
                            raw_text=match.group(0),
                            normalized_start=start,
                            normalized_end=end,
                            is_range=True,
                            confidence=0.90
                        ))
                    
                    elif pattern_type == "quarter_year":
                        quarter, year_str = match.groups()
                        year = int(year_str)
                        end = DateNormalizer.get_quarter_end(quarter, year)
                        # Quarter start is 3 months before end
                        start_month = ((end.month - 1) // 3) * 3 + 1
                        start = date(year, start_month, 1)
                        extracted.append(ExtractedDate(
                            raw_text=match.group(0),
                            normalized_start=start,
                            normalized_end=end,
                            is_range=True,
                            confidence=0.90
                        ))
                    
                    elif pattern_type == "relative_year":
                        expression, year_str = match.groups()
                        year = int(year_str)
                        start, end = DateNormalizer.normalize_relative_year(
                            expression, year
                        )
                        extracted.append(ExtractedDate(
                            raw_text=match.group(0),
                            normalized_start=start,
                            normalized_end=end,
                            is_range=True,
                            confidence=0.85
                        ))
                    
                    elif pattern_type == "by_year":
                        _, year_str = match.groups()
                        year = int(year_str)
                        extracted.append(ExtractedDate(
                            raw_text=match.group(0),
                            normalized_start=date(year, 12, 31),
                            normalized_end=None,
                            is_range=False,
                            confidence=0.85
                        ))
                
                except (ValueError, TypeError) as e:
                    logger.debug(f"Date parsing error: {e}")
                    continue
        
        # Deduplicate by raw text
        seen = set()
        unique = []
        for d in extracted:
            if d.raw_text.lower() not in seen:
                seen.add(d.raw_text.lower())
                unique.append(d)
        
        return unique


class NumericExtractor:
    """
    Extracts numeric values from text.
    """
    
    # Multiplier suffixes
    MULTIPLIERS = {
        "k": 1_000,
        "m": 1_000_000,
        "b": 1_000_000_000,
        "t": 1_000_000_000_000,
        "thousand": 1_000,
        "million": 1_000_000,
        "billion": 1_000_000_000,
        "trillion": 1_000_000_000_000
    }
    
    # Patterns for numeric extraction
    PATTERNS = [
        # Model size (parameters)
        (r'\b(\d+(?:\.\d+)?)\s*([KMBT])\s*(?:params?|parameters?)\b', "parameters"),
        (r'\b(\d+(?:\.\d+)?)\s*(billion|million|trillion)\s*(?:params?|parameters?)\b', "parameters"),
        
        # Percentages
        (r'\b(\d+(?:\.\d+)?)\s*%', "percent"),
        (r'\b(\d+(?:\.\d+)?)\s*percent\b', "percent"),
        
        # Benchmark scores (contextual)
        (r'\b(\d+(?:\.\d+)?)\s*(?:%|percent)?\s*(?:on|in|at)\s+(?:MMLU|HumanEval|HellaSwag|ARC|GSM8K|MATH|TruthfulQA)', "benchmark"),
        
        # Compute hours
        (r'\b(\d+(?:\.\d+)?)\s*([KMBT])?\s*(?:GPU|TPU|A100|H100)\s*hours?\b', "compute_hours"),
        
        # Training tokens
        (r'\b(\d+(?:\.\d+)?)\s*([KMBT])\s*tokens?\b', "tokens"),
        (r'\b(\d+(?:\.\d+)?)\s*(billion|million|trillion)\s*tokens?\b', "tokens"),
        
        # FLOPs
        (r'\b(\d+(?:\.\d+)?)\s*([KMBT])?(?:E)?(\d+)?\s*FLOPs?\b', "flops"),
    ]
    
    @classmethod
    def _parse_multiplier(cls, suffix: str) -> float:
        """Parse a multiplier suffix."""
        if not suffix:
            return 1.0
        suffix_lower = suffix.lower()
        return cls.MULTIPLIERS.get(suffix_lower, 1.0)
    
    @classmethod
    def extract(cls, text: str) -> List[ExtractedNumeric]:
        """
        Extract numeric values from text.
        
        Args:
            text: Text to extract from
        
        Returns:
            List of extracted numerics
        """
        extracted = []
        
        for pattern, context in cls.PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    groups = match.groups()
                    
                    # Get base number
                    base_num = float(groups[0])
                    
                    # Get multiplier if present
                    multiplier = 1.0
                    if len(groups) > 1 and groups[1]:
                        multiplier = cls._parse_multiplier(groups[1])
                    
                    # Calculate normalized value
                    value = base_num * multiplier
                    
                    # Determine unit
                    if context == "percent" or context == "benchmark":
                        unit = "percent"
                        # Normalize to 0-1 range
                        if value > 1:
                            value = value / 100
                    elif context == "parameters":
                        unit = "parameters"
                    elif context == "compute_hours":
                        unit = "gpu_hours"
                    elif context == "tokens":
                        unit = "tokens"
                    elif context == "flops":
                        unit = "flops"
                    else:
                        unit = "unknown"
                    
                    extracted.append(ExtractedNumeric(
                        raw_text=match.group(0),
                        normalized_value=value,
                        unit=unit,
                        context=context,
                        confidence=0.85
                    ))
                
                except (ValueError, TypeError, IndexError) as e:
                    logger.debug(f"Numeric parsing error: {e}")
                    continue
        
        # Deduplicate
        seen = set()
        unique = []
        for n in extracted:
            if n.raw_text.lower() not in seen:
                seen.add(n.raw_text.lower())
                unique.append(n)
        
        return unique


class TimeNumericExtractor:
    """
    Combined extractor for temporal and numeric information.
    """
    
    def __init__(self):
        self.time_extractor = TimeExtractor()
        self.numeric_extractor = NumericExtractor()
    
    def extract_all(self, text: str) -> Dict:
        """
        Extract all temporal and numeric information.
        
        Args:
            text: Text to extract from
        
        Returns:
            Dictionary with dates and numerics
        """
        dates = TimeExtractor.extract(text)
        numerics = NumericExtractor.extract(text)
        
        return {
            "dates": [d.to_dict() for d in dates],
            "numerics": [n.to_dict() for n in numerics]
        }
    
    def extract_deadline_relevant(
        self,
        text: str,
        event_deadline: date
    ) -> Dict:
        """
        Extract dates relevant to an event deadline.
        
        Args:
            text: Text to extract from
            event_deadline: Event's deadline date
        
        Returns:
            Dictionary with relevant date analysis
        """
        dates = TimeExtractor.extract(text)
        
        relevant = []
        for d in dates:
            # Check if date is related to deadline
            if d.normalized_start <= event_deadline:
                days_before = (event_deadline - d.normalized_start).days
                relevant.append({
                    **d.to_dict(),
                    "days_before_deadline": days_before
                })
        
        return {
            "total_dates_found": len(dates),
            "relevant_to_deadline": relevant
        }
