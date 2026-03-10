"""
LLM-Based Query Generator

Uses Google Gemini to generate highly relevant search queries for each event.
Falls back to the rule-based template generator if the LLM is unavailable.

Usage:
    gen = LLMQueryGenerator.from_env()
    queries = gen.generate(event_id, event_title, event_description)
"""

import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMQueryGenerator:
    """
    Generates relevant search queries using an LLM (Google Gemini).

    Given an event's title, description, and type, the LLM produces
    focused, high-quality search queries that are much more relevant
    than simple template expansion.
    """

    SYSTEM_PROMPT = (
        "You are a market research analyst who generates Google search queries "
        "to find the most relevant, recent news articles about a specific prediction market question.\n\n"
        "RULES:\n"
        "1. Generate queries that will find NEWS ARTICLES directly relevant to the market question\n"
        "2. Use specific names, dates, organizations — NOT generic keywords\n"
        "3. Include queries across different angles: official sources, analysis, market reactions, risks\n"
        "4. Vary query specificity: some broad, some very targeted\n"
        "5. Focus on RECENT events and developments\n"
        "6. DO NOT generate queries about unrelated topics\n"
        "7. Return ONLY a JSON array of query strings, no explanation"
    )

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        from google import genai
        from google.genai import types
        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self.model = model
        logger.info(f"LLMQueryGenerator initialised with model={model}")

    @classmethod
    def from_env(cls, env_var: str = "GEMINI_API_KEY", **kwargs) -> "LLMQueryGenerator":
        """Create from environment variable."""
        api_key = os.environ.get(env_var)
        if not api_key:
            raise EnvironmentError(f"LLM API key not found. Set {env_var!r}.")
        return cls(api_key=api_key, **kwargs)

    def generate(
        self,
        event_id: str,
        event_title: str,
        event_description: str = "",
        event_type: str = "",
        num_queries: int = 12,
        search_queries_hint: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate search queries for an event using the LLM.

        Args:
            event_id: Event identifier
            event_title: Human-readable event title (e.g. "Will the Iranian regime fall by March 31?")
            event_description: Full description/resolution criteria
            event_type: Type of event (geopolitics, technology, sports, etc.)
            num_queries: Number of queries to generate
            search_queries_hint: Optional existing queries from events.json to use as seed

        Returns:
            List of search query strings
        """
        types = self._types

        prompt = f"""Generate exactly {num_queries} Google search queries to find relevant recent news articles about this prediction market:

MARKET QUESTION: {event_title}

"""
        if event_description:
            prompt += f"DESCRIPTION: {event_description[:500]}\n\n"
        if event_type:
            prompt += f"CATEGORY: {event_type}\n\n"
        if search_queries_hint:
            prompt += f"EXISTING HINTS (expand on these): {json.dumps(search_queries_hint)}\n\n"

        prompt += f"""Generate {num_queries} search queries as a JSON array of strings.
Include a mix of:
- Breaking news queries (with "2026" or recent timeframe)
- Analysis/opinion queries 
- Key player/entity specific queries
- Risk/obstacle queries
- Official source queries

Return ONLY the JSON array, nothing else."""

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.SYSTEM_PROMPT,
                    temperature=0.3,
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                ),
            )

            text = response.text.strip()
            queries = json.loads(text)

            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                logger.info(f"LLM generated {len(queries)} queries for '{event_id}'")
                return queries[:num_queries]
            else:
                logger.warning(f"LLM returned unexpected format: {type(queries)}")
                return []

        except Exception as e:
            logger.error(f"LLM query generation failed: {e}")
            return []

    def generate_for_event_config(
        self,
        event_config: dict,
        num_queries: int = 12,
    ) -> List[str]:
        """
        Generate queries from an events.json config entry.

        Args:
            event_config: Dict from events.json with event_id, event_title, etc.
            num_queries: Number of queries to generate

        Returns:
            List of search query strings
        """
        return self.generate(
            event_id=event_config.get("event_id", ""),
            event_title=event_config.get("event_title", ""),
            event_description=event_config.get("event_description", ""),
            event_type=event_config.get("event_type", ""),
            num_queries=num_queries,
            search_queries_hint=event_config.get("search_queries"),
        )
