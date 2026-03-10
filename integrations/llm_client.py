"""
LLM Client

Provider-agnostic wrapper for LLM API calls.
Supports Google Gemini via the new google-genai SDK.

Two-step graph generation:
  Step 1 (search-grounded): Gather current real-world context about the event
  Step 2 (JSON-mode):       Structure the context into a causal graph

Usage:
    client = LLMClient.from_env()
    context = client.search_grounded_summary("Will Gemini 3.5 release by June 2026?")
    graph   = client.generate_json(prompt, context=context)
"""

import json
import logging
import os
import re
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM call fails or returns unparseable output."""
    pass


class LLMClient:
    """
    Wrapper around the Google Gemini API (google-genai SDK).

    Provides two capabilities:
    1. search_grounded_summary() — grounded web search for real-time context
    2. generate_json()           — structured JSON output (no search, clean parse)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_output_tokens: int = 16384,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.model_name = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        from google import genai
        from google.genai import types
        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=api_key)
        logger.info(f"LLMClient initialised with model={model}")

    @classmethod
    def from_env(cls, env_var: str = "GEMINI_API_KEY", **kwargs) -> "LLMClient":
        """Create client from environment variable."""
        api_key = os.environ.get(env_var)
        if not api_key:
            raise EnvironmentError(
                f"LLM API key not found. Set the {env_var!r} environment variable."
            )
        return cls(api_key=api_key, **kwargs)

    def search_grounded_summary(self, query: str) -> str:
        """
        Perform a Google-Search-grounded query to get current real-world context.

        This is Step 1: grounds the LLM's reasoning with live web data.
        Returns the raw text response (not JSON).

        Args:
            query: The search/summary query string.

        Returns:
            Grounded text summary from the LLM.
        """
        types = self._types
        tool = types.Tool(google_search=types.GoogleSearch())

        prompt = (
            f"Search the web and give me a comprehensive summary of recent developments "
            f"and key factors relevant to: {query}\n\n"
            f"Focus on: recent news, announcements, timelines, key players, risks, "
            f"and any events that could accelerate or delay this outcome. "
            f"Be specific with dates and sources where possible."
        )

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Search-grounded call attempt {attempt}")
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[tool],
                        temperature=0.1,
                        max_output_tokens=2048,
                    ),
                )
                text = response.text.strip()
                logger.info(
                    f"Search grounding successful (~{len(text)} chars)"
                )
                return text

            except Exception as e:
                last_error = e
                logger.warning(f"Search grounding error on attempt {attempt}: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        # Non-fatal: fall back to empty context (model uses its training knowledge)
        logger.warning(
            f"Search grounding failed after {self.max_retries} attempts: {last_error}. "
            "Falling back to model knowledge only."
        )
        return ""

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> Any:
        """
        Call the LLM in JSON-output mode and return parsed result.

        This is Step 2: structured output without search (clean JSON).

        Args:
            prompt: The user prompt.
            system_instruction: Optional system-level instruction.

        Returns:
            Parsed JSON (dict or list).

        Raises:
            LLMError: If the call fails after all retries.
        """
        types = self._types

        config_kwargs: dict = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "response_mime_type": "application/json",
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        config = types.GenerateContentConfig(**config_kwargs)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"JSON generation attempt {attempt}/{self.max_retries}")
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                raw = response.text.strip()

                # Strip markdown code fences if present
                if raw.startswith("```"):
                    parts = raw.split("```")
                    if len(parts) >= 3:
                        raw = parts[1]
                        raw = re.sub(r"^json\s*", "", raw).strip()
                    else:
                        raw = raw.replace("```", "").strip()

                # Attempt to repair truncated JSON by closing open structures
                raw = self._repair_json(raw)

                parsed = json.loads(raw)
                logger.debug(f"JSON generation succeeded on attempt {attempt}")
                return parsed

            except json.JSONDecodeError as e:
                last_error = LLMError(f"LLM returned non-JSON output: {e}")
                logger.warning(f"JSON parse error on attempt {attempt}: {e}")
                logger.debug(f"Raw output was: {raw[:500] if 'raw' in dir() else 'N/A'}")
            except Exception as e:
                last_error = LLMError(f"LLM API error: {e}")
                logger.warning(f"LLM API error on attempt {attempt}: {e}")

            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)

        raise last_error  # type: ignore[misc]

    @staticmethod
    def _repair_json(raw: str) -> str:
        """
        Attempt to repair a truncated JSON string by closing open structures.
        This handles the case where Gemini hits token limits mid-response.
        """
        # Count open/close braces and brackets
        depth_brace = 0
        depth_bracket = 0
        in_string = False
        escape_next = False

        for ch in raw:
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth_brace += 1
            elif ch == '}':
                depth_brace -= 1
            elif ch == '[':
                depth_bracket += 1
            elif ch == ']':
                depth_bracket -= 1

        if depth_brace == 0 and depth_bracket == 0:
            return raw  # Already valid structure

        # Truncated: close any open string, then close brackets/braces
        repaired = raw.rstrip()
        if in_string:
            repaired += '"'  # close open string
        repaired += ']' * max(0, depth_bracket)
        repaired += '}' * max(0, depth_brace)
        logger.warning(
            f"Repaired truncated JSON: added {depth_bracket}x ']' and {depth_brace}x '}}'"
        )
        return repaired

    def count_tokens_approx(self, text: str) -> int:
        """Rough token estimate (4 chars ≈ 1 token)."""
        return len(text) // 4
