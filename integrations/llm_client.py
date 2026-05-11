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
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM call fails or returns unparseable output."""
    pass


@dataclass
class GroundingChunk:
    """One web source surfaced by Gemini's Google Search grounding."""
    title: str
    uri: str
    snippet: Optional[str]
    domain: str


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
        max_retries: int = 4,
        retry_delay: float = 15.0,
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

    def _log_usage_metadata(self, response: Any, *, request_type: str) -> None:
        """Log Gemini usage metadata when available."""
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            logger.info(f"[LLM_USAGE] type={request_type} model={self.model_name} usage_metadata=missing")
            return

        def _pick(obj: Any, *names: str) -> Any:
            for name in names:
                if hasattr(obj, name):
                    return getattr(obj, name)
                if isinstance(obj, dict) and name in obj:
                    return obj[name]
            return None

        prompt_tokens = _pick(usage, "prompt_token_count", "promptTokenCount")
        output_tokens = _pick(usage, "candidates_token_count", "candidatesTokenCount")
        total_tokens = _pick(usage, "total_token_count", "totalTokenCount")
        thoughts_tokens = _pick(usage, "thoughts_token_count", "thoughtsTokenCount")
        cached_tokens = _pick(usage, "cached_content_token_count", "cachedContentTokenCount")

        logger.info(
            "[LLM_USAGE] "
            f"type={request_type} "
            f"model={self.model_name} "
            f"prompt_tokens={prompt_tokens} "
            f"output_tokens={output_tokens} "
            f"total_tokens={total_tokens} "
            f"thoughts_tokens={thoughts_tokens} "
            f"cached_tokens={cached_tokens}"
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Safely extract concatenated text from Gemini responses."""
        text = getattr(response, "text", None)
        if isinstance(text, str):
            stripped = text.strip()
            if stripped:
                return stripped

        candidates = getattr(response, "candidates", None) or []
        text_parts: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                if getattr(part, "thought", False):
                    continue
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    text_parts.append(part_text.strip())

        return "\n".join(text_parts).strip()

    @staticmethod
    def _extract_grounding_context(response: Any) -> str:
        """Build a lightweight context block from grounding metadata when text is absent."""
        candidates = getattr(response, "candidates", None) or []
        seen: set[tuple[str, str]] = set()
        lines: list[str] = []

        for candidate in candidates:
            metadata = getattr(candidate, "grounding_metadata", None)
            if not metadata:
                continue

            chunks = getattr(metadata, "grounding_chunks", None) or []
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if not web:
                    continue
                title = (getattr(web, "title", None) or "").strip()
                uri = (getattr(web, "uri", None) or "").strip()
                key = (title, uri)
                if key in seen or not (title or uri):
                    continue
                seen.add(key)
                if title and uri:
                    lines.append(f"- {title} ({uri})")
                elif title:
                    lines.append(f"- {title}")
                else:
                    lines.append(f"- {uri}")

        return "\n".join(lines[:12]).strip()

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
        text, _chunks = self.search_grounded_summary_with_sources(query)
        return text

    def search_grounded_summary_with_sources(
        self, query: str
    ) -> Tuple[str, List[GroundingChunk]]:
        """
        Same as search_grounded_summary but also returns the list of web sources
        Gemini's Google Search grounding cited.

        Use when callers want to persist the grounded URLs as evidence rows
        rather than just consuming the synthesized text.

        Returns:
            (text, chunks) — `text` is the grounded summary (possibly empty);
            `chunks` is a list of GroundingChunk (title, uri, snippet, domain).
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
                self._log_usage_metadata(response, request_type="search_grounded_summary")
                chunks = self._extract_grounding_chunks(response)
                text = self._extract_text(response)
                if text:
                    logger.info(
                        f"Search grounding successful (~{len(text)} chars, "
                        f"{len(chunks)} sources)"
                    )
                    return text, chunks

                grounded_context = self._extract_grounding_context(response)
                if grounded_context:
                    logger.info(
                        "Search grounding returned metadata-only response; "
                        f"using {len(grounded_context)} chars of grounding context "
                        f"({len(chunks)} sources)"
                    )
                    return grounded_context, chunks

                logger.warning(
                    "Search grounding returned no usable text or grounding metadata; "
                    "falling back to model knowledge only"
                )
                return "", chunks

            except Exception as e:
                last_error = e
                logger.warning(f"Search grounding error on attempt {attempt}: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        logger.warning(
            f"Search grounding failed after {self.max_retries} attempts: {last_error}. "
            "Falling back to model knowledge only."
        )
        return "", []

    @staticmethod
    def _extract_grounding_chunks(response: Any) -> List[GroundingChunk]:
        """
        Build GroundingChunks from Gemini grounding metadata.

        Gemini quirks worth knowing:
          - `web.title` is the *publisher domain* (e.g. "ibtimes.com"), not
            the article title. Treat it as the domain.
          - `web.uri` is a vertexaisearch.cloud.google.com redirect URL, not
            the original article URL. We persist this URI; users clicking it
            land on the real article via Google's redirect.
          - The actual cited text per chunk is reconstructed from
            `grounding_supports`: each support links a span of the response
            text to one or more chunk indices. We aggregate those spans into
            a per-chunk snippet so we have something to display as a headline.
        """
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return []

        # Pull the response text once — supports' segment offsets index into it.
        response_text = LLMClient._extract_text(response)

        chunks: List[GroundingChunk] = []
        seen_uris: set[str] = set()

        for candidate in candidates:
            metadata = getattr(candidate, "grounding_metadata", None)
            if not metadata:
                continue
            raw_chunks = getattr(metadata, "grounding_chunks", None) or []
            supports = getattr(metadata, "grounding_supports", None) or []

            # Aggregate snippet spans per chunk index (scoped to this candidate)
            chunk_snippets: dict[int, list[str]] = {}
            for support in supports:
                segment = getattr(support, "segment", None)
                ci_list = getattr(support, "grounding_chunk_indices", None) or []
                if not segment or not ci_list:
                    continue
                seg_text = (getattr(segment, "text", None) or "").strip()
                if not seg_text:
                    start = getattr(segment, "start_index", None)
                    end = getattr(segment, "end_index", None)
                    if (
                        isinstance(start, int)
                        and isinstance(end, int)
                        and 0 <= start < end <= len(response_text)
                    ):
                        seg_text = response_text[start:end].strip()
                if not seg_text:
                    continue
                for ci in ci_list:
                    if isinstance(ci, int):
                        chunk_snippets.setdefault(ci, []).append(seg_text)

            for ci, chunk in enumerate(raw_chunks):
                web = getattr(chunk, "web", None)
                if not web:
                    continue
                web_title = (getattr(web, "title", None) or "").strip()
                uri = (getattr(web, "uri", None) or "").strip()
                if not uri or uri in seen_uris:
                    continue
                seen_uris.add(uri)

                # web.title is the publisher domain; fall back to URI parse.
                domain = web_title.lower()
                if not domain:
                    try:
                        netloc = urlparse(uri).netloc.lower()
                        domain = netloc[4:] if netloc.startswith("www.") else netloc
                    except Exception:
                        domain = ""

                spans = chunk_snippets.get(ci, [])
                # Dedup spans, keep order, cap at 3 to avoid 5kB headlines.
                seen_spans: set[str] = set()
                deduped: list[str] = []
                for s in spans:
                    if s in seen_spans:
                        continue
                    seen_spans.add(s)
                    deduped.append(s)
                snippet = " ".join(deduped[:3]) if deduped else None

                # Display title: snippet if present, else domain
                display_title = snippet or domain or uri

                chunks.append(
                    GroundingChunk(
                        title=display_title,
                        uri=uri,
                        snippet=snippet,
                        domain=domain,
                    )
                )
        return chunks

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
                self._log_usage_metadata(response, request_type="generate_json")
                raw = self._extract_text(response)
                if not raw:
                    raise LLMError("LLM API returned no text content")

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
