"""
Article Summarizer

Generates a concise 2-4 sentence summary of scraped web articles / social posts
using facebook/bart-large-cnn (abstractive summarization).

The summary is stored alongside the raw document and is later used by the
NodeEvidenceMatcher to map articles to causal graph nodes.

Design decisions:
- Uses BART (abstractive) not extractive — produces proper compressed prose.
- Falls back to heuristic extraction if the model isn't available.
- Batch-capable: pass multiple texts for GPU-efficient inference.
- Max input: 1024 tokens (~750 words). Longer texts are truncated.
- Max output: 120 tokens (~90 words, ~3 sentences).
"""

import logging
import re
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

# ─── Model config ─────────────────────────────────────────────────────────────

_DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"   # 300MB, fast
_FALLBACK_MODEL = "facebook/bart-large-cnn"         # 400MB, highest quality

_MAX_INPUT_TOKENS = 1024
_MAX_OUTPUT_TOKENS = 120
_MIN_OUTPUT_TOKENS = 30
_BATCH_SIZE = 8


# ─── Singleton model loader ───────────────────────────────────────────────────

_pipeline = None
_model_name_loaded: Optional[str] = None


def _load_pipeline(model_name: Optional[str] = None) -> Optional[object]:
    """
    Lazy-load the BART summarization model (singleton).

    Controlled by the USE_BART=1 environment variable.
    When USE_BART is not set, heuristic extraction is used instead —
    which is fast, requires no download, and is good enough for
    sentence-transformer input and LLM prompting.

    Set USE_BART=1 to enable neural summarization for higher quality.
    """
    import os
    global _pipeline, _model_name_loaded

    if not os.getenv("USE_BART"):
        return None  # Use heuristic fallback (fast, no model download)

    target = model_name or _DEFAULT_MODEL
    if _pipeline is not None and _model_name_loaded == target:
        return _pipeline

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        logger.info(f"Loading summarization model: {target}")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(target)
        model = AutoModelForSeq2SeqLM.from_pretrained(target)
        model.eval()
        _pipeline = {"tokenizer": tokenizer, "model": model}
        _model_name_loaded = target
        logger.info(f"Summarization model loaded in {time.time()-t0:.1f}s")
        return _pipeline

    except Exception as e:
        logger.warning(f"Could not load summarization model {target}: {e}")
        return None


# ─── Heuristic fallback ───────────────────────────────────────────────────────

def _heuristic_summary(text: str, max_sentences: int = 3) -> str:
    """
    Extractive fallback: take the first N non-trivial sentences from the text.
    Used when the BART model is unavailable.
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter very short sentences (e.g., headers, nav items)
    good = [s.strip() for s in sentences if len(s.split()) >= 7]
    selected = good[:max_sentences]
    summary = " ".join(selected)
    return summary[:600] if summary else text[:300]


# ─── Public API ───────────────────────────────────────────────────────────────

class ArticleSummarizer:
    """
    Wraps a BART summarization model to produce concise article summaries.

    Usage (single):
        summarizer = ArticleSummarizer()
        summary = summarizer.summarize(raw_text, title="optional title")

    Usage (batch):
        summaries = summarizer.summarize_batch(texts)
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or _DEFAULT_MODEL
        self._pipe = None  # Loaded lazily on first call

    def _get_pipe(self):
        if self._pipe is None:
            self._pipe = _load_pipeline(self.model_name)
        return self._pipe

    @staticmethod
    def _prepare_text(raw_text: str, title: Optional[str] = None) -> str:
        """
        Clean and truncate raw text for model input.
        - Prepend title (improves summary focus).
        - Strip HTML artifacts, excessive whitespace.
        - Truncate to ~750 words (model limit).
        """
        text = raw_text or ""
        # Strip common web garbage
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(Cookie Policy|Privacy Policy|Terms of Service)[^\n]*', '', text)
        text = text.strip()

        if title:
            title_clean = title.strip().rstrip('.')
            if not text.startswith(title_clean):
                text = f"{title_clean}. {text}"

        # Truncate to ~750 words to stay under 1024 tokens
        words = text.split()
        if len(words) > 750:
            text = " ".join(words[:750])

        return text

    def summarize(
        self,
        raw_text: str,
        title: Optional[str] = None,
    ) -> str:
        """
        Summarize a single document.

        Args:
            raw_text: Full article/post text.
            title: Optional title to prepend for context.

        Returns:
            2-4 sentence summary string.
        """
        text = self._prepare_text(raw_text, title)
        if not text or len(text.split()) < 20:
            return text  # Too short to summarize

        pipe = self._get_pipe()
        if pipe is None:
            logger.debug("Summarizer model unavailable — using heuristic fallback")
            return _heuristic_summary(text)

        try:
            import torch
            tokenizer = pipe["tokenizer"]
            model = pipe["model"]
            inputs = tokenizer(
                text,
                max_length=_MAX_INPUT_TOKENS,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                ids = model.generate(
                    **inputs,
                    max_new_tokens=_MAX_OUTPUT_TOKENS,
                    min_new_tokens=_MIN_OUTPUT_TOKENS,
                    num_beams=4,
                    early_stopping=True,
                )
            return tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning(f"Summarization failed: {e} — falling back to heuristic")
            return _heuristic_summary(text)

    def summarize_batch(
        self,
        texts: List[str],
        titles: Optional[List[Optional[str]]] = None,
    ) -> List[str]:
        """
        Summarize a batch of documents efficiently.

        Args:
            texts: List of raw article texts.
            titles: Optional matching list of titles.

        Returns:
            List of summary strings, same length as texts.
        """
        if not texts:
            return []

        titles = titles or [None] * len(texts)
        prepared = [
            self._prepare_text(t, h)
            for t, h in zip(texts, titles)
        ]

        pipe = self._get_pipe()
        if pipe is None:
            return [_heuristic_summary(t) for t in prepared]

        import torch
        tokenizer = pipe["tokenizer"]
        model = pipe["model"]
        results = []
        for i in range(0, len(prepared), _BATCH_SIZE):
            batch = prepared[i : i + _BATCH_SIZE]
            try:
                inputs = tokenizer(
                    batch,
                    max_length=_MAX_INPUT_TOKENS,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    ids = model.generate(
                        **inputs,
                        max_new_tokens=_MAX_OUTPUT_TOKENS,
                        min_new_tokens=_MIN_OUTPUT_TOKENS,
                        num_beams=4,
                        early_stopping=True,
                    )
                for row in ids:
                    results.append(
                        tokenizer.decode(row, skip_special_tokens=True).strip()
                    )
            except Exception as e:
                logger.warning(f"Batch summarization failed for batch {i}: {e}")
                results.extend([_heuristic_summary(t) for t in batch])

        return results
