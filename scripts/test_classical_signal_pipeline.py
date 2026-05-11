#!/usr/bin/env python3
"""
Standalone classical (non-LLM) event-signal pipeline tester.

Purpose
-------
Validate the proposed canonical-event timeline logic independently before
integrating into the main scheduler/pipeline.

What it does
------------
1) Ingests documents + price history (from JSON input or embedded sample)
2) Clusters similar documents into canonical events (classical similarity)
3) Classifies each event as news vs rumor (rule/score based)
4) Infers direction and confidence
5) Maps events to adaptive price windows (15m / 60m / 240m)
6) Applies publish gates (provisional / confirmed / dropped)
7) Prints a metrics report and optionally writes JSON output

No LLMs are used.

Usage
-----
python scripts/test_classical_signal_pipeline.py
python scripts/test_classical_signal_pipeline.py --input-json /path/to/input.json
python scripts/test_classical_signal_pipeline.py --input-json /path/to/input.json --output-json /tmp/report.json
python scripts/test_classical_signal_pipeline.py --from-db-market <event_id_or_slug>

Input JSON shape
----------------
{
  "market_question": "Will X happen by date?",
  "market_entities": ["entity1", "entity2"],
  "documents": [
    {
      "id": "doc_1",
      "title": "Headline",
      "body": "Full text",
      "url": "https://...",
      "source": "Reuters",
      "timestamp": "2026-04-15T10:00:00Z"
    }
  ],
  "prices": [
    {"timestamp": "2026-04-15T09:00:00Z", "price": 0.42},
    {"timestamp": "2026-04-15T10:00:00Z", "price": 0.45}
  ]
}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from integrations.source_registry import get_source_credibility, get_source_type_credibility


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "was",
    "were", "will", "would", "could", "should", "about", "into", "after", "before",
    "between", "while", "their", "there", "them", "than", "then", "being", "been",
    "are", "is", "at", "to", "of", "in", "on", "a", "an", "by", "as", "it",
}

HEDGING_TERMS = {
    "reportedly", "rumor", "rumour", "unconfirmed", "allegedly",
    "sources say", "speculation", "speculative", "leaked",
    "unverified",
}

POSITIVE_TERMS = {
    "approved", "passes", "passed", "advances", "confirmed", "launch", "launched",
    "signed", "agreement", "deal", "wins", "support", "progress", "success",
}

NEGATIVE_TERMS = {
    "rejected", "delay", "delayed", "blocked", "fails", "failed", "lawsuit",
    "ban", "banned", "concern", "risk", "setback", "drop", "decline", "cuts",
}

WINDOWS_MIN = (15, 60, 240)
PROVISIONAL_CONFIDENCE_MIN = 0.50
PROVISIONAL_EVIDENCE_MIN = 0.45
CONFIRMED_CONFIDENCE_MIN = 0.72
CONFIRMED_EVIDENCE_MIN = 0.60


@dataclass
class Document:
    id: str
    title: str
    body: str
    url: str
    source: str
    timestamp: datetime


@dataclass
class PricePoint:
    timestamp: datetime
    price: float


@dataclass
class CanonicalEvent:
    event_id: str
    doc_ids: List[str]
    title: str
    timestamp: datetime
    source_names: List[str]
    source_urls: List[str]
    text: str

    event_type: str = "rumor"
    direction: str = "neutral"
    confidence: float = 0.0
    evidence_score: float = 0.0
    status: str = "dropped"

    chosen_window_min: Optional[int] = None
    price_before: Optional[float] = None
    price_after: Optional[float] = None
    price_delta: Optional[float] = None
    price_alignment_score: float = 0.0

    feature_debug: Dict[str, float] = field(default_factory=dict)


def _parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _as_async_db_url(raw: Optional[str]) -> str:
    """
    Normalize DB URL for SQLAlchemy async engine.
    """
    if raw:
        db_url = raw
    else:
        db_url = (
            os.environ.get("DATABASE_URL")
            or os.environ.get("DATABASE_URL_ASYNC")
            or os.environ.get("DATABASE_URL_SYNC")
            or "postgresql+asyncpg://causal:causal@localhost:5432/causal_interface"
        )

    if db_url.startswith("postgresql+asyncpg://"):
        return db_url
    if db_url.startswith("postgresql://"):
        return db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if db_url.startswith("postgres://"):
        return db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    return db_url


def _tokenize(text: str) -> List[str]:
    terms = re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{1,}", text.lower())
    return [t for t in terms if t not in STOPWORDS and len(t) > 2]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _domain_source_type(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]

    if domain.endswith(".gov") or domain.endswith(".int"):
        return "official"
    if any(x in domain for x in ("reuters", "apnews", "bbc", "nytimes", "bloomberg", "ft", "cnn", "cnbc")):
        return "journalist"
    if any(x in domain for x in ("reddit", "x.com", "twitter", "4chan", "telegram")):
        return "social"
    return "journalist"


def _count_term_hits(text: str, terms: Sequence[str]) -> int:
    low = text.lower()
    return sum(1 for t in terms if t in low)


def _closest_price(prices: List[PricePoint], target: datetime) -> Optional[float]:
    if not prices:
        return None
    best = min(prices, key=lambda p: abs((p.timestamp - target).total_seconds()))
    return best.price


def _direction_from_text(text: str) -> str:
    pos = _count_term_hits(text, POSITIVE_TERMS)
    neg = _count_term_hits(text, NEGATIVE_TERMS)
    if pos >= neg + 2:
        return "positive"
    if neg >= pos + 2:
        return "negative"
    return "neutral"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def cluster_documents(
    docs: List[Document],
    market_entities: List[str],
    time_window_h: int = 48,
    min_sim: float = 0.16,
) -> List[CanonicalEvent]:
    """Greedy temporal+semantic clustering into canonical events."""
    sorted_docs = sorted(docs, key=lambda d: d.timestamp)
    entities_l = [e.lower() for e in market_entities]

    clusters: List[dict] = []

    for doc in sorted_docs:
        doc_text = f"{doc.title} {doc.body}"
        doc_toks = _tokenize(doc_text)
        doc_entities = [e for e in entities_l if e in doc_text.lower()]

        best_idx = None
        best_score = -1.0

        for i, c in enumerate(clusters):
            center_time = c["timestamp"]
            if abs((doc.timestamp - center_time).total_seconds()) > time_window_h * 3600:
                continue

            sim = 0.65 * _jaccard(doc_toks, c["tokens"]) + 0.35 * _jaccard(doc_entities, c["entities"])
            if sim > best_score:
                best_score = sim
                best_idx = i

        if best_idx is not None and best_score >= min_sim:
            c = clusters[best_idx]
            c["docs"].append(doc)
            c["tokens"].update(doc_toks)
            c["entities"].update(doc_entities)
            if doc.timestamp < c["timestamp"]:
                c["timestamp"] = doc.timestamp
                c["title"] = doc.title
        else:
            clusters.append(
                {
                    "title": doc.title,
                    "timestamp": doc.timestamp,
                    "docs": [doc],
                    "tokens": set(doc_toks),
                    "entities": set(doc_entities),
                }
            )

    events: List[CanonicalEvent] = []
    for idx, c in enumerate(clusters):
        docs_in = c["docs"]
        events.append(
            CanonicalEvent(
                event_id=f"evt_{idx+1:03d}",
                doc_ids=[d.id for d in docs_in],
                title=c["title"],
                timestamp=c["timestamp"],
                source_names=list({d.source for d in docs_in}),
                source_urls=[d.url for d in docs_in],
                text="\n".join(f"{d.title}. {d.body}" for d in docs_in),
            )
        )

    return events


def score_event(
    event: CanonicalEvent,
    prices: List[PricePoint],
    market_question: str,
    market_entities: List[str],
    past_events: List[CanonicalEvent],
) -> None:
    text = event.text.lower()
    docs_count = max(1, len(event.doc_ids))

    # Credibility and corroboration
    source_creds = []
    source_types = []
    for url in event.source_urls:
        cred = get_source_credibility(url)
        source_creds.append(cred)
        source_types.append(_domain_source_type(url))

    if not source_creds:
        source_creds = [0.5]
    if not source_types:
        source_types = ["journalist"]

    cred_mean = sum(source_creds) / len(source_creds)
    type_cred_mean = statistics.mean(get_source_type_credibility(t) for t in source_types)
    credibility_score = 0.65 * cred_mean + 0.35 * type_cred_mean

    corroboration_score = min(1.0, (len(set(event.source_names)) - 1) / 3.0)

    # Rumor markers
    hedging_hits = _count_term_hits(text, HEDGING_TERMS)
    hedging_score = min(1.0, hedging_hits / max(3.0, docs_count * 2.0))

    # Entity match to market
    mq_terms = _tokenize(market_question)
    me_terms = _tokenize(" ".join(market_entities))
    event_terms = _tokenize(f"{event.title} {event.text}")
    entity_match = max(_jaccard(event_terms, mq_terms), _jaccard(event_terms, me_terms))

    # Contradiction (mixed polarity in one event)
    pos_hits = _count_term_hits(text, POSITIVE_TERMS)
    neg_hits = _count_term_hits(text, NEGATIVE_TERMS)
    contradiction = min(pos_hits, neg_hits) / max(1.0, max(pos_hits, neg_hits))

    # Novelty vs previous canonical events
    if past_events:
        max_prev_sim = max(_jaccard(event_terms, _tokenize(pe.title + " " + pe.text)) for pe in past_events)
        novelty = 1.0 - max_prev_sim
    else:
        novelty = 1.0

    # Event type and direction
    event.direction = _direction_from_text(text)

    rumor_logit = (
        1.6 * hedging_score
        + 0.7 * (1.0 - credibility_score)
        + 0.4 * (1.0 - corroboration_score)
        + 0.3 * contradiction
        - 1.0 * entity_match
    )
    rumor_prob = _sigmoid(rumor_logit)
    if credibility_score >= 0.78 and corroboration_score >= 0.30 and hedging_score <= 0.25:
        event.event_type = "news"
    else:
        event.event_type = "rumor" if rumor_prob >= 0.60 else "news"

    # Adaptive multi-window price mapping
    best_window = None
    best_score = -1.0
    best_tuple = (None, None, None)

    direction_sign = 0
    if event.direction == "positive":
        direction_sign = 1
    elif event.direction == "negative":
        direction_sign = -1

    for w in WINDOWS_MIN:
        before_t = event.timestamp - timedelta(minutes=w)
        after_t = event.timestamp + timedelta(minutes=w)
        pb = _closest_price(prices, before_t)
        pa = _closest_price(prices, after_t)
        if pb is None or pa is None:
            continue

        delta = pa - pb
        abs_move = abs(delta)
        rel_move = abs(delta / pb) if pb > 1e-6 else 0.0

        alignment = 0.0
        if direction_sign != 0:
            alignment = 1.0 if (delta * direction_sign) > 0 else 0.0
        else:
            alignment = 0.5

        price_score = min(1.0, 2.0 * abs_move + rel_move) * (0.6 + 0.4 * alignment)
        if price_score > best_score:
            best_score = price_score
            best_window = w
            best_tuple = (pb, pa, delta)

    if best_window is not None:
        pb, pa, delta = best_tuple
        event.chosen_window_min = best_window
        event.price_before = pb
        event.price_after = pa
        event.price_delta = delta
        event.price_alignment_score = max(0.0, best_score)
    else:
        event.chosen_window_min = None
        event.price_before = None
        event.price_after = None
        event.price_delta = None
        event.price_alignment_score = 0.0

    # Evidence and confidence scores
    event.evidence_score = (
        0.30 * credibility_score
        + 0.25 * corroboration_score
        + 0.20 * entity_match
        + 0.15 * event.price_alignment_score
        + 0.10 * novelty
        - 0.15 * contradiction
    )
    event.evidence_score = max(0.0, min(1.0, event.evidence_score))

    type_conf = 1.0 - abs(0.5 - rumor_prob) * 2.0

    event.confidence = (
        0.28 * event.evidence_score
        + 0.18 * credibility_score
        + 0.14 * corroboration_score
        + 0.14 * entity_match
        + 0.12 * event.price_alignment_score
        + 0.08 * (1.0 - hedging_score if event.event_type == "news" else hedging_score)
        + 0.06 * type_conf
    )
    event.confidence = max(0.0, min(1.0, event.confidence))

    # Publish gates
    if (
        event.confidence >= CONFIRMED_CONFIDENCE_MIN
        and event.evidence_score >= CONFIRMED_EVIDENCE_MIN
        and corroboration_score >= 0.34
    ):
        event.status = "confirmed"
    elif (
        event.confidence >= PROVISIONAL_CONFIDENCE_MIN
        and event.evidence_score >= PROVISIONAL_EVIDENCE_MIN
    ):
        event.status = "provisional"
    else:
        event.status = "dropped"

    if contradiction > 0.7 and event.price_alignment_score < 0.2:
        event.status = "dropped"

    event.feature_debug = {
        "credibility_score": round(credibility_score, 4),
        "corroboration_score": round(corroboration_score, 4),
        "hedging_score": round(hedging_score, 4),
        "entity_match": round(entity_match, 4),
        "contradiction": round(contradiction, 4),
        "novelty": round(novelty, 4),
        "rumor_prob": round(rumor_prob, 4),
        "type_confidence": round(type_conf, 4),
        "price_alignment_score": round(event.price_alignment_score, 4),
    }


def load_input(path: Optional[Path]) -> Tuple[str, List[str], List[Document], List[PricePoint]]:
    if path is None:
        return embedded_sample()

    raw = json.loads(path.read_text(encoding="utf-8"))
    market_question = raw.get("market_question", "Unknown market")
    market_entities = raw.get("market_entities", [])

    docs = [
        Document(
            id=d["id"],
            title=d.get("title", ""),
            body=d.get("body", ""),
            url=d.get("url", ""),
            source=d.get("source", "unknown"),
            timestamp=_parse_ts(d["timestamp"]),
        )
        for d in raw.get("documents", [])
    ]

    prices = [
        PricePoint(timestamp=_parse_ts(p["timestamp"]), price=float(p["price"]))
        for p in raw.get("prices", [])
    ]

    prices.sort(key=lambda x: x.timestamp)
    docs.sort(key=lambda x: x.timestamp)
    return market_question, market_entities, docs, prices


def embedded_sample() -> Tuple[str, List[str], List[Document], List[PricePoint]]:
    market_question = "Will Congress pass a nationwide AI regulation bill by Q4 2026?"
    market_entities = ["Congress", "AI regulation", "Senate", "House"]

    base = datetime(2026, 4, 14, 12, 0, tzinfo=timezone.utc)

    docs = [
        Document(
            id="d1",
            title="Senate committee advances AI safety bill in bipartisan vote",
            body="Committee approved draft and scheduled floor coordination next week.",
            url="https://www.reuters.com/world/us/senate-ai-bill-advance",
            source="Reuters",
            timestamp=base,
        ),
        Document(
            id="d2",
            title="AP: AI regulation proposal clears key hurdle in Senate",
            body="Officials confirmed support from several undecided senators.",
            url="https://apnews.com/article/ai-regulation-senate-hurdle",
            source="AP News",
            timestamp=base + timedelta(minutes=30),
        ),
        Document(
            id="d3",
            title="Sources say bill could be delayed by amendment fight",
            body="Unconfirmed reports suggest leadership might postpone floor debate.",
            url="https://x.com/randomaccount/status/12345",
            source="X",
            timestamp=base + timedelta(hours=5),
        ),
        Document(
            id="d4",
            title="White House statement backs timeline for AI safeguards",
            body="Official statement says administration expects progress this quarter.",
            url="https://www.whitehouse.gov/briefing-room/statement/ai-safeguards",
            source="White House",
            timestamp=base + timedelta(hours=6),
        ),
    ]

    prices = []
    price = 0.46
    for i in range(0, 14):
        ts = base - timedelta(hours=3) + timedelta(hours=i)
        # mild upward drift with one down move around rumor
        if i == 9:
            price -= 0.03
        elif i in (4, 5, 10):
            price += 0.02
        else:
            price += 0.003
        price = min(0.99, max(0.01, price))
        prices.append(PricePoint(timestamp=ts, price=round(price, 4)))

    return market_question, market_entities, docs, prices


async def load_from_db(
    market_ref: str,
    db_url: Optional[str],
    doc_limit: int,
    lookback_days: int,
) -> Tuple[str, List[str], List[Document], List[PricePoint], Dict[str, str]]:
    """
    Load market question, documents, and price points from real backend DB data.

    market_ref matches `markets.event_id` OR `markets.polymarket_slug`.
    """
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine
    except Exception as exc:
        raise RuntimeError(
            "DB mode requires sqlalchemy + asyncpg in this Python environment. "
            "Install backend deps first."
        ) from exc

    async_db_url = _as_async_db_url(db_url)
    engine = create_async_engine(async_db_url, echo=False, pool_pre_ping=True)

    try:
        async with engine.connect() as conn:
            market_row = (
                await conn.execute(
                    text(
                        """
                        SELECT id::text AS market_id, event_id, polymarket_slug, title
                        FROM markets
                        WHERE event_id = :market_ref
                           OR polymarket_slug = :market_ref
                        ORDER BY created_at DESC
                        LIMIT 1
                        """
                    ),
                    {"market_ref": market_ref},
                )
            ).mappings().first()

            if not market_row:
                raise ValueError(
                    f"No market found for '{market_ref}'. Use a valid event_id or polymarket_slug."
                )

            market_id = market_row["market_id"]
            event_id = market_row["event_id"]
            market_question = market_row["title"] or event_id or market_ref
            market_entities = _tokenize(market_question)[:8]

            feed_rows = (
                await conn.execute(
                    text(
                        """
                        SELECT *
                        FROM (
                          SELECT
                            id::text AS doc_id,
                            COALESCE(headline, summary, '[untitled]') AS title,
                            COALESCE(raw_text, summary, '') AS body,
                            COALESCE(url, '') AS url,
                            COALESCE(source, 'unknown') AS source,
                            timestamp
                          FROM feed_items
                          WHERE (market_id::text = :market_id OR event_id = :event_id)
                            AND timestamp IS NOT NULL
                            AND (headline IS NOT NULL OR summary IS NOT NULL)
                          ORDER BY timestamp DESC
                          LIMIT :doc_limit
                        ) t
                        ORDER BY timestamp ASC
                        """
                    ),
                    {
                        "market_id": market_id,
                        "event_id": event_id,
                        "doc_limit": doc_limit,
                    },
                )
            ).mappings().all()

            docs = [
                Document(
                    id=r["doc_id"],
                    title=r["title"],
                    body=r["body"],
                    url=r["url"],
                    source=r["source"],
                    timestamp=_ensure_utc(r["timestamp"]),
                )
                for r in feed_rows
            ]

            cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            snap_rows = (
                await conn.execute(
                    text(
                        """
                        SELECT captured_at AS timestamp, probability AS price
                        FROM market_snapshots
                        WHERE market_id::text = :market_id
                          AND captured_at >= :cutoff
                          AND probability IS NOT NULL
                        ORDER BY captured_at ASC
                        """
                    ),
                    {"market_id": market_id, "cutoff": cutoff},
                )
            ).mappings().all()

            prices = [
                PricePoint(
                    timestamp=_ensure_utc(r["timestamp"]),
                    price=float(r["price"]),
                )
                for r in snap_rows
            ]

            # Fallback when snapshots are sparse: derive points from timeline_events
            if len(prices) < 8:
                tl_rows = (
                    await conn.execute(
                        text(
                            """
                            SELECT timestamp, price_before, price_after
                            FROM timeline_events
                            WHERE market_id::text = :market_id
                              AND timestamp >= :cutoff
                              AND price_before IS NOT NULL
                              AND price_after IS NOT NULL
                            ORDER BY timestamp ASC
                            """
                        ),
                        {"market_id": market_id, "cutoff": cutoff},
                    )
                ).mappings().all()

                tl_points: List[PricePoint] = []
                for r in tl_rows:
                    ts = _ensure_utc(r["timestamp"])
                    tl_points.append(PricePoint(timestamp=ts - timedelta(minutes=5), price=float(r["price_before"])))
                    tl_points.append(PricePoint(timestamp=ts + timedelta(minutes=5), price=float(r["price_after"])))

                if tl_points:
                    tl_points.sort(key=lambda p: p.timestamp)
                    prices = tl_points

            meta = {
                "market_id": market_id,
                "event_id": event_id or "",
                "polymarket_slug": market_row["polymarket_slug"] or "",
                "db_url_used": async_db_url,
            }
            return market_question, market_entities, docs, prices, meta
    finally:
        await engine.dispose()


def evaluate(events: List[CanonicalEvent]) -> Dict[str, float]:
    published = [e for e in events if e.status in {"provisional", "confirmed"}]
    confirmed = [e for e in events if e.status == "confirmed"]
    rumors = [e for e in published if e.event_type == "rumor"]
    news = [e for e in published if e.event_type == "news"]

    directional = [
        e
        for e in published
        if e.direction in {"positive", "negative"} and e.price_delta is not None and abs(e.price_delta) > 1e-6
    ]

    aligned = 0
    for e in directional:
        if (e.direction == "positive" and e.price_delta > 0) or (e.direction == "negative" and e.price_delta < 0):
            aligned += 1

    return {
        "total_events": len(events),
        "published_events": len(published),
        "confirmed_events": len(confirmed),
        "published_news": len(news),
        "published_rumors": len(rumors),
        "directional_alignment_rate": round((aligned / len(directional)) if directional else 0.0, 4),
        "avg_confidence_published": round(statistics.mean([e.confidence for e in published]), 4) if published else 0.0,
        "avg_evidence_published": round(statistics.mean([e.evidence_score for e in published]), 4) if published else 0.0,
    }


def run_pipeline(market_question: str, market_entities: List[str], docs: List[Document], prices: List[PricePoint]) -> Dict:
    events = cluster_documents(docs, market_entities=market_entities)
    events.sort(key=lambda e: e.timestamp)

    scored: List[CanonicalEvent] = []
    for e in events:
        score_event(
            event=e,
            prices=prices,
            market_question=market_question,
            market_entities=market_entities,
            past_events=scored,
        )
        scored.append(e)

    metrics = evaluate(scored)

    return {
        "market_question": market_question,
        "market_entities": market_entities,
        "documents_count": len(docs),
        "price_points_count": len(prices),
        "events": [
            {
                **asdict(e),
                "timestamp": e.timestamp.isoformat(),
            }
            for e in scored
        ],
        "metrics": metrics,
    }


def _print_report(report: Dict) -> None:
    print("\n" + "=" * 92)
    print("CLASSICAL EVENT-SIGNAL PIPELINE TEST REPORT (NO LLM)")
    print("=" * 92)
    print(f"Market: {report['market_question']}")
    if report.get("run_context"):
        ctx = report["run_context"]
        print(
            f"Context: source={ctx.get('source_mode', 'unknown')}"
            f" | event_id={ctx.get('event_id', '-')}"
            f" | market_id={ctx.get('market_id', '-')}"
        )
    print(
        f"Inputs: docs={report['documents_count']} | prices={report['price_points_count']} | canonical_events={len(report['events'])}"
    )

    print("\nEvent Results")
    print("-" * 92)
    header = f"{'ID':<8} {'Type':<8} {'Status':<11} {'Dir':<8} {'Conf':>6} {'EvSc':>6} {'Win':>5} {'Δprice':>8}  Title"
    print(header)
    print("-" * len(header))

    for e in report["events"]:
        delta = e["price_delta"]
        delta_s = "n/a" if delta is None else f"{delta:+.4f}"
        win = "n/a" if e["chosen_window_min"] is None else str(e["chosen_window_min"])
        print(
            f"{e['event_id']:<8} {e['event_type']:<8} {e['status']:<11} {e['direction']:<8} "
            f"{e['confidence']:>6.3f} {e['evidence_score']:>6.3f} {win:>5} {delta_s:>8}  {e['title'][:70]}"
        )

    m = report["metrics"]
    print("\nMetrics")
    print("-" * 92)
    for k, v in m.items():
        print(f"{k}: {v}")
    print()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Standalone classical signal pipeline tester (no LLM)")
    ap.add_argument("--input-json", type=Path, default=None, help="Path to input JSON (documents + prices)")
    ap.add_argument("--output-json", type=Path, default=None, help="Optional path to write full report JSON")
    ap.add_argument(
        "--from-db-market",
        type=str,
        default=None,
        help="Load live data from DB using markets.event_id OR markets.polymarket_slug",
    )
    ap.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Optional DB URL override. If omitted, uses DATABASE_URL / DATABASE_URL_SYNC env vars.",
    )
    ap.add_argument(
        "--db-doc-limit",
        type=int,
        default=400,
        help="Max recent feed items to load from DB for the selected market.",
    )
    ap.add_argument(
        "--db-lookback-days",
        type=int,
        default=30,
        help="Price lookback window in days for market_snapshots / timeline fallback.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    run_context: Dict[str, str] = {"source_mode": "embedded_sample"}

    if args.from_db_market:
        try:
            market_question, market_entities, docs, prices, meta = asyncio.run(
                load_from_db(
                    market_ref=args.from_db_market,
                    db_url=args.db_url,
                    doc_limit=args.db_doc_limit,
                    lookback_days=args.db_lookback_days,
                )
            )
            run_context = {"source_mode": "database", **meta}
        except Exception as exc:
            print(f"DB load failed: {exc}")
            return 1
    else:
        market_question, market_entities, docs, prices = load_input(args.input_json)
        if args.input_json is not None:
            run_context = {"source_mode": "input_json", "input_path": str(args.input_json)}

    if not docs:
        print("No documents found. Provide input JSON with documents.")
        return 1

    report = run_pipeline(market_question, market_entities, docs, prices)
    report["run_context"] = run_context
    _print_report(report)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report JSON: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
