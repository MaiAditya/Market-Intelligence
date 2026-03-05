#!/usr/bin/env python3
"""
Clustering Diagnostics — Step-by-Step JSON Output

Runs each stage of the clustering pipeline separately and writes a detailed
JSON file with all intermediate results so you can inspect exactly what
happened at every step.

Usage:
    python scripts/clustering_diagnostics.py [event_id]

Output:
    data/diagnostics/clustering_<event_id>.json
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from belief_graph.event_extractor import EventExtractor
from belief_graph.event_clustering import EventClusterer
from pipeline.event_registry import EventRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("clustering_diagnostics")


def _ts(dt):
    return dt.isoformat() if dt else None


def main():
    registry = EventRegistry()

    if len(sys.argv) >= 2:
        event_id = sys.argv[1]
    else:
        events = registry.get_all_events()
        if not events:
            print("No events in registry")
            sys.exit(1)
        event_id = events[0].event_id

    event = registry.get_event(event_id)
    if event is None:
        print(f"Event not found: {event_id}")
        sys.exit(1)

    market_question = event.event_title
    result = {
        "meta": {
            "event_id": event_id,
            "market_question": market_question,
            "generated_at": datetime.utcnow().isoformat(),
            "clustering_params": {},
        },
        "steps": {},
    }

    # ── Step 1: Extract raw events ──────────────────────────────────
    t0 = time.time()
    extractor = EventExtractor()
    raw_events = extractor.extract_events_for_belief(event_id, max_events=200)
    t_extract = time.time() - t0

    result["steps"]["1_raw_events"] = {
        "description": "Events extracted from normalized documents",
        "count": len(raw_events),
        "time_seconds": round(t_extract, 2),
        "events": [
            {
                "index": i,
                "event_id": ev.event_id,
                "raw_title": ev.raw_title,
                "action": ev.action,
                "object": ev.object,
                "actors": ev.actors,
                "timestamp": _ts(ev.timestamp),
                "source": ev.source,
                "source_doc_id": ev.source_doc_id,
                "certainty": ev.certainty,
                "scope": ev.scope,
            }
            for i, ev in enumerate(raw_events)
        ],
    }

    if not raw_events:
        print("No raw events — nothing to cluster.")
        sys.exit(0)

    # ── Step 2: Build clusterer & generate text representations ─────
    clusterer = EventClusterer(
        similarity_threshold=0.75,
        time_window_hours=96,
        temporal_decay_tau=12.0,
        entity_weight=0.20,
    )
    result["meta"]["clustering_params"] = {
        "similarity_threshold": clusterer.similarity_threshold,
        "model_name": clusterer.model_name,
        "time_window_hours": clusterer.time_window_hours,
        "temporal_decay_tau": clusterer.temporal_decay_tau,
        "entity_weight": clusterer.entity_weight,
    }

    texts = [
        clusterer._get_event_text(ev, market_question=market_question)
        for ev in raw_events
    ]
    result["steps"]["2_text_representations"] = {
        "description": "Text fed into embedding model (market context + title + body snippet + actors)",
        "texts": [
            {"index": i, "text": t}
            for i, t in enumerate(texts)
        ],
    }

    # ── Step 3: Compute embeddings ──────────────────────────────────
    t0 = time.time()
    embeddings = clusterer._compute_embeddings(raw_events, market_question=market_question)
    t_embed = time.time() - t0

    result["steps"]["3_embeddings"] = {
        "description": "Sentence-transformer embeddings (all-mpnet-base-v2). Shape shown, full vectors omitted for size.",
        "time_seconds": round(t_embed, 2),
        "shape": list(embeddings.shape) if embeddings is not None else None,
        "model": clusterer.model_name,
    }

    if embeddings is None:
        print("Embeddings failed — model not available.")
        sys.exit(1)

    # ── Step 4: Semantic similarity matrix ──────────────────────────
    t0 = time.time()
    semantic_sim = clusterer._compute_similarity_matrix(embeddings)
    t_sim = time.time() - t0

    n = len(raw_events)
    # Build top pairs list
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                "i": i,
                "j": j,
                "title_i": (raw_events[i].raw_title or "")[:60],
                "title_j": (raw_events[j].raw_title or "")[:60],
                "similarity": round(float(semantic_sim[i, j]), 4),
            })
    pairs.sort(key=lambda p: p["similarity"], reverse=True)

    result["steps"]["4_semantic_similarity"] = {
        "description": "Cosine similarity between all event pairs (from embeddings)",
        "time_seconds": round(t_sim, 4),
        "stats": {
            "max": round(float(np.max(semantic_sim[np.triu_indices(n, k=1)])), 4),
            "mean": round(float(np.mean(semantic_sim[np.triu_indices(n, k=1)])), 4),
            "min": round(float(np.min(semantic_sim[np.triu_indices(n, k=1)])), 4),
            "pairs_above_075": sum(1 for p in pairs if p["similarity"] >= 0.75),
            "pairs_above_060": sum(1 for p in pairs if p["similarity"] >= 0.60),
            "total_pairs": len(pairs),
        },
        "top_30_pairs": pairs[:30],
        "bottom_10_pairs": pairs[-10:],
    }

    # ── Step 5: Entity overlap Jaccard ──────────────────────────────
    t0 = time.time()
    entity_jaccard = clusterer._compute_entity_overlap_matrix(raw_events)
    t_entity = time.time() - t0

    entity_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            jac = float(entity_jaccard[i, j])
            if jac > 0:
                entity_pairs.append({
                    "i": i,
                    "j": j,
                    "title_i": (raw_events[i].raw_title or "")[:60],
                    "title_j": (raw_events[j].raw_title or "")[:60],
                    "jaccard": round(jac, 4),
                })
    entity_pairs.sort(key=lambda p: p["jaccard"], reverse=True)

    result["steps"]["5_entity_overlap"] = {
        "description": "Jaccard similarity based on shared named entities (actors + extracted entities)",
        "time_seconds": round(t_entity, 4),
        "stats": {
            "non_zero_pairs": len(entity_pairs),
            "max_jaccard": round(float(np.max(entity_jaccard[np.triu_indices(n, k=1)])), 4) if n > 1 else 0,
            "mean_jaccard": round(float(np.mean(entity_jaccard[np.triu_indices(n, k=1)])), 4) if n > 1 else 0,
        },
        "top_20_entity_pairs": entity_pairs[:20],
    }

    # ── Step 6: Fused similarity (semantic + entity boost) ──────────
    w = clusterer.entity_weight
    blended = (1 - w) * semantic_sim + w * entity_jaccard
    fused_sim = np.maximum(semantic_sim, blended)

    fused_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(semantic_sim[i, j])
            f = float(fused_sim[i, j])
            if abs(f - s) > 0.001:  # only show pairs where entity boost made a difference
                fused_pairs.append({
                    "i": i,
                    "j": j,
                    "title_i": (raw_events[i].raw_title or "")[:60],
                    "title_j": (raw_events[j].raw_title or "")[:60],
                    "semantic_sim": round(s, 4),
                    "entity_jaccard": round(float(entity_jaccard[i, j]), 4),
                    "fused_sim": round(f, 4),
                    "boost": round(f - s, 4),
                })
    fused_pairs.sort(key=lambda p: p["boost"], reverse=True)

    result["steps"]["6_fused_similarity"] = {
        "description": f"Final similarity = max(semantic, (1-{w})*semantic + {w}*entity_jaccard). Entity overlap can only BOOST, never reduce.",
        "formula": f"fused = max(semantic, {1-w:.1f}*semantic + {w:.1f}*jaccard)",
        "pairs_where_entity_helped": len(fused_pairs),
        "top_20_boosted_pairs": fused_pairs[:20],
    }

    # ── Step 7: Temporal decay ──────────────────────────────────────
    t0 = time.time()
    decayed_sim = clusterer._apply_temporal_decay(raw_events, fused_sim)
    t_decay = time.time() - t0

    decay_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            f = float(fused_sim[i, j])
            d = float(decayed_sim[i, j])
            if abs(d - f) > 0.001:
                ts_i = raw_events[i].timestamp
                ts_j = raw_events[j].timestamp
                hours_apart = abs((ts_i - ts_j).total_seconds()) / 3600 if ts_i and ts_j else None
                decay_pairs.append({
                    "i": i,
                    "j": j,
                    "title_i": (raw_events[i].raw_title or "")[:60],
                    "title_j": (raw_events[j].raw_title or "")[:60],
                    "hours_apart": round(hours_apart, 1) if hours_apart else None,
                    "before_decay": round(f, 4),
                    "after_decay": round(d, 4),
                    "decay_factor": round(d / f, 4) if f > 0 else 0,
                })
    decay_pairs.sort(key=lambda p: p["decay_factor"])

    result["steps"]["7_temporal_decay"] = {
        "description": f"Soft exponential decay for events beyond {clusterer.time_window_hours}h window (tau={clusterer.temporal_decay_tau}h)",
        "time_seconds": round(t_decay, 4),
        "pairs_affected": len(decay_pairs),
        "top_20_most_decayed": decay_pairs[:20],
    }

    # ── Step 8: HAC clustering ──────────────────────────────────────
    t0 = time.time()
    cluster_indices = clusterer._cluster_by_similarity(raw_events, semantic_sim)
    t_hac = time.time() - t0

    result["steps"]["8_cluster_assignments"] = {
        "description": "HAC average-linkage cluster assignments (indices → cluster groups)",
        "time_seconds": round(t_hac, 2),
        "num_clusters": len(cluster_indices),
        "multi_source_clusters": sum(1 for c in cluster_indices if len(c) > 1),
        "singleton_clusters": sum(1 for c in cluster_indices if len(c) == 1),
        "clusters": [
            {
                "cluster_id": ci,
                "size": len(indices),
                "member_indices": indices,
                "members": [
                    {
                        "index": idx,
                        "title": (raw_events[idx].raw_title or "")[:80],
                        "timestamp": _ts(raw_events[idx].timestamp),
                        "actors": raw_events[idx].actors[:5],
                    }
                    for idx in indices
                ],
            }
            for ci, indices in enumerate(cluster_indices)
        ],
    }

    # ── Step 9: Final clustered events ──────────────────────────────
    t0 = time.time()
    clusters = clusterer.cluster_events(raw_events, market_question=market_question)
    t_total = time.time() - t0

    result["steps"]["9_final_clusters"] = {
        "description": "Final ClusteredEvent objects with canonical events, merged actors, and boosted certainty",
        "time_seconds": round(t_total, 2),
        "total_raw_events": len(raw_events),
        "total_clusters": len(clusters),
        "dedup_percentage": round(100 * (1 - len(clusters) / max(len(raw_events), 1)), 1),
        "clusters": [
            {
                "cluster_index": i,
                "canonical_event": {
                    "event_id": c.canonical_event.event_id,
                    "raw_title": c.canonical_event.raw_title,
                    "timestamp": _ts(c.canonical_event.timestamp),
                    "actors": c.canonical_event.actors,
                    "certainty": round(c.canonical_event.certainty, 4),
                    "source": c.canonical_event.source,
                    "scope": c.canonical_event.scope,
                },
                "num_sources": c.num_sources,
                "avg_similarity": round(c.avg_similarity, 4),
                "member_event_ids": c.member_event_ids,
                "all_actors": list(c.all_actors),
                "source_doc_ids": list(c.source_doc_ids),
            }
            for i, c in enumerate(clusters)
        ],
    }

    # ── Step 10: Timing summary ─────────────────────────────────────
    result["steps"]["10_timing_summary"] = {
        "description": "Time breakdown for each step",
        "extraction_seconds": round(t_extract, 2),
        "embedding_seconds": round(t_embed, 2),
        "similarity_matrix_seconds": round(t_sim, 4),
        "entity_overlap_seconds": round(t_entity, 4),
        "temporal_decay_seconds": round(t_decay, 4),
        "hac_clustering_seconds": round(t_hac, 2),
        "total_cluster_events_seconds": round(t_total, 2),
    }

    # ── Write output ────────────────────────────────────────────────
    out_dir = project_root / "data" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"clustering_{event_id}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*60}")
    print(f"Diagnostics written to: {out_path}")
    print(f"{'='*60}")
    print(f"  Raw events:    {len(raw_events)}")
    print(f"  Final clusters: {len(clusters)}")
    print(f"  Dedup:          {result['steps']['9_final_clusters']['dedup_percentage']}%")
    print(f"  File size:      {out_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
