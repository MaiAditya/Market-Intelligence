#!/usr/bin/env python3
"""
Test Event Clustering on Real Pipeline Data

Runs the event extractor + clustering on real normalized documents
for a given event, showing before/after cluster results.

Usage:
    python scripts/test_clustering.py [event_id]
    python scripts/test_clustering.py  # defaults to first available event
"""

import json
import logging
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from belief_graph.event_extractor import EventExtractor
from belief_graph.event_clustering import EventClusterer
from pipeline.event_registry import EventRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_clustering")


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
        print("Available:", [e.event_id for e in registry.get_all_events()])
        sys.exit(1)
    
    market_question = event.event_title
    print(f"\n{'='*80}")
    print(f"EVENT: {event_id}")
    print(f"MARKET QUESTION: {market_question}")
    print(f"{'='*80}\n")
    
    # Step 1: Extract events from documents
    extractor = EventExtractor()
    raw_events = extractor.extract_events_for_belief(event_id, max_events=200)
    
    print(f"\n--- RAW EVENTS: {len(raw_events)} ---")
    for i, ev in enumerate(raw_events):
        ts = ev.timestamp.strftime("%Y-%m-%d %H:%M") if ev.timestamp else "no-ts"
        print(f"  [{i:3d}] {ts} | {(ev.raw_title or 'no title')[:70]} | actors={ev.actors[:3]}")
    
    if not raw_events:
        print("No events extracted. Check that documents exist in data/normalized/")
        sys.exit(0)
    
    # Step 2: Cluster with the improved clusterer
    print(f"\n{'='*80}")
    print("CLUSTERING (improved: entity boost + soft decay + body text + market context)")
    print(f"{'='*80}\n")
    
    clusterer = EventClusterer(
        similarity_threshold=0.75,
        time_window_hours=96,
        temporal_decay_tau=12.0,
        entity_weight=0.20,
    )
    
    t0 = time.time()
    clusters = clusterer.cluster_events(
        raw_events,
        market_question=market_question,
    )
    t_new = time.time() - t0
    print(f"\n⏱️  NEW clustering took {t_new:.2f}s")
    
    # --- Diagnostics: show the similarity distribution ---
    print(f"\n--- SIMILARITY DIAGNOSTICS ---")
    embeddings = clusterer._compute_embeddings(raw_events, market_question=market_question)
    if embeddings is not None:
        sim_matrix = clusterer._compute_similarity_matrix(embeddings)
        # Get top-20 most similar pairs (excluding diagonal)
        n = len(raw_events)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((sim_matrix[i, j], i, j))
        pairs.sort(reverse=True)
        
        print(f"  Max similarity:  {pairs[0][0]:.4f}")
        print(f"  Mean similarity: {np.mean([p[0] for p in pairs]):.4f}")
        above_075 = sum(1 for p in pairs if p[0] >= 0.75)
        above_060 = sum(1 for p in pairs if p[0] >= 0.60)
        print(f"  Pairs >= 0.75:   {above_075}")
        print(f"  Pairs >= 0.60:   {above_060}")
        print(f"  Total pairs:     {len(pairs)}")
        print(f"\n  TOP 20 MOST SIMILAR PAIRS:")
        for sim, i, j in pairs[:20]:
            ts_i = raw_events[i].timestamp.strftime("%m-%d") if raw_events[i].timestamp else "?"
            ts_j = raw_events[j].timestamp.strftime("%m-%d") if raw_events[j].timestamp else "?"
            print(f"    sim={sim:.3f} | [{ts_i}] {(raw_events[i].raw_title or '?')[:40]}")
            print(f"           | [{ts_j}] {(raw_events[j].raw_title or '?')[:40]}")
        print()
    
    print(f"\n--- CLUSTERS: {len(clusters)} (from {len(raw_events)} raw events) ---")
    print(f"--- REDUCTION: {len(raw_events)} → {len(clusters)} ({100*(1-len(clusters)/max(len(raw_events),1)):.0f}% dedup) ---\n")
    
    for i, cluster in enumerate(clusters):
        canon = cluster.canonical_event
        ts = canon.timestamp.strftime("%Y-%m-%d %H:%M") if canon.timestamp else "no-ts"
        print(f"  CLUSTER {i+1} ({cluster.num_sources} sources, avg_sim={cluster.avg_similarity:.3f})")
        print(f"    Canonical: {ts} | {(canon.raw_title or 'no title')[:70]}")
        print(f"    Actors:    {list(cluster.all_actors)[:5]}")
        print(f"    Certainty: {canon.certainty:.3f}")
        if cluster.num_sources > 1:
            print(f"    Members:   {cluster.member_event_ids}")
        print()
    
    # Step 3 (optional): Compare with OLD clustering (no improvements)
    print(f"\n{'='*80}")
    print("COMPARISON: Old clustering (no entity boost, hard mask, 48h window, no context)")
    print(f"{'='*80}\n")
    
    old_clusterer = EventClusterer(
        similarity_threshold=0.75,
        time_window_hours=48,
        temporal_decay_tau=0.001,   # effectively hard cutoff
        entity_weight=0.0,         # no entity boost
    )
    
    t0 = time.time()
    old_clusters = old_clusterer.cluster_events(raw_events)  # no market_question
    t_old = time.time() - t0
    
    print(f"  OLD: {len(old_clusters)} clusters in {t_old:.2f}s (from {len(raw_events)} events)")
    print(f"  NEW: {len(clusters)} clusters in {t_new:.2f}s (from {len(raw_events)} events)")
    print(f"  IMPROVEMENT: {len(old_clusters) - len(clusters)} fewer clusters (better dedup)")
    if t_old > 0:
        print(f"  SPEEDUP: {t_old/t_new:.1f}x faster" if t_new < t_old else f"  TIME DELTA: +{t_new-t_old:.1f}s")
    print()


if __name__ == "__main__":
    main()
