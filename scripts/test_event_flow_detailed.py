"""
Test Event Flow Detailed

Verifies the full pipeline flow:
1. Ingestion (Mocked)
2. Event Mapping (Soft/Hard Gate)
3. Event Extraction
4. Event Clustering (Temporal + HAC + Synthesis)

Run with: python scripts/test_event_flow_detailed.py
"""

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from belief_graph.event_clustering import EventClusterer
from belief_graph.event_extractor import EventExtractor
from belief_graph.models import EventNode
from pipeline.event_mapper import EventMapper
from pipeline.event_registry import Event
from pipeline.normalizer import NormalizedDocument

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_flow")


def _utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def create_mock_documents() -> List[NormalizedDocument]:
    """Create a diverse set of documents to test all pipeline features."""
    base_time = _utc_now()
    
    docs = []
    
    # 1. Hard Gate Match (Primary + Secondary)
    docs.append(NormalizedDocument(
        doc_id="doc_hard_match_1",
        url="https://news.com/1",
        source_type="journalist",
        author_type="journalist",
        title="Google releases Gemini 1.5",
        raw_text="Google announced the release of Gemini 1.5 today. It performs well on benchmarks.",
        timestamp=base_time,
        query_used="Google Gemini",
        query_type="model_release",
        event_id="evt_gemini_release",
        extracted_entities=[
            {"text": "Google", "type": "ORG"},
            {"text": "Gemini", "type": "PRODUCT"}
        ]
    ))
    
    # 2. Soft Gate Match (Fails hard gate, passes semantic)
    # "The search giant" instead of "Google" -> fails hard gate
    # But text is highly relevant -> should pass soft gate
    docs.append(NormalizedDocument(
        doc_id="doc_soft_match_1",
        url="https://techblog.com/2",
        source_type="journalist",
        author_type="journalist",
        title="The search giant unveils new AI model",
        raw_text=(
            "The search giant unveiled its latest AI model today. "
            "It offers a 1M token context window and beats GPT-4 on reasoning. "
            "This is a major step forward for the company's AI strategy."
        ),
        timestamp=base_time,
        query_used="AI model news",
        query_type="model_release",
        event_id="evt_gemini_release",
        extracted_entities=[{"text": "AI model", "type": "misc"}]
    ))
    
    # 3. Temporal Split - Cluster A
    # "Layoffs" at time T
    docs.append(NormalizedDocument(
        doc_id="doc_layoff_jan",
        url="https://news.com/layoffs-jan",
        source_type="journalist",
        author_type="journalist",
        title="Major layoffs announced at TechCorp",
        raw_text="TechCorp executes major layoffs starting today.",
        timestamp=base_time,
        query_used="TechCorp layoffs",
        query_type="general",
        event_id="evt_layoff_jan",
        extracted_entities=[{"text": "TechCorp", "type": "ORG"}]
    ))
    
    # 4. Temporal Split - Cluster B
    # "Layoffs" at time T + 5 days (beyond 48h window) -> Should be separate cluster
    docs.append(NormalizedDocument(
        doc_id="doc_layoff_feb",
        url="https://news.com/layoffs-feb",
        source_type="journalist",
        author_type="journalist",
        title="Major layoffs announced at TechCorp",
        raw_text="TechCorp executes major layoffs starting today.",
        timestamp=base_time + timedelta(days=5),
        query_used="TechCorp layoffs",
        query_type="general",
        event_id="evt_layoff_feb",
        extracted_entities=[{"text": "TechCorp", "type": "ORG"}]
    ))
    
    # 5. Canonical Synthesis - Cluster Members
    # 3 documents about the same event to test synthesis
    for i in range(3):
        docs.append(NormalizedDocument(
            doc_id=f"doc_synthesis_{i}",
            url=f"https://source{i}.com/event",
            source_type="journalist",
            author_type="journalist",
            title=f"Event details from source {i}",
            raw_text="Details about the big event happening now.",
            # Slightly different times to test consensus
            timestamp=base_time + timedelta(hours=i),
            query_used="Big Event",
            query_type="general",
            event_id="evt_synthesis",
            extracted_entities=[{"text": "BigEvent", "type": "EVENT"}]
        ))

    return docs


def run_test():
    logger.info("Starting Detailed Flow Verification...")
    
    # 1. Setup Event Registry (Mock event)
    event = Event(
        event_id="evt_gemini_release",
        event_type="model_release",
        event_title="Google Gemini Release",
        event_description="Google releases the Gemini AI model family.",
        primary_entities=["Google", "DeepMind"],
        secondary_entities=["Gemini", "Bard", "PaLM"],
        aliases=["Gemini 1.5", "Gemini Ultra"],
        deadline=_utc_now() + timedelta(days=30),
        dependencies=["training"],
        polymarket_slug="dummy-slug"
    )
    
    # 2. Run Mapper (Soft Gate)
    logger.info("\n=== STEP 1: EVENT MAPPING (Soft Gate) ===")
    mapper = EventMapper()
    docs = create_mock_documents()
    
    mappings = []
    valid_docs = []
    
    for doc in docs:
        # We use a broad event for mapping to test the soft gate logic generally
        # In reality docs map to specific events, but here we just want to see gate behavior
        # so we'll check against our target event for the first 2 docs
        
        target_event = event
        # For the "layoff" docs, we'd need a different event, but let's just 
        # map them all to check the gate logic behavior
        
        mapping = mapper.map_document_to_event(doc, target_event)
        
        logger.info(f"Doc: {doc.doc_id}")
        logger.info(f"  - Entity Gate: {'PASSED' if mapping.entity_gate_passed else 'FAILED'}")
        logger.info(f"  - Soft Gate:   {'PASSED' if mapping.soft_gate_passed else 'FAILED'} (score={mapping.soft_entity_score:.3f})")
        logger.info(f"  - Semantic:    {mapping.relevance_score:.3f} (passed={mapping.relevance_passed})")
        
        if mapping.doc_id == "doc_soft_match_1":
             if mapping.soft_gate_passed:
                 logger.info("  SUCCESS: Soft gate correctly caught 'The search giant'!")
             else:
                 logger.warning("  FAILURE: Soft gate missed the relevant doc.")

        if mapping.relevance_passed or mapping.soft_gate_passed:
            mappings.append(mapping)
            valid_docs.append(doc)

    # 3. Run Extractor
    logger.info("\n=== STEP 2: EVENT EXTRACTION ===")
    extractor = EventExtractor()
    event_nodes = []
    
    for doc in docs:
        # Extract everything regardless of mapping for the clustering test
        node = extractor.extract_event_node(doc)
        if node:
            event_nodes.append(node)
            logger.info(f"Extracted: {node.event_id} | {node.raw_title}")
    
    # 4. Run Clusterer (Temporal + HAC + Synthesis)
    logger.info("\n=== STEP 3: EVENT CLUSTERING (Temporal + HAC + Synthesis) ===")
    clusterer = EventClusterer(
        similarity_threshold=0.75,
        time_window_hours=48  # 48h window
    )
    
    clusters = clusterer.cluster_events(event_nodes)
    
    logger.info(f"formed {len(clusters)} clusters from {len(event_nodes)} events.")
    
    for i, cluster in enumerate(clusters):
        print(f"\nCluster {i+1}:")
        print(f"  Canonical: {cluster.canonical_event.raw_title}")
        print(f"  Timestamp: {cluster.canonical_event.timestamp}")
        print(f"  Members:   {len(cluster.member_event_ids)}")
        print(f"  Certainty: {cluster.canonical_event.certainty:.3f}")
        print(f"  Sources:   {cluster.num_sources}")
        
        # Verification Checks
        member_titles = [
            next(e.raw_title for e in event_nodes if e.event_id == mid)
            for mid in cluster.member_event_ids
        ]
        
        # Check Temporal Split
        if "Layoffs" in cluster.canonical_event.raw_title:
            if len(cluster.member_event_ids) == 1:
                logger.info("  -> Temporal Check: Correctly kept separate (1 member).")
            else:
                logger.warning("  -> Temporal Check: FAILED (merged temporally distant events).")

        # Check Synthesis
        if "Event details" in cluster.canonical_event.raw_title:
             if cluster.num_sources == 3:
                 logger.info("  -> Synthesis Check: Correctly grouped 3 events.")
             else:
                 logger.warning(f"  -> Synthesis Check: FAILED (expected 3, got {cluster.num_sources}).")


if __name__ == "__main__":
    run_test()
