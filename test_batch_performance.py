#!/usr/bin/env python3
"""
Test script to verify batch processing performance improvements.
"""

import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.event_registry import get_registry
from pipeline.event_mapper import EventMapper
from pipeline.normalizer import DocumentNormalizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def test_batch_vs_sequential():
    """Compare batch vs sequential processing performance."""
    
    logger.info("=" * 60)
    logger.info("BATCH PROCESSING PERFORMANCE TEST")
    logger.info("=" * 60)
    
    # Initialize components
    registry = get_registry()
    mapper = EventMapper()
    normalizer = DocumentNormalizer()
    
    # Get first event
    event = next(iter(registry))
    logger.info(f"Testing with event: {event.event_id}")
    
    # Load documents
    documents = normalizer.load_all_for_event(event.event_id)
    logger.info(f"Loaded {len(documents)} documents")
    
    if len(documents) == 0:
        logger.error("No documents found! Run ingestion first.")
        return
    
    # Test with a subset for quick comparison
    test_size = min(50, len(documents))
    test_docs = documents[:test_size]
    logger.info(f"Testing with {test_size} documents")
    
    # Test 1: Batch processing (NEW)
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: BATCH PROCESSING (OPTIMIZED)")
    logger.info("=" * 60)
    start_time = time.time()
    
    batch_mappings = mapper.map_all_documents_to_event_batch(
        event,
        test_docs,
        save_mappings=False
    )
    
    batch_time = time.time() - start_time
    batch_relevant = sum(1 for m in batch_mappings if m.is_relevant)
    
    logger.info(f"Batch processing complete!")
    logger.info(f"  Time: {batch_time:.2f} seconds")
    logger.info(f"  Speed: {test_size / batch_time:.2f} docs/sec")
    logger.info(f"  Relevant: {batch_relevant}/{test_size}")
    
    # Test 2: Sequential processing (OLD) - only test on small subset
    if test_size <= 20:
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: SEQUENTIAL PROCESSING (OLD METHOD)")
        logger.info("=" * 60)
        start_time = time.time()
        
        sequential_mappings = []
        for doc in test_docs:
            mapping = mapper.map_document_to_event(doc, event)
            sequential_mappings.append(mapping)
        
        sequential_time = time.time() - start_time
        sequential_relevant = sum(1 for m in sequential_mappings if m.is_relevant)
        
        logger.info(f"Sequential processing complete!")
        logger.info(f"  Time: {sequential_time:.2f} seconds")
        logger.info(f"  Speed: {test_size / sequential_time:.2f} docs/sec")
        logger.info(f"  Relevant: {sequential_relevant}/{test_size}")
        
        # Comparison
        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE COMPARISON")
        logger.info("=" * 60)
        speedup = sequential_time / batch_time
        logger.info(f"Speedup: {speedup:.1f}x faster")
        logger.info(f"Time saved: {sequential_time - batch_time:.2f} seconds")
        
        # Verify results match
        if batch_relevant == sequential_relevant:
            logger.info("✓ Results match! Same number of relevant documents.")
        else:
            logger.warning(f"⚠ Results differ: batch={batch_relevant}, sequential={sequential_relevant}")
    else:
        logger.info("\n" + "=" * 60)
        logger.info("ESTIMATED PERFORMANCE GAIN")
        logger.info("=" * 60)
        # Estimate based on typical 3-4 sec per doc
        estimated_sequential = test_size * 3.5
        speedup = estimated_sequential / batch_time
        logger.info(f"Estimated sequential time: {estimated_sequential:.2f} seconds")
        logger.info(f"Estimated speedup: {speedup:.1f}x faster")
        logger.info(f"Estimated time saved: {estimated_sequential - batch_time:.2f} seconds")
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_batch_vs_sequential()
