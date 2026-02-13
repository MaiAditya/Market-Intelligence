import sys
import os
import logging
import argparse
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.event_registry import Event
from pipeline.query_generator import QueryGenerator
from pipeline.ingestion.ingestor import DataIngestor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pipeline_ingestion(
    title: str,
    description: str,
    limit: int = 5,
    event_type: str = "model_release"
):
    """
    Test the full ingestion pipeline flow for a dynamic event.
    """
    logger.info(f"Starting pipeline ingestion test for: '{title}'")
    
    # 1. Create a temporary Event object
    # We need to fill required fields
    event_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Simple heuristic for entities (just splitting title)
    primary_entities = [w for w in title.split() if len(w) > 3]
    if not primary_entities:
        primary_entities = ["AI"]
        
    event = Event(
        event_id=event_id,
        event_type=event_type,
        event_title=title,
        event_description=description,
        primary_entities=primary_entities,
        secondary_entities=[],
        aliases=[],
        deadline=datetime.now() + timedelta(days=30),
        dependencies=["training"], # Dummy
        polymarket_slug="test-slug" # Dummy
    )
    
    logger.info(f"Created temporary Event object: {event_id}")
    
    # 2. Generate Queries
    logger.info("Step 1: Generating Queries...")
    query_generator = QueryGenerator()
    query_set = query_generator.generate_queries_for_event(event)
    
    queries = [{"query": q.query, "query_type": q.query_type} for q in query_set.queries]
    
    print("\n" + "="*50)
    print(f"GENERATED QUERIES ({len(queries)})")
    print("="*50)
    for q in queries:
        print(f"[{q['query_type']}] {q['query']}")
    print("-" * 30)
    
    # 3. Ingest Data
    logger.info("Step 2: Ingesting Data...")
    ingestor = DataIngestor(max_results_per_query=limit)
    
    docs = ingestor.ingest_for_event(event.event_id, queries)
    
    print("\n" + "="*50)
    print(f"INGESTION REPORT FOR: {title}")
    print("="*50)
    print(f"Total Documents Ingested: {len(docs)}")
    print("-" * 30)
    
    for i, doc in enumerate(docs[:10], 1):
        print(f"\nDocument {i}:")
        print(f"Title: {doc.title}")
        print(f"URL: {doc.url}")
        print(f"Source: {doc.source}")
        print(f"Query: {doc.query_used} ({doc.query_type})")
        print(f"ID: {doc.doc_id}")
        
    if len(docs) > 10:
        print(f"\n... and {len(docs) - 10} more documents.")
        
    print("\n" + "="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Full Pipeline Ingestion")
    parser.add_argument("title", type=str, help="Event title")
    parser.add_argument("--desc", type=str, default="", help="Event description (optional)")
    parser.add_argument("--limit", type=int, default=3, help="Max results per query")
    parser.add_argument("--type", type=str, default="model_release", help="Event type")
    
    args = parser.parse_args()
    
    # Use title as desc if not provided
    description = args.desc if args.desc else args.title
    
    test_pipeline_ingestion(args.title, description, args.limit, args.type)
