import sys
import os
import logging
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.ingestion.ingestor import DataIngestor
from pipeline.ingestion.web_scraper import WebScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_ingestion(query: str, limit: int = 5, use_browser: bool = True):
    """
    Test data ingestion for a specific query.
    """
    logger.info(f"Starting ingestion test for query: '{query}'")
    logger.info(f"Limit: {limit} results")
    
    # Initialize Ingestor
    ingestor = DataIngestor(max_results_per_query=limit)
    
    # If we want to force browser usage, we might need to adjust WebScraper 
    # but for now we rely on the fallback logic or specific sites that block requests.
    
    # Generate a dummy event ID
    event_id = f"test_event_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Testing Web Ingestion...")
    # We strictly test web ingestion here as that's what we modified
    docs = ingestor.ingest_from_web(
        query=query,
        query_type="test_script",
        event_id=event_id,
        limit=limit
    )
    
    print("\n" + "="*50)
    print(f"INGESTION REPORT FOR: {query}")
    print("="*50)
    print(f"Total Documents Ingested: {len(docs)}")
    print("-" * 30)
    
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}:")
        print(f"Title: {doc.title}")
        print(f"URL: {doc.url}")
        print(f"Source: {doc.source}")
        print(f"Text Length: {len(doc.raw_text)} chars")
        print(f"ID: {doc.doc_id}")
        
    if not docs:
        print("\nNo documents found. Check logs for details.")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Data Ingestion Flow")
    parser.add_argument("query", type=str, help="Search query to test")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to fetch")
    
    args = parser.parse_args()
    
    test_ingestion(args.query, args.limit)
