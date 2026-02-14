import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from belief_graph.graph_builder import GraphBuilder
from belief_graph.storage import get_storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("graph_runner")

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_graph_build.py <event_id>")
        sys.exit(1)
        
    event_id = sys.argv[1]
    
    logger.info(f"Starting graph build for: {event_id}")
    
    try:
        builder = GraphBuilder()
        storage = get_storage()
        
        logger.info("Building graph...")
        graph = builder.build(event_id)
        
        logger.info("Saving graph to storage...")
        path = storage.save(graph)
        
        print(f"SUCCESS: Graph saved to {path}")
        
    except Exception as e:
        logger.error(f"Failed to build graph: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
