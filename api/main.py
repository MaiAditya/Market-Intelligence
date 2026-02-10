"""
FastAPI Application

Main entry point for the AI Market Intelligence API.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from api.endpoints import router
from api.belief_graph_endpoints import router as belief_graph_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("AI Market Intelligence API starting up...")
    
    # Startup: preload event registry
    try:
        from pipeline.event_registry import get_registry
        registry = get_registry()
        logger.info(f"Loaded {len(registry)} events from registry")
    except Exception as e:
        logger.error(f"Failed to load event registry: {e}")
    
    yield
    
    # Shutdown
    logger.info("API shutting down...")


# Create FastAPI application
app = FastAPI(
    title="AI Market Intelligence API",
    description="""
    Event-centric intelligence system for AI-related Polymarket markets.
    
    ## Features
    
    - Track AI-related events (model releases, regulations)
    - Ingest data from Reddit, Twitter, and web sources
    - Extract signals using BERT-based NLP models
    - Calculate probability delta ranges
    - Rank relevant documents
    
    ## Key Endpoints
    
    - `GET /events` - List all tracked events
    - `GET /events/{event_id}` - Get analysis for an event
    - `POST /events/{event_id}/refresh` - Trigger pipeline refresh
    - `GET /events/{event_id}/signals` - Get signals for an event
    - `GET /events/{event_id}/documents` - Get documents for an event
    
    ## Belief Graph
    
    - `GET /belief-graph/{event_id}` - Get full belief update DAG
    - `GET /belief-graph/{event_id}/upstream` - Top upstream events by impact
    - `GET /belief-graph/{event_id}/edges` - Edge explanations
    - `POST /belief-graph/{event_id}/build` - Build graph for an event
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")
app.include_router(belief_graph_router, prefix="/api/v1")

# Also mount at root for convenience
app.include_router(router)
app.include_router(belief_graph_router)


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "AI Market Intelligence API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# For running directly with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
