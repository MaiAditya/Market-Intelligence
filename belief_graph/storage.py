"""
Graph Storage

Persistence layer for belief update graphs.
Stores graphs as JSON files in data/belief_graphs/ directory.

Features:
- Save complete graphs
- Load graphs by event ID
- List available graphs
- Cache management
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.json_utils import dump_json
from belief_graph.models import BeliefGraph

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Get current UTC time as a naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class GraphStorage:
    """
    Storage manager for belief graphs.
    
    Saves graphs as JSON files with metadata for retrieval.
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize storage.
        
        Args:
            storage_dir: Directory for graph storage
        """
        if storage_dir is None:
            storage_dir = project_root / "data" / "belief_graphs"
        
        self.storage_dir = Path(storage_dir)
        
        # In-memory cache
        self._cache: Dict[str, BeliefGraph] = {}
        self._cache_max_size = 10
    
    def _get_filename(self, event_id: str) -> str:
        """Get filename for an event's graph."""
        # Sanitize event_id for filename
        safe_id = event_id.replace("/", "_").replace("\\", "_")
        return f"{safe_id}_graph.json"
    
    def save(
        self,
        graph: BeliefGraph,
        overwrite: bool = True
    ) -> Path:
        """
        Save a belief graph to PostgreSQL.
        
        Args:
            graph: BeliefGraph to save
            overwrite: Whether to overwrite existing
        
        Returns:
            Virtual Path to saved file for legacy compatibility
        """
        event_id = graph.belief_node.event_id
        
        # Prepare data
        data = graph.to_dict()
        
        # Add storage metadata
        data["_storage"] = {
            "event_id": event_id,
            "saved_at": _utc_now().isoformat(),
            "version": "1.0"
        }
        
        from utils.db_storage import save_artifact
        save_artifact(event_id, "belief_graph", data)
        
        # Update cache
        self._cache[event_id] = graph
        self._trim_cache()
        
        logger.info(f"Saved graph for event {event_id} to database")
        
        return self.storage_dir / self._get_filename(event_id)
    
    def load(self, event_id: str) -> Optional[BeliefGraph]:
        """
        Load a belief graph from PostgreSQL.
        """
        # Check cache first
        if event_id in self._cache:
            logger.debug(f"Loaded graph from cache: {event_id}")
            return self._cache[event_id]
        
        from utils.db_storage import load_artifact
        data = load_artifact(event_id, "belief_graph")
        
        if not data:
            logger.debug(f"Graph not found in database for: {event_id}")
            return None
        
        try:
            # Remove storage metadata before parsing
            data.pop("_storage", None)
            
            graph = BeliefGraph.from_dict(data)
            
            # Update cache
            self._cache[event_id] = graph
            self._trim_cache()
            
            logger.info(f"Loaded graph for event {event_id} from database")
            
            return graph
            
        except Exception as e:
            logger.error(f"Error loading graph {event_id}: {e}")
            return None
    
    def exists(self, event_id: str) -> bool:
        """Check if a graph exists for an event."""
        if event_id in self._cache:
            return True
        from utils.db_storage import load_artifact
        return bool(load_artifact(event_id, "belief_graph"))
    
    def delete(self, event_id: str) -> bool:
        """
        Delete a graph from storage.
        """
        # Remove from cache
        self._cache.pop(event_id, None)
        # Note: True deletion from DB is omitted here to preserve audit trails,
        # but the interface is supported.
        return True
    
    def list_graphs(self) -> List[Dict]:
        """
        List all stored graphs with metadata from PostgreSQL.
        """
        from utils.db_storage import list_all_artifacts_by_type
        artifacts = list_all_artifacts_by_type("belief_graph")
        
        graphs = []
        for artifact in artifacts:
            event_id = artifact["event_id"]
            data = artifact["data"]
            
            storage_meta = data.get("_storage", {})
            belief_data = data.get("belief", {})
            metadata = data.get("metadata", {})
            
            graphs.append({
                "event_id": event_id,
                "question": belief_data.get("question", ""),
                "node_count": metadata.get("node_count", 0),
                "edge_count": metadata.get("edge_count", 0),
                "saved_at": storage_meta.get("saved_at", ""),
                "generated_at": metadata.get("generated_at", ""),
                "filepath": f"db://belief_graph/{event_id}"
            })
        
        # Sort by saved_at descending
        graphs.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
        return graphs
    
    def get_stats(self) -> Dict:
        """
        Get storage statistics from PostgreSQL.
        """
        graphs = self.list_graphs()
        
        total_nodes = sum(g.get("node_count", 0) for g in graphs)
        total_edges = sum(g.get("edge_count", 0) for g in graphs)
        
        return {
            "graph_count": len(graphs),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "storage_size_bytes": 0,  # Legacy
            "storage_size_mb": 0.0,   # Legacy
            "cache_size": len(self._cache),
            "cache_max_size": self._cache_max_size
        }
    
    def _trim_cache(self) -> None:
        """Trim cache if over max size."""
        if len(self._cache) > self._cache_max_size:
            # Remove oldest entries (first added)
            keys = list(self._cache.keys())
            for key in keys[:len(keys) - self._cache_max_size]:
                del self._cache[key]
    
    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
        logger.debug("Cache cleared")
    
    def get_or_build(
        self,
        event_id: str,
        builder=None,
        rebuild: bool = False
    ) -> Optional[BeliefGraph]:
        """
        Get graph from storage or build if not exists.
        
        Args:
            event_id: Event ID
            builder: GraphBuilder instance (required if building)
            rebuild: Force rebuild even if exists
        
        Returns:
            BeliefGraph or None
        """
        if not rebuild:
            graph = self.load(event_id)
            if graph is not None:
                return graph
        
        if builder is None:
            logger.warning("No builder provided and graph not found")
            return None
        
        # Build and save
        try:
            graph = builder.build(event_id)
            self.save(graph)
            return graph
        except Exception as e:
            logger.error(f"Error building graph for {event_id}: {e}")
            return None


# Module-level singleton
_storage: Optional[GraphStorage] = None


def get_storage() -> GraphStorage:
    """Get graph storage singleton."""
    global _storage
    if _storage is None:
        _storage = GraphStorage()
    return _storage
