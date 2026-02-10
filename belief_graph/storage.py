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
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._cache: Dict[str, BeliefGraph] = {}
        self._cache_max_size = 10
        
        logger.info(f"GraphStorage initialized at {self.storage_dir}")
    
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
        Save a belief graph to storage.
        
        Args:
            graph: BeliefGraph to save
            overwrite: Whether to overwrite existing
        
        Returns:
            Path to saved file
        """
        event_id = graph.belief_node.event_id
        filename = self._get_filename(event_id)
        filepath = self.storage_dir / filename
        
        if filepath.exists() and not overwrite:
            raise FileExistsError(f"Graph already exists: {filepath}")
        
        # Prepare data
        data = graph.to_dict()
        
        # Add storage metadata
        data["_storage"] = {
            "event_id": event_id,
            "saved_at": _utc_now().isoformat(),
            "version": "1.0"
        }
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            dump_json(data, f)
        
        # Update cache
        self._cache[event_id] = graph
        self._trim_cache()
        
        logger.info(f"Saved graph for event {event_id} to {filepath}")
        
        return filepath
    
    def load(self, event_id: str) -> Optional[BeliefGraph]:
        """
        Load a belief graph from storage.
        
        Args:
            event_id: Event ID to load
        
        Returns:
            BeliefGraph or None if not found
        """
        # Check cache first
        if event_id in self._cache:
            logger.debug(f"Loaded graph from cache: {event_id}")
            return self._cache[event_id]
        
        filename = self._get_filename(event_id)
        filepath = self.storage_dir / filename
        
        if not filepath.exists():
            logger.debug(f"Graph not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Remove storage metadata before parsing
            data.pop("_storage", None)
            
            graph = BeliefGraph.from_dict(data)
            
            # Update cache
            self._cache[event_id] = graph
            self._trim_cache()
            
            logger.info(f"Loaded graph for event {event_id}")
            
            return graph
            
        except Exception as e:
            logger.error(f"Error loading graph {event_id}: {e}")
            return None
    
    def exists(self, event_id: str) -> bool:
        """Check if a graph exists for an event."""
        if event_id in self._cache:
            return True
        
        filename = self._get_filename(event_id)
        filepath = self.storage_dir / filename
        return filepath.exists()
    
    def delete(self, event_id: str) -> bool:
        """
        Delete a graph from storage.
        
        Args:
            event_id: Event ID to delete
        
        Returns:
            True if deleted, False if not found
        """
        # Remove from cache
        self._cache.pop(event_id, None)
        
        filename = self._get_filename(event_id)
        filepath = self.storage_dir / filename
        
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Deleted graph for event {event_id}")
            return True
        
        return False
    
    def list_graphs(self) -> List[Dict]:
        """
        List all stored graphs with metadata.
        
        Returns:
            List of graph metadata dictionaries
        """
        graphs = []
        
        for filepath in self.storage_dir.glob("*_graph.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                storage_meta = data.get("_storage", {})
                belief_data = data.get("belief", {})
                metadata = data.get("metadata", {})
                
                graphs.append({
                    "event_id": storage_meta.get("event_id", ""),
                    "question": belief_data.get("question", ""),
                    "node_count": metadata.get("node_count", 0),
                    "edge_count": metadata.get("edge_count", 0),
                    "saved_at": storage_meta.get("saved_at", ""),
                    "generated_at": metadata.get("generated_at", ""),
                    "filepath": str(filepath)
                })
                
            except Exception as e:
                logger.warning(f"Error reading graph {filepath}: {e}")
        
        # Sort by saved_at descending
        graphs.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
        
        return graphs
    
    def get_stats(self) -> Dict:
        """
        Get storage statistics.
        
        Returns:
            Statistics dictionary
        """
        graphs = self.list_graphs()
        
        total_nodes = sum(g.get("node_count", 0) for g in graphs)
        total_edges = sum(g.get("edge_count", 0) for g in graphs)
        
        # Calculate storage size
        total_size = sum(
            Path(g["filepath"]).stat().st_size
            for g in graphs
            if Path(g["filepath"]).exists()
        )
        
        return {
            "graph_count": len(graphs),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "storage_size_bytes": total_size,
            "storage_size_mb": round(total_size / (1024 * 1024), 2),
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
