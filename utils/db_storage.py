import os
import json
import logging
from typing import Optional, Dict, Any, List
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# Default database connection
_DEFAULT_DB_URL = "postgresql://causal:causal@localhost:5432/causal_interface"

def _get_db_url() -> str:
    return os.getenv("DATABASE_URL_SYNC") or _DEFAULT_DB_URL

def save_artifact(event_id: str, artifact_type: str, data: Any) -> None:
    """
    Save pipeline artifact data to the PostgreSQL database.
    Overwrites the artifact if it already exists for this event_id.
    """
    db_url = _get_db_url()
    try:
        # Convert non-string data to JSON
        if not isinstance(data, (str, bytes)):
            # Handle Numpy types if any exist in the pipeline
            try:
                from utils.json_utils import NumpyJSONEncoder
                data_json = json.dumps(data, cls=NumpyJSONEncoder)
            except ImportError:
                data_json = json.dumps(data, default=str)
        else:
            data_json = data

        # PostgreSQL text/jsonb fields do not support the null character (\u0000)
        # Apply strict replacement to the final stringified JSON payload
        if isinstance(data_json, str):
            data_json = data_json.replace('\x00', '').replace('\\u0000', '')

        sql = """
            INSERT INTO pipeline_artifacts (id, event_id, artifact_type, data_json, created_at, updated_at)
            VALUES (gen_random_uuid(), %s, %s, %s::jsonb, NOW(), NOW())
            ON CONFLICT ON CONSTRAINT uq_pipeline_artifact_event_type
            DO UPDATE SET 
                data_json = EXCLUDED.data_json,
                updated_at = NOW();
        """
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (event_id, artifact_type, data_json))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to save artifact '{artifact_type}' for event '{event_id}': {e}")
        raise

def load_artifact(event_id: str, artifact_type: str) -> Optional[Any]:
    """
    Load a specific pipeline artifact from the database.
    Returns the parsed JSON dictionary/list, or None if not found.
    """
    db_url = _get_db_url()
    try:
        sql = """
            SELECT data_json
            FROM pipeline_artifacts
            WHERE event_id = %s AND artifact_type = %s
        """
        with psycopg2.connect(db_url) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (event_id, artifact_type))
                row = cur.fetchone()
                
                if row and row.get("data_json"):
                    return row["data_json"]
                return None
    except Exception as e:
        logger.error(f"Failed to load artifact '{artifact_type}' for event '{event_id}': {e}")
        return None

def list_artifacts(event_id: str) -> List[str]:
    """List all artifact types available for a given event ID."""
    db_url = _get_db_url()
    try:
        sql = """
            SELECT artifact_type
            FROM pipeline_artifacts
            WHERE event_id = %s
            ORDER BY created_at ASC
        """
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (event_id,))
                return [row[0] for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Failed to list artifacts for event '{event_id}': {e}")
        return []

def list_all_artifacts_by_type(artifact_type: str) -> List[Dict[str, Any]]:
    """List all artifacts of a specific type across all events, returning the JSON data and event ID.
    Returns: [{'event_id': '...', 'data': {...}}, ...]
    """
    db_url = _get_db_url()
    try:
        sql = """
            SELECT event_id, data_json 
            FROM pipeline_artifacts
            WHERE artifact_type = %s
        """
        result = []
        with psycopg2.connect(db_url) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (artifact_type,))
                for row in cur.fetchall():
                    result.append({
                        "event_id": row["event_id"],
                        "data": row["data_json"]
                    })
        return result
    except Exception as e:
        logger.error(f"Failed to list all artifacts by type '{artifact_type}': {e}")
        return []
