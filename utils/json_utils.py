"""
JSON utilities for handling numpy and other special types.
"""

import json
from datetime import datetime
from typing import Any


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types and other special types."""
    
    def default(self, obj: Any) -> Any:
        # Handle numpy float types
        try:
            import numpy as np
            if isinstance(obj, (np.float32, np.float64, np.floating)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64, np.integer)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
        except ImportError:
            pass
        
        # Handle datetime
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        return super().default(obj)


def dump_json(obj: Any, fp, **kwargs) -> None:
    """Wrapper for json.dump that uses NumpyJSONEncoder by default."""
    kwargs.setdefault('cls', NumpyJSONEncoder)
    kwargs.setdefault('indent', 2)
    kwargs.setdefault('ensure_ascii', False)
    json.dump(obj, fp, **kwargs)


def dumps_json(obj: Any, **kwargs) -> str:
    """Wrapper for json.dumps that uses NumpyJSONEncoder by default."""
    kwargs.setdefault('cls', NumpyJSONEncoder)
    kwargs.setdefault('indent', 2)
    kwargs.setdefault('ensure_ascii', False)
    return json.dumps(obj, **kwargs)
