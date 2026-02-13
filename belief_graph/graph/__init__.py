"""
Graph structural upgrade modules.

Provides deterministic, fully-logged graph operations:
- Temporal decay modeling
- Structural influence propagation
- Causal motif detection
- DAG validation
"""

from .logging_utils import setup_graph_logger, log_edge_update, log_summary
from .temporal_decay import apply_temporal_decay, calculate_time_weight
from .propagation import propagate_influence
from .motifs import detect_and_boost_motifs
from .dag_validation import ensure_dag

__all__ = [
    'setup_graph_logger',
    'log_edge_update',
    'log_summary',
    'apply_temporal_decay',
    'calculate_time_weight',
    'propagate_influence',
    'detect_and_boost_motifs',
    'ensure_dag',
]
