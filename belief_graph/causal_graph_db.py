# This module has been removed.
#
# Previously contained CausalGraphDBWriter which wrote directly to
# causal_graphs/causal_nodes/causal_edges tables via psycopg2.
#
# Graph persistence is now handled exclusively through the artifact pipeline:
#   1. belief_graph/storage.py → db_storage.save_artifact("belief_graph")
#   2. PipelineService.ingest_from_files() reads the artifact and populates tables
