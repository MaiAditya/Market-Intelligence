# CLAUDE.md — AI Market Intelligence Pipeline

## Overview

Event-centric intelligence system for prediction markets. Ingests multi-source data (web, Reddit, Twitter), maps documents to tracked events via a 3-stage filter, extracts structured signals, calculates probability deltas, and builds causal belief graphs. Core analysis uses BERT-only models — no LLMs except in optional Phase 2 graph enrichment.

## Core Principles

- **Events are primary objects** — data flows from events, never pushed into events
- **Ingest wide, filter strictly** — collect broadly, apply rigorous 3-stage filtering
- **No final probabilities** — only output delta ranges with confidence
- **BERT-only for core logic** — NER, classification, similarity via transformer models
- **Transparent reasoning** — all intermediate outputs stored as JSONB artifacts

## Pipeline Architecture (8 Stages)

```
Event Registry → Query Generation → Data Ingestion → Normalization
  → Entity Extraction (NER) → Event Mapping (3-stage) → Signal Extraction
  → Delta Calculation
```

### Stage Details

| Stage | Module | Purpose |
|-------|--------|---------|
| 1 | `pipeline/event_registry.py` | Load events from `config/events.json`, validate, provide lookup |
| 2 | `pipeline/query_generator.py` | Template-based query expansion (official, journalist, public_opinion, critical) |
| 3 | `pipeline/ingestion/ingestor.py` | Orchestrate Reddit (PRAW), Twitter (Nitter), Web (DuckDuckGo + BS4) |
| 4 | `pipeline/normalizer.py` | Clean text, detect source type, infer author type, deduplicate |
| 5 | `pipeline/entity_extractor.py` (via `models/ner.py`) | BERT NER + regex model name detection |
| 6 | `pipeline/event_mapper.py` | **3-stage gate:** entity match → semantic relevance (adaptive: 0.35–0.45 by event type) → dependency classification |
| 7 | `pipeline/signal_extractor.py` | Extract signal type, direction, magnitude, confidence |
| 8 | `pipeline/delta_engine.py` | Rule-based aggregation: `Σ(magnitude × confidence × origin_weight × type_weight × direction_mod × time_weight)` / total_weight → scaled to delta range |

### 3-Stage Event Mapping (Critical Path)
1. **Entity Gate:** Document must contain ≥1 primary entity AND ≥1 secondary/alias — hard filter (with soft fallback at 0.85 semantic similarity)
2. **Semantic Relevance:** Cosine similarity via all-mpnet-base-v2 with adaptive thresholds by event type: regulation=0.35, model_release=0.45, capability/market/general=0.40
3. **Dependency Classification:** Zero-shot multi-label classification into: training, compute, safety, regulation, executive_statement, public_narrative

## ML Models (`models/`)

| Model | File | HuggingFace ID | Purpose |
|-------|------|----------------|---------|
| NER | `ner.py` | dslim/bert-base-NER | Entity extraction |
| Embeddings | `semantic_relevance.py` | sentence-transformers/all-mpnet-base-v2 | Document-event similarity |
| Dependency | `dependency_classifier.py` | bert-base-uncased | Zero-shot dependency classification |
| Signal | `signal_classifier.py` | roberta-base | Signal type/direction classification |

- **Model Manager** (`model_manager.py`): Lazy loading with cache directory
- **Config:** `config/model_config.json` — thresholds, cache dir, offline mode
- **All models run in offline mode** (`local_files_only: true`) — models must be pre-downloaded to `models/cache/`

### Thresholds (`config/model_config.json` + adaptive overrides in `event_mapper.py`)
- `semantic_relevance` — adaptive by event type: regulation=0.35, model_release=0.45, capability/market/general=0.40 (global default=0.40)
- `entity_confidence: 0.7` — NER confidence minimum
- `signal_confidence: 0.5` — signal extraction minimum
- `dependency_threshold: 0.3` — dependency classification minimum
- `soft_entity_gate: 0.85` — semantic similarity fallback when hard entity gate fails

## Belief Graph (`belief_graph/`)

The largest subsystem (24 files, ~8.3k LOC). Builds causal DAGs from extracted events and signals.

### Key Modules
| Module | Purpose |
|--------|---------|
| `graph_builder.py` (879 lines) | Main orchestrator: extract → candidates → classify → score → validate |
| `event_extractor.py` | Convert signals/documents into `EventNode` objects |
| `candidate_generator.py` | Rule-based edge generation between nodes |
| `mechanism_classifier.py` | Classify edge mechanisms (legal_constraint, economic_impact, signaling, etc.) |
| `evidence_scorer.py` | Quantify evidence strength per edge |
| `confidence_calculator.py` | Compute edge confidence from evidence scores |
| `event_clustering.py` | Temporal clustering of related events |
| `graph/dag_validation.py` | Cycle detection and removal |
| `graph/propagation.py` | Belief propagation through graph |
| `graph/temporal_decay.py` | Time-based confidence decay |
| `storage.py` | PostgreSQL persistence |
| `llm_causal_generator.py` | Optional LLM-based edge generation (Phase 2 only) |

### Data Structures (`belief_graph/models.py`)
- `EventNode` — event_id, type, timestamp, actors, action, object, certainty, scope
- `BeliefNode` — Final target (market outcome)
- `BeliefEdge` — source, target, mechanism, direction, latency, confidence
- `EvidenceScores` — Empirical evidence quantification

## Configuration (`config/`)

### `events.json`
Event definitions with: event_id, event_type, title, primary/secondary entities, aliases, deadline, dependencies, polymarket_slug, polymarket_token_id

### `query_templates.json`
Query families: official, journalist, public_opinion, critical. Templates use `{primary_entity}` and `{secondary_entity}` placeholders.

### `model_config.json`
Model names, thresholds, cache directory, offline mode settings.

## Data Storage

- **Pipeline artifacts:** PostgreSQL JSONB via `utils/db_storage.py`
  - `save_artifact(event_id, artifact_type, data)` — upserts on (event_id, artifact_type)
  - `load_artifact(event_id, artifact_type)` — loads from JSONB
- **Artifact types:** normalized_documents, extracted_entities, event_mappings, signals, analysis, belief_graph
- **Model cache:** `models/cache/` (HuggingFace transformers, pre-downloaded)

## CLI (`cli/`)

```bash
# Core analysis
python -m cli.run_pipeline list-events                    # List tracked events
python -m cli.run_pipeline analyze --event <event_id>     # Single event analysis
python -m cli.run_pipeline analyze-all                    # All events
python -m cli.run_pipeline show --event <event_id>        # Show saved analysis
python -m cli.run_pipeline ingest [--event <event_id>]    # Run data ingestion only
python -m cli.run_pipeline extract-signals [--event <id>] # Extract signals from existing docs
python -m cli.run_pipeline update-probabilities           # Fetch Polymarket prices
python -m cli.run_pipeline stats                          # Pipeline statistics

# Belief graph
python -m cli.run_pipeline build-graph --event <event_id> # Build belief graph for event
python -m cli.run_pipeline show-graph --event <event_id>  # Show belief graph
python -m cli.run_pipeline list-graphs                    # List all stored graphs
python -m cli.run_pipeline graph-stats                    # Graph storage statistics
python -m cli.run_pipeline build-all-graphs               # Build graphs for all events
python -m cli.run_pipeline market-report --event <id>     # Generate market report JSON

# Model management
python -m cli.run_pipeline warm-models                    # Pre-load all models into memory
```

## API (`api/`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/stats` | GET | Pipeline statistics |
| `/events` | GET | List all events |
| `/events/{event_id}` | GET | Get analysis |
| `/events/{event_id}/refresh` | POST | Trigger pipeline |
| `/events/{event_id}/run-full` | POST | Complete workflow (analysis + graph + report) |
| `/events/{event_id}/signals` | GET | Get recent signals for event |
| `/events/{event_id}/documents` | GET | Get top documents for event |
| `/probabilities` | GET | Current probabilities for all events |
| `/events/{event_id}/probability` | GET | Probability for specific event |
| `/analyze-all` | POST | Full pipeline for all events (background) |
| `/full-pipeline` | POST | End-to-end with graph building |
| `/belief-graph/{event_id}` | GET | Get complete belief update DAG |
| `/belief-graph/{event_id}/upstream` | GET | Top N upstream events by impact |
| `/belief-graph/{event_id}/edges` | GET | All edges with explanations |
| `/belief-graph/{event_id}/build` | POST | Build/rebuild graph for event |
| `/belief-graph/` | GET | List all stored belief graphs |
| `/belief-graph/stats` | GET | Graph storage statistics |
| `/belief-graph/{event_id}` | DELETE | Delete a stored belief graph |

## Testing (`tests/`)

- **Framework:** pytest + pytest-asyncio
- **Fixtures:** Session-scoped for expensive model initialization (`conftest.py`)
- **Key test files:**
  - `test_event_registry.py` — event loading, entity access
  - `test_normalizer.py` — source detection, text cleaning
  - `test_event_mapping_enhanced.py` — 3-stage mapping validation
  - `test_pipeline.py` — integration tests
  - `test_reasoning.py` — belief graph reasoning (12k LOC)
  - `test_evidence_scoring.py` — evidence quantification
  - `test_event_clustering_improved.py` — temporal clustering

```bash
pytest tests/                                # All tests
pytest tests/test_pipeline.py -v             # Single file
pytest tests/test_event_mapping_enhanced.py  # Mapping tests
```

## Code Patterns

### Dataclass Serialization
```python
@dataclass
class Event:
    event_id: str
    title: str
    # ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "Event": ...
```

### UTC Datetime Convention
```python
datetime.now(timezone.utc).replace(tzinfo=None)
```

### Zero-Shot Classification
```python
embeddings = model.encode([doc_text] + dependency_descriptions)
similarities = cosine_similarity(embeddings[0], embeddings[1:])
```

## Do's and Don'ts

- **Do** use dataclasses with `to_dict()`/`from_dict()` for domain objects
- **Do** store artifacts via `utils/db_storage.py` (PostgreSQL JSONB)
- **Do** respect all 3 stages of event mapping — the entity gate is non-negotiable
- **Do** use `model_manager.py` for lazy model loading
- **Do** keep models in offline mode (pre-downloaded to cache)
- **Don't** use LLMs for core pipeline analysis — BERT models only
- **Don't** write pipeline outputs to filesystem — use PostgreSQL
- **Don't** skip the entity gate or lower thresholds without justification
- **Don't** predict final probabilities — only output delta ranges
- **Don't** fine-tune models — use zero-shot classification
