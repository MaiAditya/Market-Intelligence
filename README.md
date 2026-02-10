# AI Market Intelligence System

Event-centric intelligence system for AI-related Polymarket markets. Pulls event-conditioned data from the web, maps documents to events, extracts structured signals, and outputs probability delta ranges with supporting documents.

## Core Principles

- **Events are primary objects** - Events pull data; data never pushes into events
- **Ingest wide, filter strictly** - Collect broadly, then apply rigorous filtering
- **No final probability predictions** - Only delta ranges
- **BERT-style models only** - No LLMs or generative reasoning
- **Transparent reasoning** - All logic visible in intermediate JSONs

## Features

- Track AI-related events (Gemini 5, GPT-5, EU AI Act)
- Multi-source data ingestion (Reddit, Twitter, Web)
- BERT-based entity extraction (dslim/bert-base-NER)
- 3-stage event-document mapping (Entity Gate → Semantic Relevance → Dependency Classification)
- Signal extraction with type/direction classification
- Rule-based probability delta calculation
- Document ranking with credibility scoring
- REST API and CLI interfaces

## Quick Start

### Installation

```bash
# Clone or navigate to the project
cd ai-market-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (optional, for Twitter scraping)
playwright install chromium
```

### Environment Variables (Optional)

For Reddit API access, set:
```bash
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"
```

### Running the CLI

```bash
# List tracked events
python -m cli.run_pipeline list-events

# Run full pipeline for all events
python -m cli.run_pipeline analyze-all

# Analyze a specific event
python -m cli.run_pipeline analyze --event gemini-5-release-2026

# Show saved analysis
python -m cli.run_pipeline show --event gemini-5-release-2026

# Fetch current probabilities from Polymarket
python -m cli.run_pipeline update-probabilities

# View statistics
python -m cli.run_pipeline stats
```

### Running the API

```bash
# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

# Or run directly
python -m api.main
```

API will be available at `http://localhost:8001`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/events` | GET | List all tracked events |
| `/events/{event_id}` | GET | Get analysis for event |
| `/events/{event_id}/refresh` | POST | Trigger pipeline refresh |
| `/events/{event_id}/signals` | GET | Get signals for event |
| `/events/{event_id}/documents` | GET | Get documents for event |
| `/events/{event_id}/probability` | GET | Get current Polymarket probability |
| `/analyze-all` | POST | Trigger full pipeline for all events |
| `/stats` | GET | Pipeline statistics |

## Project Structure

```
ai-market-intelligence/
├── config/
│   ├── events.json              # Event registry (3 AI events)
│   ├── query_templates.json     # Query generation templates
│   └── model_config.json        # BERT model configuration
├── data/
│   ├── documents/               # Raw scraped content
│   ├── normalized/              # Normalized document JSONs
│   ├── signals/                 # Extracted signals
│   ├── output/                  # Final analysis JSONs
│   └── cache/                   # API response cache
├── models/
│   ├── ner.py                   # BERT NER extraction
│   ├── semantic_relevance.py    # Sentence-transformers
│   ├── dependency_classifier.py # Dependency classification
│   ├── signal_classifier.py     # Signal type/direction
│   └── model_manager.py         # Model loading/caching
├── pipeline/
│   ├── event_registry.py        # Event schema & loading
│   ├── query_generator.py       # Template-based queries
│   ├── ingestion/               # Data ingestion
│   │   ├── reddit_api.py
│   │   ├── twitter_scraper.py
│   │   ├── web_scraper.py
│   │   └── ingestor.py
│   ├── normalizer.py            # Document normalization
│   ├── entity_extractor.py      # NER pipeline
│   ├── event_mapper.py          # 3-stage mapping
│   ├── signal_extractor.py      # Signal extraction
│   ├── time_extractor.py        # Date/numeric extraction
│   └── delta_engine.py          # Delta calculation
├── integrations/
│   ├── polymarket_client.py     # Polymarket API
│   └── source_registry.py       # Source credibility
├── api/
│   ├── main.py                  # FastAPI app
│   ├── schemas.py               # Pydantic models
│   └── endpoints.py             # Route handlers
├── cli/
│   ├── run_pipeline.py          # CLI entry point
│   └── commands.py              # Command handlers
└── tests/
    └── ...                      # Test files
```

## Tracked Events

1. **Gemini 5 Release 2026** - Google releasing Gemini 5 by end of 2026
2. **GPT-5 Release 2026** - OpenAI releasing GPT-5 by end of 2026
3. **EU AI Act Enforcement 2027** - EU AI Act fully enforced by 2027

## Pipeline Stages

### 1. Event Registry
Load events with entities, aliases, and dependencies from JSON.

### 2. Query Generation
Generate search queries using templates:
- Official (announcements, statements)
- Journalist (leaks, insider reports)
- Public Opinion (Reddit, Twitter)
- Critical (delays, problems)

### 3. Data Ingestion
Multi-source ingestion:
- Reddit: Official PRAW API
- Twitter: Nitter fallback
- Web: DuckDuckGo search + BeautifulSoup

### 4. Document Normalization
Standardize documents with source type detection and text cleaning.

### 5. Entity Extraction
Extract using `dslim/bert-base-NER`:
- Organizations (ORG)
- Model names (custom regex)
- Dates (pattern matching)
- Benchmarks (pattern matching)

### 6. Event-Document Mapping

**Stage 1: Hard Entity Gate**
- Must have ≥1 primary entity AND ≥1 secondary/alias

**Stage 2: Semantic Relevance**
- `sentence-transformers/all-mpnet-base-v2`
- Threshold: 0.45 cosine similarity

**Stage 3: Dependency Classification**
- Zero-shot classification for: training, compute, safety, regulation, executive_statement, public_narrative

### 7. Signal Extraction
Extract signals with:
- Type: training_progress, delay, rumor, official_confirmation, etc.
- Direction: positive, negative, neutral
- Magnitude and confidence scores

### 8. Delta Calculation
Rule-based aggregation:
```
delta = Σ (magnitude × confidence × origin_weight × time_weight × direction)
```

Output: Delta ranges (e.g., "+3.2% to +5.8%") with confidence scores.

## Output Format

```json
{
  "event_id": "gemini-5-release-2026",
  "current_probability": 0.61,
  "suggested_delta": "+3.2% to +5.8%",
  "confidence": 0.73,
  "dominant_signal_types": ["training_progress", "executive_statement"],
  "top_documents": [...],
  "signal_summary": {...}
}
```

## Models Used

| Purpose | Model |
|---------|-------|
| NER | `dslim/bert-base-NER` |
| Semantic Similarity | `sentence-transformers/all-mpnet-base-v2` |
| Dependency Classification | `bert-base-uncased` (zero-shot) |
| Signal Classification | `roberta-base` (zero-shot) |

## Configuration

### Model Config (`config/model_config.json`)
```json
{
  "thresholds": {
    "semantic_relevance": 0.45,
    "entity_confidence": 0.7,
    "signal_confidence": 0.5
  }
}
```

### Adding New Events
Edit `config/events.json`:
```json
{
  "event_id": "new-event-id",
  "event_type": "model_release",
  "event_title": "Event Title",
  "primary_entities": ["Entity1", "Entity2"],
  "secondary_entities": ["Related1"],
  "aliases": ["Alias1"],
  "deadline": "2027-12-31T23:59:59Z",
  "dependencies": ["training", "compute", "safety"],
  "polymarket_slug": "polymarket-event-slug"
}
```

## Non-Goals

- Auto-trading or wallet integration
- LLM-based generative summaries
- Final probability predictions (only deltas)
- Global news indexing beyond event queries
- Real-time streaming (batch processing only)

## Development

```bash
# Run tests
pytest tests/

# Format code
black .
isort .

# Type checking (optional)
mypy .
```

## License

MIT License
