# HDB Compare Flats Agent

An agentic search system for finding comparable HDB (Housing Development Board) resale transactions using natural language queries. The system uses a two-agent architecture with iterative refinement to find an optimal candidate pool of 30-200 comparable transactions.

**🎥 <a href="https://hdb-gradio-app-production.up.railway.app" target="_blank" rel="noopener noreferrer">Live Demo (featuring Jan 2024–Jan 2026 data)</a>**

## Features

- **Natural Language Queries**: Ask questions like "Find 4-room flats in Sengkang, around 95 sqm, mid floor, last 12 months"
- **Intelligent Refinement**: Automatically relaxes or tightens search constraints to find the right number of comparables
- **Hybrid Retrieval**: Combines structured filters with semantic vector search and BM25 lexical matching
- **Deterministic Ranking**: Scores results by area match, lease remaining, storey preference, and recency
- **Visual Analytics**: Price histogram with median and percentile overlays
- **Agent Trace**: See decision the agent makes during the search process
- **BrainTrust Integration**: Full observability with trace logging and LLM monitoring

## Screenshots

### Main Interface with Results

The Gradio UI provides a chat-based interface for natural language queries. Results are displayed in a sortable table with price statistics.

![UI with Results](docs/images/UI_with_dresults.jpg)

### Agent Trace Panel

See every decision the agent makes during the iterative search process - which constraints were relaxed or tightened, candidate counts at each step, and retrieval mode used.

![Trace Panel](docs/images/trace_panel.jpg)

### Price Distribution Histogram

Visual analytics showing price distribution with median (green), 25th percentile (yellow), and 75th percentile (yellow) overlays.

![Histogram](docs/images/histogram.jpg)

## Architecture

### Two-Agent Design Pattern

The system uses **two separate LLM agents** coordinated by a Python loop orchestrator:

```mermaid
graph TD
    A[User Query] --> B[Target Agent]
    B -->|Extracts Target| C[Loop Orchestrator]
    C -->|Builds Filters| D[Database Query]
    D -->|Count Results| E{Count in Range?}
    E -->|Yes: 30-200| F[Fetch & Rank]
    E -->|No| G[Planner Agent]
    G -->|Decide Action| H{Action?}
    H -->|Relax| I[Widen Constraints]
    H -->|Tighten| J[Narrow Constraints]
    H -->|Clarify| K[Ask User]
    H -->|Accept| F
    I --> C
    J --> C
    K --> L[User Response]
    L --> A
    F --> M[Return Results]

    style B fill:#4A90E2,stroke:#2E5C8A,stroke-width:3px,color:#fff
    style G fill:#E27D60,stroke:#B85C42,stroke-width:3px,color:#fff
```

### Component Responsibilities

**1. Target Agent** (`hdb_search_agents/agent/target.py`)
- Extracts structured intent from natural language
- Input: Raw user query string
- Output: `Target` Pydantic model with parsed constraints (town, flat_type, sqm, storey, lease, etc.)
- No tool calls, pure extraction

**2. Planner Agent** (`hdb_search_agents/agent/planner.py`)
- Decides iterative refinement actions
- Input: JSON payload with query, target, filters, count, stats, history, available adjustments
- Output: `PlannerDecision` with action (accept/relax/tighten/clarify) + optional adjustment
- No tool calls, pure decision-making

**3. Loop Orchestrator** (`hdb_search_agents/agent/orchestrator.py`)
- Coordinates agents and tools
- Calls Target Agent once to extract intent
- Iteratively calls Planner Agent to observe results and decide adjustments
- Directly invokes DB tools (count, fetch, stats)
- Applies adjustments and rebuilds filters between iterations
- LLMs never call tools directly - tools are Python functions only

### Iterative Retrieval Loop

Goal: Find 30-200 candidate transactions by relaxing or tightening constraints.

```
User Query → Target Agent → filters → COUNT
                                        ↓
                            ┌───────────┴─────────────┐
                            ↓                         │
                    Planner Agent observes            │
                    (count, stats, history)           │
                            ↓                         │
            ┌───────────────┼───────────────┐         │
            ↓               ↓               ↓         │
        CLARIFY         ACCEPT          RELAX/TIGHTEN │
     (ask user)      (done, fetch)    (adjust, loop)──┘
```

### Hybrid Retrieval System

The system supports three retrieval modes:

**1. Structured mode** (default):
- Pure SQL filters on town, flat_type, month_date, floor_area, storey, lease
- Used when constraints are specific and sufficient

**2. Vector similarity** (semantic search):
- pgvector with HNSW index
- 1024-dim embeddings from Jina AI (embeddings-v3)
- Triggered when: street_hint present or fuzzy location queries

**3. BM25 lexical matching** (keyword search):
- pg_textsearch for keyword-based ranking
- Boosts results matching street names, flat models, landmarks
- Triggered when: partial addresses, specific street hints

**Combined ranking**: Reciprocal Rank Fusion (RRF) merges vector and BM25 scores for optimal relevance.

### Tech Stack

- **Language**: Python 3.13
- **LLM Framework**: PydanticAI
- **Database**: PostgreSQL 17 with pgvector and [pg_textsearch](https://www.tigerdata.com/blog/introducing-pg_textsearch-true-bm25-ranking-hybrid-retrieval-postgres) extensions
- **LLM Provider**: OpenRouter (configurable model)
- **Embeddings**: Jina AI embeddings-v3 (1024 dimensions)
- **UI**: Gradio
- **Observability**: BrainTrust with OpenTelemetry

## Setup

### Prerequisites

- Docker
- Python 3.13+
- OpenRouter API key ([get one here](https://openrouter.ai/))
- Jina AI API key ([get one here](https://jina.ai/))
- (Optional) BrainTrust API key for tracing ([get one here](https://braintrust.dev/))

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/edangx100/hdb_agent_eval_v0.git
cd hdb_agent_eval_v0
```

> **Note:** This system lives in the `hdb_search_agents/` package. All commands below are run from the **repository root** (where `docker-compose.yml` and `.env` live), using `hdb_search_agents/...` paths.

2. **Create Python virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure environment variables**:
```bash
cp .env.example .env
```

**⚠️ MINIMUM REQUIRED CONFIGURATION**: You must update **3 API keys** in `.env` before running the application:

| Variable | Required | Get API Key | Purpose |
|----------|----------|-------------|---------|
| `OPENROUTER_API_KEY` | ✅ **YES** | https://openrouter.ai/keys | LLM inference (Target + Planner agents) |
| `JINA_API_KEY` | ✅ **YES** | https://jina.ai/embeddings/ | Text embeddings generation |
| `BRAINTRUST_API_KEY` | ❌ Optional | https://braintrust.dev/ | LLM tracing and observability |

Look for the `# UPDATE THIS` markers in `.env` to find these required fields.

All other variables have sensible defaults and can be left unchanged for initial setup.

4. **Start PostgreSQL**:
```bash
docker compose up -d
```

5. **Verify Postgres is running**:
```bash
docker compose exec postgres psql -U hdb -d hdb -c "SELECT version();"
```

6. **Initialize database schema**:
```bash
docker compose exec -T postgres psql -U hdb -d hdb < hdb_search_agents/db/schema.sql
```

7. **Run data ingestion** (downloads CSV, parses, generates embeddings):
```bash
PYTHONPATH=. python hdb_search_agents/db/ingest.py
```

This will:
- Filter to last 2 years (configurable via `INGEST_YEARS`, maximum is 5 years)
- Parse derived columns (storey ranges, remaining lease)
- Generate embeddings for each listing
- Load data into PostgreSQL

8. **Verify data loaded**:
```bash
docker compose exec postgres psql -U hdb -d hdb -c "SELECT COUNT(*) FROM hdb_resale;"
docker compose exec postgres psql -U hdb -d hdb -c "SELECT COUNT(*) FILTER (WHERE listing_embedding IS NOT NULL) as with_embeddings FROM hdb_resale;"
```

### Running the Application

**Launch Gradio UI** (recommended):
```bash
python hdb_search_agents/app.py
```

Then open http://localhost:7860 in your browser.

**Test agent via CLI**:
```bash
PYTHONPATH=. python hdb_search_agents/agent/orchestrator.py "Find 4-room in Sengkang, 95 sqm, mid floor, last 12 months"
```

## Example Queries

### Basic Queries

```
Find 4-room flats in Ang Mo Kio
```
Expected: Agent asks for more details (time window, size preferences)

```
Find 4-room in Sengkang, around 95 sqm, last 12 months
```
Expected: Returns 30-200 comparables with area tolerance applied

```
5-room flat in Bishan, 120-130 sqm, high floor, min 80 years lease, last 18 months
```
Expected: May need to relax constraints if too specific

## Configuration

All configuration is managed via `.env` file (single source of truth).

### Required API Keys (⚠️ Must Update)

These **3 variables** must be configured with your own API keys (marked with `# UPDATE THIS` in `.env.example`):

```env
OPENROUTER_API_KEY=your_key_here          # Get from https://openrouter.ai/keys
JINA_API_KEY=your_key_here                # Get from https://jina.ai/embeddings/
BRAINTRUST_API_KEY=your_key_here          # Optional: Get from https://braintrust.dev/
```

### Variables with Defaults (No Action Needed)

These have sensible defaults and typically don't need changes:

```env
# Database (matches docker-compose.yml)
DATABASE_URL=postgresql+psycopg2://hdb:hdb@localhost:5432/hdb

# LLM Configuration
OPENROUTER_MODEL_NAME=minimax/minimax-m2.1
OPENROUTER_FALLBACK_MODELS=["z-ai/glm-4.7", "openai/gpt-4o-mini"]

# Embedding Configuration
JINA_BASE_URL=https://api.jina.ai/v1
EMBEDDING_MODEL_NAME=jinaai/jina-embeddings-v3
EMBEDDING_DIM=1024

# Hybrid Retrieval Weights
RRF_VECTOR_WEIGHT=0.3
RRF_BM25_WEIGHT=0.7

# Data Ingestion
INGEST_YEARS=2
INGEST_CSV_PATH=hdb_search_agents/data/ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv

# Observability
BRAINTRUST_PARENT=project_name:hdb-comparable-flats
```

**📖 See `.env.example` for detailed comments explaining each variable.**

## Project Structure

The HDB search system lives in the `hdb_search_agents/` package (this folder):

```
hdb_search_agents/             # the HDB resale search system (this folder)
├── agent/
│   ├── models.py             # Pydantic models (Target, PlannerDecision, Stats, etc.)
│   ├── orchestrator.py       # Main loop coordinator
│   ├── planner.py            # Planner Agent (relax/tighten decisions)
│   ├── prompts.py            # Agent prompts
│   ├── scoring.py            # Deterministic ranking logic
│   ├── target.py             # Target Agent (intent extraction)
│   ├── tools.py              # Database tool functions
│   └── tracing.py            # BrainTrust/OpenTelemetry setup
├── data/
│   └── Resale...csv          # HDB resale data
├── db/
│   ├── ingest.py             # CSV download and embedding generation
│   ├── queries.py            # SQL query functions
│   └── schema.sql            # Database schema (idempotent)
├── docker/
│   └── Dockerfile            # Postgres 17 with extensions
├── viz/
│   └── plots.py              # Matplotlib histogram generation
├── app.py                    # Gradio UI
└── README.md                 # This file
```

The **repository root** holds the evaluation harness and shared config that
operate on this system:

```
<repo root>/
├── evals/                    # ★ evaluation harness (scorers, datasets, runner, reports)
├── tests/                    # unit tests for the evaluators
├── docs/                     # methodology, notes, incidents, images
├── hdb_search_agents/        # the system under test (described above)
├── settings.py               # configuration (pydantic-settings) — shared
├── docker-compose.yml        # local Postgres service
├── requirements.txt
└── .env / .env.example
```


## Troubleshooting

### Port 5432 (for Database) already allocated
- If you get "Bind for 0.0.0.0:5432 failed: port is already allocated", another service is using port 5432
- Check running containers: `docker ps`
- Stop the conflicting container: `docker stop <container_name>`

### No results found
- Check if data is loaded: `docker compose exec postgres psql -U hdb -d hdb -c "SELECT COUNT(*) FROM hdb_resale;"`
- Verify filters are not too restrictive
- Try a broader query first

### Embeddings not working
- Verify JINA_API_KEY is set in `.env`
- Check embeddings are populated: `docker compose exec postgres psql -U hdb -d hdb -c "SELECT COUNT(*) FILTER (WHERE listing_embedding IS NOT NULL) FROM hdb_resale;"`
- Re-run ingestion if needed: `PYTHONPATH=. python hdb_search_agents/db/ingest.py`

### LLM errors
- Verify OPENROUTER_API_KEY is valid
- Check OpenRouter dashboard for quota/rate limits
- Try a different model via OPENROUTER_MODEL_NAME, configure at .env


## Acknowledgments

- HDB resale data from [data.gov.sg](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view)
- Built with [PydanticAI](https://ai.pydantic.dev/)
- LLM inference via [OpenRouter](https://openrouter.ai/)
- Embeddings from [Jina AI](https://jina.ai/)
- Observability by [BrainTrust](https://braintrust.dev/)
