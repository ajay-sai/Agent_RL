# Semantic Data Orchestrator

Research-grade multi-agent system for semantic data querying across heterogeneous backends.

## Setup

```bash
cd examples/semantic_orchestrator
uv sync
```

## Quick Start

```bash
# Initialize databases (requires Docker)
docker-compose -f docker/docker-compose.yml up -d

# Load sample CSV data
python scripts/load_data.py --dataset sample_sales

# Start interactive query session
python scripts/demo.py
```

## Components

- **Ingestion Agent**: Loads CSV/JSON, extracts text and metadata
- **Schema Agent**: Discovers schemas, suggests semantic types and storage mapping
- **Storage**: Chroma (vector), Neo4j (graph), SQLite (structured)
- **Router Agent**: RL-trained to select appropriate backend(s)
- **Retrievers**: Execute queries against each backend
- **Synthesis Agent**: Combines results into coherent answers

## Research

See `docs/architecture.md` for detailed design and evaluation methodology.
