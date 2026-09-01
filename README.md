<<<<<<< HEAD
<<<<<<< HEAD
# RAG Pipeline

A Retrieval-Augmented Generation (RAG) system with a **hand-rolled HNSW (Hierarchical Navigable Small World) vector index** built from scratch using NumPy. No external vector databases (FAISS, Chroma, Pinecone) — pure Python implementation for learning and demonstration purposes.

## Project Status

**Phase 1 Complete**: Ingestion pipeline + Brute-force baseline ✅
- Document chunking with metadata (source file, character offsets)
- Paragraph-based splitting for .txt and .md files
- Dummy embedding provider (deterministic, no API keys)
- Flat brute-force cosine similarity index (NumPy)
- Save/load persistence (.npy + .json)
- End-to-end verification script

**Next**: Phase 2 — HNSW graph index implementation

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env

# Ingest sample documents
python -m src.cli ingest data/raw

# Search the index
python -m src.cli search "What is machine learning?"

# Run verification
python scripts/verify_baseline.py

# Run tests
pytest tests/ -v
```

## Architecture

```
src/
├── rag/
│   ├── config.py              # Centralized config & logging
│   ├── ingestion/
│   │   ├── chunker.py         # Document chunking (paragraph-based)
│   │   └── loader.py          # File loading utilities
│   ├── embeddings/
│   │   └── provider.py        # Embedding provider abstraction
│   ├── index/
│   │   ├── base.py            # Abstract VectorIndex base class
│   │   └── brute_force.py     # Flat cosine similarity baseline
│   └── query/
│       └── engine.py          # Query orchestration
└── cli/
    └── __init__.py            # CLI commands (ingest, search, stats)
```

## Key Components

### DocumentChunker (`src/rag/ingestion/chunker.py`)
- Splits .txt/.md files on paragraph boundaries (`\n\n`)
- Tracks character offsets for precise citation
- Configurable chunk size with overlap for context preservation
- Extensible metadata dictionary per chunk

### DummyEmbeddingProvider (`src/rag/embeddings/provider.py`)
- Deterministic SHA-256 based embeddings
- Reproducible results for testing
- Zero external dependencies
- Normalized vectors for cosine similarity

### BruteForceIndex (`src/rag/index/brute_force.py`)
- Exact cosine similarity via matrix multiplication
- Normalized embeddings stored in flat `np.ndarray`
- Save/load: `embeddings.npy` + `chunks.json` + `config.json`
- Filter function support for metadata-based filtering
- **Correctness anchor** for HNSW validation

### QueryEngine (`src/rag/query/engine.py`)
- High-level search interface
- Combines embedding provider + vector index
- Returns structured `QueryResult` with ranked `SearchResult` objects

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_DIMENSION` | 384 | Vector dimension |
| `EMBEDDING_PROVIDER` | dummy | Embedding provider |
| `CHUNK_SIZE` | 512 | Max characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `INDEX_TYPE` | brute_force | Index algorithm |
| `INDEX_PATH` | ./data/index | Index storage directory |
| `DEFAULT_TOP_K` | 10 | Default search results |
| `LOG_LEVEL` | INFO | Logging level |

## Testing

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_chunker.py -v
pytest tests/test_brute_force.py -v

# With coverage
pytest tests/ --cov=src/rag --cov-report=html
```

## Verification

The verification script (`scripts/verify_baseline.py`) runs a complete end-to-end test:

1. Loads sample documents from `data/raw/`
2. Chunks them with metadata
3. Generates dummy embeddings
4. Builds brute-force index
5. Saves and reloads index (roundtrip test)
6. Runs sample queries and displays results
7. Validates cosine similarity math with known vectors

## Sample Data

Three sample documents in `data/raw/`:
- `sample1.txt` — Introduction to Machine Learning
- `sample2.txt` — Vector Databases and Similarity Search
- `sample3.md` — Retrieval-Augmented Generation (RAG)

## Phase 2 Roadmap (HNSW)

- [ ] `HNSWIndex` class implementing `VectorIndex` interface
- [ ] Multi-layer graph construction
- [ ] Greedy search with entry point selection
- [ ] `efConstruction` and `efSearch` parameters
- [ ] `M` (max connections) and `M0` (layer 0 connections)
- [ ] Benchmark against brute-force baseline
- [ ] Recall@K evaluation

## License

MIT License — Educational/Research project
=======
RAG pipeline
>>>>>>> 625bf964d08a0c9cf5eab42c4c3e93b17b245bda
=======
RAG pipeline
>>>>>>> 625bf964d08a0c9cf5eab42c4c3e93b17b245bda
