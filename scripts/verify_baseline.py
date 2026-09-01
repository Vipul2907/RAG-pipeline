#!/usr/bin/env python3
"""
Verification script for Phase 1 baseline.

Runs end-to-end ingestion, indexing, and search to verify
the brute-force baseline works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.config import config, setup_logging
from rag.ingestion import DocumentChunker
from rag.embeddings import get_embedding_provider
from rag.index import BruteForceIndex
from rag.query import QueryEngine


def main():
    """Run verification pipeline."""
    print("=" * 60)
    print("Phase 1 Baseline Verification")
    print("=" * 60)

    # Setup
    setup_logging(config)
    data_dir = Path("data/raw")
    index_path = Path("data/index")

    # 1. Chunk documents
    print("\n1. Chunking documents...")
    chunker = DocumentChunker(chunk_size=config.chunk_size, overlap=config.chunk_overlap)
    chunks = chunker.chunk_directory(data_dir, glob_pattern="**/*")
    print(f"   Generated {len(chunks)} chunks from {len(list(data_dir.glob('**/*')))} files")

    if not chunks:
        print("   ERROR: No chunks generated!")
        return 1

    # Show sample chunks
    print("\n   Sample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"   [{i}] {chunk.source_file} (chars {chunk.start_char}-{chunk.end_char})")
        print(f"       '{chunk.text[:100]}...'")

    # 2. Generate embeddings
    print("\n2. Generating embeddings...")
    embedder = get_embedding_provider()
    print(f"   Provider: {type(embedder).__name__}")
    print(f"   Dimension: {embedder.dimension}")

    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed(texts)
    print(f"   Embeddings shape: {embeddings.shape}")

    # 3. Build index
    print("\n3. Building brute-force index...")
    index = BruteForceIndex(dimension=embedder.dimension)
    index.add(embeddings, chunks)
    print(f"   Index size: {index.size} vectors")

    # 4. Save index
    print("\n4. Saving index...")
    index.save(index_path)
    print(f"   Saved to {index_path}")

    # 5. Load index (roundtrip test)
    print("\n5. Testing save/load roundtrip...")
    loaded_index = BruteForceIndex.load(index_path)
    print(f"   Loaded index size: {loaded_index.size}")

    # 6. Search tests
    print("\n6. Running search tests...")
    engine = QueryEngine(loaded_index)

    test_queries = [
        "What is machine learning?",
        "How does HNSW work?",
        "What are the benefits of RAG?",
        "vector database similarity search",
        "supervised vs unsupervised learning",
    ]

    for query in test_queries:
        print(f"\n   Query: '{query}'")
        result = engine.search(query, k=3)
        for r in result.results:
            print(f"     Rank {r.rank}: score={r.score:.4f} | {r.chunk.source_file}")
            print(f"       '{r.chunk.text[:120]}...'")

    # 7. Verify cosine similarity correctness
    print("\n7. Verifying cosine similarity correctness...")
    # Test with known orthogonal vectors
    test_index = BruteForceIndex(dimension=128)
    import numpy as np
    vec_a = np.zeros(128, dtype=np.float32)
    vec_a[0] = 1.0
    vec_b = np.zeros(128, dtype=np.float32)
    vec_b[1] = 1.0
    vec_c = np.zeros(128, dtype=np.float32)
    vec_c[0] = 1.0

    from rag.ingestion import Chunk
    test_chunks = [
        Chunk(text="A", source_file="test", start_char=0, end_char=1, chunk_index=0),
        Chunk(text="B", source_file="test", start_char=0, end_char=1, chunk_index=1),
        Chunk(text="C", source_file="test", start_char=0, end_char=1, chunk_index=2),
    ]
    test_embeddings = np.vstack([vec_a, vec_b, vec_c])
    # Normalize
    test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)
    test_index.add(test_embeddings, test_chunks)

    # Query with vec_a - should match A and C with score 1.0, B with score 0.0
    results = test_index.search(vec_a, k=3)
    assert abs(results[0].score - 1.0) < 1e-5, f"Expected 1.0, got {results[0].score}"
    assert abs(results[1].score - 1.0) < 1e-5, f"Expected 1.0, got {results[1].score}"
    assert abs(results[2].score - 0.0) < 1e-5, f"Expected 0.0, got {results[2].score}"
    print("   Cosine similarity verification PASSED")

    print("\n" + "=" * 60)
    print("ALL VERIFICATION TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())