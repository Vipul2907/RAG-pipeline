"""
Test the compound query decomposition feature.

Tests the two specific failing cases:
1. "What is Artificial Narrow Intelligence and what's its main limitation?"
   (ANI compound question)
2. "Write out the formula for binary cross entropy. What is the update rule for Q-learning?"
   (BCE + Q-learning compound question)
"""

import sys
sys.path.insert(0, "src")

import numpy as np
from rag.embeddings import DummyEmbeddingProvider
from rag.index import BruteForceIndex
from rag.ingestion import Chunk, DocumentChunker
from rag.llm import DummyLLMProvider
from rag.query import QueryEngine


def create_test_documents():
    """Create test documents with ANI, BCE, and Q-learning content."""
    docs = []

    # ANI content (from user's real data)
    docs.append("""
### Interview Question 16: What are the main limitations of ANI?

Artificial Narrow Intelligence (ANI) has several key limitations: it cannot generalize beyond its narrow training domain, lacks common sense reasoning, requires massive labeled datasets, and fails on out-of-distribution inputs. These are fundamental architectural limits.
""")

    docs.append("""
ANI systems excel at specific tasks but cannot transfer knowledge to new domains. The main limitation is the inability to generalize beyond the training distribution without significant retraining or architectural changes.
""")

    # Binary Cross Entropy content
    docs.append("""
## Core Equations: Binary Cross Entropy

The binary cross entropy loss function for a single sample is:

BCE(y, ŷ) = -[y * log(ŷ) + (1 - y) * log(1 - ŷ)]

Where y is the true label (0 or 1) and ŷ is the predicted probability.

For a batch of N samples, the average BCE is:
BCE = -(1/N) * Σ[y_i * log(ŷ_i) + (1 - y_i) * log(1 - ŷ_i)]
""")

    # Q-learning content
    docs.append("""
## Q-learning Update Rule

The Q-learning update rule is:

Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

Where:
- α (alpha) is the learning rate
- γ (gamma) is the discount factor
- r is the reward
- s' is the next state
- a' is the next action

This is the temporal difference update that makes Q-learning an off-policy algorithm.
""")

    return docs


def test_compound_query_detection():
    """Test the compound query detection and splitting functions."""
    from rag.query.engine import _is_compound_query, _split_compound_query

    print("=" * 70)
    print("TESTING COMPOUND QUERY DETECTION AND SPLITTING")
    print("=" * 70)

    test_cases = [
        # (query, expected_compound, expected_min_parts)
        ("What is ANI and what's its main limitation?", True, 2),
        ("What are the main limitations of ANI?", False, 1),
        ("Write out the formula for binary cross entropy. What is the update rule for Q-learning?", True, 2),
        ("What is the BCE formula?", False, 1),
        ("How does Q-learning work and why is it off-policy?", True, 2),
        ("What is machine learning?", False, 1),
        ("What is X and what is Y and what is Z?", True, 2),  # At least 2
        ("Explain X. Also explain Y?", True, 2),
    ]

    for query, expected_compound, expected_min_parts in test_cases:
        is_compound = _is_compound_query(query)
        parts = _split_compound_query(query)

        print(f"\nQuery: '{query}'")
        print(f"  Is compound: {is_compound} (expected: {expected_compound})")
        print(f"  Parts ({len(parts)}): {parts}")

        assert is_compound == expected_compound, f"Compound detection failed for: {query}"
        assert len(parts) >= expected_min_parts, f"Split failed for: {query} - got {len(parts)} parts"

    print("\n[OK] All detection/splitting tests passed!")


def test_compound_query_retrieval():
    """Test compound query retrieval with the actual engine."""
    print("\n" + "=" * 70)
    print("TESTING COMPOUND QUERY RETRIEVAL")
    print("=" * 70)

    # Create test documents
    docs = create_test_documents()
    full_text = "\n\n".join(docs)

    # Chunk the documents
    chunker = DocumentChunker(chunk_size=512, overlap=50)
    chunks = chunker.chunk_text(full_text, "test_docs.txt")

    print(f"\nTotal chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk.text[:80]}...")

    # Embed and index
    embedder = DummyEmbeddingProvider(dimension=128)
    embeddings = embedder.embed([c.text for c in chunks])

    index = BruteForceIndex(dimension=128)
    index.add(embeddings, chunks)

    # Create query engine
    engine = QueryEngine(index=index, embedding_provider=embedder, llm_provider=DummyLLMProvider())

    # Test Case 1: ANI compound question
    print("\n" + "-" * 70)
    print("TEST CASE 1: ANI Compound Question")
    print("-" * 70)
    query1 = "What is Artificial Narrow Intelligence and what's its main limitation?"

    # Test WITHOUT decomposition
    print("\n--- WITHOUT decomposition ---")
    result_no_decomp = engine.search(query1, k=5, decompose=False)
    for i, r in enumerate(result_no_decomp.results):
        marker = " <-- ANI LIMITATION" if "limitation" in r.chunk.text.lower() else ""
        print(f"  Rank {i+1}: score={r.score:.4f} | {r.chunk.text[:80]}...{marker}")

    # Test WITH decomposition
    print("\n--- WITH decomposition ---")
    result_decomp = engine.search(query1, k=5, decompose=True)
    for i, r in enumerate(result_decomp.results):
        marker = " <-- ANI LIMITATION" if "limitation" in r.chunk.text.lower() else ""
        marker2 = " <-- ANI DEFINITION" if "artificial narrow intelligence" in r.chunk.text.lower() else ""
        print(f"  Rank {i+1}: score={r.score:.4f} | {r.chunk.text[:80]}...{marker}{marker2}")

    # Check results
    ani_definition_found = any("Artificial Narrow Intelligence" in r.chunk.text and "limitation" not in r.chunk.text.lower() for r in result_decomp.results)
    ani_limitation_found = any("limitation" in r.chunk.text.lower() for r in result_decomp.results)

    print(f"\n  ANI Definition found: {ani_definition_found}")
    print(f"  ANI Limitation found: {ani_limitation_found}")

    # Test Case 2: BCE + Q-learning compound question
    print("\n" + "-" * 70)
    print("TEST CASE 2: BCE + Q-learning Compound Question")
    print("-" * 70)
    query2 = "Write out the formula for binary cross entropy. What is the update rule for Q-learning?"

    # Test WITHOUT decomposition
    print("\n--- WITHOUT decomposition ---")
    result_no_decomp2 = engine.search(query2, k=5, decompose=False)
    for i, r in enumerate(result_no_decomp2.results):
        marker = " <-- BCE" if "cross entropy" in r.chunk.text.lower() else ""
        marker2 = " <-- Q-LEARNING" if "q-learning" in r.chunk.text.lower() else ""
        print(f"  Rank {i+1}: score={r.score:.4f} | {r.chunk.text[:80]}...{marker}{marker2}")

    # Test WITH decomposition
    print("\n--- WITH decomposition ---")
    result_decomp2 = engine.search(query2, k=5, decompose=True)
    for i, r in enumerate(result_decomp2.results):
        marker = " <-- BCE" if "cross entropy" in r.chunk.text.lower() else ""
        marker2 = " <-- Q-LEARNING" if "q-learning" in r.chunk.text.lower() else ""
        print(f"  Rank {i+1}: score={r.score:.4f} | {r.chunk.text[:80]}...{marker}{marker2}")

    # Check results
    bce_found = any("cross entropy" in r.chunk.text.lower() for r in result_decomp2.results)
    qlearning_found = any("q-learning" in r.chunk.text.lower() for r in result_decomp2.results)

    print(f"\n  BCE formula found: {bce_found}")
    print(f"  Q-learning rule found: {qlearning_found}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test 1 (ANI): Definition={ani_definition_found}, Limitation={ani_limitation_found}")
    print(f"Test 2 (BCE+QL): BCE={bce_found}, Q-learning={qlearning_found}")

    success = ani_definition_found and ani_limitation_found and bce_found and qlearning_found
    if success:
        print("\n✓ ALL TESTS PASSED - Compound query decomposition works!")
    else:
        print("\n✗ SOME TESTS FAILED")

    return success


def test_generate_with_decomposition():
    """Test the full generate pipeline with compound query decomposition."""
    print("\n" + "=" * 70)
    print("TESTING GENERATE WITH COMPOUND QUERY DECOMPOSITION")
    print("=" * 70)

    docs = create_test_documents()
    full_text = "\n\n".join(docs)

    chunker = DocumentChunker(chunk_size=512, overlap=50)
    chunks = chunker.chunk_text(full_text, "test_docs.txt")

    embedder = DummyEmbeddingProvider(dimension=128)
    embeddings = embedder.embed([c.text for c in chunks])

    index = BruteForceIndex(dimension=128)
    index.add(embeddings, chunks)

    engine = QueryEngine(index=index, embedding_provider=embedder, llm_provider=DummyLLMProvider())

    # Test ANI compound question
    query = "What is Artificial Narrow Intelligence and what's its main limitation?"

    print(f"\nQuery: '{query}'")
    print("\n--- WITH decomposition (default) ---")
    result_decomp = engine.generate(query, k=5, decompose_compound_queries=True)
    print(f"Answer: {result_decomp.answer[:200]}...")
    print(f"Citations: {len(result_decomp.citations)}")
    for c in result_decomp.citations:
        print(f"  - {c.source_file}, chunk {c.chunk_index}: {c.text_snippet[:80]}...")

    print("\n--- WITHOUT decomposition ---")
    result_no_decomp = engine.generate(query, k=5, decompose_compound_queries=False)
    print(f"Answer: {result_no_decomp.answer[:200]}...")
    print(f"Citations: {len(result_no_decomp.citations)}")
    for c in result_no_decomp.citations:
        print(f"  - {c.source_file}, chunk {c.chunk_index}: {c.text_snippet[:80]}...")


if __name__ == "__main__":
    test_compound_query_detection()
    success = test_compound_query_retrieval()
    test_generate_with_decomposition()

    if success:
        print("\n" + "=" * 70)
        print("OVERALL: COMPOUND QUERY DECOMPOSITION WORKS CORRECTLY")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("OVERALL: SOME TESTS FAILED")
        print("=" * 70)
        sys.exit(1)