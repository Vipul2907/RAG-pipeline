"""
Test the deduplication fix for near-identical chunks.

This mimics the user's scenario: 9 sections with "### Limitations" header
followed by nearly identical boilerplate text, plus 1 section with
ANI-specific content.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
from rag.embeddings import DummyEmbeddingProvider
from rag.index import HNSWIndex, BruteForceIndex
from rag.ingestion import DocumentChunker
from rag.query import QueryEngine
from rag.llm import DummyLLMProvider
from rag.config import config


def create_test_documents():
    """Create test documents mimicking user's real data pattern."""
    # 9 sections with nearly identical "Limitations" boilerplate
    boilerplate = (
        "The main limitations include data sparsity in high-dimensional spaces, "
        "distribution shift between training and deployment environments, "
        "noisy labels that corrupt supervision signals, and computational "
        "constraints that limit model capacity."
    )

    sections = []

    # 9 generic Limitations sections - exact same text (like user's real data)
    for i in range(1, 10):
        sections.append(
            f"### Limitations\n\n{boilerplate}"
        )

    # 1 ANI-specific section (the important one)
    sections.append(
        "### Interview Question 16: What are the main limitations of ANI?\n\n"
        "Artificial Narrow Intelligence (ANI) has several key limitations: "
        "it cannot generalize beyond its narrow training domain, lacks common "
        "sense reasoning, requires massive labeled datasets, and fails on "
        "out-of-distribution inputs. These are fundamental architectural limits."
    )

    # 1 ANI answer section
    sections.append(
        "ANI systems excel at specific tasks but cannot transfer knowledge "
        "to new domains. The main limitation is the inability to generalize "
        "beyond the training distribution without significant retraining or "
        "architectural changes."
    )

    return "\n\n".join(sections)


def test_deduplication_fix():
    """Test that deduplication removes boilerplate duplicates."""
    print("=" * 70)
    print("TESTING DEDUPLICATION FIX")
    print("=" * 70)

    # Create test document
    text = create_test_documents()

    # Chunk it
    chunker = DocumentChunker(chunk_size=512, overlap=50)
    chunks = chunker.chunk_text(text, "test.md")

    print(f"\nTotal chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk.text[:80]}...")

    # Embed and index
    embedder = DummyEmbeddingProvider(dimension=128)
    embeddings = embedder.embed([c.text for c in chunks])

    index = BruteForceIndex(dimension=128)
    index.add(embeddings, chunks)

    # Create query engine with dummy LLM provider
    from rag.llm import DummyLLMProvider
    engine = QueryEngine(index=index, embedding_provider=embedder, llm_provider=DummyLLMProvider())

    # Test query
    query = "What are the main limitations of ANI?"

    print(f"\nQuery: '{query}'")
    print("-" * 70)

    # Test WITHOUT deduplication
    print("\n--- WITHOUT deduplication ---")
    result_no_dedup = engine.search(query, k=10, deduplicate=False)
    for i, r in enumerate(result_no_dedup.results):
        marker = " <-- ANI QUESTION" if "Interview Question 16" in r.chunk.text else ""
        marker2 = " <-- ANI ANSWER" if "ANI systems excel" in r.chunk.text else ""
        print(f"  Rank {i+1}: score={r.score:.4f} | {r.chunk.text[:80]}...{marker}{marker2}")

    # Test WITH deduplication
    print("\n--- WITH deduplication (threshold=0.95) ---")
    result_dedup = engine.search(query, k=10, deduplicate=True, dedup_threshold=0.95)
    for i, r in enumerate(result_dedup.results):
        marker = " <-- ANI QUESTION" if "Interview Question 16" in r.chunk.text else ""
        marker2 = " <-- ANI ANSWER" if "ANI systems excel" in r.chunk.text else ""
        print(f"  Rank {i+1}: score={r.score:.4f} | {r.chunk.text[:80]}...{marker}{marker2}")

    # Find ANI chunks in results
    ani_question_rank = None
    ani_answer_rank = None

    for i, r in enumerate(result_dedup.results):
        if "Interview Question 16" in r.chunk.text:
            ani_question_rank = i + 1
        if "ANI systems excel" in r.chunk.text:
            ani_answer_rank = i + 1

    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"  ANI Question rank: {ani_question_rank}")
    print(f"  ANI Answer rank: {ani_answer_rank}")
    print(f"  Total results returned: {len(result_dedup.results)}")

    if ani_question_rank and ani_question_rank <= 5:
        print("\n[SUCCESS] ANI question now ranks in top 5!")
    elif ani_question_rank:
        print(f"\n[PARTIAL] ANI question ranks at {ani_question_rank} (was 8)")
    else:
        print("\n[FAILED] ANI question not in top 10")

    return result_dedup


def test_hnsw_with_dedup():
    """Test with HNSW index as well."""
    print("\n" + "=" * 70)
    print("TESTING WITH HNSW INDEX")
    print("=" * 70)

    text = create_test_documents()
    chunker = DocumentChunker(chunk_size=512, overlap=50)
    chunks = chunker.chunk_text(text, "test.md")

    embedder = DummyEmbeddingProvider(dimension=128)
    embeddings = embedder.embed([c.text for c in chunks])

    index = HNSWIndex(dimension=128, M=16, ef_construction=200, ef_search=100)
    index.add(embeddings, chunks)

    engine = QueryEngine(index=index, embedding_provider=embedder, llm_provider=DummyLLMProvider())

    query = "What are the main limitations of ANI?"

    result = engine.search(query, k=10, deduplicate=True, dedup_threshold=0.95)

    print(f"\nQuery: '{query}'")
    for i, r in enumerate(result.results):
        marker = " <-- ANI QUESTION" if "Interview Question 16" in r.chunk.text else ""
        marker2 = " <-- ANI ANSWER" if "ANI systems excel" in r.chunk.text else ""
        print(f"  Rank {i+1}: score={r.score:.4f} | {r.chunk.text[:80]}...{marker}{marker2}")

    ani_question_rank = None
    for i, r in enumerate(result.results):
        if "Interview Question 16" in r.chunk.text:
            ani_question_rank = i + 1
            break

    print(f"\nANI Question rank with HNSW: {ani_question_rank}")


if __name__ == "__main__":
    test_deduplication_fix()
    test_hnsw_with_dedup()