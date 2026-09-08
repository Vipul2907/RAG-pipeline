"""
Tests for the QueryEngine module.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pytest

from rag.embeddings import DummyEmbeddingProvider
from rag.index import BruteForceIndex
from rag.ingestion import Chunk, DocumentChunker
from rag.llm import DummyLLMProvider
from rag.query import QueryEngine


class TestQueryEngine:
    """Tests for QueryEngine class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.dimension = 128
        self.embedder = DummyEmbeddingProvider(dimension=self.dimension)
        self.index = BruteForceIndex(dimension=self.dimension)
        self.llm = DummyLLMProvider()

    def create_chunks(self, texts, source_file="test.txt"):
        """Helper to create chunks with metadata."""
        chunks = []
        for i, text in enumerate(texts):
            chunks.append(Chunk(
                text=text,
                source_file=source_file,
                start_char=i * 100,
                end_char=(i + 1) * 100,
                chunk_index=i,
            ))
        return chunks

    def test_init_with_providers(self):
        """Test initialization with explicit providers."""
        engine = QueryEngine(
            index=self.index,
            embedding_provider=self.embedder,
            llm_provider=self.llm,
        )
        assert engine.index is self.index
        assert engine.embedding_provider is self.embedder
        assert engine.llm_provider is self.llm

    def test_search_returns_query_result(self):
        """Test search returns QueryResult with results."""
        texts = ["Document about cats", "Document about dogs"]
        chunks = self.create_chunks(texts)
        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings, chunks)

        engine = QueryEngine(index=self.index, embedding_provider=self.embedder, llm_provider=self.llm)
        result = engine.search("cats", k=2)

        assert hasattr(result, 'query')
        assert hasattr(result, 'results')
        assert hasattr(result, 'query_embedding')
        assert len(result.results) == 2
        assert result.query == "cats"

    def test_search_respects_k(self):
        """Test search respects k parameter."""
        texts = [f"Doc {i}" for i in range(10)]
        chunks = self.create_chunks(texts)
        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings, chunks)

        engine = QueryEngine(index=self.index, embedding_provider=self.embedder, llm_provider=self.llm)
        result = engine.search("doc", k=3)

        assert len(result.results) == 3

    def test_search_with_filter(self):
        """Test search with filter function."""
        texts = ["Doc A", "Doc B", "Doc C"]
        chunks = self.create_chunks(texts)
        chunks[0].metadata["category"] = "keep"
        chunks[1].metadata["category"] = "discard"
        chunks[2].metadata["category"] = "keep"

        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings, chunks)

        engine = QueryEngine(index=self.index, embedding_provider=self.embedder, llm_provider=self.llm)
        result = engine.search("doc", k=3, filter_fn=lambda c: c.metadata.get("category") == "keep")

        assert len(result.results) == 2
        assert all(r.chunk.metadata.get("category") == "keep" for r in result.results)


class TestDeduplication:
    """Tests for search result deduplication."""

    def setup_method(self):
        """Set up test fixtures."""
        self.dimension = 128
        self.embedder = DummyEmbeddingProvider(dimension=self.dimension)
        self.index = BruteForceIndex(dimension=self.dimension)
        self.llm = DummyLLMProvider()

    def create_chunks(self, texts, source_file="test.txt"):
        """Helper to create chunks with metadata."""
        chunks = []
        for i, text in enumerate(texts):
            chunks.append(Chunk(
                text=text,
                source_file=source_file,
                start_char=i * 100,
                end_char=(i + 1) * 100,
                chunk_index=i,
            ))
        return chunks

    def test_deduplication_removes_near_identical_chunks(self):
        """Test that near-identical chunks are removed from results."""
        # Create 9 identical "Limitations" chunks + 1 unique ANI chunk
        boilerplate = (
            "The main limitations include data sparsity in high-dimensional spaces, "
            "distribution shift between training and deployment environments, "
            "noisy labels that corrupt supervision signals, and computational "
            "constraints that limit model capacity."
        )

        texts = [boilerplate] * 9  # 9 identical
        texts.append("Artificial Narrow Intelligence (ANI) cannot generalize beyond its training domain.")

        chunks = self.create_chunks(texts)
        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings, chunks)

        engine = QueryEngine(index=self.index, embedding_provider=self.embedder, llm_provider=self.llm)

        # Without deduplication - all 9 identical chunks returned
        result_no_dedup = engine.search("limitations of ANI", k=10, deduplicate=False)
        assert len(result_no_dedup.results) == 10

        # With deduplication - only 1 of the 9 identical chunks kept
        result_dedup = engine.search("limitations of ANI", k=10, deduplicate=True, dedup_threshold=0.95)

        # Should have fewer results (9 identical collapsed to 1)
        assert len(result_dedup.results) < len(result_no_dedup.results)
        # The unique ANI chunk should be present
        ani_found = any("Artificial Narrow Intelligence" in r.chunk.text for r in result_dedup.results)
        assert ani_found, "ANI-specific chunk should be in deduplicated results"

    def test_deduplication_preserves_score_order(self):
        """Test that deduplication preserves the highest-scoring chunk from each group."""
        # Create chunks with same text but different scores (via different query match)
        text = "The main limitations include data sparsity and distribution shift."

        texts = [text] * 5
        chunks = self.create_chunks(texts)
        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings, chunks)

        engine = QueryEngine(index=self.index, embedding_provider=self.embedder, llm_provider=self.llm)

        # Query that matches the text
        result_dedup = engine.search("limitations data sparsity", k=10, deduplicate=True, dedup_threshold=0.95)

        # Should return only 1 result (the highest scoring)
        assert len(result_dedup.results) == 1
        assert result_dedup.results[0].rank == 1

    def test_no_deduplication_when_disabled(self):
        """Test that deduplication can be disabled."""
        text = "Identical text repeated."
        texts = [text] * 5
        chunks = self.create_chunks(texts)
        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings, chunks)

        engine = QueryEngine(index=self.index, embedding_provider=self.embedder, llm_provider=self.llm)

        # With deduplication disabled - all 5 returned
        result = engine.search("identical", k=10, deduplicate=False)
        assert len(result.results) == 5

    def test_deduplication_with_hnsw(self):
        """Test deduplication works with HNSW index."""
        from rag.index import HNSWIndex

        boilerplate = (
            "The main limitations include data sparsity in high-dimensional spaces, "
            "distribution shift between training and deployment environments, "
            "noisy labels that corrupt supervision signals."
        )

        texts = [boilerplate] * 5
        texts.append("ANI specific content about narrow intelligence limits.")

        chunks = self.create_chunks(texts)
        embeddings = self.embedder.embed(texts)

        hnsw_index = HNSWIndex(dimension=self.dimension, M=16, ef_construction=100, ef_search=50)
        hnsw_index.add(embeddings, chunks)

        engine = QueryEngine(index=hnsw_index, embedding_provider=self.embedder, llm_provider=self.llm)

        result_dedup = engine.search("limitations ANI", k=10, deduplicate=True, dedup_threshold=0.95)

        # Should deduplicate
        assert len(result_dedup.results) < 6
        ani_found = any("ANI specific" in r.chunk.text for r in result_dedup.results)
        assert ani_found


if __name__ == "__main__":
    pytest.main([__file__, "-v"])