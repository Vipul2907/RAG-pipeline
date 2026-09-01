"""
Tests for the BruteForceIndex module.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.rag.index import BruteForceIndex, SearchResult
from src.rag.ingestion import Chunk
from src.rag.embeddings import DummyEmbeddingProvider


class TestBruteForceIndex:
    """Tests for BruteForceIndex class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.dimension = 128
        self.index = BruteForceIndex(dimension=self.dimension)
        self.embedder = DummyEmbeddingProvider(dimension=self.dimension)

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

    def test_init(self):
        """Test index initialization."""
        assert self.index.dimension == self.dimension
        assert self.index.size == 0

    def test_init_invalid_dimension(self):
        """Test that invalid dimension raises error."""
        with pytest.raises(ValueError, match="Dimension must be positive"):
            BruteForceIndex(dimension=0)

    def test_add_embeddings(self):
        """Test adding embeddings to index."""
        texts = ["First document", "Second document", "Third document"]
        chunks = self.create_chunks(texts)
        embeddings = self.embedder.embed(texts)

        self.index.add(embeddings, chunks)

        assert self.index.size == 3

    def test_add_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        chunks = self.create_chunks(["test"])
        # Wrong dimension
        embeddings = np.zeros((1, 64), dtype=np.float32)

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            self.index.add(embeddings, chunks)

    def test_add_count_mismatch(self):
        """Test that embedding/chunk count mismatch raises error."""
        chunks = self.create_chunks(["test1", "test2"])
        embeddings = self.embedder.embed(["only one"])

        with pytest.raises(ValueError, match="Embedding count mismatch"):
            self.index.add(embeddings, chunks)

    def test_add_wrong_ndim(self):
        """Test that wrong embedding ndim raises error."""
        chunks = self.create_chunks(["test"])
        embeddings = np.zeros((1, 10, 10), dtype=np.float32)  # 3D

        with pytest.raises(ValueError, match="Embeddings must be 2D array"):
            self.index.add(embeddings, chunks)

    def test_search_empty_index(self):
        """Test searching empty index returns empty list."""
        query_embedding = self.embedder.embed_single("query")
        results = self.index.search(query_embedding, k=5)

        assert results == []

    def test_search_basic(self):
        """Test basic search functionality."""
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "A completely different topic about cats",
            "Another document about dogs and pets",
        ]
        chunks = self.create_chunks(texts)
        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings, chunks)

        # Search for something similar to first doc
        query_embedding = self.embedder.embed_single("quick brown fox")
        results = self.index.search(query_embedding, k=3)

        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        # First result should be most similar
        assert results[0].rank == 1
        assert results[0].score >= results[1].score >= results[2].score

    def test_search_top_k(self):
        """Test that search respects k parameter."""
        texts = [f"Document {i}" for i in range(10)]
        chunks = self.create_chunks(texts)
        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings, chunks)

        query_embedding = self.embedder.embed_single("document")
        results = self.index.search(query_embedding, k=3)

        assert len(results) == 3
        assert results[0].rank == 1
        assert results[2].rank == 3

    def test_search_k_larger_than_index(self):
        """Test search with k larger than index size."""
        texts = ["Only one document"]
        chunks = self.create_chunks(texts)
        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings, chunks)

        query_embedding = self.embedder.embed_single("document")
        results = self.index.search(query_embedding, k=10)

        assert len(results) == 1

    def test_search_score_range(self):
        """Test that scores are in valid range [-1, 1]."""
        texts = ["Test document"]
        chunks = self.create_chunks(texts)
        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings, chunks)

        query_embedding = self.embedder.embed_single("test")
        results = self.index.search(query_embedding, k=1)

        assert -1.0 <= results[0].score <= 1.0

    def test_search_with_filter(self):
        """Test search with filter function."""
        texts = ["Doc A", "Doc B", "Doc C"]
        chunks = self.create_chunks(texts)
        chunks[0].metadata["category"] = "keep"
        chunks[1].metadata["category"] = "discard"
        chunks[2].metadata["category"] = "keep"

        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings, chunks)

        query_embedding = self.embedder.embed_single("doc")
        results = self.index.search(query_embedding, k=3, filter_fn=lambda c: c.metadata.get("category") == "keep")

        assert len(results) == 2
        assert all(r.chunk.metadata.get("category") == "keep" for r in results)

    def test_search_query_dimension_mismatch(self):
        """Test that query dimension mismatch raises error."""
        texts = ["test"]
        chunks = self.create_chunks(texts)
        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings, chunks)

        # Wrong dimension query
        query_embedding = np.zeros(64, dtype=np.float32)

        with pytest.raises(ValueError, match="Query dimension mismatch"):
            self.index.search(query_embedding, k=1)

    def test_search_zero_norm_query(self):
        """Test that zero-norm query raises error."""
        texts = ["test"]
        chunks = self.create_chunks(texts)
        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings, chunks)

        query_embedding = np.zeros(self.dimension, dtype=np.float32)

        with pytest.raises(ValueError, match="Query vector has zero norm"):
            self.index.search(query_embedding, k=1)

    def test_save_load_roundtrip(self):
        """Test save/load preserves index state."""
        texts = ["Document one", "Document two", "Document three"]
        chunks = self.create_chunks(texts)
        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings, chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test_index"
            self.index.save(index_path)

            # Load and verify
            loaded_index = BruteForceIndex.load(index_path)

            assert loaded_index.dimension == self.dimension
            assert loaded_index.size == 3

            # Verify search results match
            query_embedding = self.embedder.embed_single("document")
            original_results = self.index.search(query_embedding, k=3)
            loaded_results = loaded_index.search(query_embedding, k=3)

            assert len(original_results) == len(loaded_results)
            for orig, loaded in zip(original_results, loaded_results):
                assert orig.chunk.text == loaded.chunk.text
                assert abs(orig.score - loaded.score) < 1e-5
                assert orig.rank == loaded.rank

    def test_save_load_embeddings_preserved(self):
        """Test that embeddings are exactly preserved after save/load."""
        texts = ["Test document"]
        chunks = self.create_chunks(texts)
        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings, chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test_index"
            self.index.save(index_path)
            loaded_index = BruteForceIndex.load(index_path)

            # Embeddings should be identical (normalized)
            np.testing.assert_array_almost_equal(
                self.index.get_embeddings(),
                loaded_index.get_embeddings()
            )

    def test_save_creates_files(self):
        """Test that save creates all required files."""
        texts = ["Test"]
        chunks = self.create_chunks(texts)
        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings, chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test_index"
            self.index.save(index_path)

            assert (index_path / "embeddings.npy").exists()
            assert (index_path / "chunks.json").exists()
            assert (index_path / "config.json").exists()

    def test_load_missing_config(self):
        """Test loading from directory without config raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="Config not found"):
                BruteForceIndex.load(Path(tmpdir))

    def test_load_missing_embeddings(self):
        """Test loading from directory without embeddings raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir)
            # Create config only
            import json
            with open(index_path / "config.json", "w") as f:
                json.dump({"dimension": 128, "size": 0, "index_type": "brute_force"}, f)

            with pytest.raises(FileNotFoundError, match="Embeddings not found"):
                BruteForceIndex.load(index_path)

    def test_load_missing_chunks(self):
        """Test loading from directory without chunks raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir)
            import json
            with open(index_path / "config.json", "w") as f:
                json.dump({"dimension": 128, "size": 0, "index_type": "brute_force"}, f)
            np.save(index_path / "embeddings.npy", np.zeros((0, 128), dtype=np.float32))

            with pytest.raises(FileNotFoundError, match="Chunks not found"):
                BruteForceIndex.load(index_path)

    def test_load_dimension_mismatch(self):
        """Test loading with dimension mismatch raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir)
            import json
            with open(index_path / "config.json", "w") as f:
                json.dump({"dimension": 256, "size": 0, "index_type": "brute_force"}, f)
            np.save(index_path / "embeddings.npy", np.zeros((0, 128), dtype=np.float32))
            with open(index_path / "chunks.json", "w") as f:
                json.dump([], f)

            with pytest.raises(ValueError, match="Embedding dimension mismatch"):
                BruteForceIndex.load(index_path)

    def test_cosine_similarity_correctness(self):
        """Test that cosine similarity computation is mathematically correct."""
        # Create known vectors
        vec1 = np.array([1.0, 0.0, 0.0] + [0.0] * (self.dimension - 3), dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0] + [0.0] * (self.dimension - 3), dtype=np.float32)
        vec3 = np.array([1.0, 0.0, 0.0] + [0.0] * (self.dimension - 3), dtype=np.float32)  # Same as vec1

        embeddings = np.vstack([vec1, vec2, vec3])
        chunks = self.create_chunks(["v1", "v2", "v3"])

        # Normalize manually
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(embeddings, chunks)

        # Query with vec1 - should match vec1 and vec3 perfectly (score=1.0)
        query = vec1 / np.linalg.norm(vec1)
        results = self.index.search(query, k=3)

        # First and third should have score ~1.0
        assert abs(results[0].score - 1.0) < 1e-5
        assert abs(results[1].score - 1.0) < 1e-5
        # Second should have score ~0.0 (orthogonal)
        assert abs(results[2].score - 0.0) < 1e-5

    def test_get_all_chunks(self):
        """Test get_all_chunks returns copy."""
        texts = ["Test"]
        chunks = self.create_chunks(texts)
        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings, chunks)

        all_chunks = self.index.get_all_chunks()
        assert len(all_chunks) == 1
        assert all_chunks[0].text == "Test"

        # Modify copy shouldn't affect original
        all_chunks[0].text = "Modified"
        assert self.index.get_all_chunks()[0].text == "Test"

    def test_get_embeddings(self):
        """Test get_embeddings returns copy."""
        texts = ["Test"]
        chunks = self.create_chunks(texts)
        embeddings = self.embedder.embed(texts)
        self.index.add(embeddings, chunks)

        emb_copy = self.index.get_embeddings()
        assert emb_copy.shape == (1, self.dimension)

        # Modify copy shouldn't affect original
        emb_copy[0, 0] = 999
        assert self.index.get_embeddings()[0, 0] != 999


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_valid_search_result(self):
        """Test creating valid SearchResult."""
        chunk = Chunk(text="test", source_file="test.txt", start_char=0, end_char=4, chunk_index=0)
        result = SearchResult(chunk=chunk, score=0.5, rank=1)
        assert result.score == 0.5
        assert result.rank == 1

    def test_invalid_score_high(self):
        """Test that score > 1 raises error."""
        chunk = Chunk(text="test", source_file="test.txt", start_char=0, end_char=4, chunk_index=0)
        with pytest.raises(ValueError, match="Score must be in \\[-1, 1\\]"):
            SearchResult(chunk=chunk, score=1.5, rank=1)

    def test_invalid_score_low(self):
        """Test that score < -1 raises error."""
        chunk = Chunk(text="test", source_file="test.txt", start_char=0, end_char=4, chunk_index=0)
        with pytest.raises(ValueError, match="Score must be in \\[-1, 1\\]"):
            SearchResult(chunk=chunk, score=-1.5, rank=1)

    def test_invalid_rank(self):
        """Test that rank < 1 raises error."""
        chunk = Chunk(text="test", source_file="test.txt", start_char=0, end_char=4, chunk_index=0)
        with pytest.raises(ValueError, match="Rank must be >= 1"):
            SearchResult(chunk=chunk, score=0.5, rank=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])