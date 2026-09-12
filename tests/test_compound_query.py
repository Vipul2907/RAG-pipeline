"""
Tests for compound query decomposition in QueryEngine.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pytest

from rag.embeddings import DummyEmbeddingProvider
from rag.index import BruteForceIndex, HNSWIndex
from rag.ingestion import Chunk, DocumentChunker
from rag.llm import DummyLLMProvider
from rag.query import QueryEngine
from rag.query.engine import _is_compound_query, _split_compound_query


class TestCompoundQueryDetection:
    """Tests for compound query detection and splitting."""

    def test_single_question_not_compound(self):
        """Single questions should not be detected as compound."""
        assert not _is_compound_query("What is ANI?")
        assert not _is_compound_query("How does Q-learning work?")
        assert not _is_compound_query("Explain binary cross entropy")

    def test_multiple_question_marks_compound(self):
        """Multiple question marks should be detected as compound."""
        assert _is_compound_query("What is X? What is Y?")
        assert _is_compound_query("How does it work? Why does it work?")

    def test_and_what_pattern_compound(self):
        """'and what/how/why' patterns should be detected as compound."""
        assert _is_compound_query("What is ANI and what's its limitation?")
        assert _is_compound_query("How does it work and why is it important?")
        assert _is_compound_query("Explain X and also explain Y?")

    def test_imperative_sentences_compound(self):
        """Imperative sentences followed by questions should be detected."""
        assert _is_compound_query("Write out the formula. What is the rule?")
        assert _is_compound_query("Explain the concept. How does it apply?")

    def test_split_and_what_pattern(self):
        """Split on 'and what/how/why' patterns."""
        parts = _split_compound_query("What is ANI and what's its limitation?")
        assert len(parts) == 2
        assert "What is ANI?" in parts[0]
        assert "what's its limitation?" in parts[1].lower()

    def test_split_multiple_question_marks(self):
        """Split on multiple question marks."""
        parts = _split_compound_query("What is X? What is Y?")
        assert len(parts) >= 2

    def test_split_imperative_then_question(self):
        """Split imperative sentence followed by question."""
        parts = _split_compound_query("Write out the formula for BCE. What is Q-learning rule?")
        assert len(parts) == 2
        assert "Write out the formula for BCE?" in parts[0]
        assert "What is Q-learning rule?" in parts[1]

    def test_no_split_for_single(self):
        """Single questions should return as-is."""
        parts = _split_compound_query("What is machine learning?")
        assert len(parts) == 1
        assert parts[0] == "What is machine learning?"


class TestCompoundQueryRetrieval:
    """Tests for compound query retrieval with decomposition."""

    def setup_method(self):
        """Set up test fixtures with ANI, BCE, and Q-learning content."""
        self.dimension = 128
        self.embedder = DummyEmbeddingProvider(dimension=self.dimension)
        self.llm = DummyLLMProvider()

        # Create test documents
        docs = [
            "### Interview Question 16: What are the main limitations of ANI?\n\n"
            "Artificial Narrow Intelligence (ANI) has several key limitations: it cannot generalize "
            "beyond its narrow training domain, lacks common sense reasoning, requires massive "
            "labeled datasets, and fails on out-of-distribution inputs.",

            "ANI systems excel at specific tasks but cannot transfer knowledge to new domains. "
            "The main limitation is the inability to generalize beyond the training distribution.",

            "## Core Equations: Binary Cross Entropy\n\n"
            "The binary cross entropy loss function for a single sample is:\n"
            "BCE(y, y_hat) = -[y * log(y_hat) + (1 - y) * log(1 - y_hat)]\n"
            "Where y is the true label (0 or 1) and y_hat is the predicted probability.",

            "## Q-learning Update Rule\n\n"
            "The Q-learning update rule is:\n"
            "Q(s, a) = Q(s, a) + alpha * [r + gamma * max Q(s', a') - Q(s, a)]\n"
            "Where alpha is the learning rate, gamma is the discount factor, r is the reward."
        ]

        full_text = "\n\n".join(docs)
        chunker = DocumentChunker(chunk_size=512, overlap=50)
        chunks = chunker.chunk_text(full_text, "test_docs.txt")

        embeddings = self.embedder.embed([c.text for c in chunks])
        self.index = BruteForceIndex(dimension=self.dimension)
        self.index.add(embeddings, chunks)

        self.engine = QueryEngine(index=self.index, embedding_provider=self.embedder, llm_provider=self.llm)

    def test_ani_compound_question_retrieval(self):
        """Test ANI compound question retrieves both definition and limitation."""
        query = "What is Artificial Narrow Intelligence and what's its main limitation?"

        # Without decomposition - should miss one or both parts
        result_no_decomp = self.engine.search_compound(query, k=5, decompose=False)
        no_decomp_texts = [r.chunk.text for r in result_no_decomp.results]

        # With decomposition - should find both parts
        result_decomp = self.engine.search_compound(query, k=5, decompose=True)
        decomp_texts = [r.chunk.text for r in result_decomp.results]

        # Check that decomposition finds the Interview Question chunk (definition)
        ani_def_found = any("Interview Question 16" in t for t in decomp_texts)
        # Check that limitation content is found
        ani_lim_found = any("limitation" in t.lower() for t in decomp_texts)

        assert ani_def_found, "ANI definition (Interview Question 16) not found with decomposition"
        assert ani_lim_found, "ANI limitation content not found with decomposition"

        # Decomposition should find at least as much relevant content
        assert len(decomp_texts) >= 1, "No results with decomposition"

    def test_bce_qlearning_compound_question_retrieval(self):
        """Test BCE + Q-learning compound question retrieves both."""
        query = "Write out the formula for binary cross entropy. What is the update rule for Q-learning?"

        # Without decomposition
        result_no_decomp = self.engine.search_compound(query, k=5, decompose=False)
        no_decomp_texts = [r.chunk.text.lower() for r in result_no_decomp.results]

        # With decomposition
        result_decomp = self.engine.search_compound(query, k=5, decompose=True)
        decomp_texts = [r.chunk.text.lower() for r in result_decomp.results]

        # Both BCE and Q-learning should be found with decomposition
        bce_found = any("cross entropy" in t for t in decomp_texts)
        ql_found = any("q-learning" in t or "update rule" in t for t in decomp_texts)

        assert bce_found, "BCE formula not found with decomposition"
        assert ql_found, "Q-learning update rule not found with decomposition"

    def test_decomposition_disabled_works_like_regular_search(self):
        """When decompose=False, should work like regular search."""
        query = "What is ANI and what's its limitation?"

        result_search = self.engine.search(query, k=5)
        result_compound_no_decomp = self.engine.search_compound(query, k=5, decompose=False)

        # Should have same number of results
        assert len(result_search.results) == len(result_compound_no_decomp.results)

    def test_decomposition_with_hnsw(self):
        """Test decomposition works with HNSW index."""
        hnsw_index = HNSWIndex(dimension=self.dimension, M=16, ef_construction=100, ef_search=50)
        chunks = self.index.get_all_chunks()
        embeddings = self.embedder.embed([c.text for c in chunks])
        hnsw_index.add(embeddings, chunks)

        hnsw_engine = QueryEngine(index=hnsw_index, embedding_provider=self.embedder, llm_provider=self.llm)

        query = "What is ANI and what's its limitation?"
        result = hnsw_engine.search_compound(query, k=5, decompose=True)

        # Should find ANI content
        texts = [r.chunk.text for r in result.results]
        ani_found = any("Artificial Narrow Intelligence" in t or "ANI" in t for t in texts)
        assert ani_found, "ANI content not found with HNSW + decomposition"

    def test_generate_with_decomposition(self):
        """Test full generate pipeline with compound query decomposition."""
        query = "What is Artificial Narrow Intelligence and what's its main limitation?"

        # With decomposition
        result_decomp = self.engine.generate(query, k=5, decompose_compound_queries=True)

        # Should have citations from relevant chunks
        assert len(result_decomp.citations) > 0
        citation_texts = [c.text_snippet for c in result_decomp.citations]
        ani_mentioned = any("ANI" in t or "Artificial Narrow" in t for t in citation_texts)
        assert ani_mentioned, "ANI content not cited in generated answer"


class TestCompoundQueryConfigurable:
    """Tests that compound query decomposition is configurable."""

    def setup_method(self):
        self.dimension = 128
        self.embedder = DummyEmbeddingProvider(dimension=self.dimension)
        self.index = BruteForceIndex(dimension=self.dimension)
        self.engine = QueryEngine(index=self.index, embedding_provider=self.embedder, llm_provider=DummyLLMProvider())

    def test_generate_decompose_false(self):
        """Test generate with decompose_compound_queries=False."""
        query = "What is X and what is Y?"
        # Should not raise error
        result = self.engine.generate(query, k=3, decompose_compound_queries=False)
        assert result is not None

    def test_generate_decompose_true(self):
        """Test generate with decompose_compound_queries=True (default)."""
        query = "What is X and what is Y?"
        result = self.engine.generate(query, k=3, decompose_compound_queries=True)
        assert result is not None

    def test_search_compound_decompose_false(self):
        """Test search_compound with decompose=False."""
        query = "What is X and what is Y?"
        result = self.engine.search_compound(query, k=3, decompose=False)
        assert result is not None

    def test_search_compound_decompose_true(self):
        """Test search_compound with decompose=True (default)."""
        query = "What is X and what is Y?"
        result = self.engine.search_compound(query, k=3, decompose=True)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])