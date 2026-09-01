"""
Query engine for the RAG pipeline.

Orchestrates the retrieval process: embedding query, searching index,
and returning formatted results.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import config, logger
from ..embeddings import EmbeddingProvider, get_embedding_provider
from ..index import VectorIndex, SearchResult
from ..ingestion import Chunk


@dataclass(frozen=True)
class QueryResult:
    """
    Result of a RAG query.

    Attributes:
        query: The original query text.
        results: List of search results with chunks and scores.
        query_embedding: The embedded query vector (for debugging).
    """
    query: str
    results: List[SearchResult]
    query_embedding: Optional[np.ndarray] = None


class QueryEngine:
    """
    High-level query engine for RAG retrieval.

    Combines embedding provider and vector index to provide
    a simple search interface.
    """

    def __init__(
        self,
        index: VectorIndex,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        """
        Initialize the query engine.

        Args:
            index: Vector index to search.
            embedding_provider: Embedding provider (default from config).
        """
        self._index = index
        self._embedding_provider = embedding_provider or get_embedding_provider()

        # Verify dimension compatibility
        if self._index.dimension != self._embedding_provider.dimension:
            raise ValueError(
                f"Dimension mismatch: index={self._index.dimension}, "
                f"provider={self._embedding_provider.dimension}"
            )

        logger.info(
            f"Initialized QueryEngine: index={type(self._index).__name__}, "
            f"provider={type(self._embedding_provider).__name__}"
        )

    def search(
        self,
        query: str,
        k: int = None,
        filter_fn: Optional[callable] = None,
    ) -> QueryResult:
        """
        Search the index for relevant chunks.

        Args:
            query: Query text string.
            k: Number of results (default from config).
            filter_fn: Optional filter function for results.

        Returns:
            QueryResult with search results.
        """
        k = k or config.default_top_k

        # Embed query
        query_embedding = self._embedding_provider.embed_single(query)

        # Search index
        results = self._index.search(query_embedding, k=k, filter_fn=filter_fn)

        logger.info(f"Query '{query[:50]}...': found {len(results)} results")

        return QueryResult(
            query=query,
            results=results,
            query_embedding=query_embedding,
        )

    def search_with_embedding(
        self,
        query_embedding: np.ndarray,
        k: int = None,
        filter_fn: Optional[callable] = None,
    ) -> QueryResult:
        """
        Search using a pre-computed query embedding.

        Args:
            query_embedding: Pre-computed query vector.
            k: Number of results (default from config).
            filter_fn: Optional filter function for results.

        Returns:
            QueryResult with search results.
        """
        k = k or config.default_top_k

        results = self._index.search(query_embedding, k=k, filter_fn=filter_fn)

        return QueryResult(
            query="<pre-computed embedding>",
            results=results,
            query_embedding=query_embedding,
        )

    @property
    def index(self) -> VectorIndex:
        """Access the underlying index."""
        return self._index

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Access the embedding provider."""
        return self._embedding_provider