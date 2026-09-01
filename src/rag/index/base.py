"""
Abstract base class for vector indices.

Defines the interface that all vector index implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..ingestion import Chunk
from ..config import logger


@dataclass(frozen=True)
class SearchResult:
    """
    Result of a vector similarity search.

    Attributes:
        chunk: The matched chunk with metadata.
        score: Cosine similarity score in [-1, 1].
        rank: Rank in the result set (1 = best match).
    """
    chunk: Chunk
    score: float
    rank: int

    def __post_init__(self):
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be in [-1, 1], got {self.score}")
        if self.rank < 1:
            raise ValueError(f"Rank must be >= 1, got {self.rank}")


class VectorIndex(ABC):
    """
    Abstract base class for vector similarity indices.

    All implementations must provide:
    - add(): Add embeddings with associated metadata
    - search(): Find top-k similar vectors
    - save()/load(): Persistence
    """

    def __init__(self, dimension: int):
        """
        Initialize the index.

        Args:
            dimension: Dimensionality of vectors to be indexed.
        """
        if dimension <= 0:
            raise ValueError("Dimension must be positive")
        self._dimension = dimension
        self._size = 0
        logger.debug(f"Initialized {self.__class__.__name__} with dimension={dimension}")

    @property
    def dimension(self) -> int:
        """Return the vector dimension."""
        return self._dimension

    @property
    def size(self) -> int:
        """Return the number of vectors in the index."""
        return self._size

    @abstractmethod
    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """
        Add vectors to the index.

        Args:
            embeddings: Array of shape (n, dimension) with vectors to add.
            chunks: List of Chunk objects corresponding to each embedding.

        Raises:
            ValueError: If embeddings shape doesn't match dimension or count mismatch.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_fn: Optional[callable] = None,
    ) -> List[SearchResult]:
        """
        Search for top-k similar vectors.

        Args:
            query_embedding: Query vector of shape (dimension,).
            k: Number of results to return.
            filter_fn: Optional callable(chunk) -> bool to filter results.

        Returns:
            List of SearchResult objects, sorted by score descending.
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save index to disk.

        Args:
            path: Directory path to save index files.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "VectorIndex":
        """
        Load index from disk.

        Args:
            path: Directory path containing index files.

        Returns:
            Loaded VectorIndex instance.
        """
        pass

    def _validate_embeddings(self, embeddings: np.ndarray, expected_count: int) -> None:
        """Validate embeddings array shape and type."""
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D array, got {embeddings.ndim}D")
        if embeddings.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimension}, "
                f"got {embeddings.shape[1]}"
            )
        if embeddings.shape[0] != expected_count:
            raise ValueError(
                f"Embedding count mismatch: expected {expected_count}, "
                f"got {embeddings.shape[0]}"
            )
        if embeddings.dtype != np.float32:
            logger.warning(
                f"Embeddings dtype is {embeddings.dtype}, converting to float32"
            )