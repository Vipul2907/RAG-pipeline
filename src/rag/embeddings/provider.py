"""
Embedding provider abstraction.

Provides a unified interface for generating vector embeddings from text.
Includes a deterministic dummy provider for testing and development.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..config import config, logger


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            Array of shape (n_texts, dimension) with embeddings.
        """
        pass

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.embed([text])[0]


class DummyEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic hash-based embedding provider for testing.

    Generates pseudo-random but reproducible embeddings using SHA-256 hashing.
    No external dependencies or API keys required.
    """

    def __init__(self, dimension: Optional[int] = None, seed: int = 42):
        """
        Initialize the dummy provider.

        Args:
            dimension: Embedding dimension (default from config).
            seed: Random seed for reproducibility.
        """
        self._dimension = dimension or config.embedding_dimension
        self._seed = seed
        logger.debug(f"Initialized DummyEmbeddingProvider: dimension={self._dimension}")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate deterministic embeddings using hashing.

        Each text produces a unique but reproducible vector based on its content.
        Vectors are normalized to unit length for cosine similarity.
        """
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)

        embeddings = np.zeros((len(texts), self._dimension), dtype=np.float32)

        for i, text in enumerate(texts):
            # Create a deterministic hash from the text
            hasher = hashlib.sha256()
            hasher.update(f"{self._seed}:{text}".encode("utf-8"))
            hash_bytes = hasher.digest()

            # Convert hash bytes to float32 values in [-1, 1]
            # Use multiple hash iterations if dimension > 32 bytes
            vec = np.zeros(self._dimension, dtype=np.float32)
            for j in range(self._dimension):
                byte_idx = j % len(hash_bytes)
                # Convert byte to float in [-1, 1]
                vec[j] = (hash_bytes[byte_idx] / 127.5) - 1.0

            # Normalize to unit length for cosine similarity
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            embeddings[i] = vec

        return embeddings


def get_embedding_provider(provider_name: Optional[str] = None) -> EmbeddingProvider:
    """
    Factory function to get an embedding provider by name.

    Args:
        provider_name: Name of provider ("dummy" or from config).

    Returns:
        EmbeddingProvider instance.

    Raises:
        ValueError: If provider name is unknown.
    """
    name = provider_name or config.embedding_provider

    if name == "dummy":
        return DummyEmbeddingProvider()
    else:
        raise ValueError(
            f"Unknown embedding provider: {name}. "
            f"Supported: 'dummy'. "
            f"Set EMBEDDING_PROVIDER env var or pass provider_name."
        )