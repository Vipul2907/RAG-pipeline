"""
Embedding provider abstraction.

Provides a unified interface for generating vector embeddings from text.
Includes a deterministic dummy provider for testing and real providers
for production use (HuggingFace sentence-transformers, OpenAI).
"""

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

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


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """
    HuggingFace sentence-transformers embedding provider.

    Uses local sentence-transformers models to generate embeddings.
    No API key required - runs entirely locally.

    Popular models:
    - "sentence-transformers/all-MiniLM-L6-v2" (384 dim, fast, good quality)
    - "sentence-transformers/all-mpnet-base-v2" (768 dim, higher quality)
    - "sentence-transformers/all-MiniLM-L12-v2" (384 dim, better quality)
    """

    _model_cache: dict[str, "HuggingFaceEmbeddingProvider"] = {}

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: Optional[int] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        """
        Initialize the HuggingFace embedding provider.

        Args:
            model_name: Name of the sentence-transformers model to use.
            dimension: Expected embedding dimension (auto-detected if None).
            device: Device to run on ("cpu", "cuda", "mps", or None for auto).
            batch_size: Batch size for encoding.
            normalize: Whether to normalize embeddings to unit length.
        """
        self.model_name = model_name
        self._batch_size = batch_size
        self._normalize = normalize

        # Lazy import to avoid dependency if not used
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for HuggingFaceEmbeddingProvider. "
                "Install with: pip install sentence-transformers torch"
            ) from e

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        logger.info(f"Loading HuggingFace model: {model_name} on {device}")

        self._model = SentenceTransformer(model_name, device=device)

        # Detect dimension from model
        if dimension is None:
            # Get dimension by encoding a test string
            test_emb = self._model.encode(["test"], convert_to_numpy=True, show_progress_bar=False)
            self._dimension = test_emb.shape[1]
        else:
            self._dimension = dimension

        logger.info(f"Initialized HuggingFaceEmbeddingProvider: model={model_name}, dim={self._dimension}")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using the sentence-transformers model.

        Args:
            texts: List of text strings to embed.

        Returns:
            Array of shape (n_texts, dimension) with embeddings.
        """
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)

        # Encode in batches
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )

        return embeddings.astype(np.float32)

    @classmethod
    def get_cached(cls, model_name: str, **kwargs) -> "HuggingFaceEmbeddingProvider":
        """Get or create a cached model instance."""
        if model_name not in cls._model_cache:
            cls._model_cache[model_name] = cls(model_name, **kwargs)
        return cls._model_cache[model_name]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding API provider.

    Uses OpenAI's text-embedding models via their API.
    Requires OPENAI_API_KEY environment variable.
    """

    # Known model dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimension: Optional[int] = None,
        api_key: Optional[str] = None,
        batch_size: int = 100,
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        """
        Initialize the OpenAI embedding provider.

        Args:
            model_name: OpenAI embedding model name.
            dimension: Expected embedding dimension (auto-detected if None).
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
            batch_size: Batch size for API requests.
            max_retries: Maximum number of retries for failed requests.
            timeout: Request timeout in seconds.
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required for OpenAIEmbeddingProvider. "
                "Install with: pip install openai"
            ) from e

        self.model_name = model_name
        self._batch_size = batch_size
        self._max_retries = max_retries

        # Get API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._client = OpenAI(api_key=self.api_key, timeout=timeout, max_retries=max_retries)

        # Set dimension
        if dimension is not None:
            self._dimension = dimension
        elif model_name in self.MODEL_DIMENSIONS:
            self._dimension = self.MODEL_DIMENSIONS[model_name]
        else:
            # Default to ada-002 dimension
            self._dimension = 1536

        logger.info(f"Initialized OpenAIEmbeddingProvider: model={model_name}, dim={self._dimension}")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using OpenAI API.

        Args:
            texts: List of text strings to embed.

        Returns:
            Array of shape (n_texts, dimension) with embeddings.
        """
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]

            try:
                response = self._client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    encoding_format="float",
                )

                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"OpenAI embedding failed for batch {i//self._batch_size}: {e}")
                raise

        embeddings = np.array(all_embeddings, dtype=np.float32)

        # Normalize for cosine similarity (OpenAI embeddings are already normalized,
        # but we ensure it for consistency)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        return embeddings


def get_embedding_provider(provider_name: Optional[str] = None, **kwargs) -> EmbeddingProvider:
    """
    Factory function to get an embedding provider by name.

    Args:
        provider_name: Name of provider ("dummy", "huggingface", "openai", or from config).
        **kwargs: Additional arguments passed to the provider constructor.

    Returns:
        EmbeddingProvider instance.

    Raises:
        ValueError: If provider name is unknown.
    """
    name = (provider_name or config.embedding_provider).lower()

    if name == "dummy":
        return DummyEmbeddingProvider(**kwargs)
    elif name in ("huggingface", "hf", "sentence-transformers"):
        model_name = kwargs.pop("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddingProvider(model_name=model_name, **kwargs)
    elif name in ("openai",):
        model_name = kwargs.pop("model_name", "text-embedding-3-small")
        return OpenAIEmbeddingProvider(model_name=model_name, **kwargs)
    else:
        raise ValueError(
            f"Unknown embedding provider: {name}. "
            f"Supported: 'dummy', 'huggingface', 'openai'. "
            f"Set EMBEDDING_PROVIDER env var or pass provider_name."
        )