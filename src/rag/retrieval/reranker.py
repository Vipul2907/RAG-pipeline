"""Cross-encoder reranker implementations for improved retrieval quality.

Cross-encoders score (query, document) pairs jointly, providing better
relevance scoring than bi-encoder dot products.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ..config import config, logger
from ..index import SearchResult
from ..ingestion import Chunk


class Reranker(ABC):
    """Abstract base class for cross-encoder rerankers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: Original query text
            results: Search results to rerank
            top_k: Number of top results to return after reranking

        Returns:
            Reranked SearchResult list (length <= top_k)
        """
        pass

    def _prepare_pairs(self, query: str, results: List[SearchResult]) -> List[tuple]:
        """Prepare (query, chunk_text) pairs for cross-encoder."""
        return [(query, r.chunk.text) for r in results]


class BGEReranker(Reranker):
    """
    BGE Reranker (BAAI/bge-reranker-v2-m3).

    State-of-the-art cross-encoder for reranking.
    Supports multiple languages, long contexts (up to 8192 tokens).
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        batch_size: int = None,
        max_length: int = 512,
    ):
        """
        Initialize BGE reranker.

        Args:
            model_name: HuggingFace model name (default from config)
            device: Device to run on (cuda, cpu, mps, or None for auto)
            batch_size: Batch size for inference (default from config)
            max_length: Max sequence length for cross-encoder
        """
        self._model_name = model_name or config.reranker_model
        self._batch_size = batch_size or config.reranker_batch_size
        self._max_length = max_length

        # Determine device
        if device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"
        self._device = device or (config.reranker_device or "cpu")

        # Lazy load model
        self._model = None
        self._tokenizer = None
        self._load_model()

        logger.info(f"Initialized BGEReranker: model={self._model_name}, device={self._device}")

    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for BGEReranker. "
                "Install with: pip install sentence-transformers torch"
            ) from e

        self._model = CrossEncoder(
            self._model_name,
            device=self._device,
            max_length=self._max_length,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """Rerank using BGE cross-encoder."""
        if not results:
            return results

        # Prepare pairs
        pairs = self._prepare_pairs(query, results)

        # Score in batches
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Combine with original results and sort by cross-encoder score
        scored_results = list(zip(scores, results))
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Return top-k with updated scores and ranks
        reranked = []
        for rank, (score, result) in enumerate(scored_results[:top_k], 1):
            reranked.append(SearchResult(
                chunk=result.chunk,
                score=float(score),  # Cross-encoder score (logits)
                rank=rank,
            ))

        logger.debug(f"BGE reranked {len(results)} -> {len(reranked)} results")

        return reranked


class MSMarcoReranker(Reranker):
    """
    MS MARCO MiniLM Reranker (cross-encoder/ms-marco-MiniLM-L-12-v2).

    Smaller, faster cross-encoder trained on MS MARCO.
    Good quality/speed tradeoff for CPU inference.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: str = None,
        batch_size: int = None,
        max_length: int = 512,
    ):
        """
        Initialize MS MARCO reranker.

        Args:
            model_name: HuggingFace model name
            device: Device to run on (cuda, cpu, mps, or None for auto)
            batch_size: Batch size for inference
            max_length: Max sequence length
        """
        self._model_name = model_name
        self._batch_size = batch_size or config.reranker_batch_size
        self._max_length = max_length

        # Determine device
        if device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"
        self._device = device or (config.reranker_device or "cpu")

        self._model = None
        self._load_model()

        logger.info(f"Initialized MSMarcoReranker: model={self._model_name}, device={self._device}")

    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for MSMarcoReranker. "
                "Install with: pip install sentence-transformers torch"
            ) from e

        self._model = CrossEncoder(
            self._model_name,
            device=self._device,
            max_length=self._max_length,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """Rerank using MS MARCO cross-encoder."""
        if not results:
            return results

        pairs = self._prepare_pairs(query, results)

        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        scored_results = list(zip(scores, results))
        scored_results.sort(key=lambda x: x[0], reverse=True)

        reranked = []
        for rank, (score, result) in enumerate(scored_results[:top_k], 1):
            reranked.append(SearchResult(
                chunk=result.chunk,
                score=float(score),
                rank=rank,
            ))

        logger.debug(f"MS MARCO reranked {len(results)} -> {len(reranked)} results")

        return reranked


class DummyReranker(Reranker):
    """
    Dummy reranker for testing.

    Returns results in original order with dummy scores.
    """

    def __init__(self, model_name: str = "dummy-reranker"):
        self._model_name = model_name
        logger.debug("Initialized DummyReranker")

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """Return results as-is (no reranking)."""
        return results[:top_k]


def get_reranker(reranker_name: str = None, **kwargs) -> Reranker:
    """
    Factory function to get a reranker by name.

    Args:
        reranker_name: Name of reranker ("bge", "msmarco", "dummy", or from config)
        **kwargs: Additional arguments passed to reranker constructor

    Returns:
        Reranker instance

    Raises:
        ValueError: If reranker name is unknown
    """
    name = (reranker_name or "").lower()

    if name in ("bge", "bge-reranker", "bge-reranker-v2-m3", "bge-reranker-v2"):
        return BGEReranker(**kwargs)
    elif name in ("msmarco", "ms-marco", "mini-lm", "cross-encoder/ms-marco-minilm-l-12-v2"):
        return MSMarcoReranker(**kwargs)
    elif name in ("dummy", "test", "none", ""):
        return DummyReranker(**kwargs)
    else:
        # Default to BGE for unknown names
        logger.warning(f"Unknown reranker: {name}, defaulting to BGE")
        return BGEReranker(**kwargs)