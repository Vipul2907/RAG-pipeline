"""Embeddings module."""

from .provider import EmbeddingProvider, DummyEmbeddingProvider, get_embedding_provider

__all__ = ["EmbeddingProvider", "DummyEmbeddingProvider", "get_embedding_provider"]