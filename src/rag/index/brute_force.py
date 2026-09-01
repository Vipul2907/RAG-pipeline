"""
Brute-force flat cosine similarity index.

Simple, correct baseline implementation for validating HNSW results.
Stores normalized embeddings in a flat array and computes exact cosine similarity.
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import VectorIndex, SearchResult
from ..ingestion import Chunk
from ..config import logger


class BruteForceIndex(VectorIndex):
    """
    Flat brute-force cosine similarity index.

    Stores all embeddings in a single normalized matrix.
    Search computes exact cosine similarity via matrix multiplication.
    Guarantees correct results (modulo floating point) for validation.
    """

    def __init__(self, dimension: int):
        super().__init__(dimension)
        self._embeddings: np.ndarray = np.zeros((0, dimension), dtype=np.float32)
        self._chunks: List[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """
        Add embeddings and chunks to the index.

        Args:
            embeddings: Array of shape (n, dimension).
            chunks: List of Chunk objects, one per embedding.

        Raises:
            ValueError: If validation fails.
        """
        self._validate_embeddings(embeddings, len(chunks))

        # Ensure embeddings are normalized (cosine similarity = dot product)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        normalized = embeddings / norms

        # Append to existing index
        if self._embeddings.size == 0:
            self._embeddings = normalized
        else:
            self._embeddings = np.vstack([self._embeddings, normalized])

        self._chunks.extend(chunks)
        self._size = len(self._chunks)

        logger.info(f"Added {len(chunks)} vectors to BruteForceIndex (total: {self._size})")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_fn: Optional[callable] = None,
    ) -> List[SearchResult]:
        """
        Search for top-k most similar vectors using exact cosine similarity.

        Args:
            query_embedding: Query vector of shape (dimension,).
            k: Number of results to return.
            filter_fn: Optional callable(chunk) -> bool to filter results.

        Returns:
            List of SearchResult sorted by score descending.
        """
        if self._size == 0:
            logger.warning("Search on empty index")
            return []

        # Validate and normalize query
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError(f"Query must be 1D array, got {query.ndim}D")
        if query.shape[0] != self._dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self._dimension}, got {query.shape[0]}"
            )

        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            raise ValueError("Query vector has zero norm")
        query = query / query_norm

        # Compute cosine similarities: dot product of normalized vectors
        # Shape: (n_vectors,)
        scores = self._embeddings @ query

        # Get top-k indices
        k = min(k, self._size)
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])][::-1]  # Sort descending

        # Build results with filtering
        results = []
        for rank, idx in enumerate(top_indices, 1):
            chunk = self._chunks[idx]
            if filter_fn and not filter_fn(chunk):
                continue
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(scores[idx]),
                    rank=rank,
                )
            )

        logger.debug(f"BruteForceIndex search: k={k}, returned={len(results)} results")
        return results

    def save(self, path: Path) -> None:
        """
        Save index to disk as .npy (embeddings) + .json (metadata).

        Args:
            path: Directory path to save index files.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        embeddings_path = path / "embeddings.npy"
        np.save(embeddings_path, self._embeddings)

        # Save chunk metadata as JSON
        chunks_path = path / "chunks.json"
        chunks_data = []
        for chunk in self._chunks:
            chunks_data.append({
                "text": chunk.text,
                "source_file": chunk.source_file,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            })

        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        # Save config
        config_path = path / "config.json"
        config_data = {
            "dimension": self._dimension,
            "size": self._size,
            "index_type": "brute_force",
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Saved BruteForceIndex to {path} ({self._size} vectors)")

    @classmethod
    def load(cls, path: Path) -> "BruteForceIndex":
        """
        Load index from disk.

        Args:
            path: Directory containing index files.

        Returns:
            Loaded BruteForceIndex instance.

        Raises:
            FileNotFoundError: If required files are missing.
            ValueError: If data is corrupted or incompatible.
        """
        path = Path(path)

        # Load config
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = json.load(f)

        dimension = config_data.get("dimension")
        if dimension is None:
            raise ValueError("Config missing 'dimension' field")

        # Load embeddings
        embeddings_path = path / "embeddings.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

        embeddings = np.load(embeddings_path)
        if embeddings.shape[1] != dimension:
            raise ValueError(
                f"Embedding dimension mismatch: config={dimension}, "
                f"data={embeddings.shape[1]}"
            )

        # Load chunks
        chunks_path = path / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks not found: {chunks_path}")

        with open(chunks_path, "r") as f:
            chunks_data = json.load(f)

        chunks = []
        for data in chunks_data:
            chunks.append(Chunk(
                text=data["text"],
                source_file=data["source_file"],
                start_char=data["start_char"],
                end_char=data["end_char"],
                chunk_index=data["chunk_index"],
                metadata=data.get("metadata", {}),
            ))

        # Create index and populate
        index = cls(dimension)
        index._embeddings = embeddings
        index._chunks = chunks
        index._size = len(chunks)

        logger.info(f"Loaded BruteForceIndex from {path} ({index._size} vectors)")
        return index

    def get_all_chunks(self) -> List[Chunk]:
        """Return all chunks in the index (for inspection/debugging)."""
        return self._chunks.copy()

    def get_embeddings(self) -> np.ndarray:
        """Return all embeddings (for inspection/debugging)."""
        return self._embeddings.copy()