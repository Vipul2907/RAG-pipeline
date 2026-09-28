"""Hybrid search combining BM25 sparse retrieval with dense vector search.

Uses Reciprocal Rank Fusion (RRF) to combine results from both systems.
"""

import logging
from typing import List, Optional, Callable, Dict, Any

import numpy as np

from ..config import config, logger
from ..index import VectorIndex, SearchResult
from ..embeddings import EmbeddingProvider
from ..ingestion import Chunk


try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    BM25Okapi = None


class HybridRetriever:
    """
    Combines BM25 sparse retrieval with dense vector search via RRF.

    BM25 excels at exact keyword matching (entity names, codes, acronyms).
    Dense vectors excel at semantic similarity.
    RRF provides parameter-free fusion that works well in practice.
    """

    def __init__(
        self,
        dense_index: VectorIndex,
        embedding_provider: EmbeddingProvider,
        chunks: List[Chunk],
        bm25_weight: float = None,
        dense_weight: float = None,
        rrf_k: int = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            dense_index: Vector index for dense search (HNSW or BruteForce)
            embedding_provider: Provider for query embeddings
            chunks: All chunks in the index (for BM25 corpus)
            bm25_weight: Weight for BM25 scores in RRF (default from config)
            dense_weight: Weight for dense scores in RRF (default from config)
            rrf_k: RRF constant k (default from config)
        """
        if not _BM25_AVAILABLE:
            raise ImportError(
                "rank-bm25 is required for HybridRetriever. "
                "Install with: pip install rank-bm25"
            )

        self.dense_index = dense_index
        self.embedding_provider = embedding_provider
        self.chunks = chunks

        self.bm25_weight = bm25_weight or config.bm25_weight
        self.dense_weight = dense_weight or config.dense_weight
        self.rrf_k = rrf_k or config.rrf_k

        # Build BM25 index from chunk texts
        self._build_bm25_index()

        logger.info(
            f"Initialized HybridRetriever: {len(chunks)} chunks, "
            f"bm25_weight={self.bm25_weight}, dense_weight={self.dense_weight}, rrf_k={self.rrf_k}"
        )

    def _build_bm25_index(self) -> None:
        """Build BM25 index from chunk texts."""
        # Tokenize all chunk texts
        corpus = [chunk.text.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(corpus)
        logger.debug(f"Built BM25 index with {len(corpus)} documents")

    def _bm25_search(self, query: str, k: int) -> List[SearchResult]:
        """Search using BM25 only."""
        query_tokens = query.lower().split()
        if not query_tokens:
            return []

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        k = min(k, len(self.chunks))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])][::-1]

        # Normalize BM25 scores to [-1, 1] range for SearchResult validation
        # BM25 scores can be arbitrarily large, so we use a soft normalization
        if len(top_indices) > 0:
            max_score = scores[top_indices[0]]
            min_score = scores[top_indices[-1]]
            # If all scores are the same, normalize to 0
            if max_score == min_score:
                normalized_scores = np.zeros_like(scores[top_indices])
            else:
                # Normalize to [0, 1] then scale to [-1, 1]
                normalized_scores = 2 * (scores[top_indices] - min_score) / (max_score - min_score) - 1
        else:
            normalized_scores = np.array([])

        # Convert to SearchResult
        results = []
        for rank, (idx, norm_score) in enumerate(zip(top_indices, normalized_scores), 1):
            if scores[idx] <= 0:
                continue
            results.append(SearchResult(
                chunk=self.chunks[idx],
                score=float(norm_score),
                rank=rank,
            ))

        return results

    def _dense_search(self, query: str, k: int, filter_fn: Optional[Callable] = None) -> List[SearchResult]:
        """Search using dense vector index."""
        query_embedding = self.embedding_provider.embed_single(query)
        return self.dense_index.search(query_embedding, k=k, filter_fn=filter_fn)

    def _rrf_fusion(
        self,
        bm25_results: List[SearchResult],
        dense_results: List[SearchResult],
        k: int,
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF score = sum(weight_i / (rank_i + k))
        """
        # Create mapping from chunk to ranks in each system
        chunk_to_bm25_rank = {}
        for r in bm25_results:
            chunk_to_bm25_rank[id(r.chunk)] = r.rank

        chunk_to_dense_rank = {}
        for r in dense_results:
            chunk_to_dense_rank[id(r.chunk)] = r.rank

        # All unique chunks
        all_chunks = {}
        for r in bm25_results + dense_results:
            cid = id(r.chunk)
            if cid not in all_chunks:
                all_chunks[cid] = r.chunk

        # Compute RRF scores
        fused_scores = []
        for cid, chunk in all_chunks.items():
            score = 0.0

            if cid in chunk_to_bm25_rank:
                score += self.bm25_weight / (chunk_to_bm25_rank[cid] + self.rrf_k)

            if cid in chunk_to_dense_rank:
                score += self.dense_weight / (chunk_to_dense_rank[cid] + self.rrf_k)

            fused_scores.append((score, chunk))

        # Sort by fused score descending
        fused_scores.sort(key=lambda x: x[0], reverse=True)

        # Convert to SearchResult
        results = []
        for rank, (score, chunk) in enumerate(fused_scores[:k], 1):
            results.append(SearchResult(
                chunk=chunk,
                score=score,
                rank=rank,
            ))

        return results

    def search(
        self,
        query: str,
        k: int = 10,
        filter_fn: Optional[Callable[[Chunk], bool]] = None,
    ) -> List[SearchResult]:
        """
        Hybrid search combining BM25 and dense retrieval.

        Args:
            query: Query text
            k: Number of results to return
            filter_fn: Optional filter function applied to results

        Returns:
            List of SearchResult objects sorted by fused score
        """
        # Search with both methods (get more candidates for better fusion)
        search_k = k * 3

        bm25_results = self._bm25_search(query, search_k)
        dense_results = self._dense_search(query, search_k, filter_fn)

        # Apply filter to BM25 results if provided
        if filter_fn:
            bm25_results = [r for r in bm25_results if filter_fn(r.chunk)]

        # Fuse results
        fused_results = self._rrf_fusion(bm25_results, dense_results, k)

        logger.debug(f"Hybrid search: {len(bm25_results)} BM25 + {len(dense_results)} dense -> {len(fused_results)} fused")

        return fused_results

    def search_with_embedding(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_fn: Optional[Callable[[Chunk], bool]] = None,
    ) -> List[SearchResult]:
        """
        Search using pre-computed query embedding (dense only, no BM25).

        Useful when query embedding is already available (e.g., from HyDE).
        """
        return self.dense_index.search(query_embedding, k=k, filter_fn=filter_fn)


def create_hybrid_retriever(
    dense_index: VectorIndex,
    embedding_provider: EmbeddingProvider,
    chunks: List[Chunk],
) -> HybridRetriever:
    """Factory function to create HybridRetriever from config."""
    return HybridRetriever(
        dense_index=dense_index,
        embedding_provider=embedding_provider,
        chunks=chunks,
    )