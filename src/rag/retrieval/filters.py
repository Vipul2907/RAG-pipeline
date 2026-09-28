"""Metadata filtering utilities for vector search.

Provides filter functions and filtered search mixin for HNSW.
"""

import logging
from typing import Callable, List, Optional, Set, Dict, Any

import numpy as np

from ..config import logger
from ..index import SearchResult
from ..ingestion import Chunk


def apply_filter(
    results: List[SearchResult],
    filter_fn: Callable[[Chunk], bool],
) -> List[SearchResult]:
    """
    Apply a filter function to search results.

    Args:
        results: Search results to filter
        filter_fn: Function taking a Chunk and returning bool

    Returns:
        Filtered results with updated ranks
    """
    filtered = []
    for rank, result in enumerate(results, 1):
        if filter_fn(result.chunk):
            # Create new result with updated rank
            filtered.append(SearchResult(
                chunk=result.chunk,
                score=result.score,
                rank=rank,
            ))

    logger.debug(f"Filter applied: {len(results)} -> {len(filtered)} results")
    return filtered


def metadata_filter(metadata_key: str, allowed_values: Set[Any]) -> Callable[[Chunk], bool]:
    """
    Create a filter function that matches metadata values.

    Args:
        metadata_key: Key in chunk.metadata to check
        allowed_values: Set of allowed values

    Returns:
        Filter function
    """
    def _filter(chunk: Chunk) -> bool:
        value = chunk.metadata.get(metadata_key)
        return value in allowed_values

    return _filter


def source_file_filter(allowed_files: Set[str]) -> Callable[[Chunk], bool]:
    """Create a filter for specific source files."""
    def _filter(chunk: Chunk) -> bool:
        return chunk.source_file in allowed_files
    return _filter


def chunk_index_filter(allowed_indices: Set[int]) -> Callable[[Chunk], bool]:
    """Create a filter for specific chunk indices."""
    def _filter(chunk: Chunk) -> bool:
        return chunk.chunk_index in allowed_indices
    return _filter


def text_contains_filter(substring: str, case_sensitive: bool = False) -> Callable[[Chunk], bool]:
    """Create a filter that checks if chunk text contains substring."""
    if not case_sensitive:
        substring = substring.lower()
    def _filter(chunk: Chunk) -> bool:
        text = chunk.text if case_sensitive else chunk.text.lower()
        return substring in text
    return _filter


def combine_filters(*filters: Callable[[Chunk], bool], operator: str = "and") -> Callable[[Chunk], bool]:
    """
    Combine multiple filter functions.

    Args:
        *filters: Filter functions to combine
        operator: "and" or "or"

    Returns:
        Combined filter function
    """
    if operator == "and":
        def _filter(chunk: Chunk) -> bool:
            return all(f(chunk) for f in filters)
    elif operator == "or":
        def _filter(chunk: Chunk) -> bool:
            return any(f(chunk) for f in filters)
    else:
        raise ValueError(f"Unknown operator: {operator}. Use 'and' or 'or'")

    return _filter


class FilteredSearchMixin:
    """
    Mixin to add filtered search capability to vector indices.

    For HNSW: uses larger ef_search and applies filter during traversal.
    For BruteForce: applies filter after scoring (already efficient).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._filter_multiplier = 3.0  # Default multiplier for ef_search

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_fn: Optional[Callable[[Chunk], bool]] = None,
        ef_search: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Search with optional metadata filtering.

        For filtered searches, uses larger ef to find enough valid candidates.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter_fn: Optional filter function
            ef_search: Override ef_search (for filtered searches)

        Returns:
            Filtered search results
        """
        if filter_fn is None:
            # No filter - use standard search
            return self._search_unfiltered(query_embedding, k, ef_search)

        # Filtered search - use larger ef to compensate for filtered-out candidates
        effective_ef = ef_search or int(getattr(self, 'ef_search', 50) * self._filter_multiplier)

        # Search with larger ef
        candidates = self._search_unfiltered(query_embedding, effective_ef * 2, effective_ef)

        # Apply filter
        return apply_filter(candidates, filter_fn)[:k]

    def _search_unfiltered(
        self,
        query_embedding: np.ndarray,
        k: int,
        ef_search: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Unfiltered search (to be implemented by subclass).

        This should be the original search method without filtering.
        """
        raise NotImplementedError("Subclass must implement _search_unfiltered")

    def set_filter_multiplier(self, multiplier: float) -> None:
        """Set the ef_search multiplier for filtered searches."""
        self._filter_multiplier = multiplier