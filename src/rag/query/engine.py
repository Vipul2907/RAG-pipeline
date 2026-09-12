"""
Query engine for the RAG pipeline.

Orchestrates the retrieval process: embedding query, searching index,
LLM generation with grounded answers and citations.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..config import config, logger
from ..embeddings import EmbeddingProvider, get_embedding_provider
from ..index import VectorIndex, SearchResult
from ..ingestion import Chunk
from ..llm import LLMProvider, GenerationResult, get_llm_provider


def _deduplicate_results(
    results: List[SearchResult],
    similarity_threshold: float = 0.95,
) -> List[SearchResult]:
    """
    Remove near-duplicate chunks from search results.

    Keeps the highest-scoring chunk from each near-duplicate group.
    Two chunks are considered near-duplicates if their text cosine
    similarity (using simple word overlap) exceeds the threshold.

    Args:
        results: List of search results from the index.
        similarity_threshold: Threshold for considering chunks as duplicates (0-1).

    Returns:
        Deduplicated list of results, preserving original score order.
    """
    if not results:
        return results

    # Simple word-based Jaccard similarity for deduplication
    # (avoids needing to re-embed; fast and effective for boilerplate)
    def _word_set(text: str) -> set:
        return set(text.lower().split())

    def _jaccard_similarity(text1: str, text2: str) -> float:
        set1 = _word_set(text1)
        set2 = _word_set(text2)
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    deduplicated = []
    seen_texts = []

    for result in results:
        chunk_text = result.chunk.text
        is_duplicate = False

        for seen_text in seen_texts:
            if _jaccard_similarity(chunk_text, seen_text) >= similarity_threshold:
                is_duplicate = True
                logger.debug(
                    f"Dropping near-duplicate chunk (score={result.score:.4f}) "
                    f"similar to already-selected chunk"
                )
                break

        if not is_duplicate:
            deduplicated.append(result)
            seen_texts.append(chunk_text)

    logger.info(f"Deduplicated {len(results)} -> {len(deduplicated)} results")
    return deduplicated


def _is_compound_query(query: str) -> bool:
    """
    Heuristic detection of compound queries (multiple distinct questions in one).

    Looks for patterns like:
    - Multiple question marks
    - "and what" / "and how" / "and why" patterns
    - Sentences separated by periods that each look like questions
    - "also" / "additionally" between question-like clauses

    Args:
        query: The user query string.

    Returns:
        True if the query appears to contain multiple distinct questions.
    """
    # Count question marks
    question_marks = query.count('?')
    if question_marks >= 2:
        return True

    # Check for "and what" / "and how" / "and why" / "and when" / "and where" / "and who"
    # These often indicate a second question appended to the first
    compound_patterns = [
        r'\band\s+(?:what|how|why|when|where|who)\b',
        r'\b(?:also|additionally|furthermore|moreover)\b.*\?',
        r'\b(?:then|next)\b.*\?',
    ]
    for pattern in compound_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True

    # Check for multiple question-like sentences separated by periods
    # Split on sentence boundaries and check if multiple parts look like questions
    sentences = re.split(r'[.!?]+\s+', query.strip())
    question_like_count = 0
    # Question words + imperative verbs that indicate information-seeking
    question_starters = (
        'what|how|why|when|where|who|which|can|could|would|should|'
        'is|are|do|does|did|write|explain|describe|show|tell|give|list|define'
    )
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Check if sentence looks like a question (starts with question word or ends with ?)
        if (sent.endswith('?') or
            re.match(rf'^\s*({question_starters})\b', sent, re.IGNORECASE)):
            question_like_count += 1

    if question_like_count >= 2:
        return True

    return False


def _split_compound_query(query: str) -> List[str]:
    """
    Split a compound query into sub-questions.

    Uses simple heuristics to split on common patterns:
    - Split on "and what/how/why/when/where/who" patterns
    - Split on multiple question marks
    - Split on sentence boundaries where each part looks like a question

    Args:
        query: The compound query string.

    Returns:
        List of sub-questions. Returns [query] if no split is possible.
    """
    # First try: split on "and what/how/why/when/where/who" pattern
    # This handles "What is X and what is Y?" -> ["What is X?", "What is Y?"]
    split_pattern = r'\s+(?:and|also)\s+(what|how|why|when|where|who)\b'
    match = re.search(split_pattern, query, re.IGNORECASE)
    if match:
        # Split at the match position
        split_pos = match.start()
        part1 = query[:split_pos].strip()
        part2 = query[split_pos:].strip()
        # Ensure both parts end with question marks
        if not part1.endswith('?'):
            part1 += '?'
        if not part2.endswith('?'):
            part2 += '?'
        return [part1, part2]

    # Second try: split on multiple question marks
    # "What is X? What is Y?" -> ["What is X?", "What is Y?"]
    if query.count('?') >= 2:
        parts = [p.strip() + '?' for p in query.split('?') if p.strip()]
        if len(parts) >= 2:
            return parts

    # Third try: split on sentence boundaries where each part looks like a question
    sentences = re.split(r'[.!?]+\s+', query.strip())
    question_parts = []
    question_starters = (
        'what|how|why|when|where|who|which|can|could|would|should|'
        'is|are|do|does|did|write|explain|describe|show|tell|give|list|define'
    )
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if (sent.endswith('?') or
            re.match(rf'^\s*({question_starters})\b', sent, re.IGNORECASE)):
            if not sent.endswith('?'):
                sent += '?'
            question_parts.append(sent)

    if len(question_parts) >= 2:
        return question_parts

    # No split possible
    return [query]


@dataclass(frozen=True)
class QueryResult:
    """
    Result of a RAG query (retrieval only).

    Attributes:
        query: The original query text.
        results: List of search results with chunks and scores.
        query_embedding: The embedded query vector (for debugging).
    """
    query: str
    results: List[SearchResult]
    query_embedding: Optional[np.ndarray] = None


@dataclass(frozen=True)
class Citation:
    """Citation for a grounded answer."""
    source_file: str
    chunk_index: int
    text_snippet: str
    score: float


@dataclass(frozen=True)
class RAGResult:
    """
    Complete RAG result with grounded answer and citations.

    Attributes:
        query: The original query text.
        answer: The LLM-generated answer grounded in retrieved context.
        citations: List of citations supporting the answer.
        retrieval_results: The raw search results used for context.
        generation_result: The LLM generation metadata.
    """
    query: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    retrieval_results: List[SearchResult] = field(default_factory=list)
    generation_result: Optional[GenerationResult] = None


class QueryEngine:
    """
    High-level query engine for RAG retrieval and generation.

    Combines embedding provider, vector index, and LLM provider
    to provide a complete RAG pipeline.
    """

    # System prompt for grounded RAG generation
    RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.
Follow these rules strictly:
1. Answer based ONLY on the provided context documents
2. If the context doesn't contain enough information, say "I don't have enough information to answer this question"
3. Cite your sources using [Source: filename, chunk N] format after each claim
4. Be concise and accurate
5. Do not use external knowledge

Context documents:
{context}"""

    def __init__(
        self,
        index: VectorIndex,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
    ):
        """
        Initialize the query engine.

        Args:
            index: Vector index to search.
            embedding_provider: Embedding provider (default from config).
            llm_provider: LLM provider for generation (default from config).
        """
        self._index = index
        self._embedding_provider = embedding_provider or get_embedding_provider()

        # Verify dimension compatibility
        if self._index.dimension != self._embedding_provider.dimension:
            raise ValueError(
                f"Dimension mismatch: index={self._index.dimension}, "
                f"provider={self._embedding_provider.dimension}"
            )

        # Initialize LLM provider
        self._llm_provider = llm_provider or get_llm_provider()

        logger.info(
            f"Initialized QueryEngine: index={type(self._index).__name__}, "
            f"embedder={type(self._embedding_provider).__name__}, "
            f"llm={type(self._llm_provider).__name__} ({self._llm_provider.model_name})"
        )

    def search(
        self,
        query: str,
        k: int = None,
        filter_fn: Optional[callable] = None,
        deduplicate: bool = True,
        dedup_threshold: float = 0.95,
    ) -> QueryResult:
        """
        Search the index for relevant chunks (retrieval only).

        Args:
            query: Query text string.
            k: Number of results to return after deduplication (default from config).
            filter_fn: Optional filter function for results.
            deduplicate: Whether to remove near-duplicate chunks (default: True).
            dedup_threshold: Jaccard similarity threshold for deduplication (default: 0.95).

        Returns:
            QueryResult with search results.
        """
        k = k or config.default_top_k

        # Embed query
        query_embedding = self._embedding_provider.embed_single(query)

        # Search index with extra results to account for deduplication
        # Fetch 2*k to ensure we have enough after dropping duplicates
        search_k = k * 2 if deduplicate else k
        results = self._index.search(query_embedding, k=search_k, filter_fn=filter_fn)

        # Deduplicate near-identical chunks (e.g., boilerplate "Limitations" sections)
        if deduplicate and results:
            results = _deduplicate_results(results, similarity_threshold=dedup_threshold)
            # Trim to requested k after deduplication
            results = results[:k]

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
        deduplicate: bool = True,
        dedup_threshold: float = 0.95,
    ) -> QueryResult:
        """
        Search using a pre-computed query embedding.

        Args:
            query_embedding: Pre-computed query vector.
            k: Number of results to return after deduplication (default from config).
            filter_fn: Optional filter function for results.
            deduplicate: Whether to remove near-duplicate chunks (default: True).
            dedup_threshold: Jaccard similarity threshold for deduplication (default: 0.95).

        Returns:
            QueryResult with search results.
        """
        k = k or config.default_top_k

        # Search index with extra results to account for deduplication
        search_k = k * 2 if deduplicate else k
        results = self._index.search(query_embedding, k=search_k, filter_fn=filter_fn)

        # Deduplicate near-identical chunks
        if deduplicate and results:
            results = _deduplicate_results(results, similarity_threshold=dedup_threshold)
            results = results[:k]

        return QueryResult(
            query="<pre-computed embedding>",
            results=results,
            query_embedding=query_embedding,
        )

    def search_compound(
        self,
        query: str,
        k: int = None,
        filter_fn: Optional[callable] = None,
        deduplicate: bool = True,
        dedup_threshold: float = 0.95,
        decompose: bool = True,
    ) -> QueryResult:
        """
        Search with automatic compound query decomposition.

        If the query appears to contain multiple distinct questions and decompose=True,
        splits the query into sub-questions, searches each separately, and merges
        results with deduplication.

        Args:
            query: Query text string (may be compound).
            k: Number of results to return after deduplication (default from config).
            filter_fn: Optional filter function for results.
            deduplicate: Whether to remove near-duplicate chunks (default: True).
            dedup_threshold: Jaccard similarity threshold for deduplication (default: 0.95).
            decompose: Whether to attempt compound query decomposition (default: True).

        Returns:
            QueryResult with merged search results from all sub-queries.
        """
        k = k or config.default_top_k

        # Check if we should decompose
        if not decompose or not _is_compound_query(query):
            # Not a compound query or decomposition disabled - use regular search
            return self.search(query, k=k, filter_fn=filter_fn, deduplicate=deduplicate, dedup_threshold=dedup_threshold)

        # Decompose into sub-questions
        sub_queries = _split_compound_query(query)
        logger.info(f"Compound query detected, decomposed into {len(sub_queries)} sub-queries: {sub_queries}")

        if len(sub_queries) <= 1:
            # Decomposition didn't produce multiple parts
            return self.search(query, k=k, filter_fn=filter_fn, deduplicate=deduplicate, dedup_threshold=dedup_threshold)

        # Search each sub-query separately
        all_results = []
        per_subquery_k = max(k, k // len(sub_queries) + 1)  # Get enough from each

        for sub_query in sub_queries:
            sub_result = self.search(
                sub_query,
                k=per_subquery_k,
                filter_fn=filter_fn,
                deduplicate=False,  # Don't dedupe per sub-query, do it globally after merge
            )
            all_results.extend(sub_result.results)
            logger.debug(f"Sub-query '{sub_query[:50]}...': {len(sub_result.results)} results")

        # Deduplicate merged results
        if deduplicate and all_results:
            all_results = _deduplicate_results(all_results, similarity_threshold=dedup_threshold)

        # Trim to k
        all_results = all_results[:k]

        logger.info(f"Compound query '{query[:50]}...': merged {len(all_results)} results from {len(sub_queries)} sub-queries")

        return QueryResult(
            query=query,
            results=all_results,
            query_embedding=None,  # No single embedding for compound query
        )

    def generate(
        self,
        query: str,
        k: int = None,
        filter_fn: Optional[callable] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        decompose_compound_queries: bool = True,
    ) -> RAGResult:
        """
        Full RAG pipeline: retrieve + generate grounded answer.

        Args:
            query: User query text.
            k: Number of chunks to retrieve.
            filter_fn: Optional filter for retrieval.
            max_tokens: Max tokens for LLM generation.
            temperature: LLM sampling temperature.
            decompose_compound_queries: Whether to decompose compound queries into sub-questions
                and search each separately before merging results (default: True).

        Returns:
            RAGResult with grounded answer and citations.
        """
        # Step 1: Retrieve relevant chunks (with compound query decomposition if enabled)
        retrieval_result = self.search_compound(
            query, k=k, filter_fn=filter_fn, decompose=decompose_compound_queries
        )

        if not retrieval_result.results:
            return RAGResult(
                query=query,
                answer="I don't have enough information to answer this question.",
                citations=[],
                retrieval_results=[],
            )

        # Step 2: Build context from retrieved chunks
        context_parts = []
        citations = []

        for result in retrieval_result.results:
            chunk = result.chunk
            # Build context entry with citation marker
            source_info = f"Source: {chunk.source_file}, chunk {chunk.chunk_index}"
            context_parts.append(f"[{source_info}]\n{chunk.text}")

            # Create citation
            citations.append(Citation(
                source_file=chunk.source_file,
                chunk_index=chunk.chunk_index,
                text_snippet=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                score=result.score,
            ))

        context = "\n\n---\n\n".join(context_parts)

        # Step 3: Generate answer with LLM
        system_prompt = self.RAG_SYSTEM_PROMPT.format(context=context)

        gen_result = self._llm_provider.generate(
            prompt=query,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return RAGResult(
            query=query,
            answer=gen_result.text,
            citations=citations,
            retrieval_results=retrieval_result.results,
            generation_result=gen_result,
        )

    def generate_streaming(
        self,
        query: str,
        k: int = None,
        filter_fn: Optional[callable] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        decompose_compound_queries: bool = True,
    ):
        """
        Streaming RAG generation (if provider supports it).
        Currently falls back to non-streaming.
        """
        # For now, just yield the full result
        result = self.generate(query, k, filter_fn, max_tokens, temperature, decompose_compound_queries)
        yield result

    @property
    def index(self) -> VectorIndex:
        """Access the underlying index."""
        return self._index

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Access the embedding provider."""
        return self._embedding_provider

    @property
    def llm_provider(self) -> LLMProvider:
        """Access the LLM provider."""
        return self._llm_provider