"""
Query engine for the RAG pipeline.

Orchestrates the retrieval process: embedding query, searching index,
LLM generation with grounded answers and citations.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..config import config, logger
from ..embeddings import EmbeddingProvider, get_embedding_provider
from ..index import VectorIndex, SearchResult
from ..ingestion import Chunk
from ..llm import LLMProvider, GenerationResult, get_llm_provider


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
    ) -> QueryResult:
        """
        Search the index for relevant chunks (retrieval only).

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

    def generate(
        self,
        query: str,
        k: int = None,
        filter_fn: Optional[callable] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> RAGResult:
        """
        Full RAG pipeline: retrieve + generate grounded answer.

        Args:
            query: User query text.
            k: Number of chunks to retrieve.
            filter_fn: Optional filter for retrieval.
            max_tokens: Max tokens for LLM generation.
            temperature: LLM sampling temperature.

        Returns:
            RAGResult with grounded answer and citations.
        """
        # Step 1: Retrieve relevant chunks
        retrieval_result = self.search(query, k=k, filter_fn=filter_fn)

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
    ):
        """
        Streaming RAG generation (if provider supports it).
        Currently falls back to non-streaming.
        """
        # For now, just yield the full result
        result = self.generate(query, k, filter_fn, max_tokens, temperature)
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