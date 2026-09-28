"""Query expansion techniques for improved retrieval.

Includes HyDE (Hypothetical Document Embeddings) and Multi-Query expansion.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ..config import config, logger
from ..llm import LLMProvider, get_llm_provider
from ..embeddings import EmbeddingProvider, get_embedding_provider


class QueryExpander(ABC):
    """Abstract base class for query expanders."""

    @abstractmethod
    def expand(self, query: str) -> List[str]:
        """
        Expand a query into multiple query variations.

        Args:
            query: Original query text

        Returns:
            List of expanded queries (including original)
        """
        pass


class HyDEExpander(QueryExpander):
    """
    HyDE (Hypothetical Document Embeddings) expander.

    Generates a hypothetical document that would answer the query,
    then uses that document's embedding for retrieval.

    Reference: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    (Gao et al., 2022)
    """

    HYDE_PROMPT = """Write a detailed, informative passage that directly answers the following question.
The passage should be written as if it were a document from a knowledge base.
Include specific facts, details, and explanations.

Question: {query}

Passage:"""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_provider_name: str = None,
    ):
        """
        Initialize HyDE expander.

        Args:
            llm_provider: LLM provider for generating hypothetical documents
            embedding_provider: Embedding provider (for interface compatibility)
            llm_provider_name: Name of LLM provider to use (from config if None)
        """
        self.llm_provider = llm_provider or get_llm_provider(llm_provider_name or config.hyde_llm_provider)
        self.embedding_provider = embedding_provider
        logger.info(f"Initialized HyDEExpander: llm={type(self.llm_provider).__name__}")

    def expand(self, query: str) -> List[str]:
        """
        Generate hypothetical document for the query.

        Returns:
            List containing the hypothetical document as a single expanded query
        """
        try:
            prompt = self.HYDE_PROMPT.format(query=query)
            response = self.llm_provider.generate(
                prompt=prompt,
                system_prompt="You are a helpful assistant that writes informative passages.",
                max_tokens=512,
                temperature=0.3,
            )
            hypothetical_doc = response.text.strip()

            if not hypothetical_doc:
                logger.warning("HyDE generated empty document, falling back to original query")
                return [query]

            logger.debug(f"HyDE generated document for '{query[:50]}...': {len(hypothetical_doc)} chars")
            return [hypothetical_doc]

        except Exception as e:
            logger.error(f"HyDE expansion failed: {e}")
            return [query]  # Fallback to original query


class MultiQueryExpander(QueryExpander):
    """
    Multi-Query expansion using LLM.

    Generates multiple alternative phrasings of the same question
    to improve recall by capturing different vocabulary.
    """

    MULTI_QUERY_PROMPT = """Generate {n} different ways to ask the same question.
Each variation should use different vocabulary and phrasing but seek the same information.
Return ONLY the questions, one per line, no numbering or extra text.

Original question: {query}

Variations:"""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        n_variations: int = None,
        llm_provider_name: str = None,
    ):
        """
        Initialize MultiQueryExpander.

        Args:
            llm_provider: LLM provider for generating variations
            n_variations: Number of variations to generate (default from config)
            llm_provider_name: Name of LLM provider to use
        """
        self.llm_provider = llm_provider or get_llm_provider(llm_provider_name or config.llm_provider)
        self.n_variations = n_variations or config.multi_query_variations
        logger.info(f"Initialized MultiQueryExpander: n={self.n_variations}, llm={type(self.llm_provider).__name__}")

    def expand(self, query: str) -> List[str]:
        """
        Generate multiple query variations.

        Returns:
            List containing original query + generated variations
        """
        try:
            prompt = self.MULTI_QUERY_PROMPT.format(n=self.n_variations, query=query)
            response = self.llm_provider.generate(
                prompt=prompt,
                system_prompt="You are a helpful assistant that generates query variations.",
                max_tokens=256,
                temperature=0.5,
            )

            # Parse variations (one per line)
            variations = [line.strip() for line in response.text.strip().split("\n") if line.strip()]

            # Ensure we have the right number
            variations = variations[:self.n_variations]

            # Combine with original query
            all_queries = [query] + variations

            logger.debug(f"Multi-query generated {len(variations)} variations for '{query[:50]}...'")
            return all_queries

        except Exception as e:
            logger.error(f"Multi-query expansion failed: {e}")
            return [query]


class CombinedExpander(QueryExpander):
    """
    Combined HyDE + Multi-Query expander.

    Applies both expansions and combines the results.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        n_variations: int = None,
    ):
        self.hyde = HyDEExpander(llm_provider, embedding_provider)
        self.multi = MultiQueryExpander(llm_provider, n_variations)
        logger.info("Initialized CombinedExpander (HyDE + Multi-Query)")

    def expand(self, query: str) -> List[str]:
        """Apply both HyDE and Multi-Query expansion."""
        hyde_queries = self.hyde.expand(query)
        multi_queries = self.multi.expand(query)

        # Combine and deduplicate
        all_queries = list(dict.fromkeys(hyde_queries + multi_queries))
        return all_queries


class NoExpander(QueryExpander):
    """Identity expander - returns query unchanged."""

    def expand(self, query: str) -> List[str]:
        return [query]


def get_query_expander(
    expander_type: str = None,
    llm_provider: Optional[LLMProvider] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
    **kwargs,
) -> QueryExpander:
    """
    Factory function to get a query expander.

    Args:
        expander_type: Type of expander ("hyde", "multi_query", "combined", "none", or from config)
        llm_provider: LLM provider instance
        embedding_provider: Embedding provider instance
        **kwargs: Additional arguments

    Returns:
        QueryExpander instance
    """
    # Determine expander type from config if not provided
    if expander_type is None:
        if config.use_hyde and config.use_multi_query:
            expander_type = "combined"
        elif config.use_hyde:
            expander_type = "hyde"
        elif config.use_multi_query:
            expander_type = "multi_query"
        else:
            expander_type = "none"

    name = expander_type.lower()

    if name in ("hyde", "hypothetical"):
        return HyDEExpander(llm_provider=llm_provider, embedding_provider=embedding_provider, **kwargs)
    elif name in ("multi_query", "multi-query", "multiquery"):
        return MultiQueryExpander(llm_provider=llm_provider, **kwargs)
    elif name in ("combined", "hyde+multi", "hyde_multi"):
        return CombinedExpander(llm_provider=llm_provider, embedding_provider=embedding_provider, **kwargs)
    elif name in ("none", "null", "identity", ""):
        return NoExpander()
    else:
        logger.warning(f"Unknown expander type: {name}, using NoExpander")
        return NoExpander()