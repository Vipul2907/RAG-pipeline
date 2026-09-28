"""
Centralized configuration and logging for the RAG pipeline.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment variables."""

    # Embedding settings
    embedding_dimension: int = 384
    embedding_provider: str = "dummy"  # "dummy", "huggingface", "openai"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Index settings
    index_type: str = "brute_force"  # "brute_force" or "hnsw"
    index_path: Path = Path("./data/index")

    # Search settings
    default_top_k: int = 10

    # Hybrid Search (BM25 + Dense)
    use_hybrid_search: bool = False
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    rrf_k: int = 60

    # Reranker (Cross-Encoder)
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 50
    reranker_batch_size: int = 32
    reranker_device: str = None  # None = auto-detect

    # Query Expansion
    use_hyde: bool = False
    hyde_llm_provider: str = "dummy"
    use_multi_query: bool = False
    multi_query_variations: int = 3

    # Filtered Search
    filtered_search_multiplier: float = 3.0

    # LLM Provider
    llm_provider: str = "anthropic"  # "anthropic", "openai", "groq", "dummy"
    anthropic_model: str = "claude-3-5-haiku-20241022"
    openai_model: str = "gpt-4o-mini"
    groq_model: str = "openai/gpt-oss-120b"

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls(
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "dummy"),
            hf_embedding_model=os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            index_type=os.getenv("INDEX_TYPE", "brute_force"),
            index_path=Path(os.getenv("INDEX_PATH", "./data/index")),
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "10")),
            # Hybrid Search
            use_hybrid_search=os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true",
            bm25_weight=float(os.getenv("BM25_WEIGHT", "0.5")),
            dense_weight=float(os.getenv("DENSE_WEIGHT", "0.5")),
            rrf_k=int(os.getenv("RRF_K", "60")),
            # Reranker
            use_reranker=os.getenv("USE_RERANKER", "false").lower() == "true",
            reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            reranker_top_k=int(os.getenv("RERANKER_TOP_K", "50")),
            reranker_batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "32")),
            reranker_device=os.getenv("RERANKER_DEVICE") or None,
            # Query Expansion
            use_hyde=os.getenv("USE_HYDE", "false").lower() == "true",
            hyde_llm_provider=os.getenv("HYDE_LLM_PROVIDER", "dummy"),
            use_multi_query=os.getenv("USE_MULTI_QUERY", "false").lower() == "true",
            multi_query_variations=int(os.getenv("MULTI_QUERY_VARIATIONS", "3")),
            # Filtered Search
            filtered_search_multiplier=float(os.getenv("FILTERED_SEARCH_MULTIPLIER", "3.0")),
            # LLM Provider
            llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        )


def setup_logging(config: Optional[Config] = None) -> logging.Logger:
    """Configure application logging."""
    if config is None:
        config = Config.from_env()

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format=config.log_format,
        force=True,
    )

    logger = logging.getLogger("rag")
    logger.setLevel(getattr(logging, config.log_level.upper()))
    return logger


# Global config instance
config = Config.from_env()
logger = setup_logging(config)