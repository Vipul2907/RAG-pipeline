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
    embedding_provider: str = "dummy"  # "dummy" or "openai"

    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Index settings
    index_type: str = "brute_force"  # "brute_force" or "hnsw"
    index_path: Path = Path("./data/index")

    # Search settings
    default_top_k: int = 10

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls(
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "dummy"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            index_type=os.getenv("INDEX_TYPE", "brute_force"),
            index_path=Path(os.getenv("INDEX_PATH", "./data/index")),
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "10")),
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