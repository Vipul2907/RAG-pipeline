"""
File loading utilities for the ingestion pipeline.
"""

from pathlib import Path
from typing import Optional

from ..config import logger


def load_text_file(file_path: Path) -> str:
    """
    Load a plain text file.

    Args:
        file_path: Path to the .txt file.

    Returns:
        File contents as string.

    Raises:
        FileNotFoundError: If file doesn't exist.
        UnicodeDecodeError: If file isn't valid UTF-8.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    logger.debug(f"Loading text file: {file_path}")
    return file_path.read_text(encoding="utf-8")


def load_markdown_file(file_path: Path) -> str:
    """
    Load a markdown file.

    Currently identical to load_text_file but provides semantic clarity
    and a hook for future markdown-specific processing (frontmatter, etc.).

    Args:
        file_path: Path to the .md file.

    Returns:
        File contents as string.
    """
    return load_text_file(file_path)


def load_file(file_path: Path) -> str:
    """
    Load a file based on its extension.

    Args:
        file_path: Path to the file.

    Returns:
        File contents as string.

    Raises:
        ValueError: If file extension is not supported.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix in [".txt", ".text"]:
        return load_text_file(file_path)
    elif suffix in [".md", ".markdown", ".mkd"]:
        return load_markdown_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .txt, .md")


def discover_files(
    dir_path: Path,
    extensions: Optional[list[str]] = None,
    recursive: bool = True,
) -> list[Path]:
    """
    Discover files matching extensions in a directory.

    Args:
        dir_path: Directory to search.
        extensions: List of file extensions (e.g., [".txt", ".md"]).
        recursive: Whether to search recursively.

    Returns:
        List of matching file paths.
    """
    if extensions is None:
        extensions = [".txt", ".md"]

    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    pattern = dir_path.rglob if recursive else dir_path.glob
    files = []

    for ext in extensions:
        files.extend(pattern(f"*{ext}"))

    # Filter to only files (not directories)
    files = [f for f in files if f.is_file()]

    logger.debug(f"Discovered {len(files)} files in {dir_path}")
    return sorted(files)