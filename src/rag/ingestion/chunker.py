"""
Document chunking module.

Splits .txt and .md files into paragraphs with rich metadata including
source file path, character offsets, and chunk sequence index.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..config import config, logger


@dataclass(frozen=True)
class Chunk:
    """
    A text chunk with metadata.

    Attributes:
        text: The chunk text content.
        source_file: Path to the source file.
        start_char: Starting character offset in the source file.
        end_char: Ending character offset in the source file.
        chunk_index: Sequential index of this chunk within the source file.
        metadata: Extensible dictionary for additional metadata.
    """
    text: str
    source_file: str
    start_char: int
    end_char: int
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate chunk invariants."""
        if self.start_char < 0:
            raise ValueError("start_char must be non-negative")
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        if not self.text:
            raise ValueError("text cannot be empty")


class DocumentChunker:
    """
    Splits documents into overlapping chunks with metadata preservation.

    Uses paragraph-based splitting (double newline) for .txt and .md files,
    with configurable chunk size and overlap for context preservation.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum characters per chunk (default from config).
            overlap: Character overlap between consecutive chunks (default from config).
        """
        self.chunk_size = chunk_size or config.chunk_size
        self.overlap = overlap or config.chunk_overlap

        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")

        logger.debug(
            f"Initialized DocumentChunker: chunk_size={self.chunk_size}, overlap={self.overlap}"
        )

    def chunk_text(
        self,
        text: str,
        source_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Split text into chunks with metadata.

        Args:
            text: The text to chunk.
            source_path: Path to the source file (for metadata).
            metadata: Additional metadata to include in each chunk.

        Returns:
            List of Chunk objects with metadata.
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for {source_path}")
            return []

        base_metadata = metadata or {}
        base_metadata.update({"source_file": source_path})

        # Split on double newlines (paragraphs) - natural for .txt/.md
        paragraphs = self._split_paragraphs(text)

        chunks = []
        char_offset = 0
        chunk_index = 0

        for para in paragraphs:
            para_start = text.find(para, char_offset)
            if para_start == -1:
                # Fallback if paragraph not found at expected position
                para_start = char_offset
            para_end = para_start + len(para)

            # If paragraph fits in chunk_size, add as single chunk
            if len(para) <= self.chunk_size:
                chunk_meta = base_metadata.copy()
                chunk_meta.update({"paragraph_index": chunk_index})
                chunks.append(
                    Chunk(
                        text=para,
                        source_file=source_path,
                        start_char=para_start,
                        end_char=para_end,
                        chunk_index=chunk_index,
                        metadata=chunk_meta,
                    )
                )
                chunk_index += 1
            else:
                # Split long paragraph into overlapping windows
                sub_chunks = self._split_long_text(para, para_start, base_metadata, chunk_index)
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)

            char_offset = para_end

        # Post-process: merge header chunks with their following content
        chunks = self._merge_header_chunks(chunks)

        logger.info(f"Chunked {source_path}: {len(chunks)} chunks from {len(paragraphs)} paragraphs")
        return chunks

    def _merge_header_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Merge short header-like chunks with the following content chunk.

        Detects chunks that are likely headers (short, start with #) and merges
        them with the immediately following chunk to keep header + content together.

        Args:
            chunks: List of initial chunks.

        Returns:
            List of chunks with headers merged into following content.
        """
        if len(chunks) < 2:
            return chunks

        merged = []
        i = 0

        while i < len(chunks):
            current = chunks[i]

            # Check if current chunk is a header-like chunk:
            # - Short (<= 120 chars)
            # - Starts with markdown header (# ## ### etc.) or "Interview Question" or "Advanced Note"
            # - Not the last chunk
            is_header = (
                len(current.text) <= 120
                and i + 1 < len(chunks)
                and (
                    current.text.lstrip().startswith(("#", "##", "###", "####", "#####", "######"))
                    or current.text.lstrip().lower().startswith("interview question")
                    or current.text.lstrip().lower().startswith("advanced note")
                    or current.text.lstrip().lower().startswith("note:")
                    or current.text.lstrip().lower().startswith("question:")
                )
            )

            if is_header:
                # Merge with next chunk
                next_chunk = chunks[i + 1]
                merged_text = current.text + "\n\n" + next_chunk.text

                # Create merged chunk with combined metadata
                merged_meta = current.metadata.copy()
                merged_meta.update({
                    "merged_from_header": True,
                    "original_header_chunk_index": current.chunk_index,
                    "original_content_chunk_index": next_chunk.chunk_index,
                })

                merged.append(
                    Chunk(
                        text=merged_text,
                        source_file=current.source_file,
                        start_char=current.start_char,
                        end_char=next_chunk.end_char,
                        chunk_index=len(merged),
                        metadata=merged_meta,
                    )
                )
                i += 2  # Skip both chunks
            else:
                # Keep chunk as-is, but update chunk_index
                if current.chunk_index != len(merged):
                    # Need to create new chunk with updated index
                    merged.append(
                        Chunk(
                            text=current.text,
                            source_file=current.source_file,
                            start_char=current.start_char,
                            end_char=current.end_char,
                            chunk_index=len(merged),
                            metadata=current.metadata,
                        )
                    )
                else:
                    merged.append(current)
                i += 1

        return merged

    def _split_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs on double newlines.

        Preserves the paragraph text exactly as it appears.
        """
        # Split on double newlines, keep the delimiter for accurate offsets
        # Using regex to handle various newline combinations
        paragraphs = re.split(r"\n{2,}", text)

        # Filter out empty paragraphs but keep track of them for offsets
        result = []
        for para in paragraphs:
            stripped = para.strip()
            if stripped:
                result.append(stripped)

        return result

    def _split_long_text(
        self,
        text: str,
        base_offset: int,
        base_metadata: Dict[str, Any],
        start_index: int,
    ) -> List[Chunk]:
        """
        Split long text into overlapping windows of chunk_size.

        Args:
            text: Text to split.
            base_offset: Starting character offset in source file.
            base_metadata: Base metadata to include.
            start_index: Starting chunk index.

        Returns:
            List of Chunk objects.
        """
        chunks = []
        step = self.chunk_size - self.overlap

        for i in range(0, len(text), step):
            window = text[i : i + self.chunk_size]
            if not window.strip():
                continue

            chunk_meta = base_metadata.copy()
            chunk_meta.update({"paragraph_index": start_index, "window_index": i // step})

            chunks.append(
                Chunk(
                    text=window,
                    source_file=base_metadata.get("source_file", ""),
                    start_char=base_offset + i,
                    end_char=base_offset + i + len(window),
                    chunk_index=start_index + len(chunks),
                    metadata=chunk_meta,
                )
            )

        return chunks

    def chunk_file(self, file_path: Path) -> List[Chunk]:
        """
        Load and chunk a single file.

        Args:
            file_path: Path to the file to chunk.

        Returns:
            List of Chunk objects.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = file_path.read_text(encoding="utf-8")
        return self.chunk_text(text, str(file_path))

    def chunk_directory(
        self,
        dir_path: Path,
        glob_pattern: str = "**/*.txt",
        recursive: bool = True,
    ) -> List[Chunk]:
        """
        Chunk all matching files in a directory.

        Args:
            dir_path: Directory to search.
            glob_pattern: Glob pattern for file matching.
            recursive: Whether to search recursively.

        Returns:
            List of Chunk objects from all matched files.
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        all_chunks = []
        pattern = dir_path.rglob if recursive else dir_path.glob

        for file_path in pattern(glob_pattern):
            if file_path.is_file():
                try:
                    chunks = self.chunk_file(file_path)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Failed to chunk {file_path}: {e}")

        logger.info(f"Chunked directory {dir_path}: {len(all_chunks)} total chunks")
        return all_chunks


# Convenience function for simple use cases
def chunk_text(text: str, source_path: str, **kwargs) -> List[Chunk]:
    """Convenience function to chunk text with default settings."""
    chunker = DocumentChunker(**kwargs)
    return chunker.chunk_text(text, source_path)


def chunk_file(file_path: Path, **kwargs) -> List[Chunk]:
    """Convenience function to chunk a file with default settings."""
    chunker = DocumentChunker(**kwargs)
    return chunker.chunk_file(file_path)