"""
Tests for the DocumentChunker module.
"""

import tempfile
from pathlib import Path

import pytest

from src.rag.ingestion import DocumentChunker, Chunk, chunk_text, chunk_file


class TestDocumentChunker:
    """Tests for DocumentChunker class."""

    def test_init_defaults(self):
        """Test initialization with default config values."""
        chunker = DocumentChunker()
        assert chunker.chunk_size == 512
        assert chunker.overlap == 50

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        chunker = DocumentChunker(chunk_size=256, overlap=25)
        assert chunker.chunk_size == 256
        assert chunker.overlap == 25

    def test_init_invalid_overlap(self):
        """Test that overlap >= chunk_size raises error."""
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            DocumentChunker(chunk_size=100, overlap=100)

    def test_init_invalid_chunk_size(self):
        """Test that non-positive chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            DocumentChunker(chunk_size=0)

    def test_init_negative_overlap(self):
        """Test that negative overlap raises error."""
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            DocumentChunker(chunk_size=100, overlap=-10)

    def test_chunk_empty_text(self):
        """Test chunking empty text returns empty list."""
        chunker = DocumentChunker()
        chunks = chunker.chunk_text("", "test.txt")
        assert chunks == []

    def test_chunk_whitespace_only(self):
        """Test chunking whitespace-only text returns empty list."""
        chunker = DocumentChunker()
        chunks = chunker.chunk_text("   \n\n  \n  ", "test.txt")
        assert chunks == []

    def test_chunk_single_paragraph(self):
        """Test chunking a single short paragraph."""
        chunker = DocumentChunker(chunk_size=512, overlap=50)
        text = "This is a single paragraph with some text content."
        chunks = chunker.chunk_text(text, "test.txt")

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].source_file == "test.txt"
        assert chunks[0].chunk_index == 0
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == len(text)

    def test_chunk_multiple_paragraphs(self):
        """Test chunking multiple paragraphs."""
        chunker = DocumentChunker(chunk_size=512, overlap=50)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunker.chunk_text(text, "test.txt")

        assert len(chunks) == 3
        assert chunks[0].text == "Paragraph one."
        assert chunks[1].text == "Paragraph two."
        assert chunks[2].text == "Paragraph three."
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1
        assert chunks[2].chunk_index == 2

    def test_chunk_paragraphs_with_extra_newlines(self):
        """Test chunking handles multiple consecutive newlines."""
        chunker = DocumentChunker(chunk_size=512, overlap=50)
        text = "Para one.\n\n\n\nPara two.\n\n\nPara three."
        chunks = chunker.chunk_text(text, "test.txt")

        assert len(chunks) == 3
        assert chunks[0].text == "Para one."
        assert chunks[1].text == "Para two."
        assert chunks[2].text == "Para three."

    def test_chunk_long_paragraph_splits(self):
        """Test that long paragraphs are split into overlapping windows."""
        chunker = DocumentChunker(chunk_size=50, overlap=10)
        text = "A" * 120  # Longer than chunk_size
        chunks = chunker.chunk_text(text, "test.txt")

        # Should split into overlapping windows
        assert len(chunks) > 1
        # Check overlap
        for i in range(len(chunks) - 1):
            # The end of chunk i should overlap with start of chunk i+1
            overlap_text = chunks[i].text[-10:]
            assert chunks[i + 1].text.startswith(overlap_text)

    def test_chunk_metadata_preserved(self):
        """Test that custom metadata is preserved in chunks."""
        chunker = DocumentChunker()
        custom_meta = {"author": "Test Author", "category": "test"}
        chunks = chunker.chunk_text("Test paragraph.", "test.txt", metadata=custom_meta)

        assert len(chunks) == 1
        assert chunks[0].metadata["author"] == "Test Author"
        assert chunks[0].metadata["category"] == "test"
        assert chunks[0].metadata["source_file"] == "test.txt"

    def test_chunk_file(self):
        """Test chunking a file directly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Paragraph one.\n\nParagraph two.\n\nParagraph three.")
            temp_path = Path(f.name)

        try:
            chunker = DocumentChunker()
            chunks = chunker.chunk_file(temp_path)

            assert len(chunks) == 3
            assert all(c.source_file == str(temp_path) for c in chunks)
        finally:
            temp_path.unlink()

    def test_chunk_file_not_found(self):
        """Test chunk_file raises FileNotFoundError for missing file."""
        chunker = DocumentChunker()
        with pytest.raises(FileNotFoundError):
            chunker.chunk_file(Path("/nonexistent/file.txt"))

    def test_chunk_directory(self):
        """Test chunking a directory of files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "file1.txt").write_text("File one content.\n\nMore content.")
            (tmpdir / "file2.txt").write_text("File two content.")
            (tmpdir / "ignore.md").write_text("Markdown file.")  # Not matched by default glob

            chunker = DocumentChunker()
            chunks = chunker.chunk_directory(tmpdir, glob_pattern="**/*.txt")

            assert len(chunks) == 3  # 2 from file1, 1 from file2
            source_files = {c.source_file for c in chunks}
            assert str(tmpdir / "file1.txt") in source_files
            assert str(tmpdir / "file2.txt") in source_files

    def test_chunk_directory_recursive(self):
        """Test recursive directory chunking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "file1.txt").write_text("Root file.")
            subdir = tmpdir / "subdir"
            subdir.mkdir()
            (subdir / "file2.txt").write_text("Subdir file.")

            chunker = DocumentChunker()
            chunks = chunker.chunk_directory(tmpdir, glob_pattern="**/*.txt", recursive=True)

            assert len(chunks) == 2

    def test_chunk_directory_non_recursive(self):
        """Test non-recursive directory chunking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "file1.txt").write_text("Root file.")
            subdir = tmpdir / "subdir"
            subdir.mkdir()
            (subdir / "file2.txt").write_text("Subdir file.")

            chunker = DocumentChunker()
            chunks = chunker.chunk_directory(tmpdir, glob_pattern="**/*.txt", recursive=False)

            assert len(chunks) == 1  # Only root file

    def test_chunk_convenience_functions(self):
        """Test chunk_text and chunk_file convenience functions."""
        chunks = chunk_text("Test paragraph.\n\nAnother paragraph.", "test.txt")
        assert len(chunks) == 2

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("File content.")
            temp_path = Path(f.name)

        try:
            chunks = chunk_file(temp_path)
            assert len(chunks) == 1
        finally:
            temp_path.unlink()


class TestChunkDataclass:
    """Tests for Chunk dataclass validation."""

    def test_valid_chunk(self):
        """Test creating a valid chunk."""
        chunk = Chunk(
            text="Test text",
            source_file="test.txt",
            start_char=0,
            end_char=9,
            chunk_index=0,
        )
        assert chunk.text == "Test text"

    def test_invalid_negative_start_char(self):
        """Test that negative start_char raises error."""
        with pytest.raises(ValueError, match="start_char must be non-negative"):
            Chunk(text="Test", source_file="test.txt", start_char=-1, end_char=4, chunk_index=0)

    def test_invalid_end_char(self):
        """Test that end_char <= start_char raises error."""
        with pytest.raises(ValueError, match="end_char must be greater than start_char"):
            Chunk(text="Test", source_file="test.txt", start_char=5, end_char=3, chunk_index=0)

    def test_invalid_chunk_index(self):
        """Test that negative chunk_index raises error."""
        with pytest.raises(ValueError, match="chunk_index must be non-negative"):
            Chunk(text="Test", source_file="test.txt", start_char=0, end_char=4, chunk_index=-1)

    def test_empty_text(self):
        """Test that empty text raises error."""
        with pytest.raises(ValueError, match="text cannot be empty"):
            Chunk(text="", source_file="test.txt", start_char=0, end_char=0, chunk_index=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])