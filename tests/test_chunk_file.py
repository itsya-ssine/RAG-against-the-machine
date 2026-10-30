"""Unit tests for the chunk_file dispatcher."""

import logging
from pathlib import Path

import pytest

from src.chunking import chunk_file
from src.io_utils import iter_corpus_files, to_corpus_path


def _write(path: Path, content: str) -> Path:
    """Create a file (and its parents) and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")
    return path


def test_python_file_uses_python_chunker(tmp_path: Path) -> None:
    """A .py file is cut along its definitions with Class context."""
    body = "\n".join(f"        value_{i} = {i}" for i in range(60))
    path = _write(tmp_path / "mod.py",
                  f"class A:\n    def run(self):\n{body}\n")
    chunks = chunk_file(path, 150)
    assert any(c.context == "A.run" for c in chunks)
    assert all(c.file_path == to_corpus_path(path) for c in chunks)


def test_markdown_file_uses_markdown_chunker(tmp_path: Path) -> None:
    """A .md file is cut at headings with a heading-path context."""
    path = _write(tmp_path / "doc.md", "# Title\n\n## Part\n\nSome text.\n")
    chunks = chunk_file(path)
    assert [c.context for c in chunks] == ["Title > Part"]


def test_extension_is_case_insensitive(tmp_path: Path) -> None:
    """An upper-case .PY extension is still treated as Python."""
    path = _write(tmp_path / "MOD.PY", "def f():\n    return 1\n")
    assert [c.text for c in chunk_file(path)] == ["def f():\n    return 1"]


def test_offsets_match_file_content(tmp_path: Path) -> None:
    """Chunk offsets index into the exact file text (CRLF kept)."""
    text = "# T\r\n\r\nline é🚀\r\n\r\n" + "word " * 600 + "\r\n"
    path = _write(tmp_path / "doc.md", text)
    chunks = chunk_file(path, 500)
    assert len(chunks) > 1
    for chunk in chunks:
        first, last = chunk.first_character_index, chunk.last_character_index
        assert text[first:last] == chunk.text


def test_size_is_clamped_to_the_allowed_maximum(tmp_path: Path) -> None:
    """A limit above 2000 is lowered to 2000."""
    path = _write(tmp_path / "doc.md", "word " * 3000)
    chunks = chunk_file(path, 5000)
    assert len(chunks) > 1
    assert all(len(c.text) <= 2000 for c in chunks)


def test_missing_file_is_skipped_with_warning(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A missing file gives no chunks and a warning, not an exception."""
    with caplog.at_level(logging.WARNING):
        assert chunk_file(tmp_path / "nope.py") == []
    assert "Skipping unreadable file" in caplog.text


def test_non_utf8_file_is_skipped(tmp_path: Path) -> None:
    """A binary file gives no chunks."""
    path = tmp_path / "bad.md"
    path.write_bytes(b"\xff\xfe\x00binary")
    assert chunk_file(path) == []


def test_empty_file_gives_no_chunks(tmp_path: Path) -> None:
    """An empty file (e.g. an empty __init__.py) gives no chunks."""
    assert chunk_file(_write(tmp_path / "__init__.py", "")) == []


def test_invalid_max_chunk_size(tmp_path: Path) -> None:
    """A non-positive limit raises ValueError."""
    path = _write(tmp_path / "a.py", "x = 1\n")
    with pytest.raises(ValueError):
        chunk_file(path, 0)


def test_whole_corpus_walk(tmp_path: Path) -> None:
    """Walking a small corpus and chunking every file works end to end."""
    _write(tmp_path / "docs" / "a.md", "# A\n\nText.\n")
    _write(tmp_path / "pkg" / "b.py", "def g():\n    return 2\n")
    _write(tmp_path / "pkg" / "__init__.py", "")
    _write(tmp_path / "pkg" / "data.json", "{}")
    chunks = [c for f in iter_corpus_files(tmp_path) for c in chunk_file(f)]
    assert len(chunks) == 2
    assert {Path(c.file_path).name for c in chunks} == {"a.md", "b.py"}
