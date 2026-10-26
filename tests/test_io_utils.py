"""Unit tests for the file-system helpers."""

from pathlib import Path

import pytest

from src.io_utils import iter_corpus_files, read_text, to_corpus_path


def _touch(path: Path, content: str = "x") -> None:
    """Create a file (and its parents) with the given content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_walker_filters_extensions_and_excluded_dirs(tmp_path: Path) -> None:
    """Only .py/.md files outside excluded directories are returned."""
    _touch(tmp_path / "docs" / "a.md")
    _touch(tmp_path / "pkg" / "b.py")
    _touch(tmp_path / "pkg" / "conf.json")
    _touch(tmp_path / "pkg" / "logo.png")
    _touch(tmp_path / "pkg" / "__pycache__" / "b.cpython-312.py")
    _touch(tmp_path / "csrc" / "kernel.py")
    found = [p.relative_to(tmp_path).as_posix()
             for p in iter_corpus_files(tmp_path)]
    assert found == ["docs/a.md", "pkg/b.py"]


def test_walker_is_deterministic(tmp_path: Path) -> None:
    """Two walks return the same sorted list."""
    for name in ("z.py", "a.py", "m.md"):
        _touch(tmp_path / name)
    first = iter_corpus_files(tmp_path)
    assert first == iter_corpus_files(tmp_path)
    assert [p.name for p in first] == ["a.py", "m.md", "z.py"]


def test_walker_missing_root_raises(tmp_path: Path) -> None:
    """A missing corpus directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        iter_corpus_files(tmp_path / "nope")


def test_to_corpus_path_uses_forward_slashes() -> None:
    """Corpus paths are POSIX-style and keep their prefix."""
    path = Path("data/raw/vllm-0.10.1/docs/features/lora.md")
    assert to_corpus_path(path) == (
        "data/raw/vllm-0.10.1/docs/features/lora.md")


def test_read_text_preserves_crlf_and_unicode(tmp_path: Path) -> None:
    """Line endings are untouched so character offsets stay exact."""
    target = tmp_path / "f.md"
    target.write_bytes("é\r\nlà 🚀\r\n".encode("utf-8"))
    assert read_text(target) == "é\r\nlà 🚀\r\n"


def test_read_text_invalid_utf8_raises(tmp_path: Path) -> None:
    """Non UTF-8 content raises UnicodeDecodeError for the caller."""
    target = tmp_path / "bad.py"
    target.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(UnicodeDecodeError):
        read_text(target)
