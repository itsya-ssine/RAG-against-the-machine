"""Unit tests for the indexer."""

import json
from pathlib import Path

import pytest

from src.config import CHUNKS_FILENAME, INDEX_FILENAME
from src.indexer import build_index, load_index
from src.retrieval import tokenize


def _write(path: Path, content: str) -> None:
    """Create a file (and its parents)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A tiny corpus: one doc, one module, ignored and empty files."""
    raw = tmp_path / "raw"
    _write(raw / "vllm" / "docs" / "lora.md",
           "# LoRA\n\nUse enable_lora to serve adapters.\n")
    _write(raw / "vllm" / "engine.py",
           "def start_engine():\n    return 'running'\n")
    _write(raw / "vllm" / "config.json", '{"ignored": true}')
    _write(raw / "vllm" / "__init__.py", "")
    return raw


def test_build_index_writes_both_files(corpus: Path, tmp_path: Path) -> None:
    """Two chunks are indexed and both files are written."""
    processed = tmp_path / "processed"
    assert build_index(corpus, processed, 2000) == 2
    assert (processed / INDEX_FILENAME).is_file()
    data = json.loads((processed / CHUNKS_FILENAME).read_text("utf-8"))
    assert data["max_chunk_size"] == 2000
    assert len(data["chunks"]) == 2


def test_load_index_round_trip_and_search(
        corpus: Path, tmp_path: Path) -> None:
    """Loaded documents line up with chunks and are searchable."""
    processed = tmp_path / "processed"
    build_index(corpus, processed)
    index, chunks = load_index(processed)
    assert index.num_docs == len(chunks) == 2

    doc_id, _ = index.search(tokenize("how to enable_lora adapters"), 1)[0]
    assert chunks[doc_id].file_path.endswith("lora.md")
    doc_id, _ = index.search(tokenize("start_engine"), 1)[0]
    assert chunks[doc_id].file_path.endswith("engine.py")


def test_file_path_and_context_are_searchable(
        corpus: Path, tmp_path: Path) -> None:
    """A query naming the file or the heading finds its chunk."""
    processed = tmp_path / "processed"
    build_index(corpus, processed)
    index, chunks = load_index(processed)
    best = index.search(tokenize("engine.py"), 1)[0][0]
    assert chunks[best].file_path.endswith("engine.py")


def test_chunks_point_into_the_files(corpus: Path, tmp_path: Path) -> None:
    """Chunk offsets index the real file content."""
    processed = tmp_path / "processed"
    build_index(corpus, processed)
    _, chunks = load_index(processed)
    for chunk in chunks:
        content = Path(chunk.file_path).read_text("utf-8")
        first, last = chunk.first_character_index, chunk.last_character_index
        assert content[first:last] == chunk.text


def test_smaller_chunk_size_gives_more_chunks(
        corpus: Path, tmp_path: Path) -> None:
    """max_chunk_size is applied while indexing."""
    big = build_index(corpus, tmp_path / "a", 2000)
    small = build_index(corpus, tmp_path / "b", 20)
    assert small > big


def test_empty_corpus_gives_empty_index(tmp_path: Path) -> None:
    """A corpus without indexable files still produces a valid index."""
    raw = tmp_path / "raw"
    raw.mkdir()
    assert build_index(raw, tmp_path / "processed") == 0
    index, chunks = load_index(tmp_path / "processed")
    assert chunks == [] and index.search(["cat"], 5) == []


def test_missing_corpus_directory(tmp_path: Path) -> None:
    """A missing raw folder raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        build_index(tmp_path / "nope", tmp_path / "processed")


def test_invalid_max_chunk_size(corpus: Path, tmp_path: Path) -> None:
    """A non-positive size is rejected."""
    with pytest.raises(ValueError):
        build_index(corpus, tmp_path / "processed", 0)


def test_load_index_errors(corpus: Path, tmp_path: Path) -> None:
    """Missing, corrupt or mismatched index files are reported."""
    processed = tmp_path / "processed"
    with pytest.raises(FileNotFoundError):
        load_index(processed)

    build_index(corpus, processed)
    chunks_file = processed / CHUNKS_FILENAME

    chunks_file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_index(processed)

    chunks_file.write_text('{"chunks": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_index(processed)

    chunks_file.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_index(processed)
