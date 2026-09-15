"""Indexing stage: chunk the raw corpus and persist a searchable index.

Walks ``data/raw/``, splits every supported file into chunks using the
chunking strategy that matches its type (Python code vs.
Markdown/text), tokenizes each chunk for lexical (BM25) retrieval, and
persists the result under ``data/processed/``. Persisting both the
chunk metadata and the pre-computed tokens means the retrieval stage
only has to unpickle a file and build a BM25 index in memory, rather
than re-walking and re-chunking the whole corpus on every query.
"""

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from src.chunking import RawChunk, chunk_python_file, chunk_text_file
from src.tokenizer import tokenize

DEFAULT_MAX_CHUNK_SIZE = 2000
INDEX_FILENAME = "index.pkl"

_CODE_EXTENSIONS = {".py"}
_TEXT_EXTENSIONS = {".md", ".rst", ".txt"}
_SKIPPED_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
}

_Chunker = Callable[[str, str, int], "list[RawChunk]"]


@dataclass
class IndexStats:
    """Summary of a completed indexing run."""

    files_seen: int
    files_indexed: int
    files_skipped: int
    chunks_indexed: int
    max_chunk_size: int
    elapsed_seconds: float


def _iter_source_files(raw_dir: Path) -> "list[Path]":
    """Recursively list the files under raw_dir worth indexing.

    Args:
        raw_dir: Root directory of the ingested corpus (e.g.
            ``data/raw``).

    Returns:
        Sorted list of file paths with a supported extension, skipping
        VCS/cache/virtualenv directories.
    """
    supported = _CODE_EXTENSIONS | _TEXT_EXTENSIONS
    files = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in supported
        and not any(part in _SKIPPED_DIR_NAMES for part in path.parts)
    ]
    return sorted(files)


def _chunker_for(path: Path) -> "_Chunker | None":
    """Pick the chunking strategy matching a file's extension.

    Args:
        path: Path of the candidate source file.

    Returns:
        ``chunk_python_file`` for ``.py`` files, ``chunk_text_file``
        for Markdown/text files, or ``None`` if the extension is not
        supported.
    """
    suffix = path.suffix.lower()
    if suffix in _CODE_EXTENSIONS:
        return chunk_python_file
    if suffix in _TEXT_EXTENSIONS:
        return chunk_text_file
    return None


def _read_source(path: Path) -> "str | None":
    """Read a source file as text, tolerating encoding issues.

    Args:
        path: File to read.

    Returns:
        The file's text content, decoding as UTF-8 with invalid bytes
        replaced, or ``None`` if the file could not be read at all.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def build_index(
    raw_dir: "Path | str" = Path("data/raw"),
    processed_dir: "Path | str" = Path("data/processed"),
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> IndexStats:
    """Chunk every supported file under raw_dir and persist the index.

    This is the sole entry point behind the ``index`` CLI command. It
    is safe to call repeatedly: each run rebuilds the index from
    scratch and overwrites whatever was previously written under
    ``processed_dir``.

    Args:
        raw_dir: Root directory of the ingested corpus.
        processed_dir: Directory the index is written to.
        max_chunk_size: Maximum number of characters per chunk. Must
            not exceed the grader's ``max_context_length`` (2000
            characters); smaller values are fine.

    Returns:
        Summary statistics for the run.

    Raises:
        FileNotFoundError: If ``raw_dir`` does not exist.
        ValueError: If ``max_chunk_size`` is not a positive integer.
    """
    start_time = time.monotonic()
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"raw corpus directory not found: {raw_dir}"
        )
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer")

    files = _iter_source_files(raw_dir)

    chunks: "list[RawChunk]" = []
    files_indexed = 0
    files_skipped = 0

    for path in tqdm(files, desc="Chunking", unit="file"):
        chunker = _chunker_for(path)
        source = _read_source(path) if chunker is not None else None
        if chunker is None or source is None:
            files_skipped += 1
            continue

        file_path = path.as_posix()
        try:
            file_chunks = chunker(file_path, source, max_chunk_size)
        except Exception:
            # A single unparsable/malformed file must never abort the
            # whole indexing run; skip it and keep going.
            files_skipped += 1
            continue

        if file_chunks:
            chunks.extend(file_chunks)
            files_indexed += 1
        else:
            files_skipped += 1

    tokens = [
        tokenize(chunk.content)
        for chunk in tqdm(chunks, desc="Tokenizing", unit="chunk")
    ]

    processed_dir.mkdir(parents=True, exist_ok=True)
    index_path = processed_dir / INDEX_FILENAME
    payload = {
        "max_chunk_size": max_chunk_size,
        "chunks": [
            {
                "file_path": chunk.file_path,
                "content": chunk.content,
                "first_character_index": chunk.first_character_index,
                "last_character_index": chunk.last_character_index,
            }
            for chunk in chunks
        ],
        "tokens": tokens,
    }
    with index_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.monotonic() - start_time
    print(
        f"Ingestion complete! Indexed {len(chunks)} chunks "
        f"under {processed_dir}/"
    )

    return IndexStats(
        files_seen=len(files),
        files_indexed=files_indexed,
        files_skipped=files_skipped,
        chunks_indexed=len(chunks),
        max_chunk_size=max_chunk_size,
        elapsed_seconds=elapsed,
    )
