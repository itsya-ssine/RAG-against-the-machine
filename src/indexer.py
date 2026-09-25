"""Indexing stage: chunk the raw corpus and persist a searchable index.

Walks ``data/raw/``, splits every text file into chunks using the
chunking strategy that matches its type (Python code, C/C++/CUDA/JS/CSS,
Markdown, or a generic paragraph/line splitter for everything else:
YAML, JSON, shell, Dockerfiles, CMake, templates, extension-less
files...), tokenizes each chunk for lexical (BM25) retrieval, and
persists the result under ``data/processed/``. Only binary files are
left out. Persisting both the chunk metadata and the pre-computed tokens
means the retrieval stage only has to unpickle a file and build a BM25
index in memory, rather than re-walking and re-chunking the whole corpus
on every query.

Files are decoded from their raw bytes (no newline translation), so the
character offsets stored in the index refer to the file exactly as it
is on disk.
"""

import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from src.chunking import (
    RawChunk,
    chunk_c_file,
    chunk_css_file,
    chunk_generic_file,
    chunk_js_file,
    chunk_python_file,
    chunk_text_file,
)
from src.tokenizer import tokenize

DEFAULT_MAX_CHUNK_SIZE = 2000
INDEX_FILENAME = "index.pkl"

_Chunker = Callable[[str, str, int], "list[RawChunk]"]

# Extension -> dedicated chunker. Anything not listed here goes through
# the generic paragraph/line chunker.
_CHUNKERS: "dict[str, _Chunker]" = {
    ".py": chunk_python_file,
    ".md": chunk_text_file,
    ".rst": chunk_text_file,
    ".c": chunk_c_file,
    ".cc": chunk_c_file,
    ".cpp": chunk_c_file,
    ".cxx": chunk_c_file,
    ".cu": chunk_c_file,
    ".cuh": chunk_c_file,
    ".h": chunk_c_file,
    ".hh": chunk_c_file,
    ".hpp": chunk_c_file,
    ".hxx": chunk_c_file,
    ".inl": chunk_c_file,
    ".js": chunk_js_file,
    ".mjs": chunk_js_file,
    ".css": chunk_css_file,
}

# Files that are never meaningful as text. Anything else is sniffed for
# NUL bytes (see _read_source). SVG is text but is path-data noise.
_BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
    ".pdf", ".zip", ".gz", ".tar", ".tgz", ".bz2", ".xz", ".whl",
    ".so", ".a", ".o", ".dll", ".dylib", ".exe", ".pyc", ".bin",
    ".pt", ".pth", ".safetensors", ".npy", ".npz", ".pkl",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
}
_BINARY_SNIFF_BYTES = 8192

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
    """Recursively list every file under raw_dir.

    Skipped directories are pruned during the walk, and only directory
    names *below* raw_dir are considered, so the location of the corpus
    itself never causes files to be dropped.

    Args:
        raw_dir: Root directory of the ingested corpus (e.g.
            ``data/raw``).

    Returns:
        Sorted list of file paths, excluding VCS/cache/virtualenv
        directories.
    """
    files: "list[Path]" = []
    for root, dirnames, filenames in os.walk(raw_dir):
        dirnames[:] = [
            name for name in dirnames if name not in _SKIPPED_DIR_NAMES
        ]
        files.extend(Path(root) / name for name in filenames)
    return sorted(files)


def _chunker_for(path: Path) -> _Chunker:
    """Pick the chunking strategy matching a file's extension.

    Args:
        path: Path of the candidate source file.

    Returns:
        The dedicated chunker for the extension (Python, C-family,
        JavaScript, CSS, Markdown), or the generic paragraph/line
        chunker for every other file.
    """
    return _CHUNKERS.get(path.suffix.lower(), chunk_generic_file)


def _read_source(path: Path) -> "str | None":
    """Read a file as text, tolerating encoding issues.

    The bytes are decoded directly (not through text mode) so that
    ``\\r\\n`` line endings are preserved and character offsets match
    the file on disk.

    Args:
        path: File to read.

    Returns:
        The file's text, decoded as UTF-8 with invalid bytes replaced,
        or ``None`` if the file is binary or could not be read.
    """
    if path.suffix.lower() in _BINARY_EXTENSIONS:
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:_BINARY_SNIFF_BYTES]:
        return None
    return data.decode("utf-8", errors="replace")


def build_index(
    raw_dir: "Path | str" = Path("data/raw"),
    processed_dir: "Path | str" = Path("data/processed"),
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> IndexStats:
    """Chunk every text file under raw_dir and persist the index.

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
        Summary statistics for the run. ``files_skipped`` counts
        binary, unreadable, empty and unchunkable files.

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
        source = _read_source(path)
        if source is None:
            files_skipped += 1
            continue

        file_path = path.as_posix()
        try:
            file_chunks = _chunker_for(path)(
                file_path, source, max_chunk_size
            )
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
