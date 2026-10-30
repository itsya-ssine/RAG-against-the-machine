"""Build and load the search index (BM25 index plus the chunk list)."""

import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from src.chunking import Chunk, chunk_file
from src.config import (
    CHUNKS_FILENAME,
    DEFAULT_MAX_CHUNK_SIZE,
    INDEX_FILENAME,
    PROCESSED_DIR,
    RAW_DATA_DIR,
)
from src.io_utils import iter_corpus_files
from src.retrieval import BM25Index, tokenize

logger = logging.getLogger(__name__)


def _searchable_text(chunk: Chunk) -> str:
    """Return the text indexed for a chunk.

    The file path and the context (heading path or class name) are added
    to the chunk text, so a question that names a document or a class
    can match it.
    """
    return f"{chunk.file_path} {chunk.context} {chunk.text}"


def build_index(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DIR,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> int:
    """Chunk the corpus, build the BM25 index and save both.

    Two files are written in ``processed_dir``: the BM25 index and the
    list of chunks (the retriever maps a document id back to a chunk).

    Args:
        raw_dir: Folder containing the corpus.
        processed_dir: Folder where the index files are written.
        max_chunk_size: Maximum chunk length in characters.

    Returns:
        The number of chunks indexed.

    Raises:
        FileNotFoundError: If ``raw_dir`` does not exist.
        ValueError: If ``max_chunk_size`` is lower than 1.
        OSError: If the index files cannot be written.
    """
    started = time.monotonic()
    files = iter_corpus_files(raw_dir)

    chunks: List[Chunk] = []
    for path in tqdm(files, desc="Chunking", unit="file"):
        chunks.extend(chunk_file(path, max_chunk_size))

    index = BM25Index()
    index.build(tokenize(_searchable_text(chunk)) for chunk in chunks)

    processed_dir.mkdir(parents=True, exist_ok=True)
    index.save(processed_dir / INDEX_FILENAME)
    data = {
        "max_chunk_size": max_chunk_size,
        "chunks": [chunk.model_dump() for chunk in chunks],
    }
    with open(processed_dir / CHUNKS_FILENAME, "w", encoding="utf-8") as f:
        json.dump(data, f)

    logger.info("Indexed %d chunks from %d files in %.1f s",
                len(chunks), len(files), time.monotonic() - started)

    return len(chunks)


def load_index(
    processed_dir: Path = PROCESSED_DIR,
) -> Tuple[BM25Index, List[Chunk]]:
    """Load the index written by ``build_index``.

    Args:
        processed_dir: Folder containing the index files.

    Returns:
        The BM25 index and the chunks; document ``i`` of the index is
        ``chunks[i]``.

    Raises:
        FileNotFoundError: If an index file is missing (the corpus has
            not been indexed yet).
        ValueError: If a file is corrupt or the two files do not match.
    """
    index = BM25Index.load(processed_dir / INDEX_FILENAME)

    with open(processed_dir / CHUNKS_FILENAME, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        chunks = [Chunk(**item) for item in data["chunks"]]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid chunks file in {processed_dir}") from error

    if len(chunks) != index.num_docs:
        raise ValueError("The index and the chunk list do not match; "
                         "run the index command again")

    return index, chunks
