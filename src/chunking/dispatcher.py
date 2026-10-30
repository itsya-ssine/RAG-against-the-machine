"""Chunk a corpus file with the strategy that fits its type."""

import logging
from pathlib import Path
from typing import List

from src.chunking.common import Chunk
from src.chunking.markdown_chunker import chunk_markdown
from src.chunking.python_chunker import chunk_python
from src.config import (
    DEFAULT_MAX_CHUNK_SIZE,
    MAX_ALLOWED_CHUNK_SIZE,
    PYTHON_EXTENSIONS,
)
from src.io_utils import read_text, to_corpus_path

logger = logging.getLogger(__name__)


def chunk_file(
    path: Path,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> List[Chunk]:
    """Read a corpus file and split it into chunks.

    Python files (``.py``) use the Python chunker, every other file the
    Markdown chunker. A file that cannot be read is skipped with a
    warning, so one bad file never stops the indexing.

    Args:
        path: File to chunk, relative to the project root so that the
            stored ``file_path`` matches the corpus
            (e.g. ``data/raw/vllm-0.10.1/docs/features/lora.md``).
        max_chunk_size: Maximum chunk length in characters. Values above
            ``MAX_ALLOWED_CHUNK_SIZE`` are lowered to that limit.

    Returns:
        The chunks of the file in order, or an empty list if the file
        is empty or unreadable.

    Raises:
        ValueError: If ``max_chunk_size`` is lower than 1.
    """
    if max_chunk_size < 1:
        raise ValueError("max_chunk_size must be at least 1")

    size = min(max_chunk_size, MAX_ALLOWED_CHUNK_SIZE)

    try:
        text = read_text(path)
    except (OSError, UnicodeDecodeError) as error:
        logger.warning("Skipping unreadable file %s: %s", path, error)
        return []

    file_path = to_corpus_path(path)

    if path.suffix.lower() in PYTHON_EXTENSIONS:
        return chunk_python(file_path, text, size)

    return chunk_markdown(file_path, text, size)
