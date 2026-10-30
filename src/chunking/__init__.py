"""Chunking package: split corpus files into searchable chunks."""

from src.chunking.common import Chunk, split_by_size
from src.chunking.dispatcher import chunk_file
from src.chunking.markdown_chunker import chunk_markdown
from src.chunking.python_chunker import chunk_python

__all__ = [
    "Chunk",
    "split_by_size",
    "chunk_file",
    "chunk_markdown",
    "chunk_python",
]
