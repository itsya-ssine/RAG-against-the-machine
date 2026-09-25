"""Chunking strategies used by the indexer."""

from src.chunking.brace_chunker import (
    chunk_c_file,
    chunk_css_file,
    chunk_js_file,
)
from src.chunking.code_chunker import chunk_python_file
from src.chunking.common import RawChunk
from src.chunking.generic_chunker import chunk_generic_file
from src.chunking.text_chunker import chunk_text_file

__all__ = [
    "RawChunk",
    "chunk_c_file",
    "chunk_css_file",
    "chunk_generic_file",
    "chunk_js_file",
    "chunk_python_file",
    "chunk_text_file",
]
