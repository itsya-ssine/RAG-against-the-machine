"""Generic plain-text chunker.

Fallback for every file type without a dedicated chunker: YAML, JSON,
shell scripts, Dockerfiles, CMake, Jinja templates, requirements files,
extension-less files (LICENSE, CODEOWNERS...), and Python files that
``ast`` cannot parse.

Chunks are aligned on blank-line paragraph breaks first; a paragraph
wider than max_chunk_size is split on line boundaries, and a single line
wider still (minified JSON, for instance) falls back to fixed-size
windows.
"""

from src.chunking.common import (
    RawChunk,
    chunk_by_boundaries,
    paragraph_ends,
    split_by_lines,
)


def chunk_generic_file(
    file_path: str,
    source: str,
    max_chunk_size: int = 2000,
) -> "list[RawChunk]":
    """Chunk an arbitrary text file on paragraph, then line boundaries.

    Args:
        file_path: Path of the source file, as stored in the corpus.
        source: Full content of the file.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        A list of RawChunk covering the whole file in order, with no
        chunk longer than max_chunk_size characters. Returns an empty
        list for an empty or whitespace-only file.
    """
    if not source.strip():
        return []

    return chunk_by_boundaries(
        file_path,
        source,
        0,
        paragraph_ends(source, 0, len(source)),
        max_chunk_size,
        oversized_handler=split_by_lines,
    )
