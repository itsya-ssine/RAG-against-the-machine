"""Markdown chunker.

Splits a Markdown (or reStructuredText) file along its header structure
so each chunk stays a coherent section, then further splits any section
wider than max_chunk_size along paragraph (blank-line) breaks, then
along lines (so table rows stay whole), and only as a last resort into
fixed-size windows.

Lines starting with ``#`` inside fenced code blocks (shell comments,
Python comments...) are not headers and never start a section.
Files with no headers at all are treated as a single section.
"""

import re

from src.chunking.common import (
    RawChunk,
    chunk_by_boundaries,
    line_start_offsets,
    paragraph_ends,
    split_by_lines,
)

_HEADER_RE = re.compile(r"#{1,6}[ \t]+\S")
_FENCE_RE = re.compile(r"[ \t]{0,3}(`{3,}|~{3,})")


def _section_start_offsets(source: str) -> "list[int]":
    """Return the offsets where each Markdown section begins.

    A section starts at a header line (`#` through `######`) that is
    not inside a fenced code block. The file start is always a section
    start, so a file with no headers is a single section.

    Args:
        source: Full file content.

    Returns:
        Sorted, deduplicated list of section start offsets, always
        including 0.
    """
    offsets = line_start_offsets(source)
    starts = {0}
    fence = ""
    for index in range(len(offsets) - 1):
        line = source[offsets[index]:offsets[index + 1]]
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if not fence:
                fence = marker
            elif (marker[0] == fence[0]
                    and len(marker) >= len(fence)
                    and set(line.strip()) == {marker[0]}):
                fence = ""
        elif not fence and _HEADER_RE.match(line):
            starts.add(offsets[index])
    return sorted(starts)


def _split_section_by_paragraphs(
    file_path: str,
    source: str,
    first: int,
    last: int,
    max_chunk_size: int,
) -> "list[RawChunk]":
    """Split an oversized section along blank-line paragraph breaks.

    Used as the oversized-unit handler when a whole section is wider
    than max_chunk_size: grouping paragraphs preserves more structure
    than an immediate fixed-size cut. Any single paragraph that is
    itself still too wide (a big table, for example) is split on line
    boundaries.

    Args:
        file_path: Path of the source file, as stored in the corpus.
        source: Full file content.
        first: Start offset of the section.
        last: End offset (exclusive) of the section.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        A list of RawChunk covering [first, last).
    """
    return chunk_by_boundaries(
        file_path,
        source,
        first,
        paragraph_ends(source, first, last),
        max_chunk_size,
        oversized_handler=split_by_lines,
    )


def chunk_text_file(
    file_path: str,
    source: str,
    max_chunk_size: int = 2000,
) -> "list[RawChunk]":
    """Chunk a Markdown file along its section structure.

    Consecutive sections are grouped together while the running chunk
    stays under max_chunk_size. A single section wider than
    max_chunk_size on its own is split further along paragraph breaks,
    then line breaks, then fixed-size windows.

    Args:
        file_path: Path of the source file, as stored in the corpus
            (used verbatim in the resulting chunk offsets).
        source: Full content of the Markdown file.
        max_chunk_size: Maximum number of characters per chunk. Must
            match the value used consistently across indexing and
            retrieval.

    Returns:
        A list of RawChunk covering the whole file in order, with no
        chunk longer than max_chunk_size characters. Returns an empty
        list for an empty or whitespace-only file.
    """
    if not source.strip():
        return []

    source_len = len(source)
    section_starts = _section_start_offsets(source)
    # A section's "boundary" for the shared greedy grouping logic is
    # the start of the next section (or end of file for the last one).
    boundaries = section_starts[1:] + [source_len]

    return chunk_by_boundaries(
        file_path,
        source,
        0,
        boundaries,
        max_chunk_size,
        oversized_handler=_split_section_by_paragraphs,
    )
