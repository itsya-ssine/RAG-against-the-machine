"""Markdown / plain-text chunker.

Splits a Markdown file along its header structure so each chunk stays a
coherent section, then further splits any section wider than
max_chunk_size along paragraph (blank-line) breaks. Files with no
headers at all (plain .txt) are treated as a single section, so the
same paragraph-splitting fallback applies to them too.
"""

import re

from src.chunking.common import RawChunk, chunk_by_boundaries, split_fixed_size

_HEADER_RE = re.compile(r"^#{1,6}[ \t]+\S", re.MULTILINE)
_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n")


def _section_start_offsets(source: str) -> "list[int]":
    """Return the offsets where each Markdown section begins.

    A section starts at a header line (`#` through `######`). If the
    file has no headers, the whole file is treated as one section
    starting at offset 0.

    Args:
        source: Full file content.

    Returns:
        Sorted, deduplicated list of section start offsets, always
        including 0.
    """
    starts = {match.start() for match in _HEADER_RE.finditer(source)}
    starts.add(0)
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
    itself still too wide falls through to a fixed-size split.

    Args:
        file_path: Path of the source file, as stored in the corpus.
        source: Full file content.
        first: Start offset of the section.
        last: End offset (exclusive) of the section.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        A list of RawChunk covering [first, last).
    """
    section = source[first:last]
    paragraph_ends = [
        first + match.end()
        for match in _PARAGRAPH_BREAK_RE.finditer(section)
    ]
    paragraph_ends.append(last)

    return chunk_by_boundaries(
        file_path,
        source,
        first,
        paragraph_ends,
        max_chunk_size,
        oversized_handler=split_fixed_size,
    )


def chunk_text_file(
    file_path: str,
    source: str,
    max_chunk_size: int = 2000,
) -> "list[RawChunk]":
    """Chunk a Markdown or plain-text file along its section structure.

    Consecutive sections are grouped together while the running chunk
    stays under max_chunk_size. A single section wider than
    max_chunk_size on its own is split further along paragraph breaks,
    and a single paragraph wider still falls back to a fixed-size
    split.

    Args:
        file_path: Path of the source file, as stored in the corpus
            (used verbatim in the resulting chunk offsets).
        source: Full content of the Markdown/text file.
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
