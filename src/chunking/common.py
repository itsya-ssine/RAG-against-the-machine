"""Shared building blocks used by every chunking strategy.

All chunkers (Python AST, Markdown/text, C-family/JS/CSS, generic text)
face the same underlying problem: given a source file and a list of
offsets marking the end of each semantic unit (a top-level statement, a
paragraph, a section, a function...), group consecutive units into
chunks that stay under `max_chunk_size`, and fall back to a finer split
for any single unit that alone is already too wide. This module
implements that once so each chunker only needs to compute its own
notion of "unit".

The fallback ladder used across the package is, from coarse to fine:
semantic units -> paragraphs -> lines -> fixed-size windows.
"""

import re
from dataclasses import dataclass
from typing import Callable

_LINE_END_RE = re.compile(r"\r\n|\r|\n")
_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t\r]*\n")


@dataclass
class RawChunk:
    """A chunk of raw text extracted from a source file.

    This is an internal chunker output, not one of the pydantic models
    from the subject: it never crosses a CLI/JSON boundary, it is only
    consumed by the indexer. A lightweight dataclass avoids pydantic
    validation overhead over the tens of thousands of chunks a full
    corpus produces.

    Attributes:
        file_path: Path of the source file, exactly as stored in the
            corpus (e.g. "data/raw/vllm-0.10.1/docs/features/lora.md").
        content: The chunk's text, i.e. source[first:last].
        first_character_index: Offset of the chunk's first character.
        last_character_index: Offset just past the chunk's last
            character (so that content == source[first:last]).
    """

    file_path: str
    content: str
    first_character_index: int
    last_character_index: int


OversizedHandler = Callable[[str, str, int, int, int], "list[RawChunk]"]


def line_start_offsets(source: str) -> "list[int]":
    """Compute the character offset at which each line starts.

    Only ``\\n``, ``\\r\\n`` and ``\\r`` are line terminators. Unlike
    ``str.splitlines`` this ignores form feeds, ``\\x85``, ``\\u2028``
    and friends, so the numbering agrees with what ``ast`` reports.

    Args:
        source: The full file content.

    Returns:
        A list where index i holds the offset of the start of line
        i + 1, with a final entry equal to len(source). An empty
        source yields ``[0]``.
    """
    offsets = [0]
    for match in _LINE_END_RE.finditer(source):
        offsets.append(match.end())
    if offsets[-1] != len(source):
        offsets.append(len(source))
    return offsets


def paragraph_ends(source: str, first: int, last: int) -> "list[int]":
    """Return the end offsets of blank-line separated paragraphs.

    A paragraph includes the blank line(s) that follow it, so that the
    separator stays attached to the paragraph it terminates.

    Args:
        source: Full file content.
        first: Start offset of the region.
        last: End offset (exclusive) of the region.

    Returns:
        Strictly increasing offsets in (first, last], always ending
        with ``last``. Empty if the region is empty.
    """
    if first >= last:
        return []
    ends = [
        match.end()
        for match in _PARAGRAPH_BREAK_RE.finditer(source, first, last)
        if match.end() < last
    ]
    ends.append(last)
    return ends


def split_fixed_size(
    file_path: str,
    source: str,
    first: int,
    last: int,
    max_chunk_size: int,
) -> "list[RawChunk]":
    """Split a span into fixed-size windows of at most max_chunk_size.

    Last-resort fallback used when nothing finer can be found (for
    instance a single line wider than max_chunk_size).

    Args:
        file_path: Path of the source file, as stored in the corpus.
        source: Full content of the file.
        first: Start offset of the span to split.
        last: End offset (exclusive) of the span to split.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        A list of RawChunk covering exactly [first, last), each at most
        max_chunk_size characters wide.
    """
    chunks: "list[RawChunk]" = []
    cursor = first
    while cursor < last:
        end = min(cursor + max_chunk_size, last)
        chunks.append(
            RawChunk(
                file_path=file_path,
                content=source[cursor:end],
                first_character_index=cursor,
                last_character_index=end,
            )
        )
        cursor = end
    return chunks


def chunk_by_boundaries(
    file_path: str,
    source: str,
    start: int,
    boundaries: "list[int]",
    max_chunk_size: int,
    oversized_handler: OversizedHandler = split_fixed_size,
) -> "list[RawChunk]":
    """Greedily group semantic units into chunks under max_chunk_size.

    `boundaries` must be sorted and end with the offset marking the end
    of the region being chunked (typically len(source) or the end of a
    section). Boundaries that do not move past the running cursor
    (duplicates, for instance) are ignored, so no empty chunk is ever
    produced. Starting from `start`, this function extends a running
    group as far as the boundaries allow while staying within
    max_chunk_size, then flushes it as one chunk. If a single unit
    (from the current cursor to the next boundary) already overflows
    max_chunk_size on its own, `oversized_handler` is used to split it
    further instead.

    Args:
        file_path: Path of the source file, as stored in the corpus.
        source: Full content of the file.
        start: Offset at which to start grouping (0 for a whole file,
            or a section's start offset when called recursively).
        boundaries: Sorted list of unit-end offsets, all greater than
            start, with the last entry equal to the region's end.
        max_chunk_size: Maximum number of characters per chunk.
        oversized_handler: Called as (file_path, source, first, last,
            max_chunk_size) whenever a single unit is too wide to fit
            in one chunk on its own. Defaults to a fixed-size split.

    Returns:
        A list of RawChunk covering [start, boundaries[-1]) in order,
        with no chunk longer than max_chunk_size characters.
    """
    if not boundaries:
        return []

    chunks: "list[RawChunk]" = []
    cursor = start
    index = 0
    total = len(boundaries)

    while index < total:
        reach = index
        while (reach + 1 < total
                and boundaries[reach + 1] - cursor <= max_chunk_size):
            reach += 1
        end = boundaries[reach]

        if end <= cursor:
            # Duplicate / already-consumed boundary: nothing to emit.
            index = reach + 1
            continue

        if end - cursor <= max_chunk_size:
            chunks.append(
                RawChunk(
                    file_path=file_path,
                    content=source[cursor:end],
                    first_character_index=cursor,
                    last_character_index=end,
                )
            )
        else:
            chunks.extend(
                oversized_handler(
                    file_path,
                    source,
                    cursor,
                    end,
                    max_chunk_size
                )
            )

        cursor = end
        index = reach + 1

    return chunks


def split_by_lines(
    file_path: str,
    source: str,
    first: int,
    last: int,
    max_chunk_size: int,
) -> "list[RawChunk]":
    """Split a span on line boundaries, packing lines up to the limit.

    Preferred over `split_fixed_size` as the last resort for text: it
    keeps table rows, statements and config entries whole. Only a
    single line wider than max_chunk_size is cut mid-line.

    Args:
        file_path: Path of the source file, as stored in the corpus.
        source: Full content of the file.
        first: Start offset of the span to split.
        last: End offset (exclusive) of the span to split.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        A list of RawChunk covering exactly [first, last), each at most
        max_chunk_size characters wide.
    """
    if first >= last:
        return []
    ends = [
        first + offset
        for offset in line_start_offsets(source[first:last])[1:]
    ]
    return chunk_by_boundaries(
        file_path,
        source,
        first,
        ends,
        max_chunk_size,
        oversized_handler=split_fixed_size,
    )
