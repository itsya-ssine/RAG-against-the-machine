"""Markdown chunking strategy for the vLLM docs.

The text is cut into blocks (paragraphs, headings, fenced code blocks),
which are packed into chunks of at most ``max_chunk_size`` characters.
A new chunk always starts at a heading, and each chunk keeps the path of
the headings above it as search-only context.
"""

import re
from typing import List, Optional, Tuple

from src.chunking.common import Chunk, Piece, build_chunks, split_by_size

# '# --8<-- [start:x]' lines of the *.inc.md files are mkdocs markers,
# not headings.
_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(?![ \t]|--8<--)(\S.*?)[ \t]*#*$")
_FENCE = re.compile(r"^\s*(```|~~~)")

# (level, title)
Heading = Tuple[int, str]
# (start_index, end_index, heading)
Block = Tuple[int, int, Optional[Heading]]


def _blocks(text: str) -> List[Block]:
    """Cut a document into blocks.

    A block starts at a heading, or at the first non-blank line after a
    blank line. Blank lines and ``#`` lines inside a fenced code block
    never start a block.

    Args:
        text: Full document text.

    Returns:
        ``(start, end, heading)`` for each block, contiguous and in
        order; ``heading`` is ``(level, title)`` if the block begins
        with a heading.
    """
    blocks: List[Block] = []
    start = 0
    current: Optional[Heading] = None
    in_fence = False
    prev_blank = True
    pos = 0

    for line in text.splitlines(keepends=True):
        fence = _FENCE.match(line) is not None
        if in_fence:
            in_fence = not fence
            prev_blank = False
        else:
            blank = not line.strip()
            match = None if fence else _HEADING.match(line.rstrip("\r\n"))

            if match or (prev_blank and not blank):
                if pos > start:
                    blocks.append((start, pos, current))
                start = pos
                current = (len(match.group(1)), match.group(2)) \
                    if match else None

            in_fence = fence
            prev_blank = blank

        pos += len(line)

    if pos > start:
        blocks.append((start, pos, current))

    return blocks


def _pieces(text: str, max_size: int) -> List[Piece]:
    """Pack blocks into pieces of at most ``max_size`` characters.

    Args:
        text: Full document text.
        max_size: Maximum piece length.

    Returns:
        ``(start, end, context)`` pieces, where context is the heading
        path such as ``"Parent > Child"``.
    """
    pieces: List[Piece] = []
    stack: List[Heading] = []
    current: Optional[List[int]] = None
    context = ""

    def flush() -> None:
        nonlocal current
        if current is not None:
            pieces.append((current[0], current[1], context))
            current = None

    for start, end, heading in _blocks(text):
        if heading is not None:
            flush()

            while stack and stack[-1][0] >= heading[0]:
                stack.pop()

            stack.append(heading)
            context = " > ".join(title for _, title in stack)

        if end - start > max_size:
            # Huge block (table, code): split it together with the
            # pending text so a heading is never left alone.
            first = current[0] if current is not None else start
            current = None
            for s, e in split_by_size(text[first:end], max_size, first):
                pieces.append((s, e, context))
        elif current is None:
            current = [start, end]
        elif end - current[0] > max_size:
            flush()
            current = [start, end]
        else:
            current[1] = end

    flush()

    return pieces


def chunk_markdown(
    file_path: str,
    text: str,
    max_chunk_size: int,
) -> List[Chunk]:
    """Split a Markdown document into chunks.

    Args:
        file_path: Corpus path stored in every chunk.
        text: Full content of the file.
        max_chunk_size: Maximum chunk length in characters.

    Returns:
        Chunks in document order, trimmed of surrounding whitespace,
        with ``text[first:last] == chunk.text`` and at most
        ``max_chunk_size`` characters. ``context`` holds the heading
        path. Empty for a blank document.

    Raises:
        ValueError: If ``max_chunk_size`` is lower than 1.
    """
    if max_chunk_size < 1:
        raise ValueError("max_chunk_size must be at least 1")

    pieces = _pieces(text, max_chunk_size)

    return build_chunks(file_path, text, pieces, max_chunk_size)
