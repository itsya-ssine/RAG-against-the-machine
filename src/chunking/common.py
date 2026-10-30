"""Shared chunking primitives: the Chunk model and a size-based splitter.

Offset convention used everywhere in the project: a chunk covers the
half-open character range ``[first_character_index,
last_character_index)`` of the file, so that
``file_text[first:last] == chunk.text``.
"""

from typing import List, Tuple

from pydantic import BaseModel, Field, model_validator

# Separators tried in order of preference when a text must be cut.
_SEPARATORS: Tuple[str, ...] = ("\n\n", "\r\n\r\n", "\n", " ")

# A boundary is only accepted if it leaves at least this share of the
# window in the piece, to avoid producing tiny chunks.
_MIN_PIECE_RATIO: float = 0.5

# build_chunks merges a piece smaller than this share of the limit into
# the next one.
_MERGE_RATIO: float = 0.15

# (start_index, end_index, context_string)
Piece = Tuple[int, int, str]


class Chunk(BaseModel):
    """A contiguous piece of a corpus file.

    Attributes:
        file_path: Path of the file, written exactly as in the corpus.
        first_character_index: Offset of the first character (inclusive).
        last_character_index: Offset just after the last character
            (exclusive).
        text: The exact file content of the range.
        context: Extra searchable text (e.g. a class name or heading
            path). It is never part of the offsets nor of ``text``.
    """

    file_path: str
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)
    text: str
    context: str = ""

    @model_validator(mode="after")
    def _check_span(self) -> "Chunk":
        """Ensure the offsets are consistent with the chunk text.

        Returns:
            The validated chunk.

        Raises:
            ValueError: If the range is empty/inverted or its length
                differs from the length of ``text``.
        """
        span = self.last_character_index - self.first_character_index
        if span <= 0:
            raise ValueError("last_character_index must be greater than "
                             "first_character_index")
        if span != len(self.text):
            raise ValueError("text length does not match the offsets")

        return self


def split_by_size(
    text: str,
    max_size: int,
    offset: int = 0,
) -> List[Tuple[int, int]]:
    """Cut a text into contiguous spans of at most ``max_size`` characters.

    Each cut is made at the best boundary found in the last part of the
    window: a blank line first, then a line break, then a space. If none
    is found, the text is cut hard at ``max_size``. The spans cover the
    whole text without gaps or overlaps.

    Args:
        text: Text to split.
        max_size: Maximum span length in characters (at least 1).
        offset: Position of ``text`` inside its file, added to every
            returned index.

    Returns:
        List of ``(start, end)`` spans (end exclusive), in file
        coordinates. Empty if ``text`` is empty.

    Raises:
        ValueError: If ``max_size`` is lower than 1 or ``offset`` is
            negative.
    """
    if max_size < 1:
        raise ValueError("max_size must be at least 1")
    if offset < 0:
        raise ValueError("offset must not be negative")

    spans: List[Tuple[int, int]] = []
    length = len(text)
    pos = 0
    while pos < length:
        if length - pos <= max_size:
            spans.append((pos + offset, length + offset))
            break

        window_end = pos + max_size
        min_cut = pos + int(max_size * _MIN_PIECE_RATIO)
        cut = window_end

        for sep in _SEPARATORS:
            idx = text.rfind(sep, pos, window_end)
            if idx != -1 and idx + len(sep) > min_cut:
                cut = idx + len(sep)
                break

        spans.append((pos + offset, cut + offset))
        pos = cut

    return spans


def build_chunks(
    file_path: str,
    text: str,
    pieces: List[Piece],
    max_size: int,
) -> List[Chunk]:
    """Turn contiguous pieces of a file into validated chunks.

    A piece smaller than 15% of ``max_size`` is first merged with the
    piece that follows (when the result fits), because a tiny chunk such
    as a lone heading or a few imports is scored too high by BM25. The
    merged piece keeps the context of the later piece. Each piece is then
    trimmed of surrounding whitespace, and blank pieces are dropped.

    Args:
        file_path: Corpus path stored in every chunk.
        text: Full content of the file.
        pieces: ``(start, end, context)`` spans in file order, each at
            most ``max_size`` long.
        max_size: Maximum chunk length in characters.

    Returns:
        The chunks, in file order.
    """
    minimum = max(1, int(max_size * _MERGE_RATIO))
    merged: List[Piece] = []

    for start, end, context in pieces:
        if (merged and merged[-1][1] - merged[-1][0] < minimum
                and end - merged[-1][0] <= max_size):
            start = merged.pop()[0]

        merged.append((start, end, context))

    chunks: List[Chunk] = []
    for start, end, context in merged:
        raw = text[start:end]
        body = raw.strip()

        if not body:
            continue

        first = start + len(raw) - len(raw.lstrip())

        chunks.append(
            Chunk(
                file_path=file_path,
                first_character_index=first,
                last_character_index=first + len(body),
                text=body,
                context=context,
            )
        )

    return chunks
