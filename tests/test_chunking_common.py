"""Unit tests for the Chunk model and split_by_size."""

from typing import List, Tuple

import pytest
from pydantic import ValidationError

from src.chunking import Chunk, split_by_size


def _check_cover(text: str, spans: List[Tuple[int, int]],
                 max_size: int, offset: int = 0) -> None:
    """Assert spans are contiguous, non-empty and within max_size."""
    assert spans[0][0] == offset
    assert spans[-1][1] == len(text) + offset
    for (start, end), nxt in zip(spans, spans[1:] + [(0, 0)]):
        assert 0 < end - start <= max_size
        if nxt != (0, 0):
            assert end == nxt[0]


def test_chunk_valid() -> None:
    """A consistent chunk is accepted and context defaults to empty."""
    chunk = Chunk(file_path="data/raw/a.py", first_character_index=3,
                  last_character_index=8, text="hello")
    assert chunk.context == ""


def test_chunk_length_mismatch_rejected() -> None:
    """Offsets must match the length of the text."""
    with pytest.raises(ValidationError):
        Chunk(file_path="a", first_character_index=0,
              last_character_index=10, text="hello")


def test_chunk_empty_or_negative_range_rejected() -> None:
    """Empty, inverted or negative ranges are rejected."""
    with pytest.raises(ValidationError):
        Chunk(file_path="a", first_character_index=5,
              last_character_index=5, text="")
    with pytest.raises(ValidationError):
        Chunk(file_path="a", first_character_index=-1,
              last_character_index=4, text="abcde")


def test_split_empty_and_short_text() -> None:
    """Empty text gives no span; short text gives a single span."""
    assert split_by_size("", 100) == []
    assert split_by_size("abc", 100) == [(0, 3)]
    assert split_by_size("abc", 3) == [(0, 3)]


def test_split_prefers_blank_line() -> None:
    """The cut happens right after the last blank line of the window."""
    text = "aaaa aaaa\n\nbbbb bbbb bbbb"
    spans = split_by_size(text, 16)
    assert spans[0] == (0, 11)
    _check_cover(text, spans, 16)


def test_split_falls_back_to_line_then_space_then_hard_cut() -> None:
    """Without a blank line, cut at a newline, a space or hard."""
    by_line = "aaaaaaaaa\nbbbbbbbbbbbb"
    assert split_by_size(by_line, 12)[0] == (0, 10)
    by_space = "aaaaaaaaa bbbbbbbbbbbb"
    assert split_by_size(by_space, 12)[0] == (0, 10)
    hard = "a" * 25
    spans = split_by_size(hard, 10)
    assert spans == [(0, 10), (10, 20), (20, 25)]


def test_split_ignores_boundary_too_early() -> None:
    """A boundary in the first half of the window is not used."""
    text = "a\n" + "b" * 30
    spans = split_by_size(text, 10)
    assert spans[0] == (0, 10)


def test_split_applies_offset_and_covers_text() -> None:
    """Indices are shifted by the offset; coverage has no gaps."""
    text = ("word " * 200).strip()
    spans = split_by_size(text, 50, offset=1000)
    _check_cover(text, spans, 50, offset=1000)


def test_split_unicode_offsets() -> None:
    """Offsets count characters, not bytes."""
    text = "é🚀" * 30
    spans = split_by_size(text, 7)
    _check_cover(text, spans, 7)
    rebuilt = "".join(text[s:e] for s, e in spans)
    assert rebuilt == text


def test_split_invalid_arguments() -> None:
    """Invalid size or offset raises ValueError."""
    with pytest.raises(ValueError):
        split_by_size("abc", 0)
    with pytest.raises(ValueError):
        split_by_size("abc", 5, offset=-1)
