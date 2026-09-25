"""Chunker for brace-delimited languages (C/C++/CUDA, JavaScript, CSS).

There is no parser for these languages here; a light lexical scan is
enough to find where top-level definitions end. Each line is annotated
with the brace depth after it, ignoring braces inside comments, string
and character literals, and (for C-family code) preprocessor lines. A
line ends a unit at depth ``d`` when the depth after it is ``d`` and it
is blank, closes a block (``}``), ends a statement (``;``) or is a
preprocessor directive.

Chunking starts at depth 0 (functions, classes, namespaces' siblings...).
A unit wider than max_chunk_size - a whole ``namespace { ... }`` or a
huge function - is split again at the next depth level that yields more
than one unit, recursively; when no deeper structure exists it falls
back to line boundaries, then fixed-size windows.

The scan is deliberately forgiving: unbalanced braces from ``#if``
branches or regex literals can only make units coarser, never break the
invariants (contiguous chunks, each within max_chunk_size).
"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from src.chunking.common import (
    RawChunk,
    chunk_by_boundaries,
    line_start_offsets,
    split_by_lines,
)

_UNIT_END_CHARS = frozenset("};#")


@dataclass(frozen=True)
class _Syntax:
    """Lexical features that differ between brace languages."""

    preprocessor: bool
    line_comments: bool
    backticks: bool


C_SYNTAX = _Syntax(preprocessor=True, line_comments=True, backticks=False)
JS_SYNTAX = _Syntax(preprocessor=False, line_comments=True, backticks=True)
CSS_SYNTAX = _Syntax(
    preprocessor=False, line_comments=False, backticks=False
)


@dataclass
class _Line:
    """Scan result for one source line.

    Attributes:
        end: Offset just past the line (including its terminator).
        depth: Brace depth after the line.
        blank: True for a whitespace-only line outside any comment.
        last: Last significant character of the line ("" for a
            comment-only line, "#" for a preprocessor line).
    """

    end: int
    depth: int
    blank: bool
    last: str


def _is_digit_separator(text: str, pos: int) -> bool:
    """Tell a C++14 digit separator (1'000) or prefix from a quote.

    Args:
        text: The current line.
        pos: Index of an apostrophe in ``text``.

    Returns:
        True if the apostrophe follows an identifier/number character
        and therefore does not open a character literal.
    """
    return pos > 0 and (text[pos - 1].isalnum() or text[pos - 1] == "_")


def _scan_lines(source: str, syntax: _Syntax) -> "list[_Line]":
    """Annotate every line with its brace depth and last character.

    Args:
        source: Full file content.
        syntax: Lexical features of the language being scanned.

    Returns:
        One ``_Line`` per source line, in order.
    """
    offsets = line_start_offsets(source)
    lines: "list[_Line]" = []
    depth = 0
    in_block = False
    in_template = False
    continued = False

    for index in range(len(offsets) - 1):
        text = source[offsets[index]:offsets[index + 1]]
        is_directive = continued or (
            syntax.preprocessor
            and not in_block
            and not in_template
            and text.lstrip().startswith("#")
        )
        quote = ""
        last = ""
        pos = 0
        size = len(text)

        while pos < size:
            char = text[pos]
            if in_block:
                close = text.find("*/", pos)
                if close < 0:
                    break
                in_block = False
                pos = close + 2
            elif quote or in_template:
                if char == "\\":
                    pos += 2
                    continue
                if char == (quote or "`"):
                    quote = ""
                    in_template = False
                    last = char
                pos += 1
            elif text.startswith("/*", pos):
                in_block = True
                pos += 2
            elif syntax.line_comments and text.startswith("//", pos):
                break
            elif char == '"' or (
                char == "'" and not _is_digit_separator(text, pos)
            ):
                quote = char
                pos += 1
            elif char == "`" and syntax.backticks:
                in_template = True
                pos += 1
            else:
                if not char.isspace():
                    last = char
                    if not is_directive:
                        if char == "{":
                            depth += 1
                        elif char == "}" and depth > 0:
                            depth -= 1
                pos += 1

        continued = is_directive and text.rstrip().endswith("\\")
        if is_directive:
            last = "#"
        lines.append(
            _Line(
                end=offsets[index + 1],
                depth=depth,
                blank=not text.strip() and not in_block and not in_template,
                last=last,
            )
        )
    return lines


def _unit_ends(
    lines: "list[_Line]",
    line_ends: "list[int]",
    first: int,
    last: int,
    level: int,
) -> "list[int]":
    """List the unit-end offsets at one depth level inside a span.

    Args:
        lines: Scan result for the whole file.
        line_ends: ``line.end`` for every line (for bisecting).
        first: Start offset of the span.
        last: End offset (exclusive) of the span.
        level: Brace depth at which units end.

    Returns:
        Strictly increasing offsets in (first, last], ending with last.
    """
    low = bisect_right(line_ends, first)
    high = bisect_left(line_ends, last)
    ends = [
        line.end
        for line in lines[low:high + 1]
        if line.end <= last
        and line.depth == level
        and (line.blank or line.last in _UNIT_END_CHARS)
    ]
    if not ends or ends[-1] != last:
        ends.append(last)
    return ends


def _chunk_brace_file(
    file_path: str,
    source: str,
    max_chunk_size: int,
    syntax: _Syntax,
) -> "list[RawChunk]":
    """Chunk a brace-language file (see module docstring).

    Args:
        file_path: Path of the source file, as stored in the corpus.
        source: Full file content.
        max_chunk_size: Maximum number of characters per chunk.
        syntax: Lexical features of the language.

    Returns:
        Chunks covering the whole file, or [] for a blank file.
    """
    if not source.strip():
        return []

    lines = _scan_lines(source, syntax)
    line_ends = [line.end for line in lines]

    def descend(
        path: str,
        text: str,
        first: int,
        last: int,
        limit: int,
    ) -> "list[RawChunk]":
        low = bisect_right(line_ends, first)
        high = bisect_left(line_ends, last)
        base = lines[low - 1].depth if low > 0 else 0
        peak = max(
            (line.depth for line in lines[low:high + 1]), default=base
        )
        for level in range(base + 1, peak + 1):
            ends = _unit_ends(lines, line_ends, first, last, level)
            if len(ends) > 1:
                return chunk_by_boundaries(
                    path, text, first, ends, limit,
                    oversized_handler=descend,
                )
        return split_by_lines(path, text, first, last, limit)

    return chunk_by_boundaries(
        file_path,
        source,
        0,
        _unit_ends(lines, line_ends, 0, len(source), 0),
        max_chunk_size,
        oversized_handler=descend,
    )


def chunk_c_file(
    file_path: str,
    source: str,
    max_chunk_size: int = 2000,
) -> "list[RawChunk]":
    """Chunk a C/C++/CUDA source or header file.

    Args:
        file_path: Path of the source file, as stored in the corpus.
        source: Full file content.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        Chunks covering the whole file in order, or [] if it is blank.
    """
    return _chunk_brace_file(file_path, source, max_chunk_size, C_SYNTAX)


def chunk_js_file(
    file_path: str,
    source: str,
    max_chunk_size: int = 2000,
) -> "list[RawChunk]":
    """Chunk a JavaScript file.

    Args:
        file_path: Path of the source file, as stored in the corpus.
        source: Full file content.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        Chunks covering the whole file in order, or [] if it is blank.
    """
    return _chunk_brace_file(file_path, source, max_chunk_size, JS_SYNTAX)


def chunk_css_file(
    file_path: str,
    source: str,
    max_chunk_size: int = 2000,
) -> "list[RawChunk]":
    """Chunk a CSS file.

    Args:
        file_path: Path of the source file, as stored in the corpus.
        source: Full file content.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        Chunks covering the whole file in order, or [] if it is blank.
    """
    return _chunk_brace_file(file_path, source, max_chunk_size, CSS_SYNTAX)
