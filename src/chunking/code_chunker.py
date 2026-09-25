"""Python source code chunker.

Splits a Python file into chunks aligned on top-level statement
boundaries (imports, functions, classes, ...) rather than arbitrary
character windows. A top-level statement wider than max_chunk_size (a
big class, typically) is split recursively along its own body, so a
chunk only cuts through a method when that method's individual
statements are still too wide.
"""

import ast
import warnings
from typing import Callable

from src.chunking.common import (
    RawChunk,
    chunk_by_boundaries,
    line_start_offsets,
    split_by_lines,
)
from src.chunking.generic_chunker import chunk_generic_file

_CHILD_FIELDS = ("body", "handlers", "orelse", "finalbody")
_Handler = Callable[[str, str, int, int, int], "list[RawChunk]"]


def _node_end_offset(
    node: ast.AST,
    line_offsets: "list[int]",
    source_len: int
) -> int:
    """Return the character offset just past a node's last line.

    Using only the *end* of each node (rather than its start) means
    everything between two statements - decorators, comments, blank
    lines - naturally stays attached to the chunk that follows it,
    without needing special-case handling.

    Args:
        node: An AST node with line information.
        line_offsets: Precomputed line-start offsets for the source.
        source_len: Total length of the source, used as a safe bound.

    Returns:
        The offset immediately after the node's last character.
    """
    end_line: "int | None" = getattr(node, "end_lineno", None)
    if end_line is None or end_line >= len(line_offsets):
        return source_len
    return min(line_offsets[end_line], source_len)


def _child_nodes(node: ast.AST) -> "list[ast.AST]":
    """Return the statement-level children of a compound node.

    Covers class/function bodies as well as if/try/with/for/while
    blocks (body, except handlers, else and finally branches).

    Args:
        node: Any AST node.

    Returns:
        Child nodes carrying line information, or an empty list if the
        node has no nested statements.
    """
    children: "list[ast.AST]" = []
    for field in _CHILD_FIELDS:
        value = getattr(node, field, None)
        if isinstance(value, list):
            children.extend(
                child
                for child in value
                if isinstance(child, ast.AST)
                and hasattr(child, "end_lineno")
            )
    return children


def _make_descender(
    line_offsets: "list[int]",
    source_len: int,
    node_by_end: "dict[int, ast.AST]",
) -> _Handler:
    """Build an oversized-unit handler that descends into node bodies.

    Args:
        line_offsets: Precomputed line-start offsets for the source.
        source_len: Total length of the source.
        node_by_end: Maps the end offset of each unit being split to
            the AST node that produced it.

    Returns:
        A handler suitable for ``chunk_by_boundaries``. Units whose
        node has no nested statements are split on line boundaries.
    """

    def handler(
        file_path: str,
        source: str,
        first: int,
        last: int,
        max_chunk_size: int,
    ) -> "list[RawChunk]":
        node = node_by_end.get(last)
        children = _child_nodes(node) if node is not None else []
        if not children:
            return split_by_lines(
                file_path, source, first, last, max_chunk_size
            )

        pairs = sorted(
            (
                (_node_end_offset(child, line_offsets, source_len), child)
                for child in children
            ),
            key=lambda pair: pair[0],
        )
        # The last child owns everything up to the end of the unit.
        pairs[-1] = (last, pairs[-1][1])
        child_by_end = dict(pairs)
        ends = sorted(end for end in child_by_end if first < end <= last)
        return chunk_by_boundaries(
            file_path,
            source,
            first,
            ends,
            max_chunk_size,
            oversized_handler=_make_descender(
                line_offsets, source_len, child_by_end
            ),
        )

    return handler


def chunk_python_file(
    file_path: str,
    source: str,
    max_chunk_size: int = 2000,
) -> "list[RawChunk]":
    """Chunk a Python source file at statement boundaries.

    Consecutive top-level statements are grouped together while the
    running chunk stays under max_chunk_size. A single statement wider
    than max_chunk_size on its own is split along its body (methods of
    a class, statements of a function), recursively, and only then on
    line boundaries.

    Args:
        file_path: Path of the source file, as stored in the corpus
            (used verbatim in the resulting chunk offsets).
        source: Full content of the Python file.
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

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # A leading BOM makes ast.parse fail; it does not change
            # any line number, so parsing without it keeps offsets valid.
            tree = ast.parse(
                source[1:] if source.startswith("\ufeff") else source
            )
    except (SyntaxError, ValueError, RecursionError):
        # Not parseable as Python (templated file, stray snippet, ...):
        # fall back to paragraph/line based chunking of the raw text.
        return chunk_generic_file(file_path, source, max_chunk_size)

    top_level_nodes = list(ast.iter_child_nodes(tree))
    if not top_level_nodes:
        return chunk_generic_file(file_path, source, max_chunk_size)

    line_offsets = line_start_offsets(source)
    boundaries = [
        _node_end_offset(node, line_offsets, source_len)
        for node in top_level_nodes
    ]
    # Extend the last boundary to the true end of file so trailing
    # comments/blank lines after the final statement are not dropped.
    boundaries[-1] = source_len

    node_by_end = dict(zip(boundaries, top_level_nodes))
    return chunk_by_boundaries(
        file_path,
        source,
        0,
        boundaries,
        max_chunk_size,
        oversized_handler=_make_descender(
            line_offsets, source_len, node_by_end
        ),
    )
