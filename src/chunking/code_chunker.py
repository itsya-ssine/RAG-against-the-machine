"""Python source code chunker.

Splits a Python file into chunks aligned on top-level statement
boundaries (imports, functions, classes, ...) rather than arbitrary
character windows, so a chunk never cuts a function in half unless that
single function alone is already wider than max_chunk_size.
"""

import ast

from src.chunking.common import RawChunk, chunk_by_boundaries, split_fixed_size


def _line_start_offsets(source: str) -> "list[int]":
    """Compute the character offset at which each line starts.

    Args:
        source: The full file content.

    Returns:
        A list where index i holds the character offset of line i + 1
        (AST line numbers are 1-indexed), with a final entry equal to
        len(source).
    """
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _node_end_offset(
    node: ast.AST,
    line_offsets: "list[int]",
    source_len: int
) -> int:
    """Return the character offset just past a top-level node's last line.

    Using only the *end* of each node (rather than its start) means
    everything between two statements - decorators, comments, blank
    lines - naturally stays attached to the chunk that follows it,
    without needing special-case handling.

    Args:
        node: A top-level AST node.
        line_offsets: Precomputed line-start offsets for the source.
        source_len: Total length of the source, used as a safe bound.

    Returns:
        The offset immediately after the node's last character.
    """
    end_line: "int | None" = getattr(node, "end_lineno", None)
    if end_line is None or end_line >= len(line_offsets):
        return source_len
    return min(line_offsets[end_line], source_len)


def chunk_python_file(
    file_path: str,
    source: str,
    max_chunk_size: int = 2000,
) -> "list[RawChunk]":
    """Chunk a Python source file at top-level statement boundaries.

    Consecutive top-level statements are grouped together while the
    running chunk stays under max_chunk_size. A single statement wider
    than max_chunk_size on its own (a very long function, for instance)
    is split further into fixed-size windows.

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
        tree = ast.parse(source)
    except SyntaxError:
        # Not parseable as Python (templated file, stray snippet, ...):
        # fall back to plain fixed-size windows over the raw text.
        return split_fixed_size(
            file_path,
            source,
            0,
            source_len,
            max_chunk_size
        )

    top_level_nodes = list(ast.iter_child_nodes(tree))
    if not top_level_nodes:
        return split_fixed_size(
            file_path,
            source,
            0,
            source_len,
            max_chunk_size
        )

    line_offsets = _line_start_offsets(source)
    boundaries = [
        _node_end_offset(node, line_offsets, source_len)
        for node in top_level_nodes
    ]
    # Extend the last boundary to the true end of file so trailing
    # comments/blank lines after the final statement are not dropped.
    boundaries[-1] = source_len

    return chunk_by_boundaries(
        file_path,
        source,
        0,
        boundaries,
        max_chunk_size
    )
