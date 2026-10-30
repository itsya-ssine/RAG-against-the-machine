"""Python chunking strategy for the vLLM source code.

The module is cut along its top-level statements (imports, functions,
classes), using ``ast``. Comments and decorators stay with the
definition that follows them. Consecutive small units are packed into
chunks of at most ``max_chunk_size`` characters. A class that is too
big is cut by method, and a function that is too big is cut by size.
"""

import ast
import re
from typing import List, Optional, Sequence, Tuple

from src.chunking.common import Chunk, Piece, build_chunks, split_by_size

# Same line endings as the Python tokenizer, so ast line numbers match.
_NEWLINE = re.compile(r"\r\n|\r|\n")

# (start_index, end_index, statement)
Unit = Tuple[int, int, Optional[ast.stmt]]


def _join(context: str, name: str) -> str:
    """Append a name to a dotted context such as ``Class.method``."""
    return f"{context}.{name}" if context else name


class _Splitter:
    """Cut a parsed module into pieces along its definitions."""

    def __init__(self, text: str, max_size: int) -> None:
        """Store the text and precompute the offset of each line."""
        self.text = text
        self.max_size = max_size
        self.line_starts = [0] + [m.end() for m in _NEWLINE.finditer(text)]

    def _end_offset(self, node: ast.stmt) -> int:
        """Return the offset just after the last line of a statement."""
        line = node.end_lineno or node.lineno

        if line < len(self.line_starts):
            return self.line_starts[line]

        return len(self.text)

    def _units(
        self,
        body: Sequence[ast.stmt],
        start: int,
        end: int,
    ) -> List[Unit]:
        """Cut ``[start, end)`` into one unit per statement.

        A unit runs from the end of the previous statement to the end of
        its own, so the comments and decorators above a definition belong
        to it. The last unit also takes the trailing text up to ``end``.
        """
        units: List[Unit] = []

        for node in body:
            stop = min(self._end_offset(node), end)

            if stop > start:
                units.append((start, stop, node))
                start = stop

        if start < end:
            if units:
                units[-1] = (units[-1][0], end, units[-1][2])
            else:
                units.append((start, end, None))

        return units

    def split(
        self,
        body: Sequence[ast.stmt],
        start: int,
        end: int,
        context: str = "",
    ) -> List[Piece]:
        """Pack the units of ``[start, end)`` into pieces.

        Args:
            body: Statements covering the range.
            start: Offset where the first unit begins.
            end: Offset where the range ends.
            context: Dotted name of the enclosing class, if any.

        Returns:
            Contiguous ``(start, end, context)`` pieces of at most
            ``max_size`` characters.
        """
        pieces: List[Piece] = []
        pending: Optional[Tuple[int, int]] = None

        for s, e, node in self._units(body, start, end):
            if e - s > self.max_size:
                if pending is not None:
                    pieces.append((pending[0], pending[1], context))
                    pending = None

                pieces.extend(self._split_big(s, e, node, context))
            elif pending is None:
                pending = (s, e)
            elif e - pending[0] <= self.max_size:
                pending = (pending[0], e)
            else:
                pieces.append((pending[0], pending[1], context))
                pending = (s, e)

        if pending is not None:
            pieces.append((pending[0], pending[1], context))

        return pieces

    def _split_big(
        self,
        start: int,
        end: int,
        node: Optional[ast.stmt],
        context: str,
    ) -> List[Piece]:
        """Split a unit longer than ``max_size``."""
        if isinstance(node, ast.ClassDef):
            return self.split(node.body, start, end,
                              _join(context, node.name))

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            context = _join(context, node.name)

        spans = split_by_size(self.text[start:end], self.max_size, start)

        return [(s, e, context) for s, e in spans]


def chunk_python(
    file_path: str,
    text: str,
    max_chunk_size: int,
) -> List[Chunk]:
    """Split a Python source file into chunks.

    Args:
        file_path: Corpus path stored in every chunk.
        text: Full content of the file.
        max_chunk_size: Maximum chunk length in characters.

    Returns:
        Chunks in file order, trimmed of surrounding whitespace, with
        ``text[first:last] == chunk.text`` and at most
        ``max_chunk_size`` characters. ``context`` is the dotted name of
        the enclosing class or function when a definition was split
        (e.g. ``LoRAModel.from_pretrained``). A file that cannot be
        parsed is cut by size only.

    Raises:
        ValueError: If ``max_chunk_size`` is lower than 1.
    """
    if max_chunk_size < 1:
        raise ValueError("max_chunk_size must be at least 1")

    try:
        body = ast.parse(text).body
    except (SyntaxError, ValueError, RecursionError):
        spans = split_by_size(text, max_chunk_size)
        pieces = [(s, e, "") for s, e in spans]
    else:
        pieces = _Splitter(text, max_chunk_size).split(body, 0, len(text))

    return build_chunks(file_path, text, pieces, max_chunk_size)
