"""Unit tests for the Python chunker."""

import inspect
import random
from typing import List

import pytest

from src.chunking import Chunk, chunk_python

PATH = "data/raw/vllm-0.10.1/vllm/sample.py"

CODE = '''# SPDX-License-Identifier: Apache-2.0
"""Module docstring."""
import os
from typing import List

LIMIT = 10


def small() -> int:
    """A small function."""
    return 1


# Compute something heavy.
@decorator
def heavy(x: int) -> int:
    total = 0
    for i in range(x):
        total += i
    return total


class Engine:
    """An engine."""

    def start(self) -> None:
        print("start")

    def stop(self) -> None:
        print("stop")

    def restart(self) -> None:
        self.stop()
        self.start()
'''


def _check(text: str, chunks: List[Chunk], size: int) -> None:
    """Assert offsets, size limit, trimming and ordering."""
    previous_end = 0
    for chunk in chunks:
        first, last = chunk.first_character_index, chunk.last_character_index
        assert text[first:last] == chunk.text
        assert 0 < len(chunk.text) <= size
        assert chunk.text == chunk.text.strip()
        assert first >= previous_end
        previous_end = last


@pytest.mark.parametrize("size", [1, 20, 60, 120, 300, 2000])
def test_invariants_for_many_sizes(size: int) -> None:
    """Offsets, size and order hold for any max_chunk_size."""
    _check(CODE, chunk_python(PATH, CODE, size), size)


def test_small_units_are_packed_together() -> None:
    """Everything fits in one chunk when the limit is large."""
    chunks = chunk_python(PATH, CODE, 2000)
    assert [c.text for c in chunks] == [CODE.strip()]


def test_comment_and_decorator_stay_with_their_function() -> None:
    """The comment and decorator above a def are in the same chunk."""
    for size in (80, 120, 200):
        chunks = chunk_python(PATH, CODE, size)
        holder = [c for c in chunks if "def heavy" in c.text]
        assert len(holder) == 1
        assert "# Compute something heavy." in holder[0].text
        assert "@decorator" in holder[0].text


def test_big_class_is_cut_by_method_with_class_context() -> None:
    """A class over the limit is split at method boundaries."""
    chunks = chunk_python(PATH, CODE, 90)
    methods = [c for c in chunks if "def stop" in c.text]
    assert len(methods) == 1
    assert methods[0].context == "Engine"
    assert methods[0].text.count("def ") <= 2


def test_big_function_is_split_by_size_with_its_name() -> None:
    """An oversized function is cut by size and keeps its name."""
    body = "\n".join(f"    value_{i} = {i}" for i in range(60))
    text = f"def big():\n{body}\n"
    chunks = chunk_python(PATH, text, 150)
    _check(text, chunks, 150)
    assert len(chunks) > 3
    assert all(c.context == "big" for c in chunks)


def test_method_context_is_dotted() -> None:
    """A huge method inside a class gets Class.method as context."""
    body = "\n".join(f"        value_{i} = {i}" for i in range(60))
    text = f"class A:\n    def run(self):\n{body}\n"
    chunks = chunk_python(PATH, text, 150)
    _check(text, chunks, 150)
    assert any(c.context == "A.run" for c in chunks)


def test_syntax_error_falls_back_to_size_split() -> None:
    """Unparseable code is still chunked, within the size limit."""
    text = "def broken(:\n" + "    x = 1\n" * 100
    chunks = chunk_python(PATH, text, 120)
    _check(text, chunks, 120)
    assert len(chunks) > 5


def test_blank_comment_only_and_no_final_newline() -> None:
    """Edge-case files do not crash."""
    assert chunk_python(PATH, "", 100) == []
    assert chunk_python(PATH, "\n\n  \n", 100) == []
    assert [c.text for c in chunk_python(PATH, "# just a comment\n", 100)] \
        == ["# just a comment"]
    assert [c.text for c in chunk_python(PATH, "x = 1", 100)] == ["x = 1"]


def test_crlf_unicode_and_semicolons() -> None:
    """CRLF endings, non-BMP characters and one-line statements."""
    text = ('name = "é🚀"\r\n\r\ndef f():\r\n    return "à"\r\n\r\n'
            'a = 1; b = 2\r\n')
    for size in (10, 25, 2000):
        _check(text, chunk_python(PATH, text, size), size)


def test_real_module_keeps_invariants() -> None:
    """A real, large stdlib module chunks correctly at several sizes."""
    text = inspect.getsource(inspect)
    for size in (300, 1000, 2000):
        chunks = chunk_python(PATH, text, size)
        _check(text, chunks, size)
        assert len(chunks) > len(text) // size


def test_invalid_max_chunk_size() -> None:
    """A non-positive limit raises ValueError."""
    with pytest.raises(ValueError):
        chunk_python(PATH, CODE, 0)


def test_random_programs_keep_invariants() -> None:
    """Randomly assembled programs never break the invariants."""
    rng = random.Random(3)
    parts = ["import os\n", "X = 1\n", "# comment\n", "\n",
             "def f():\n    return 1\n",
             "@d\ndef g(a):\n    x = a\n    return x\n",
             "class C:\n    def m(self):\n        pass\n    y = 2\n",
             "class D:\n    '''doc'''\n", "if True:\n    z = 'é🚀'\n",
             "a = 1; b = 2\n", "def broken(:\n"]
    for _ in range(300):
        text = "".join(rng.choice(parts) for _ in range(rng.randint(1, 15)))
        if rng.random() < 0.3:
            text = text.replace("\n", "\r\n")
        size = rng.randint(1, 150)
        _check(text, chunk_python(PATH, text, size), size)
