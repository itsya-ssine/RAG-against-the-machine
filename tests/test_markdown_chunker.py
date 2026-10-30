"""Unit tests for the Markdown chunker."""

import random
from typing import List

import pytest

from src.chunking import Chunk, chunk_markdown

PATH = "data/raw/vllm-0.10.1/docs/sample.md"

DOC = """# OpenAI Server

Intro paragraph about the server.

## Serving

Run the server with:

```bash
# this is a comment, not a heading

vllm serve model
```

## Options

### Sampling

Temperature and top-p.

### Penalties

Frequency penalty.
"""


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


@pytest.mark.parametrize("size", [1, 10, 50, 200, 500, 2000])
def test_invariants_for_many_sizes(size: int) -> None:
    """Offsets, size and order hold for any max_chunk_size."""
    _check(DOC, chunk_markdown(PATH, DOC, size), size)


def test_chunks_start_at_headings_with_heading_path() -> None:
    """Sections become chunks whose context is their heading path."""
    chunks = chunk_markdown(PATH, DOC, 90)
    by_context = {c.context: c for c in chunks}
    sampling = by_context["OpenAI Server > Options > Sampling"]
    assert "### Sampling" in sampling.text
    assert "OpenAI Server > Options > Penalties" in by_context


def test_fenced_block_is_atomic_and_hides_comments() -> None:
    """'#' lines and blank lines inside a fence do not split it."""
    chunks = chunk_markdown(PATH, DOC, 130)
    holder = [c for c in chunks if "# this is a comment" in c.text]
    assert len(holder) == 1
    assert "vllm serve model" in holder[0].text
    assert all("this is a comment" not in c.context for c in chunks)


def test_indented_fence_in_admonition() -> None:
    """Fences indented inside '??? code' blocks are recognised."""
    text = ("??? code\n\n    ```bash\n    # not a heading\n\n    run\n"
            "    ```\n\n## Real\n\nBody.\n")
    chunks = chunk_markdown(PATH, text, 40)
    _check(text, chunks, 40)
    assert all("not a heading" not in c.context for c in chunks)


def test_mkdocs_markers_are_not_headings() -> None:
    """'# --8<-- [start:x]' lines of *.inc.md files stay in the text."""
    text = ("# --8<-- [start:installation]\n\nInstall it.\n\n"
            "# --8<-- [end:installation]\n")
    chunks = chunk_markdown(PATH, text, 2000)
    assert len(chunks) == 1
    assert chunks[0].context == ""
    assert "--8<-- [end:installation]" in chunks[0].text


def test_long_section_is_split_between_paragraphs() -> None:
    """An oversized section is cut at blank lines, keeping context."""
    paragraphs = [f"Paragraph {i}. " + "word " * 10 for i in range(6)]
    text = "# Title\n\n" + "\n\n".join(paragraphs) + "\n"
    chunks = chunk_markdown(PATH, text, 150)
    _check(text, chunks, 150)
    assert len(chunks) > 1
    assert all(c.context == "Title" for c in chunks)
    assert all(c.text.split()[0] in ("#", "Paragraph") for c in chunks)


def test_huge_block_keeps_its_heading() -> None:
    """A heading is not left alone before a block over the limit."""
    table = "\n".join(f"| model_{i} | yes |" for i in range(100))
    text = f"## Models\n\n{table}\n"
    chunks = chunk_markdown(PATH, text, 200)
    _check(text, chunks, 200)
    assert chunks[0].text.startswith("## Models")
    assert len(chunks[0].text) > 100


def test_tiny_sections_are_merged() -> None:
    """A lone heading is merged with the section that follows."""
    text = "# A\n\n## B\n\ntext\n"
    chunks = chunk_markdown(PATH, text, 2000)
    assert [c.text for c in chunks] == [text.strip()]
    assert chunks[0].context == "A > B"


def test_blank_documents_give_no_chunks() -> None:
    """Empty and whitespace-only documents return []."""
    assert chunk_markdown(PATH, "", 100) == []
    assert chunk_markdown(PATH, " \n\n\t\n", 100) == []


def test_crlf_unicode_and_no_final_newline() -> None:
    """CRLF, non-BMP characters and a missing last newline are fine."""
    text = "# Titre é\r\n\r\nPremier 🚀\r\n\r\n## Suite\r\n\r\nFin à ça"
    for size in (20, 40, 2000):
        chunks = chunk_markdown(PATH, text, size)
        _check(text, chunks, size)
    assert chunks[-1].last_character_index == len(text)


def test_invalid_max_chunk_size() -> None:
    """A non-positive limit raises ValueError."""
    with pytest.raises(ValueError):
        chunk_markdown(PATH, DOC, 0)


def test_random_documents_keep_invariants() -> None:
    """Random documents never break the invariants."""
    rng = random.Random(7)
    lines = ["# H", "## Sub", "```", "~~~", "text", "", "  ", "---",
             "word word word", "    ```py", "é🚀", "#nospace", "# --8<--"]
    for _ in range(400):
        body = [rng.choice(lines) for _ in range(rng.randint(1, 40))]
        sep = rng.choice(["\n", "\r\n"])
        text = sep.join(body) + rng.choice(["", sep])
        size = rng.randint(1, 120)
        _check(text, chunk_markdown(PATH, text, size), size)
