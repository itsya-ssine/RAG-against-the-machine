"""Shared tokenizer for lexical retrieval (BM25).

A single, deterministic tokenization function is used both when
building the persisted index and when tokenizing a query at search
time. Using the same function in both places is what makes lexical
matching work at all: an index built with one tokenization scheme and
queried with another would silently return nothing useful.
"""

import re


def tokenize(text: str) -> "list[str]":
    """Split text into lowercase alphanumeric/underscore tokens.

    Args:
        text: Raw text to tokenize (source code, Markdown, or a
            natural-language question).

    Returns:
        A list of lowercase tokens. Identifiers such as
        ``trust_remote_code`` are kept whole (underscore is a token
        character) so exact identifier lookups still work; splitting
        further (e.g. on case changes) is a possible enhancement, not
        a requirement of the lexical baseline.
    """
    _TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
    return _TOKEN_RE.findall(text.lower())
