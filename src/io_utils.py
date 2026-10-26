"""File-system helpers: corpus walking and text reading."""

import os
from pathlib import Path
from typing import FrozenSet, List, Optional

from src.config import EXCLUDED_DIRS, INCLUDED_EXTENSIONS


def iter_corpus_files(
    root: Path,
    extensions: Optional[FrozenSet[str]] = None,
    excluded_dirs: Optional[FrozenSet[str]] = None,
) -> List[Path]:
    """List the files to index under a corpus root.

    The walk is deterministic: directories and files are visited in
    sorted order, so the resulting index is reproducible.

    Args:
        root: Directory containing the corpus (e.g. ``data/raw``).
        extensions: Lowercase extensions to keep (with the dot).
            Defaults to ``INCLUDED_EXTENSIONS``.
        excluded_dirs: Directory names never entered.
            Defaults to ``EXCLUDED_DIRS``.

    Returns:
        Sorted list of file paths, each prefixed with ``root``.

    Raises:
        FileNotFoundError: If ``root`` does not exist or is not a
            directory.
    """
    keep = INCLUDED_EXTENSIONS if extensions is None else extensions
    skip = EXCLUDED_DIRS if excluded_dirs is None else excluded_dirs

    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root}")

    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix.lower() in keep:
                files.append(path)

    return files


def to_corpus_path(path: Path) -> str:
    """Convert a path to the string stored in ``MinimalSource.file_path``.

    Args:
        path: Path of a corpus file, relative to the project root.

    Returns:
        The path with forward slashes, written exactly as in the corpus
        (e.g. ``data/raw/vllm-0.10.1/docs/features/lora.md``).
    """
    return path.as_posix()


def read_text(path: Path) -> str:
    """Read a file as UTF-8 text without altering line endings.

    ``newline=""`` disables newline translation, so character offsets
    computed on the returned string match the file content exactly.

    Args:
        path: File to read.

    Returns:
        The file content.

    Raises:
        OSError: If the file cannot be opened or read.
        UnicodeDecodeError: If the file is not valid UTF-8.
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()
