"""Central configuration: default paths and corpus selection rules."""

from pathlib import Path
from typing import FrozenSet

# Default locations (all overridable from the CLI).
RAW_DATA_DIR: Path = Path("data/raw")
PROCESSED_DIR: Path = Path("data/processed")
DATASETS_DIR: Path = Path("data/datasets")
OUTPUT_DIR: Path = Path("data/output")

# Files written by the indexer inside the processed directory.
INDEX_FILENAME: str = "bm25_index.json"
CHUNKS_FILENAME: str = "chunks.json"

# Chunking limits. The moulinette rejects any source longer than
# MAX_ALLOWED_CHUNK_SIZE characters (its max_context_length).
DEFAULT_MAX_CHUNK_SIZE: int = 2000
MAX_ALLOWED_CHUNK_SIZE: int = 2000

# Corpus selection: only text files worth searching are indexed.
INCLUDED_EXTENSIONS: FrozenSet[str] = frozenset({".py", ".md"})

# Directories never entered while walking the corpus.
EXCLUDED_DIRS: FrozenSet[str] = frozenset({
    "__pycache__",
    ".git",
    ".venv",
    "node_modules",
    "csrc",
})

# Python files use the Python chunker; everything else the text chunker.
PYTHON_EXTENSIONS: FrozenSet[str] = frozenset({".py"})
