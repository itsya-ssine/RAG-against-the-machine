"""Command-line interface for the RAG pipeline, built with Python Fire.

Only the ``index`` command is wired up so far. ``search``,
``search_dataset``, ``answer``, ``answer_dataset`` and ``evaluate``
will be added once retrieval and generation are implemented.
"""

from pathlib import Path

from src.indexer import DEFAULT_MAX_CHUNK_SIZE, build_index


class RagCLI:
    """Entry point exposing the RAG pipeline as CLI commands."""

    def index(
        self,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        raw_directory: str = "data/raw",
        save_directory: str = "data/processed",
    ) -> None:
        """Ingest ``data/raw/`` and build the index under ``data/processed/``.

        Args:
            max_chunk_size: Maximum number of characters per chunk.
            raw_directory: Root directory of the corpus to ingest.
            save_directory: Directory the index is persisted to.
        """
        try:
            build_index(
                raw_dir=Path(raw_directory),
                processed_dir=Path(save_directory),
                max_chunk_size=max_chunk_size,
            )
        except (FileNotFoundError, ValueError) as error:
            print(f"index failed: {error}")
