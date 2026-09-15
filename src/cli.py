"""Command-line interface for the RAG pipeline, built with Python Fire.

``index``, ``search`` and ``search_dataset`` are wired up so far.
``answer``, ``answer_dataset`` and ``evaluate`` will follow once
generation is implemented.
"""

from pathlib import Path

from src.indexer import DEFAULT_MAX_CHUNK_SIZE, build_index
from src.models import StudentSearchResults
from src.retriever import (
    DEFAULT_K,
    IndexNotFoundError,
    Retriever,
    format_source_line,
)


class RagCLI:
    """Entry point exposing the RAG pipeline as CLI commands."""

    def index(
        self,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        raw_directory: str = "data/raw",
        save_directory: str = "data/processed",
    ) -> None:
        """Ingest data/raw/ and build the index under data/processed/.

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

    def search(
        self,
        query: str,
        k: int = DEFAULT_K,
        index_directory: str = "data/processed",
    ) -> None:
        """Return the top-k sources for a single query.

        Args:
            query: The question to search for.
            k: Number of results to return.
            index_directory: Directory produced by the index command.
        """
        try:
            retriever = Retriever(Path(index_directory))
        except (IndexNotFoundError, ValueError) as error:
            print(f"search failed: {error}")
            return

        if not query.strip():
            print("search failed: query is empty")
            return
        if k <= 0:
            print("search failed: k must be a positive integer")
            return

        for source in retriever.search(query, k):
            print(format_source_line(source))

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        k: int = DEFAULT_K,
        index_directory: str = "data/processed",
    ) -> None:
        """Run search over a dataset and write a StudentSearchResults file.

        Args:
            dataset_path: Path to the input dataset JSON file.
            save_directory: Directory the output JSON file is written
                to (named after the input file).
            k: Number of results to return per question.
            index_directory: Directory produced by the index command.
        """
        try:
            retriever = Retriever(Path(index_directory))
        except (IndexNotFoundError, ValueError) as error:
            print(f"search_dataset failed: {error}")
            return
        if k <= 0:
            print("search_dataset failed: k must be a positive integer")
            return

        try:
            results = retriever.search_dataset(Path(dataset_path), k)
        except (FileNotFoundError, ValueError) as error:
            print(f"search_dataset failed: {error}")
            return

        output_path = Path(save_directory) / Path(dataset_path).name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = StudentSearchResults(search_results=results, k=k)
        output_path.write_text(
            payload.model_dump_json(indent=2), encoding="utf-8"
        )
        print(f"Saved student_search_results to {output_path}")
