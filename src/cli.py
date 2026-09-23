"""Command-line interface for the RAG pipeline, built with Python Fire."""

import json
from pathlib import Path

from pydantic import ValidationError

from src.evaluate import evaluate as compute_recall
from src.evaluate import format_report
from src.generator import DEFAULT_MODEL_NAME, Generator
from src.indexer import DEFAULT_MAX_CHUNK_SIZE, build_index
from src.models import (
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
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

    def answer(
        self,
        query: str,
        k: int = DEFAULT_K,
        index_directory: str = "data/processed",
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        """Answer a single query using the retrieved context.

        Args:
            query: The question to answer.
            k: Number of retrieved sources to ground the answer in.
            index_directory: Directory produced by the index command.
            model_name: Hugging Face model id used for generation.
        """
        try:
            retriever = Retriever(Path(index_directory))
        except (IndexNotFoundError, ValueError) as error:
            print(f"answer failed: {error}")
            return

        if not query.strip():
            print("answer failed: query is empty")
            return
        if k <= 0:
            print("answer failed: k must be a positive integer")
            return

        sources = retriever.search(query, k)

        try:
            generator = Generator(model_name=model_name)
        except RuntimeError as error:
            print(f"answer failed: {error}")
            return

        answer_text = generator.generate_answer(query, sources)

        for source in sources:
            print(format_source_line(source))
        print(f"\n{answer_text}")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        """Generate answers for a dataset of already-retrieved questions.

        Reads a StudentSearchResults JSON file (as produced by
        ``search_dataset``) and writes a StudentSearchResultsAndAnswer
        JSON file with one generated answer per question.

        Args:
            student_search_results_path: Path to a StudentSearchResults
                JSON file.
            save_directory: Directory the output JSON file is written
                to (named after the input file).
            model_name: Hugging Face model id used for generation.
        """
        input_path = Path(student_search_results_path)
        if not input_path.is_file():
            print(
                "answer_dataset failed: student search results not "
                f"found: {input_path}"
            )
            return

        try:
            raw = json.loads(input_path.read_text(encoding="utf-8"))
            student = StudentSearchResults.model_validate(raw)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
            print(f"answer_dataset failed: malformed JSON: {error}")
            return
        except ValidationError as error:
            print(
                "answer_dataset failed: input does not match "
                f"StudentSearchResults: {error}"
            )
            return

        try:
            generator = Generator(model_name=model_name)
        except RuntimeError as error:
            print(f"answer_dataset failed: {error}")
            return

        print(f"Loaded {len(student.search_results)} questions")
        answers = generator.answer_search_results(student.search_results)
        print(
            f"Processed {len(answers)} of "
            f"{len(student.search_results)} questions"
        )

        output_path = Path(save_directory) / input_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = StudentSearchResultsAndAnswer(
            search_results=answers, k=student.k
        )
        output_path.write_text(
            payload.model_dump_json(indent=2), encoding="utf-8"
        )
        print(f"Saved student_search_results_and_answer to {output_path}")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Report your own recall@k against a ground-truth dataset.

        This is for local iteration only: the official recall@k used
        during the defense is computed by the provided moulinette
        executable, not by this command.

        Args:
            student_search_results_path: Path to a StudentSearchResults
                JSON file, as produced by ``search_dataset``.
            dataset_path: Path to a ground-truth AnsweredQuestions
                dataset JSON file.
        """
        try:
            recall_by_k = compute_recall(
                Path(student_search_results_path), Path(dataset_path)
            )
        except (FileNotFoundError, ValueError) as error:
            print(f"evaluate failed: {error}")
            return

        print(format_report(recall_by_k))
