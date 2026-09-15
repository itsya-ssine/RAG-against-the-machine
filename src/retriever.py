"""Retrieval stage: rank persisted chunks against a query with BM25.

Loads the index built by ``indexer.build_index``, wraps its
pre-tokenized chunks in a ``rank_bm25.BM25Okapi`` model, and answers
top-k queries either one at a time or in batch over a whole dataset.
The BM25 model is rebuilt in memory once per process from the
persisted tokens (fast: no re-reading or re-chunking of source files),
so that repeated queries against the same ``Retriever`` instance are
what stays in the millisecond range.
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.indexer import INDEX_FILENAME
from src.models import MinimalSearchResults, MinimalSource, RagDataset
from src.tokenizer import tokenize

DEFAULT_K = 10


class IndexNotFoundError(RuntimeError):
    """Raised when ``data/processed/index.pkl`` does not exist yet."""


@dataclass
class _IndexedChunk:
    """A single persisted chunk, as loaded back from data/processed/."""

    file_path: str
    first_character_index: int
    last_character_index: int


class Retriever:
    """BM25 lexical retriever over a persisted chunk index."""

    def __init__(
        self, processed_dir: "Path | str" = Path("data/processed")
    ) -> None:
        """Load the persisted index and build the BM25 model.

        Args:
            processed_dir: Directory produced by the ``index``
                command.

        Raises:
            IndexNotFoundError: If no index has been built yet.
            ValueError: If the persisted index is empty or corrupted.
        """
        processed_dir = Path(processed_dir)
        index_path = processed_dir / INDEX_FILENAME
        if not index_path.is_file():
            raise IndexNotFoundError(
                f"no index found at {index_path}; run `index` first"
            )

        try:
            with index_path.open("rb") as handle:
                payload = pickle.load(handle)
            chunk_dicts = payload["chunks"]
            tokens = payload["tokens"]
        except (pickle.UnpicklingError, KeyError, EOFError, OSError) as e:
            raise ValueError(f"corrupted index at {index_path}: {e}") from e

        if len(chunk_dicts) != len(tokens):
            raise ValueError(
                f"corrupted index at {index_path}: "
                f"{len(chunk_dicts)} chunks but {len(tokens)} token lists"
            )

        self._chunks = [
            _IndexedChunk(
                file_path=chunk["file_path"],
                first_character_index=chunk["first_character_index"],
                last_character_index=chunk["last_character_index"],
            )
            for chunk in chunk_dicts
        ]
        self._bm25 = BM25Okapi(tokens) if tokens else None

    def search(self, query: str, k: int) -> "list[MinimalSource]":
        """Return the top-k sources most relevant to a query.

        Args:
            query: Natural-language or code-ish question text.
            k: Number of results to return. ``k <= 0`` returns an
                empty list.

        Returns:
            Up to k MinimalSource, ranked by BM25 score (descending,
            ties broken by chunk order). An empty list if the index
            is empty or the query tokenizes to nothing.
        """
        if k <= 0 or self._bm25 is None:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        top = ranked[:k]

        return [
            MinimalSource(
                file_path=self._chunks[i].file_path,
                first_character_index=self._chunks[i].first_character_index,
                last_character_index=self._chunks[i].last_character_index,
            )
            for i in top
        ]

    def search_dataset(
        self,
        dataset_path: "Path | str",
        k: int,
    ) -> "list[MinimalSearchResults]":
        """Run search over every question in a dataset file.

        Args:
            dataset_path: Path to a JSON file conforming to
                RagDataset (questions may be Answered or Unanswered -
                only question_id/question are used for retrieval).
            k: Number of results to return per question.

        Returns:
            One MinimalSearchResults per question, in file order.

        Raises:
            FileNotFoundError: If dataset_path does not exist.
            ValueError: If the file is not valid JSON or does not
                match the expected dataset shape.
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.is_file():
            raise FileNotFoundError(f"dataset not found: {dataset_path}")

        try:
            raw = json.loads(dataset_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            raise ValueError(f"malformed JSON in {dataset_path}: {e}") from e

        try:
            dataset = RagDataset.model_validate(raw)
        except ValidationError as e:
            raise ValueError(
                f"{dataset_path} does not match the expected dataset "
                f"format: {e}"
            ) from e

        results: "list[MinimalSearchResults]" = []
        for question in tqdm(
            dataset.rag_questions, desc="Searching", unit="question"
        ):
            sources = self.search(question.question, k)
            results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=sources,
                )
            )
        return results


def format_source_line(source: MinimalSource) -> str:
    """Format a source the way the subject's single-query CLI does.

    Args:
        source: A retrieved source location.

    Returns:
        A ``"<file_path> [<first>:<last>]"`` line.
    """
    return (
        f"{source.file_path} "
        f"[{source.first_character_index}:{source.last_character_index}]"
    )
