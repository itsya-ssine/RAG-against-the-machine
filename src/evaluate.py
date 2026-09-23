"""Self-evaluation stage: recall@k against a ground-truth dataset.

This mirrors a simplified version of what the official moulinette
computes, for local iteration only - the score reported here is never
the official one used during the defense. A retrieved source counts as
finding a reference source when both hold:

  * an exact match on ``file_path`` (compared verbatim), and
  * an IoU (intersection over union) of at least ``IOU_THRESHOLD``
    between the retrieved and reference character ranges.
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

IOU_THRESHOLD = 0.05
RECALL_LEVELS: "tuple[int, ...]" = (1, 3, 5, 10)


def _iou(a: MinimalSource, b: MinimalSource) -> float:
    """Intersection-over-union of two character ranges in the same file.

    Args:
        a: A retrieved source.
        b: A reference (ground-truth) source.

    Returns:
        ``0.0`` if the files differ or the ranges do not overlap,
        otherwise the IoU of ``[a.first, a.last)`` and
        ``[b.first, b.last)``.
    """
    if a.file_path != b.file_path:
        return 0.0

    start = max(a.first_character_index, b.first_character_index)
    end = min(a.last_character_index, b.last_character_index)
    intersection = max(0, end - start)
    if intersection == 0:
        return 0.0

    union = (
        (a.last_character_index - a.first_character_index)
        + (b.last_character_index - b.first_character_index)
        - intersection
    )
    if union <= 0:
        return 0.0
    return intersection / union


def _is_match(retrieved: MinimalSource, reference: MinimalSource) -> bool:
    """Whether a retrieved source counts as finding a reference source.

    Args:
        retrieved: A single retrieved source location.
        reference: A single ground-truth source location.

    Returns:
        True if both the file path matches exactly and the IoU of the
        two ranges is at least ``IOU_THRESHOLD``.
    """
    return _iou(retrieved, reference) >= IOU_THRESHOLD


def _recall_at_k(
    retrieved: "list[MinimalSource]",
    references: "list[MinimalSource]",
    k: int,
) -> "Optional[float]":
    """Share of a question's reference sources found in its top-k results.

    Args:
        retrieved: Ranked retrieved sources for one question.
        references: Ground-truth sources for that question.
        k: How many of the top retrieved sources to consider.

    Returns:
        ``found / len(references)``, or ``None`` if the question has
        no reference sources at all (such questions are excluded from
        the average rather than silently counted as a perfect or a
        zero score).
    """
    if not references:
        return None

    top = retrieved[:k]
    found = sum(
        1
        for reference in references
        if any(_is_match(candidate, reference) for candidate in top)
    )
    return found / len(references)


def _load_student_results(path: Path) -> "dict[str, MinimalSearchResults]":
    """Load a StudentSearchResults file, indexed by question_id.

    Args:
        path: Path to the JSON file to load.

    Returns:
        A mapping from question_id to its retrieval result.

    Raises:
        ValueError: If the file is not valid JSON or does not match
            the expected StudentSearchResults shape.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
        raise ValueError(f"malformed JSON in {path}: {error}") from error

    try:
        student = StudentSearchResults.model_validate(raw)
    except ValidationError as error:
        raise ValueError(
            f"{path} does not match StudentSearchResults: {error}"
        ) from error

    return {result.question_id: result for result in student.search_results}


def _load_reference_dataset(path: Path) -> "list[AnsweredQuestion]":
    """Load a ground-truth dataset and keep only its answered questions.

    Args:
        path: Path to a JSON file matching the RagDataset shape.

    Returns:
        The subset of questions that carry ground-truth sources
        (``AnsweredQuestion`` entries) - only those can be scored.

    Raises:
        ValueError: If the file is not valid JSON, does not match the
            expected shape, or contains no AnsweredQuestion entries.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
        raise ValueError(f"malformed JSON in {path}: {error}") from error

    try:
        dataset = RagDataset.model_validate(raw)
    except ValidationError as error:
        raise ValueError(
            f"{path} does not match RagDataset: {error}"
        ) from error

    answered = [
        question
        for question in dataset.rag_questions
        if isinstance(question, AnsweredQuestion)
    ]
    if not answered:
        raise ValueError(
            f"{path} contains no AnsweredQuestion entries to evaluate "
            "against"
        )
    return answered


def evaluate(
    student_search_results_path: "Path | str",
    dataset_path: "Path | str",
) -> "dict[int, float]":
    """Compute recall@k at several k values against a ground-truth dataset.

    Args:
        student_search_results_path: Path to a StudentSearchResults
            JSON file, e.g. produced by ``search_dataset``.
        dataset_path: Path to a ground-truth dataset JSON file whose
            questions carry ``sources`` (AnsweredQuestions).

    Returns:
        A mapping from k to the average recall@k across every question
        id present in both files, for k in ``RECALL_LEVELS``.

    Raises:
        FileNotFoundError: If either input file does not exist.
        ValueError: If either file is malformed, or there is no
            question_id overlap between the two files.
    """
    student_path = Path(student_search_results_path)
    dataset_path = Path(dataset_path)
    if not student_path.is_file():
        raise FileNotFoundError(
            f"student search results not found: {student_path}"
        )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    student_by_id = _load_student_results(student_path)
    reference_questions = _load_reference_dataset(dataset_path)

    per_k_scores: "dict[int, list[float]]" = {k: [] for k in RECALL_LEVELS}
    evaluated = 0

    for question in reference_questions:
        result = student_by_id.get(question.question_id)
        if result is None:
            continue
        evaluated += 1
        for k in RECALL_LEVELS:
            score = _recall_at_k(result.retrieved_sources, question.sources, k)
            if score is not None:
                per_k_scores[k].append(score)

    if evaluated == 0:
        raise ValueError(
            "no matching question_id between the student results and "
            "the reference dataset"
        )

    return {
        k: (sum(scores) / len(scores) if scores else 0.0)
        for k, scores in per_k_scores.items()
    }


def format_report(recall_by_k: "dict[int, float]") -> str:
    """Format a recall@k report similar to the moulinette's own output.

    Args:
        recall_by_k: Mapping from k to average recall@k, as returned
            by :func:`evaluate`.

    Returns:
        A multi-line, human-readable report string.
    """
    lines = ["Evaluation Results", "=" * 40]
    lines.append(
        "  ".join(
            f"Recall@{k}: {recall_by_k[k]:.3f}" for k in sorted(recall_by_k)
        )
    )
    return "\n".join(lines)
