"""Unit tests for the pydantic data models."""

from typing import Any, Dict

import pytest
from pydantic import ValidationError

from src.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)

SOURCE: Dict[str, Any] = {
    "file_path": "data/raw/vllm-0.10.1/docs/features/lora.md",
    "first_character_index": 0,
    "last_character_index": 100,
}


def test_unanswered_question_generates_unique_ids() -> None:
    """Each question without an id gets its own UUID."""
    first = UnansweredQuestion(question="What is LoRA?")
    second = UnansweredQuestion(question="What is LoRA?")
    assert first.question_id != second.question_id


def test_dataset_parses_answered_and_unanswered() -> None:
    """A dataset accepts both kinds of questions."""
    dataset = RagDataset.model_validate({
        "rag_questions": [
            {"question_id": "q1", "question": "Q?"},
            {"question_id": "q2", "question": "Q?",
             "sources": [SOURCE], "answer": "A"},
        ]
    })
    assert len(dataset.rag_questions) == 2
    assert isinstance(dataset.rag_questions[1], AnsweredQuestion)


def test_invalid_source_is_rejected() -> None:
    """A source with a non-integer index fails validation."""
    with pytest.raises(ValidationError):
        MinimalSource.model_validate({**SOURCE, "first_character_index": "x"})


def test_search_results_round_trip() -> None:
    """Search results and answers serialize and parse back unchanged."""
    answer = MinimalAnswer(
        question_id="q1", question="Q?",
        retrieved_sources=[MinimalSource(**SOURCE)], answer="A",
    )
    out = StudentSearchResultsAndAnswer(search_results=[answer], k=5)
    parsed = StudentSearchResultsAndAnswer.model_validate_json(
        out.model_dump_json())
    assert parsed == out
    assert StudentSearchResults(search_results=[], k=0).k == 0
