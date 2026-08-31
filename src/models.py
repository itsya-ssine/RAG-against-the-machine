"""Pydantic data models exchanged between the RAG pipeline stages."""

import uuid

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A single retrieved source location inside the indexed corpus."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question from a dataset, before retrieval/generation."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question paired with its ground-truth sources and answer."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions, answered or not."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Retrieval output for a single question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieval output plus a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Top-level output written by the `search_dataset` command."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Top-level output written by the `answer_dataset` command."""

    search_results: list[MinimalAnswer]
    k: int
