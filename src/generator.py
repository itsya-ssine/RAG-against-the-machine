"""Answer generation stage: produce grounded answers with Qwen/Qwen3-0.6B.

Re-reads the retrieved source spans from disk, assembles them into a
prompt that fits the model's context window, and asks a local
``Qwen/Qwen3-0.6B`` instance to answer the question strictly from that
context. The model and tokenizer are loaded once per ``Generator``
instance and reused across every question handed to it, since reloading
per question would dominate runtime on CPU.
"""

import re
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.models import MinimalAnswer, MinimalSearchResults, MinimalSource

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_NEW_TOKENS = 512
# Character budget for the assembled context block. Kept well under the
# model's token window (a token is ~4 characters on average for English
# text, so this leaves headroom for the prompt template, the question,
# and the generated answer itself).
DEFAULT_MAX_CONTEXT_CHARS = 6000

_NO_CONTEXT_ANSWER = (
    "I don't have any retrieved sources to answer this question from."
)

_SYSTEM_PROMPT = (
    "You are a documentation assistant answering questions about a "
    "codebase. Answer ONLY using the excerpts provided below. If the "
    "excerpts do not contain the answer, say you don't know instead of "
    "guessing. Be concise, factual, and do not invent file names, "
    "function names, or behaviour that is not shown in the excerpts."
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _read_source_span(
    source: MinimalSource, project_root: Path
) -> "Optional[str]":
    """Re-read a retrieved chunk's exact text from disk.

    Args:
        source: A retrieved source location (file_path + character
            range), as produced by the retrieval stage.
        project_root: Root directory ``source.file_path`` is relative
            to (normally the current working directory, since
            ``file_path`` already starts with ``data/raw/...``).

    Returns:
        The exact substring the source refers to, or ``None`` if the
        file can no longer be read (e.g. moved or deleted since
        indexing).
    """
    path = project_root / source.file_path
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return text[source.first_character_index:source.last_character_index]


def _build_context(
    sources: "list[MinimalSource]",
    project_root: Path,
    max_context_chars: int,
) -> str:
    """Assemble retrieved spans into a single, budget-limited context block.

    Args:
        sources: Retrieved source locations, in ranked order.
        project_root: Root directory sources are relative to.
        max_context_chars: Hard cap on the total size of the assembled
            context, to keep the final prompt inside the model's token
            budget.

    Returns:
        A block of excerpts separated by blank lines, each prefixed
        with its file path so the model can refer to it, truncated to
        stay under ``max_context_chars``.
    """
    parts: "list[str]" = []
    used = 0
    for source in sources:
        text = _read_source_span(source, project_root)
        if not text:
            continue
        header = f"### {source.file_path}\n"
        budget_left = max_context_chars - used - len(header)
        if budget_left <= 0:
            break
        block = f"{header}{text[:budget_left]}"
        parts.append(block)
        used += len(block)
        if used >= max_context_chars:
            break
    return "\n\n".join(parts)


def _build_messages(question: str, context: str) -> "list[dict[str, str]]":
    """Build the chat messages sent to the model.

    Args:
        question: The user's question.
        context: Assembled retrieved source excerpts.

    Returns:
        A chat-style message list (system + user) suitable for
        ``tokenizer.apply_chat_template``.
    """
    user = (
        f"Context excerpts:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _strip_thinking(text: str) -> str:
    """Remove a Qwen3 ``<think>...</think>`` reasoning block, if present.

    Args:
        text: Raw decoded model output.

    Returns:
        The text with any leading reasoning block removed and
        surrounding whitespace stripped.
    """
    return _THINK_BLOCK_RE.sub("", text).strip()


class Generator:
    """Wraps a local Qwen/Qwen3-0.6B model for grounded answer generation."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        project_root: "Path | str" = Path("."),
    ) -> None:
        """Load the tokenizer and model once for reuse across questions.

        Args:
            model_name: Hugging Face model id to load.
            max_new_tokens: Maximum number of tokens to generate per
                answer.
            max_context_chars: Character budget for the assembled
                retrieved context.
            project_root: Root directory retrieved ``file_path``
                values are relative to.

        Raises:
            RuntimeError: If ``transformers``/``torch`` are missing or
                the model fails to load (e.g. no network access on a
                machine with no cached weights).
        """
        self._max_new_tokens = max_new_tokens
        self._max_context_chars = max_context_chars
        self._project_root = Path(project_root)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "transformers and torch are required for answer "
                "generation; install project dependencies with "
                "`uv sync`"
            ) from error

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32
            )
        except Exception as error:  # noqa: BLE001 - surfaced to the CLI
            raise RuntimeError(
                f"failed to load model '{model_name}': {error}"
            ) from error

        self._model.eval()
        self._torch = torch

    def generate_answer(
        self, question: str, sources: "list[MinimalSource]"
    ) -> str:
        """Generate a grounded answer for a single question.

        Args:
            question: The question text.
            sources: Retrieved source locations to ground the answer
                in, in ranked order.

        Returns:
            The generated answer text, or a fixed fallback message if
            no usable context could be assembled (e.g. empty retrieval
            or unreadable source files).
        """
        context = _build_context(
            sources, self._project_root, self._max_context_chars
        )
        if not context.strip():
            return _NO_CONTEXT_ANSWER

        messages = _build_messages(question, context)
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older chat templates may not accept enable_thinking.
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = self._tokenizer(prompt, return_tensors="pt")

        with self._torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=(
                    self._tokenizer.pad_token_id
                    or self._tokenizer.eos_token_id
                ),
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        answer = _strip_thinking(text)
        return answer if answer else _NO_CONTEXT_ANSWER

    def answer_search_results(
        self, search_results: "list[MinimalSearchResults]"
    ) -> "list[MinimalAnswer]":
        """Generate answers for a batch of already-retrieved questions.

        Args:
            search_results: Retrieval output for a dataset, as
                produced by the ``search_dataset`` command.

        Returns:
            One ``MinimalAnswer`` per input result, in the same order.
        """
        answers: "list[MinimalAnswer]" = []
        for result in tqdm(
            search_results, desc="Generating answers", unit="question"
        ):
            answer_text = self.generate_answer(
                result.question, result.retrieved_sources
            )
            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer_text,
                )
            )
        return answers
