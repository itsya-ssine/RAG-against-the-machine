"""End-to-end tests for the RAG pipeline built so far.

Covers ``indexer.py`` (VI.1), ``retriever.py`` (VI.2), the ``RagCLI``
class in ``cli.py``, and a subprocess smoke test of ``__main__.py``
itself (i.e. the exact ``uv run python -m src <command>`` invocation
the grader uses). These are ad hoc sanity tests, not the official
moulinette - they exist to catch regressions locally before a defense.

Run with::

    uv run pytest tests/test_pipeline.py -v
"""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.cli import RagCLI
from src.indexer import INDEX_FILENAME, build_index
from src.models import StudentSearchResults
from src.retriever import IndexNotFoundError, Retriever

SAMPLER_PY = '''"""A fake module mimicking vllm's sampler, for testing."""

import math


def softmax(x):
    """Compute softmax."""
    m = max(x)
    exps = [math.exp(i - m) for i in x]
    s = sum(exps)
    return [e / s for e in exps]


class Sampler:
    """A dummy sampler class."""

    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def sample(self, logits):
        """Sample from logits using softmax."""
        scaled = [logit / self.temperature for logit in logits]
        return softmax(scaled)
'''

LORA_MD = """# LoRA Support

vLLM supports LoRA adapters for efficient fine-tuning deployment.

## Configuration

Pass `--enable-lora` to enable LoRA support at server startup.

## Limitations

Only a subset of models currently support LoRA.
"""


@pytest.fixture()
def project_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a tiny corpus under data/raw/ and chdir into it.

    Mirrors the real repository layout (data/raw/<repo>/...) and
    chdir's the test into tmp_path so that code under test using the
    spec's relative defaults ("data/raw", "data/processed") behaves
    exactly as it would when run for real from the repository root.
    """
    root = tmp_path / "data" / "raw" / "vllm-0.10.1"
    (root / "vllm").mkdir(parents=True)
    (root / "docs").mkdir(parents=True)
    (root / "vllm" / "sampler.py").write_text(SAMPLER_PY, encoding="utf-8")
    (root / "docs" / "lora.md").write_text(LORA_MD, encoding="utf-8")
    # unsupported extension: never even counted, must not crash indexing
    (root / "notes.yaml").write_text("key: value\n", encoding="utf-8")
    # supported extension but empty: counted as seen, then skipped
    (root / "docs" / "empty.md").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _load_index(processed_dir: Path) -> "dict[str, Any]":
    with (processed_dir / INDEX_FILENAME).open("rb") as handle:
        payload: "dict[str, Any]" = pickle.load(handle)
    return payload


# ---------------------------------------------------------------------
# Indexing (VI.1)
# ---------------------------------------------------------------------


class TestIndexing:
    """Tests for src/indexer.py."""

    def test_offsets_reconstruct_the_original_source(
        self, project_dir: Path
    ) -> None:
        stats = build_index(
            Path("data/raw"), Path("data/processed"), max_chunk_size=200
        )

        # notes.yaml has an unsupported extension: it is filtered out by
        # _iter_source_files before it is ever counted, so it shows up
        # in neither files_seen nor files_skipped. empty.md passes the
        # extension filter but chunks to nothing, so it is skipped.
        assert stats.files_seen == 3  # sampler.py + lora.md + empty.md
        assert stats.files_indexed == 2
        assert stats.files_skipped == 1
        assert stats.chunks_indexed > 0

        payload = _load_index(Path("data/processed"))
        assert len(payload["chunks"]) == len(payload["tokens"])
        assert payload["max_chunk_size"] == 200

        for chunk in payload["chunks"]:
            assert chunk["file_path"].startswith("data/raw/vllm-0.10.1/")
            source = Path(chunk["file_path"]).read_text(encoding="utf-8")
            first = chunk["first_character_index"]
            last = chunk["last_character_index"]
            assert source[first:last] == chunk["content"]
            assert len(chunk["content"]) <= 200

    def test_missing_raw_dir_raises(self, project_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_index(Path("data/does_not_exist"), Path("data/processed"))

    @pytest.mark.parametrize("bad_size", [0, -1])
    def test_non_positive_max_chunk_size_raises(
        self, project_dir: Path, bad_size: int
    ) -> None:
        with pytest.raises(ValueError):
            build_index(
                Path("data/raw"), Path("data/processed"), bad_size
            )

    def test_survives_a_syntactically_broken_python_file(
        self, project_dir: Path
    ) -> None:
        (Path("data/raw/vllm-0.10.1/vllm") / "broken.py").write_text(
            "def broken(:\n    pass\n", encoding="utf-8"
        )
        stats = build_index(
            Path("data/raw"), Path("data/processed"), max_chunk_size=200
        )
        # must not raise, and the good files must still be indexed
        assert stats.chunks_indexed > 0

    def test_empty_corpus_produces_zero_chunks_without_crashing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "data" / "raw").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        stats = build_index(Path("data/raw"), Path("data/processed"))
        assert stats.chunks_indexed == 0
        assert _load_index(Path("data/processed"))["chunks"] == []

    def test_rebuilding_overwrites_the_previous_index(
        self, project_dir: Path
    ) -> None:
        build_index(Path("data/raw"), Path("data/processed"), 2000)
        first = _load_index(Path("data/processed"))
        build_index(Path("data/raw"), Path("data/processed"), 200)
        second = _load_index(Path("data/processed"))
        assert second["max_chunk_size"] == 200
        assert first["chunks"] != second["chunks"]


# ---------------------------------------------------------------------
# Retrieval (VI.2)
# ---------------------------------------------------------------------


class TestRetriever:
    """Tests for src/retriever.py."""

    @pytest.fixture()
    def retriever(self, project_dir: Path) -> Retriever:
        build_index(Path("data/raw"), Path("data/processed"), 200)
        return Retriever(Path("data/processed"))

    def test_missing_index_raises(self, project_dir: Path) -> None:
        with pytest.raises(IndexNotFoundError):
            Retriever(Path("data/processed"))

    def test_relevant_chunk_is_ranked_first(
        self, retriever: Retriever
    ) -> None:
        results = retriever.search("how do I enable lora support", k=3)
        assert results
        assert "lora.md" in results[0].file_path
        assert results[0].first_character_index == 0

    def test_different_query_ranks_a_different_file_first(
        self, retriever: Retriever
    ) -> None:
        results = retriever.search("what does softmax compute", k=3)
        assert results
        assert "sampler.py" in results[0].file_path

    def test_results_respect_k(self, retriever: Retriever) -> None:
        assert len(retriever.search("lora", k=1)) == 1
        assert len(retriever.search("lora", k=0)) == 0

    def test_empty_query_returns_no_results(
        self, retriever: Retriever
    ) -> None:
        assert retriever.search("", k=5) == []
        assert retriever.search("   ", k=5) == []

    def test_search_dataset_returns_one_result_per_question(
        self, retriever: Retriever, project_dir: Path
    ) -> None:
        dataset_path = Path("dataset.json")
        dataset_path.write_text(
            json.dumps(
                {
                    "rag_questions": [
                        {"question_id": "q1", "question": "lora support"},
                        {"question_id": "q2", "question": "softmax"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        results = retriever.search_dataset(dataset_path, k=2)
        assert [r.question_id for r in results] == ["q1", "q2"]
        assert all(len(r.retrieved_sources) <= 2 for r in results)

    def test_search_dataset_missing_file_raises(
        self, retriever: Retriever
    ) -> None:
        with pytest.raises(FileNotFoundError):
            retriever.search_dataset(Path("does_not_exist.json"), k=3)

    def test_search_dataset_malformed_json_raises(
        self, retriever: Retriever, project_dir: Path
    ) -> None:
        bad_path = Path("broken.json")
        bad_path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError):
            retriever.search_dataset(bad_path, k=3)

    def test_search_dataset_wrong_shape_raises(
        self, retriever: Retriever, project_dir: Path
    ) -> None:
        wrong_shape = Path("wrong_shape.json")
        wrong_shape.write_text(json.dumps({"not": "a dataset"}))
        with pytest.raises(ValueError):
            retriever.search_dataset(wrong_shape, k=3)


# ---------------------------------------------------------------------
# CLI (src/cli.py, called as a plain Python class - no subprocess)
# ---------------------------------------------------------------------


class TestRagCLI:
    """Tests for the RagCLI class methods directly."""

    def test_index_then_search_end_to_end(
        self, project_dir: Path, capsys: pytest.CaptureFixture
    ) -> None:
        cli = RagCLI()
        cli.index(max_chunk_size=200)
        assert Path("data/processed", INDEX_FILENAME).is_file()

        capsys.readouterr()  # discard the index command's output
        cli.search("lora support", k=3)
        out = capsys.readouterr().out
        assert "lora.md" in out
        assert "[" in out and "]" in out  # "[first:last]" formatting

    def test_index_missing_raw_dir_prints_message_not_traceback(
        self, project_dir: Path, capsys: pytest.CaptureFixture
    ) -> None:
        RagCLI().index(raw_directory="does/not/exist")
        assert "index failed" in capsys.readouterr().out

    def test_search_without_index_prints_message_not_traceback(
        self, project_dir: Path, capsys: pytest.CaptureFixture
    ) -> None:
        RagCLI().search("lora", k=3)
        assert "search failed" in capsys.readouterr().out

    def test_search_empty_query_prints_message(
        self, project_dir: Path, capsys: pytest.CaptureFixture
    ) -> None:
        RagCLI().index(max_chunk_size=200)
        capsys.readouterr()
        RagCLI().search("", k=3)
        assert "search failed" in capsys.readouterr().out

    def test_search_dataset_writes_a_valid_student_search_results_file(
        self, project_dir: Path, capsys: pytest.CaptureFixture
    ) -> None:
        RagCLI().index(max_chunk_size=200)
        capsys.readouterr()

        dataset_path = Path("data/datasets/UnansweredQuestions/mini.json")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text(
            json.dumps(
                {
                    "rag_questions": [
                        {"question_id": "q1", "question": "lora support"},
                    ]
                }
            ),
            encoding="utf-8",
        )

        save_dir = Path("data/output/search_results/UnansweredQuestions")
        RagCLI().search_dataset(
            dataset_path=str(dataset_path),
            save_directory=str(save_dir),
            k=3,
        )

        output_path = save_dir / "mini.json"
        assert output_path.is_file()
        assert "Saved student_search_results" in capsys.readouterr().out

        raw = json.loads(output_path.read_text(encoding="utf-8"))
        model = StudentSearchResults.model_validate(raw)
        assert model.k == 3
        assert model.search_results[0].question_id == "q1"


# ---------------------------------------------------------------------
# __main__.py entry point (subprocess: exactly how the grader runs it)
# ---------------------------------------------------------------------


class TestMainEntryPoint:
    """Smoke-tests `uv run python -m src <command>` itself.

    The class-level tests above exercise RagCLI's logic directly and
    run fast; this exercises the actual Fire wiring in __main__.py, so
    a broken argument name or import only visible at the CLI boundary
    doesn't slip through.
    """

    def _run(self, args: "list[str]", cwd: Path) -> "subprocess.CompletedProcess[str]":
        repo_root = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONPATH": str(repo_root)}
        return subprocess.run(
            [sys.executable, "-m", "src", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_index_command(self, project_dir: Path) -> None:
        result = self._run(
            ["index", "--max_chunk_size", "200"], cwd=project_dir
        )
        assert result.returncode == 0, result.stderr
        assert "Ingestion complete!" in result.stdout
        assert (project_dir / "data/processed" / INDEX_FILENAME).is_file()

    def test_search_command(self, project_dir: Path) -> None:
        self._run(["index", "--max_chunk_size", "200"], cwd=project_dir)
        result = self._run(
            ["search", "lora support", "--k", "3"], cwd=project_dir
        )
        assert result.returncode == 0, result.stderr
        assert "lora.md" in result.stdout
