PYTHON  := uv run python
ARGS    ?=

.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	$(PYTHON) -m src $(ARGS)

debug:
	$(PYTHON) -m pdb -m src $(ARGS)

clean:
	find . -type d \( -name "__pycache__" -o -name ".mypy_cache" \
		-o -name ".pytest_cache" -o -name ".ruff_cache" \) \
		-not -path "./.venv/*" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -not -path "./.venv/*" -delete
	rm -rf *.egg-info build dist

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
