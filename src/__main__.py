"""CLI entry point: ``uv run python -m src <command> [options]``."""

import fire

from src.cli import RagCLI


def main() -> None:
    """Run the Python Fire CLI."""
    fire.Fire(RagCLI)


if __name__ == "__main__":
    main()
