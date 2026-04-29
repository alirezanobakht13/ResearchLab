# Copilot Instructions for Research Lab (rlab)

## Build, Test, and Lint

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/tracking/test_tracker.py

# Run a single test by name
uv run pytest -k "test_experiment_tracker_logging"

# Lint and format
uv run ruff check .
uv run ruff format .

# Run the CLI
uv run rlab list
```

## Architecture

Project has one module under `src/researchlab/`:

**`tracking/`** — Captures Git state (base commit + dirty patch) per MLflow run so experiments are reproducible even from uncommitted code. `ExperimentTracker` is a context manager that starts an MLflow run, uses `git add -N` to include untracked files in the diff, saves patch as artifact, and tags run with `rlab.base_commit` and `rlab.run_id`. CLI (`rlab`) restores runs by creating `experiment/<run_id>` branches and applying patches.

## Conventions

- Python 3.12+ syntax required — use `class Foo[T]:` generic syntax, `X | Y` unions, etc.
- Google-style docstrings when writing docstrings (enforced by ruff `D` rules), but docstrings are optional — no warnings for missing ones.
- Line length is 100 characters.
- All commands run through `uv run` (not bare `python` or `pytest`).
- Tests use `tmp_path` fixtures with mock git repos and local MLflow tracking URIs (see `tests/conftest.py`).
