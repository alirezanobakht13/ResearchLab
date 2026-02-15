# Copilot Instructions: Research Lab (rlab)

## Project Overview

`researchlab` (CLI: `rlab`) manages "dirty" research runs by bridging MLflow experiment results with the exact Git state that produced them—even uncommitted code. Three-phase workflow:

1. **Capture** — `ExperimentTracker` context manager logs the HEAD SHA + a full patch (staged, unstaged, untracked) to MLflow.
2. **Restore** — `rlab restore <run_id>` creates branch `experiment/<run_id>` from the base commit and applies the saved patch.
3. **Compare** — `rlab diff <id1> <id2>` reconstructs both code states in temp dirs and runs a three-way diff.

## Commands

```bash
uv sync                          # Install dependencies
uv run pytest                    # Run all tests
uv run pytest tests/test_cli.py  # Run one test file
uv run pytest -k test_cli_diff   # Run a single test by name
uv run ruff check .              # Lint
uv run ruff format .             # Format
```

## Architecture

- **`src/researchlab/utils.py`** — Core logic: `get_git_state()` captures HEAD SHA + patch using intent-to-add (`git add -N`) for untracked files, then resets the index. `find_run_by_rlab_id()` searches MLflow by the `rlab.run_id` tag. `generate_run_id()` produces `YYYY-MM-DD_slug` identifiers via the `coolname` library.
- **`src/researchlab/tracker.py`** — `ExperimentTracker` context manager that starts an MLflow run, calls `get_git_state()`, logs the base commit as tag `rlab.base_commit` and the patch as artifact `run.patch`. Also supports logging YAML configs as flattened MLflow params.
- **`src/researchlab/cli.py`** — Typer CLI (`app`) with commands: `restore`, `list`, `delete`, `diff`. Entry point registered as `rlab` in `pyproject.toml`.

## Key Conventions

- **Python 3.12+** with type hints on all function signatures.
- **`pathlib.Path`** for all file system operations (never raw `os.path`).
- **`uv`** as the package manager — always use `uv run` to invoke tools.
- **Ruff** for linting and formatting: line length 100, Google-style docstrings (optional but enforced if present), `F401`/`F841` are unfixable (never auto-removed). See `ruff.toml`.
- **MLflow tags**: `rlab.base_commit` (HEAD SHA), `rlab.run_id` (human-readable ID). **Artifact**: `run.patch`.
- **Branch naming**: restored runs go to `experiment/<run_id>`.
- **Git intent-to-add pattern**: untracked files are temporarily marked with `git add -N`, diffed, then reset — the index is always restored in a `finally` block.

## Testing

Tests use `pytest` with fixtures in `tests/conftest.py`:
- `mock_repo` — creates a temp Git repo with an initial commit and `chdir`s into it.
- `mlflow_setup` (autouse) — sets MLflow tracking URI to a temp directory.

Tests validate the full lifecycle: capture → restore → list → delete, and the diff workflow. CLI tests use `typer.testing.CliRunner`.
