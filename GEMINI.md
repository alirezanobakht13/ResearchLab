# Gemini Context: Research Lab (rlab)

This document provides essential context and instructions for working on the `researchlab` project.

## Project Overview

`researchlab` (CLI: `rlab`) focuses on reproducible "dirty run" experiment tracking.

1. `tracking`: Manages dirty research runs by capturing Git state (commits + patches).

## Technical Stack

- **Language:** Python 3.12+.
- **Dependency Management:** [uv](https://github.com/astral-sh/uv)
- **Experiment Tracking:** [MLflow](https://mlflow.org/)
- **Git Integration:** [GitPython](https://gitpython.readthedocs.io/)
- **CLI Framework:** [Typer](https://typer.tiangolo.com/)
- **Testing:** [pytest](https://docs.pytest.org/)

## Key Files and Directory Structure

- `src/researchlab/`
  - `tracking/`: **Dirty Run Tracking**
    - `tracker.py`: `ExperimentTracker` context manager.
    - `cli.py`: `rlab` CLI commands.
    - `utils.py`: Git capture and MLflow helpers.
- `tests/`
  - `tracking/`: Tests for the tracking module (`test_tracker.py`, `test_cli.py`).
  - `conftest.py`: Shared fixtures.

## Development Commands

### Environment Setup

```bash
# Install dependencies
uv sync
```

### Running the CLI

```bash
uv run rlab list
```

### Testing

```bash
# Run all tests
uv run pytest

# Run tracking tests
uv run pytest tests/tracking/
```

### Linting and Formatting

The project uses `ruff` for linting and formatting.

```bash
uv run ruff check .
uv run ruff format .
```

### Documentation

The project uses `mkdocs` with `invoke` tasks for documentation.

```bash
# Build documentation
uv run inv build-docs

# Serve documentation locally
uv run inv serve-docs
```

## Implementation Details & Conventions

### Tracking Module

- **Tags:** `rlab.base_commit` (Git SHA), `rlab.run_id` (Readable ID).
- **Artifacts:** `run.patch` (Git patch).
- **Git Strategy:** Uses `git add -N` for untracked files.

### Code Style

- Use Python 3.12+ syntax (e.g., `class Foo[T]:`).
- Google-style docstrings for public APIs.
- Strict typing using standard library.

## Versioning and Releases

This project follows [Semantic Versioning (SemVer)](https://semver.org/).

- **Versioning Tool:** [python-semantic-release](https://python-semantic-release.readthedocs.io/) is used to automate versioning and changelog generation.
- **Commit Standards:** Commit messages **must** follow [Conventional Commits](https://www.conventionalcommits.org/). This is mandatory for `python-semantic-release` to function.
  - `feat: ...` -> MINOR version bump.
  - `fix: ...` -> PATCH version bump.
  - `docs:`, `style:`, `refactor:`, `perf:`, `test:`, `chore:`, `ci:`, `build:` -> No version bump (usually).
  - `BREAKING CHANGE:` in the footer or `!` after the type -> MAJOR version bump.
- **Automation:** Version strings are maintained in `pyproject.toml` and `src/researchlab/__version__.py`.
