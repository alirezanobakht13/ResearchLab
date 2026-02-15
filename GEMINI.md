# Gemini Context: Research Lab (rlab)

This document provides essential context and instructions for working on the `researchlab` project.

## Project Overview

`researchlab` (CLI: `rlab`) is a tool designed to manage "dirty" research runs. It bridges the gap between experimental results in MLflow and the exact code state that produced them, even if the code was not committed at the time of the run.

### Core Workflow
1.  **Capture:** The `ExperimentTracker` context manager captures the current Git HEAD SHA and a full patch (staged + unstaged + untracked changes) and logs them to MLflow.
2.  **Restore:** The `rlab restore` command creates a new git branch from the recorded base commit and applies the saved patch.
3.  **Compare:** The `rlab diff` command performs a three-way diff between two MLflow runs by reconstructing their code states in temporary directories.

## Technical Stack
- **Language:** Python 3.12+
- **Dependency Management:** [uv](https://github.com/astral-sh/uv)
- **Experiment Tracking:** [MLflow](https://mlflow.org/)
- **Git Integration:** [GitPython](https://gitpython.readthedocs.io/)
- **CLI Framework:** [Typer](https://typer.tiangolo.com/)
- **Testing:** [pytest](https://docs.pytest.org/)

## Key Files and Directory Structure
- `src/researchlab/`
    - `tracker.py`: Contains the `ExperimentTracker` context manager used in training scripts.
    - `cli.py`: Implements the `rlab` command-line interface.
    - `utils.py`: Utility functions for Git state capture (`get_git_state`) and MLflow run lookup.
- `tests/`
    - `conftest.py`: Shared fixtures, including a mocked git repository and local MLflow tracking.
    - `test_tracker.py` & `test_cli.py`: Comprehensive tests for core functionality.

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync
```

### Running the CLI
```bash
# List experiment branches
uv run rlab list

# Restore a run
uv run rlab restore <run_id>

# Diff two runs
uv run rlab diff <run_id_1> <run_id_2>
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage (if configured)
uv run pytest --cov=researchlab
```

### Linting and Formatting
The project uses `ruff` for linting and formatting (see `ruff.toml`).
```bash
uv run ruff check .
uv run ruff format .
```

## Implementation Details & Conventions

### MLflow Integration
- **Tags:** 
    - `rlab.base_commit`: The Git SHA of HEAD at run time.
    - `rlab.run_id`: A human-readable ID (e.g., `2026-02-15_radiant-octopus`).
- **Artifacts:**
    - `run.patch`: A git-compatible patch file containing all local changes.

### Git Strategy
- **Intent-to-Add:** To include untracked files in the patch, `rlab` temporarily marks them with `git add -N` before generating the diff, then resets the index.
- **Branch Naming:** Restored runs are placed in branches named `experiment/<run_id>`.

### Code Style
- Use type hints for all function signatures.
- Prefer `pathlib.Path` for file system operations.
- Follow the formatting rules defined in `ruff.toml`.
