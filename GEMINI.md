# Gemini Context: Research Lab (rlab)

This document provides essential context and instructions for working on the `researchlab` project.

## Project Overview

`researchlab` (CLI: `rlab`) is a comprehensive tool for ML research. It consists of two main modules:
1.  `tracking`: Manages "dirty" research runs by capturing Git state (commits + patches).
2.  `design`: Provides patterns and abstractions (State, Config, Loop) for structured, functional-style research code using JAX/Equinox.

## Technical Stack
-   **Language:** Python 3.12+ (Uses new type parameter syntax `class Foo[T]:`).
-   **Dependency Management:** [uv](https://github.com/astral-sh/uv)
-   **Experiment Tracking:** [MLflow](https://mlflow.org/)
-   **Git Integration:** [GitPython](https://gitpython.readthedocs.io/)
-   **CLI Framework:** [Typer](https://typer.tiangolo.com/)
-   **Core ML Framework (Design module):** [JAX](https://github.com/google/jax), [Equinox](https://github.com/patrick-kidger/equinox)
-   **Configuration:** [Pydantic](https://docs.pydantic.dev/)
-   **Testing:** [pytest](https://docs.pytest.org/)

## Key Files and Directory Structure
-   `src/researchlab/`
    -   `tracking/`: **Module 1: Dirty Run Tracking**
        -   `tracker.py`: `ExperimentTracker` context manager.
        -   `cli.py`: `rlab` CLI commands.
        -   `utils.py`: Git capture and MLflow helpers.
    -   `design/`: **Module 2: Research Design Patterns**
        -   `core.py`: Core abstractions (`State`, `Config`, `Selector`).
        -   `infra.py`: Infrastructure interfaces (`Telemetry`, `Persister`, `DataProvider`).
        -   `orchestrator.py`: The training `Loop`.
        -   `utils.py`: Flattening/Unflattening utilities for PyTrees and Configs.
-   `tests/`
    -   `tracking/`: Tests for the tracking module (`test_tracker.py`, `test_cli.py`).
    -   `design/`: Tests for the design module (`test_core.py`, `test_infra.py`, `test_utils.py`).
    -   `conftest.py`: Shared fixtures.

## Development Commands

### Environment Setup
```bash
# Install all dependencies (including design extras)
uv sync --all-extras
```

### Running the CLI
```bash
uv run rlab list
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific module tests
uv run pytest tests/design/
```

### Linting and Formatting
The project uses `ruff` for linting and formatting.
```bash
uv run ruff check .
uv run ruff format .
```

## Implementation Details & Conventions

### Tracking Module
-   **Tags:** `rlab.base_commit` (Git SHA), `rlab.run_id` (Readable ID).
-   **Artifacts:** `run.patch` (Git patch).
-   **Git Strategy:** Uses `git add -N` for untracked files.

### Design Module
-   **State:** Inherits from `equinox.Module`. Must be a valid JAX PyTree.
-   **Config:** Inherits from `pydantic.BaseModel` (frozen).
-   **Selectors:** Use `FieldSelector` to map `state.x` -> `kernel_arg` via dot notation.
-   **Serialization:** `EquinoxPersister` uses `equinox.tree_serialise_leaves` (Safetensors) for State. Config is handled separately or via Pydantic dump.

### Code Style
-   Use Python 3.12+ syntax (e.g., `class Foo[T]:`).
-   Google-style docstrings for public APIs.
-   Strict typing using standard library and `jaxtyping` where applicable.
