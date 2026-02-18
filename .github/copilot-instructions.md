# Copilot Instructions for Research Lab (rlab)

## Build, Test, and Lint

```bash
# Install all dependencies (including design extras)
uv sync --all-extras

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/design/test_core.py

# Run a single test by name
uv run pytest -k "test_field_selector"

# Lint and format
uv run ruff check .
uv run ruff format .

# Run the CLI
uv run rlab list
```

## Architecture

The project has two decoupled modules under `src/researchlab/`:

**`tracking/`** — Captures Git state (base commit + dirty patch) per MLflow run so experiments are reproducible even from uncommitted code. `ExperimentTracker` is a context manager that starts an MLflow run, uses `git add -N` to include untracked files in the diff, saves the patch as an artifact, and tags the run with `rlab.base_commit` and `rlab.run_id`. The CLI (`rlab`) restores runs by creating `experiment/<run_id>` branches and applying patches.

**`design/`** — Functional-style abstractions for JAX/Equinox research code. The pattern is:
1. **State** (`equinox.Module`) — single JAX PyTree holding all mutable simulation state.
2. **Config** (`pydantic.BaseModel`, frozen) — immutable hyperparameters.
3. **Kernels** — pure functions operating on data, decoupled from State/Config shape.
4. **Selectors** (`FieldSelector`) — decorators that map `state.x` / `config.y` dot-notation paths to kernel arguments, acting as lenses between the data structure and pure logic.
5. **Loop** (orchestrator) — coordinates State, Config, Kernels, and infrastructure (DataProvider, Telemetry, Persister, Visualizer).

Infrastructure interfaces in `design/infra.py` are abstract base classes; concrete implementations (e.g., `MLFlowTelemetry`, `EquinoxPersister`) handle side effects.

## Conventions

- Python 3.12+ syntax required — use `class Foo[T]:` generic syntax, `X | Y` unions, etc.
- Google-style docstrings when writing docstrings (enforced by ruff `D` rules), but docstrings are optional — no warnings for missing ones.
- Line length is 100 characters.
- All commands run through `uv run` (not bare `python` or `pytest`).
- State classes must be valid JAX PyTrees (inherit `equinox.Module`). Config classes must be frozen Pydantic models (inherit `Config`).
- `design` dependencies (JAX, Equinox, etc.) are optional extras — core tracking works without them.
- Tests use `tmp_path` fixtures with mock git repos and local MLflow tracking URIs (see `tests/conftest.py`).
- `EquinoxPersister` serializes only State (via Safetensors); Config is handled separately through Pydantic.
