# Research Lab (rlab)

Manage dirty research runs and structure reproducible ML experiments.

## Why Research Lab?

Research code often faces two problems:
1.  **Dirty Runs:** You make changes, run a script, and continue editing. Later, you can't reproduce the exact state of code that generated the results.
2.  **Boilerplate & Coupling:** Research logic (math) gets tangled with infrastructure (logging, saving, data loading), making code hard to test and reuse.

`rlab` solves this with two decoupled modules:
-   **Tracking:** Automatically captures Git state (base commit + dirty patch) for every MLflow run.
-   **Design:** A set of lightweight, functional-style abstractions (Config, State, Loop) to structure your research code using JAX and Equinox.

## Installation

```bash
# Install core (tracking only)
uv add researchlab

# Install with design features (JAX, Equinox, etc.)
uv add researchlab[design]
```

## Module 1: Tracking (Dirty Run Management)

Automatically tracks the exact code state of your experiments.

### Usage

```python
from researchlab import ExperimentTracker

# Use as a context manager
with ExperimentTracker(experiment_name="my_project") as tracker:
    # Logs params and captures changes to the config file in the patch
    tracker.log_config("params.yaml")

    # Your training logic here...
    print(f"Running experiment: {tracker.run_name}")
```

### CLI for Code Management

`rlab` provides a CLI to restore experiment branches.

```bash
# List restored branches
rlab list

# Restore a specific run (creates a branch and applies the patch)
rlab restore 2026-02-15_radiant-octopus

# See the diff between two runs
rlab diff run_id_1 run_id_2
```

## Module 2: Design (Functional Research Patterns)

A functional, state-centric approach to research code, heavily inspired by JAX patterns.

### Core Concepts

1.  **State (The Snapshot):** Single source of truth for simulation state (parameters, optimizer, step). Must be a JAX PyTree.
2.  **Config (Hyperparameters):** Immutable configuration object.
3.  **Pure Kernels (The Logic):** Pure functions that operate on data.
4.  **Selectors (The Mapping):** Decorators that map `State` and `Config` to kernel arguments.
5.  **Infrastructure (The Side Effects):** Data loading, Telemetry, Persistence (isolated from core logic).

### Usage Example

```python
import jax.numpy as jnp
from researchlab.design.core import Config, State, FieldSelector
from researchlab.design.orchestrator import Loop
from researchlab.design.infra import MLFlowTelemetry, EquinoxPersister

# 1. Define State and Config
class TrainState(State):
    params: jnp.ndarray
    step: int

class TrainConfig(Config):
    lr: float = 1e-3

# 2. Define Pure Kernel
# Use FieldSelector to map State/Config fields to function arguments
@FieldSelector(w="state.params", lr="config.lr")
def update_step(w, lr):
    return w - lr * 0.1, {"loss": 0.5}

# 3. Setup Loop
state = TrainState(params=jnp.ones(10), step=0)
config = TrainConfig()

loop = Loop(
    config=config,
    initial_state=state,
    step_fn=update_step,
    data_provider=MyDataProvider(),
    telemetry=MLFlowTelemetry(),
    persister=EquinoxPersister(),
)

loop.run(num_steps=100)
```

## How it works

-   **Tracking:** Uses `git add -N` (intent-to-add) to include untracked files in the patch. Stores `rlab.base_commit` and `rlab.run_id` as MLflow tags.
-   **Design:** Leverages `equinox` for PyTree management and serialization (Safetensors). Configurations are Pydantic models for robust validation.
