# Design Module

The `researchlab.design` module provides a set of abstractions to structure machine learning research code. It is not just a utility library; it embodies a specific **design philosophy** inspired by functional programming and the principles of separation of concerns.

This module is most useful if you adopt this philosophy, which strictly separates your "Pure Math" from your "Infrastructure".

## Philosophy: Core vs. Infrastructure

Research code often becomes a tangled mess where training logic is mixed with logging, checkpointing, and data loading. This makes code hard to test, debug, and reuse.

`rlab` proposes a clean split:

1.  **The Core (Pure & Deterministic):** Contains only data and math. It has _no knowledge_ of disks, networks, or GPUs. It is side-effect free.
2.  **The Infrastructure (Impure & Resourceful):** Handles the "Real World" (saving files, logging to MLflow, loading data). It has internal state (like cache or network connections) but is never part of the mathematical snapshot.
3.  **The Orchestrator:** The only place where Core and Infrastructure meet.

## 1. The Core (Pure & Deterministic)

This layer defines _what_ your experiment is mathematically. It should be easily unit-testable without mocking any external services.

### State (`State`)

The **State** is the minimum set of variables required to perfectly resume a run. It represents the "snapshot" of your experiment at any tick.

- **Must be a JAX PyTree:** We use `equinox.Module` to ensure it works with JAX transformations (`jit`, `grad`, `vmap`).
- **Contains:** Model parameters, optimizer state, step counter, RNG keys.
- **Does NOT Contain:** Data loaders, loggers, or file handles.

### Config (`Config`)

The **Config** holds immutable metadata and hyperparameters defined at the start of a run.

- **Immutable:** Should not change during execution.
- **Validated:** We use `pydantic` to ensure types and constraints.

### Kernels & Selectors (`Selector`)

**Kernels** are pure functions that perform the actual logic (e.g., `update_step`, `compute_loss`).

- **Granular:** They should take specific primitive arguments (e.g., `params`, `batch`, `learning_rate`) rather than the whole `State` object. This makes them trivial to test.

**Selectors** act as "lenses" or adapters. They map the complex `State` and `Config` structure to the granular arguments required by Kernels.

- **Decoupling:** Your math doesn't need to know the shape of your `State`. `FieldSelector` lets you wire them up declaratively.

## 2. Infrastructure (Impure & Resourceful)

This layer handles side effects. These components observe the Core but typically do not modify the mathematical state directly (except via data injection).

### Infrastructure Components

- **DataProvider:** Handles high-frequency input (e.g., streaming batches).
- **Telemetry:** Handles medium-frequency output (logging metrics/params). `MLFlowTelemetry` is provided out-of-the-box.
- **Persister:** Handles low-frequency persistence (saving checkpoints). `EquinoxPersister` handles saving the `State` PyTree.
- **Visualizer:** Handles rendering (creating images/videos from State).

## 3. The Orchestrator (`Loop`)

The **Loop** is the glue code. It manages the flow of time and interaction between layers.

It follows a standard cycle:

1.  **Input:** Request data from `DataProvider`.
2.  **Update:** Call the Pure Kernel (wrapped by Selector) with `State`, `Config`, and Data.
3.  **Output:** Receive new `State` and Metrics.
4.  **Side Effects:** Send Metrics to `Telemetry`, save `State` via `Persister`.

By using `researchlab.design`, you simply define your Core and Infrastructure components, and the `Loop` handles the orchestration boilerplate for you.
