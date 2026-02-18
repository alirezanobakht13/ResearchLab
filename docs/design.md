# Design Module

The `researchlab.design` module provides a set of abstractions to structure machine learning research code. It is not just a utility library; it embodies a specific **design philosophy** inspired by functional programming and the principles of separation of concerns and Reinforcement Learning in mind.

## 1. The core (Pure & Deterministic)

This section contains only data and math. It has no knowledge of disks, networks, or GPUs.

### Hyperparameters (config)

- Immutable metadata defined at the start of a run.
- Constant during the run.
- Should be easily saved and loaded (e.g. yaml, json, etc.)

### State (The Snapshot)

- The minimum set of variables required to perfectly resume a run.
- Should support nested structures and be easily vectorized. (e.g. using PyTree)
- Should be easily saved and loaded (e.g. safetensors, msgpack, etc.)
- Some included components:
  - Model params (e.g. equinox object)
  - Optimizer state
  - Experience (Memory) state
  - Clock (step, episode, epoch, wall time) state
  - RNG state

### Pure Kernels (The Logic)

- Granular Functions: Functions that take specific primitive arguments (e.g., compute_loss(pixels, weights)). This makes testing trivial.
- State-Aware Wrappers (Lenses): Higher-order functions (decorators) or "selectors" that extract the necessary data from the State and Hyperparameters to call the granular functions (e.g., compute_loss(state)). This provides convenience for the main logic while keeping the core computations testable and profiled.
- They should be side-effect free and vectorizable.

## 2. Infrastructure (Impure & Resourceful)

This section handles the "Real World." These objects can have internal state (caching) but are never included in the "Core State" snapshot.

### Data Providers (The Source)

- High-frequency input (e.g. handles the loading of large image patches).

### Telemetry (The Reporter)

- Medium-frequency output. Handles logging, metrics, and debugging.
- Exact use or a wrapper around stuff like mlflow, tensorboard, wandb, etc.
- Observes the State and Config but cannot modify them.

### Persister (The Vault)

- Low-frequency persistence. Handles saving/loading.
- Ensures atomic saves.

### Visualizer (The Renderer)

- Handles rendering and visualization of the environments, video recording, etc.
- Operates on the State to produce frames (not able to change State); strictly separated from the training math.

## 3. The Orchestrator (The Loop)

This is the only place where the Core and Infrastructure meet.

- Example logic:
  1. Request data from Provider based on current State.clock.
  2. Call the Pure Kernel with State, Config, and the Provider Payload.
  3. Receive a New State and Metrics.
  4. Send Metrics to Telemetry.
  5. Periodically send State and Config to Persister.
