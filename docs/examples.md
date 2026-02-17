# Examples

## Combined Usage: Tracking + Design

This example demonstrates how to use the `ExperimentTracker` to ensure reproducibility while using the `design` module's abstractions for a clean training loop.

```python
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import Any

from researchlab import ExperimentTracker
from researchlab.design.core import State, Config, FieldSelector
from researchlab.design.infra import MLFlowTelemetry, EquinoxPersister, DataProvider
from researchlab.design.orchestrator import Loop

# --- 1. Define State and Config ---

class TrainConfig(Config):
    learning_rate: float = 1e-3
    num_steps: int = 100
    batch_size: int = 32

class TrainState(State):
    params: jnp.ndarray
    opt_state: optax.OptState
    step: int

# --- 2. Define Core Logic (Kernels) ---

def loss_fn(params, batch):
    return jnp.mean((params - batch) ** 2)

# Pure update function
# Note: It only takes what it needs (params, opt_state, batch, learning_rate)
# It doesn't know about 'TrainState' or 'TrainConfig' classes.
def update_step(params, opt_state, batch, learning_rate):
    optimizer = optax.sgd(learning_rate)
    grads = jax.grad(loss_fn)(params, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    metrics = {"loss": loss_fn(params, batch)}
    return (new_params, new_opt_state), metrics

# --- 3. Wire it up with Selectors ---

# We use FieldSelector to map:
# state.params -> params
# state.opt_state -> opt_state
# config.learning_rate -> learning_rate
# batch is passed implicitly as the 3rd argument from DataProvider
@FieldSelector(
    params="state.params",
    opt_state="state.opt_state",
    learning_rate="config.learning_rate"
)
def step_wrapper(params, opt_state, learning_rate, batch):
    # This wrapper matches the signature expected by Loop: (state, config, batch) -> (new_state, metrics)
    # But wait, the Selector passes extracted args to the decorated function.
    # The decorated function 'step_wrapper' receives (params, opt_state, learning_rate).
    # Where does 'batch' come from?
    #
    # The Loop calls: step_fn(state, config, batch).
    # The Selector extracts (params, opt_state, learning_rate) from state/config.
    # It passes them as *args to the decorated function.
    # Any *extra* arguments passed to the call (like 'batch') are passed as *args AFTER the extracted ones.
    # So step_wrapper receives: (params, opt_state, learning_rate, batch).

    (new_params, new_opt_state), metrics = update_step(params, opt_state, batch, learning_rate)

    # We must return a new State object (and metrics)
    # Since we are inside the wrapper, we don't have access to the original 'state' object directly
    # to use eqx.tree_at or similar easily unless we pass it.
    # A common pattern is to reconstruct the state or use a helper.
    # For this example, let's assume we reconstruct it (simple) or the user handles state updates differently.

    # Simplification: Let's assume we just return the raw new values and the Loop or State handles update?
    # No, Loop expects (new_state, metrics).
    # So we need to construct TrainState.

    # Actually, a better pattern for JAX/Equinox state updates often involves
    # returning the modified state.

    # Let's adjust the wrapper to simply assume we can reconstruct:
    new_state = TrainState(params=new_params, opt_state=new_opt_state, step=0) # Step is updated by Loop
    return new_state, metrics

# --- 4. Dummy Infrastructure ---

class RandomDataProvider(DataProvider[TrainState]):
    def __next__(self):
        return jnp.ones(32)
    def get_batch(self, state):
        return jnp.ones(32)

# --- 5. The Main Execution ---

def main():
    # Start tracking
    with ExperimentTracker(experiment_name="demo_project") as tracker:

        # Log config
        config = TrainConfig()
        tracker.log_config("config.yaml") # Or use telemetry.log_params(config.model_dump())

        # Initialize state
        params = jnp.zeros(10)
        optimizer = optax.sgd(config.learning_rate)
        opt_state = optimizer.init(params)
        state = TrainState(params=params, opt_state=opt_state, step=0)

        # Setup infrastructure
        # MLFlowTelemetry will automatically log to the run started by ExperimentTracker
        telemetry = MLFlowTelemetry()
        telemetry.log_params(config.model_dump())

        # Setup Loop
        loop = Loop(
            config=config,
            initial_state=state,
            step_fn=step_wrapper,
            data_provider=RandomDataProvider(),
            telemetry=telemetry,
            persister=EquinoxPersister()
        )

        # Run
        print("Starting training...")
        loop.run(num_steps=config.num_steps)
        print("Done.")

if __name__ == "__main__":
    main()
```
