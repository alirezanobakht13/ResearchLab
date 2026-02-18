from typing import Any

from .core import Config, Kernel, State
from .infra import DataProvider, Persister, Telemetry, Visualizer


class Loop[S: State, C: Config]:
    """A generic training/simulation loop orchestrator.

    The `Loop` class coordinates the interaction between the core components
    (State, Config, Kernel) and the infrastructure components (Data, Telemetry,
    Persistence, Visualization). It manages the execution flow, step counting,
    and periodic tasks.

    Example:
        ```python
        # Assuming components are defined
        loop = Loop(
            config=config,
            initial_state=state,
            step_fn=step_fn,
            data_provider=provider,
            telemetry=telemetry
        )
        loop.run(num_steps=100)
        ```
    """

    def __init__(
        self,
        config: C,
        initial_state: S,
        step_fn: Kernel[tuple[S, C, Any], tuple[S, dict[str, Any]]],
        data_provider: DataProvider[S],
        telemetry: Telemetry[S, C] | None = None,
        persister: Persister[S, C] | None = None,
        visualizer: Visualizer[S] | None = None,
    ):
        """Initializes the loop.

        Args:
            config: Immutable hyperparameters.
            initial_state: Initial simulation state.
            step_fn: A pure kernel function that takes (state, config, batch) and returns (new_state, metrics).
            data_provider: Source of data.
            telemetry: Optional logger.
            persister: Optional checkpointer.
            visualizer: Optional renderer.
        """
        self.config = config
        self.state = initial_state
        self.step_fn = step_fn
        self.data_provider = data_provider
        self.telemetry = telemetry
        self.persister = persister
        self.visualizer = visualizer
        self.step = 0

    def run(self, num_steps: int):
        """Run the loop for a specified number of steps."""
        for _ in range(num_steps):
            self.step += 1
            
            # 1. Get Data
            try:
                batch = self.data_provider.get_batch(self.state)
            except NotImplementedError:
                batch = next(self.data_provider) # Fallback to iterator protocol if implemented
            except StopIteration:
                break

            # 2. Step (Pure Kernel)
            self.state, metrics = self.step_fn(self.state, self.config, batch)

            # 3. Telemetry
            if self.telemetry:
                self.telemetry.log_metrics(metrics, self.step)

            # 4. Persistence (Example: every 1000 steps or similar logic could be added)
            # For simplicity, we expose a manual save method or rely on the user to call it.
            # Here we just show how it *could* be used.

            # 5. Visualization (Example: every N steps)
            if self.visualizer:
                # self.visualizer.render(self.state)
                # self.visualizer.show()
                pass

    def save_checkpoint(self, path: Any): # using Any for path to avoid circular imports if Path is needed
        if self.persister:
            self.persister.save(self.state, self.config, self.step, path)

    def load_checkpoint(self, path: Any):
        if self.persister:
            self.state, self.config, self.step = self.persister.load(path, self.state, self.config)
