from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import equinox as eqx
import mlflow

from researchlab.tracking.utils import log_flattened_params

from .core import Config, State

# -----------------------------------------------------------------------------
# Data Provider
# -----------------------------------------------------------------------------


class DataProvider[S: State](ABC):
    """Abstract base class for providing data batches to the training loop.

    Implementations should handle data loading, preprocessing, and batching.
    The `get_batch` method can optionally depend on the current `State`,
    allowing for curriculum learning or state-dependent sampling.
    """

    @abstractmethod
    def __next__(self) -> Any:
        """Returns the next batch of data from the iterator.

        This method supports the iterator protocol.

        Returns:
            A batch of data (structure depends on implementation).

        Raises:
            StopIteration: If the data stream is exhausted.
        """
        ...

    @abstractmethod
    def get_batch(self, state: S) -> Any:
        """Returns a batch of data, potentially dependent on the current state.

        Args:
            state: The current simulation state.

        Returns:
            A batch of data.
        """
        ...


# -----------------------------------------------------------------------------
# Telemetry
# -----------------------------------------------------------------------------


class Telemetry[S: State, C: Config](ABC):
    """Abstract base class for logging metrics and parameters.

    Telemetry components handle the reporting of experiment results. They should
    be non-intrusive and only observe the state/metrics.
    """

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Logs scalar metrics for a given step.

        Args:
            metrics: A dictionary mapping metric names to values.
            step: The current global step or iteration.
        """
        ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Logs hyperparameters or configuration settings.

        Args:
            params: A dictionary of parameters to log. Nested dictionaries are supported.
        """
        ...


class MLFlowTelemetry[S: State, C: Config](Telemetry[S, C]):
    """Concrete implementation of Telemetry using MLflow.

    This class logs metrics and parameters to an active MLflow run. It assumes
    that an MLflow run has already been started (e.g., by `ExperimentTracker`).

    It automatically flattens nested parameter dictionaries and converts JAX
    scalar arrays to Python floats to ensure compatibility with MLflow.

    Example:
        >>> # Inside an active MLflow run
        >>> telemetry = MLFlowTelemetry()
        >>> telemetry.log_metrics({"loss": 0.5}, step=10)
    """

    def __init__(self):
        """Initializes the MLFlowTelemetry instance."""

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Logs scalar metrics to MLflow.

        Converts JAX/NumPy scalars to Python floats before logging.

        Args:
            metrics: Dictionary of metrics.
            step: Current step.
        """
        # Convert JAX/Numpy scalars to python float for MLflow
        clean_metrics = {}
        for k, v in metrics.items():
            if hasattr(v, "item"):
                try:
                    clean_metrics[k] = float(v.item())
                except (ValueError, TypeError):
                    # Fallback if item() doesn't return a float (e.g. array with >1 element)
                    # We might log it as is and let mlflow handle or error?
                    # Better to ignore or log warning? For now, try best effort.
                    clean_metrics[k] = v
            else:
                clean_metrics[k] = v
        mlflow.log_metrics(clean_metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Logs parameters to MLflow, flattening nested dictionaries.

        Args:
            params: Dictionary of parameters.
        """
        log_flattened_params(params)


# -----------------------------------------------------------------------------
# Persister
# -----------------------------------------------------------------------------


class Persister[S: State, C: Config](ABC):
    """Abstract base class for saving and loading checkpoints.

    Persisters handle the serialization and deserialization of the simulation
    state and configuration.
    """

    @abstractmethod
    def save(self, state: S, config: C, step: int, path: Path) -> None:
        """Saves the current state and config to the specified path.

        Args:
            state: The current simulation state.
            config: The experiment configuration.
            step: The current step count.
            path: The file path or directory to save to.
        """
        ...

    @abstractmethod
    def load(self, path: Path, state_structure: S, config_structure: C) -> tuple[S, C, int]:
        """Loads state and config from a checkpoint.

        Args:
            path: Path to the checkpoint file/directory.
            state_structure: A structure (PyTree) matching the state to load.
            config_structure: A structure matching the config to load.

        Returns:
            A tuple containing `(loaded_state, loaded_config, step)`.
        """
        ...


class EquinoxPersister[S: State, C: Config](Persister[S, C]):
    """Concrete implementation using Equinox's serialization (Safetensors).

    This persister saves the `State` object to a `.eqx` file using `equinox.tree_serialise_leaves`.
    It assumes that the `Config` object is handled separately or reconstructible, as Equinox
    serialization primarily handles array data (leaves).

    Attributes:
        None
    """

    def save(self, state: S, config: C, step: int, path: Path) -> None:
        """Saves state to a .eqx file.

        Config is not saved by EquinoxPersister as it is usually static,
        but for completeness we could pickle it or verify if config is a PyTree.

        For now, we only serialize the State using equinox, assuming Config is handled separately
        or is reconstructible.

        Args:
            state: The state to save.
            config: The config (ignored).
            step: The step (ignored/not saved in file).
            path: The path to save the .eqx file.
        """
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(path, state)

    def load(self, path: Path, state_structure: S, config_structure: C) -> tuple[S, C, int]:
        """Loads state from a .eqx file.

        Note: This implementation currently only restores State.
        It returns the passed `config_structure` as is, and assumes step=0
        (since step is not intrinsically stored in the tree leaves unless added to State).

        Args:
            path: Path to the .eqx file.
            state_structure: Structure of the state to load into.
            config_structure: Structure of the config (returned as is).

        Returns:
            (loaded_state, config_structure, 0)
        """
        loaded_state = eqx.tree_deserialise_leaves(path, state_structure)
        return loaded_state, config_structure, 0


# -----------------------------------------------------------------------------
# Visualizer
# -----------------------------------------------------------------------------


class Visualizer[S: State](ABC):
    """Abstract base class for rendering the simulation state.

    Visualizers produce visual representations of the state, such as images,
    videos, or plots.
    """

    @abstractmethod
    def render(self, state: S) -> Any:
        """Render the current state.

        Args:
            state: The state to render.

        Returns:
            The rendered artifact (e.g., image array).
        """
        ...
