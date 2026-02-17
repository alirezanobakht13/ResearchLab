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
    """Abstract base class for data loading."""
    
    @abstractmethod
    def __next__(self) -> Any:
        """Return the next batch of data."""
        ...
    
    @abstractmethod
    def get_batch(self, state: S) -> Any:
        """Return a batch of data, potentially dependent on the current state."""
        ...

# -----------------------------------------------------------------------------
# Telemetry
# -----------------------------------------------------------------------------

class Telemetry[S: State, C: Config](ABC):
    """Abstract base class for logging."""
    
    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log scalar metrics."""
        ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        ...

class MLFlowTelemetry[S: State, C: Config](Telemetry[S, C]):
    """Concrete implementation of Telemetry using MLflow.
    
    Assumes an active MLflow run exists (e.g. started by ExperimentTracker).
    """
    
    def __init__(self):
        pass

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
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
        log_flattened_params(params)


# -----------------------------------------------------------------------------
# Persister
# -----------------------------------------------------------------------------

class Persister[S: State, C: Config](ABC):
    """Abstract base class for saving/loading checkpoints."""
    
    @abstractmethod
    def save(self, state: S, config: C, step: int, path: Path) -> None:
        """Save the current state and config."""
        ...

    @abstractmethod
    def load(self, path: Path, state_structure: S, config_structure: C) -> tuple[S, C, int]:
        """Load state and config from a checkpoint.
        
        Args:
            path: Path to the checkpoint file/directory.
            state_structure: A structure (PyTree) matching the state to load.
            config_structure: A structure matching the config to load.
            
        Returns:
            (loaded_state, loaded_config, step)
        """
        ...

class EquinoxPersister[S: State, C: Config](Persister[S, C]):
    """Concrete implementation using equinox.tree_serialise_leaves (safetensors)."""
    
    def save(self, state: S, config: C, step: int, path: Path) -> None:
        """Saves state to a .eqx file.
        
        Config is not saved by EquinoxPersister as it is usually static, 
        but for completeness we could pickle it or verify if config is a PyTree. 

        For now, we only serialize the State using equinox, assuming Config is handled separately
        or is reconstructible.
        """
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(path, state)

    def load(self, path: Path, state_structure: S, config_structure: C) -> tuple[S, C, int]:
        """Loads state from a .eqx file.
        
        Note: This implementation currently only restores State.
        It returns the passed `config_structure` as is, and assumes step=0 
        (since step is not intrinsically stored in the tree leaves unless added to State).
        """
        loaded_state = eqx.tree_deserialise_leaves(path, state_structure)
        return loaded_state, config_structure, 0


# -----------------------------------------------------------------------------
# Visualizer
# -----------------------------------------------------------------------------

class Visualizer[S: State](ABC):
    """Abstract base class for rendering."""
    
    @abstractmethod
    def render(self, state: S) -> Any:
        """Render the current state (e.g., return an image array)."""
        ...
