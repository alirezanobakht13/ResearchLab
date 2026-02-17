from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
from pydantic import BaseModel, ConfigDict

# -----------------------------------------------------------------------------
# Core Components
# -----------------------------------------------------------------------------


class Config(BaseModel):
    """Base class for immutable hyperparameters.

    This class serves as a foundation for configuration objects. It uses Pydantic
    for schema validation and serialization. Configurations are immutable (frozen)
    to ensure reproducibility.

    Example:
        >>> class TrainingConfig(Config):
        ...     learning_rate: float = 1e-3
        ...     batch_size: int = 32
        >>>
        >>> config = TrainingConfig(learning_rate=0.01)
        >>> config.model_dump()
        {'learning_rate': 0.01, 'batch_size': 32}
    """

    model_config = ConfigDict(frozen=True)


class State(eqx.Module):
    """Abstract base class for simulation state.

    States must be JAX PyTrees, which is handled automatically by inheriting
    from `equinox.Module`. This ensures compatibility with JAX transformations
    like `jax.jit`, `jax.grad`, and `jax.vmap`.

    Example:
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>>
        >>> class EnvState(State):
        ...     position: jnp.ndarray
        ...     velocity: jnp.ndarray
        >>>
        >>> state = EnvState(position=jnp.zeros(2), velocity=jnp.zeros(2))
    """


@runtime_checkable
class Kernel[P, R](Protocol):
    """Protocol for pure functions representing the core logic.

    Kernels are pure functions that take specific inputs (defined by `P`) and
    return a result (defined by `R`). They should be free of side effects and
    depend only on their inputs, making them easy to test and transform.
    """

    def __call__(self, *args: P, **kwargs: Any) -> R: ...


# -----------------------------------------------------------------------------
# Selector / Lenses
# -----------------------------------------------------------------------------


class SelectedKernel[S: State, C: Config, R](eqx.Module):
    """The wrapped kernel returned by a Selector.

    This class wraps a pure kernel function and an extractor. When called with
    `State` and `Config` objects, it uses the extractor to derive the arguments
    for the kernel and then executes the kernel.

    Attributes:
        raw: The original pure kernel function.
    """

    _func: Callable[..., R]
    _extractor: Callable[[S, C], tuple[tuple[Any, ...], dict[str, Any]]]

    def __call__(self, state: S, config: C) -> R:
        """Executes the kernel using data extracted from state and config.

        Args:
            state: The current simulation state.
            config: The experiment configuration.

        Returns:
            The result of the kernel function.
        """
        args, kwargs = self._extractor(state, config)
        return self._func(*args, **kwargs)

    @property
    def raw(self) -> Callable[..., R]:
        """Access the original pure function.

        Returns:
            The underlying kernel function before it was wrapped.
        """
        return self._func


class Selector[S: State, C: Config]:
    """A decorator/higher-order function to bind State and Config to Kernel arguments.

    A `Selector` defines how to extract arguments for a kernel from the broader
    `State` and `Config` context. It decouples the kernel's signature from the
    application's data structure.

    Args:
        extractor: A function that takes `(state, config)` and returns a tuple
            `((args...), {kwargs...})` containing the arguments to be passed
            to the kernel.

    Example:
        >>> class MyState(State):
        ...     val: int
        >>> class MyConfig(Config):
        ...     factor: int
        >>>
        >>> # Define how to extract arguments
        >>> def my_extractor(state, config):
        ...     return (state.val, config.factor), {}
        >>>
        >>> # Wrap the kernel
        >>> @Selector(my_extractor)
        ... def compute(val, factor):
        ...     return val * factor
        >>>
        >>> state = MyState(val=10)
        >>> config = MyConfig(factor=2)
        >>> compute(state, config)
        20
    """

    def __init__(
        self,
        extractor: Callable[[S, C], tuple[tuple[Any, ...], dict[str, Any]]],
    ):
        """Initializes the Selector with an extractor function."""
        self.extractor = extractor

    def __call__[R](self, func: Callable[..., R]) -> SelectedKernel[S, C, R]:
        """Wraps a kernel function.

        Args:
            func: The pure kernel function to wrap.

        Returns:
            A `SelectedKernel` that can be called with `(state, config)`.
        """
        return SelectedKernel(func, self.extractor)


class FieldSelector[S: State, C: Config](Selector[S, C]):
    """A simplified Selector using dot-notation strings for mapping.

    This selector allows you to define a mapping from kernel argument names to
    dot-notation paths within the `State` or `Config` objects. It supports
    nested attributes (e.g., `state.sub.val`) and dictionary/sequence access
    (e.g., `state.dict.key`, `state.list.0`).

    Args:
        **mappings: Keyword arguments where keys are the kernel argument names
            and values are dot-notation strings starting with 'state.' or 'config.'.

    Example:
        >>> class MyState(State):
        ...     val: int
        >>> class MyConfig(Config):
        ...     factor: int
        >>>
        >>> @FieldSelector(a="state.val", b="config.factor")
        ... def multiply(a, b):
        ...     return a * b
        >>>
        >>> state = MyState(val=5)
        >>> config = MyConfig(factor=3)
        >>> multiply(state, config)
        15
    """

    def __init__(self, **mappings: str):
        """Initializes the FieldSelector with path mappings."""

        def extractor(state: S, config: C) -> tuple[tuple[Any, ...], dict[str, Any]]:
            kwargs = {}
            for arg_name, path in mappings.items():
                parts = path.split(".")
                root_name = parts[0]
                if root_name == "state":
                    obj = state
                elif root_name == "config":
                    obj = config
                else:
                    raise ValueError(f"Path must start with 'state' or 'config', got '{path}'")

                for part in parts[1:]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        # Handle dictionary/list/tuple access
                        try:
                            obj = obj[part]
                        except (TypeError, KeyError, IndexError):
                            # Try integer index for sequences
                            try:
                                idx = int(part)
                                obj = obj[idx]
                            except (ValueError, TypeError, IndexError, KeyError):
                                raise AttributeError(
                                    f"Could not resolve path '{path}': '{part}' not found."
                                ) from None
                kwargs[arg_name] = obj
            return (), kwargs

        super().__init__(extractor)
