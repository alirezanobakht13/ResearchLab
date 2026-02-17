from typing import Any, Callable, Generic, Protocol, TypeVar, runtime_checkable

import equinox as eqx
from pydantic import BaseModel, ConfigDict

# -----------------------------------------------------------------------------
# Type Variables
# -----------------------------------------------------------------------------
S = TypeVar("S", bound="State")
C = TypeVar("C", bound="Config")
P = TypeVar("P")
R = TypeVar("R")


# -----------------------------------------------------------------------------
# Core Components
# -----------------------------------------------------------------------------

class Config(BaseModel):
    """Base class for immutable hyperparameters.

    Implemented as a Pydantic model for schema validation and serialization.
    Should be treated as immutable.
    """
    model_config = ConfigDict(frozen=True)


class State(eqx.Module):
    """Abstract base class for simulation state.

    Must be a JAX PyTree (handled by Equinox).
    """
    pass


@runtime_checkable
class Kernel(Protocol[P, R]):
    """Protocol for pure functions."""
    def __call__(self, *args: P, **kwargs: Any) -> R: ...


# -----------------------------------------------------------------------------
# Selector / Lenses
# -----------------------------------------------------------------------------

class SelectedKernel(eqx.Module, Generic[S, C, R]):
    """The wrapped kernel returned by Selector.
    
    It is an Equinox Module, so it can be JIT-compiled or differentiated
    if the underlying function supports it.
    """
    _func: Callable[..., R]
    _extractor: Callable[[S, C], tuple[tuple[Any, ...], dict[str, Any]]]

    def __call__(self, state: S, config: C) -> R:
        args, kwargs = self._extractor(state, config)
        return self._func(*args, **kwargs)

    @property
    def raw(self) -> Callable[..., R]:
        """Access the original pure function."""
        return self._func


class Selector(Generic[S, C]):
    """A decorator/higher-order function to bind State and Config to Kernel arguments.

    Example:
        @Selector(lambda s, c: ((s.x, c.y), {}))
        def my_kernel(x, y):
            return x + y
        
        # Usage
        result = my_kernel(state, config)
        original_result = my_kernel.raw(1, 2)
    """
    def __init__(
        self,
        extractor: Callable[[S, C], tuple[tuple[Any, ...], dict[str, Any]]],
    ):
        """
        Args:
            extractor: A function that takes (state, config) and returns
                       ((args...), {kwargs...}) to be passed to the kernel.
        """
        self.extractor = extractor

    def __call__(self, func: Callable[..., R]) -> SelectedKernel[S, C, R]:
        return SelectedKernel(func, self.extractor)


class FieldSelector(Selector[S, C]):
    """A simplified Selector that maps kernel arguments to State/Config fields using dot-notation strings.
    
    Example:
        @FieldSelector(x="state.x", y="config.y")
        def my_kernel(x, y):
            return x + y
            
        # state.x -> passed to x
        # config.y -> passed to y
    """
    def __init__(self, **mappings: str):
        """
        Args:
            **mappings: keys are kernel argument names, values are dot-notation paths
                        starting with 'state.' or 'config.'.
        """
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
                                 raise AttributeError(f"Could not resolve path '{path}': '{part}' not found.")
                kwargs[arg_name] = obj
            return (), kwargs
            
        super().__init__(extractor)
