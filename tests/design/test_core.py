import pytest
from pydantic import ValidationError
import equinox as eqx
import jax
import jax.numpy as jnp

from researchlab.design.core import Config, Selector, FieldSelector, State, SelectedKernel

def test_config_immutability():
    class MyConfig(Config):
        lr: float = 0.01

    c = MyConfig()
    
    with pytest.raises(ValidationError):
        c.lr = 0.02

def test_selector_functionality():
    class MyState(State):
        x: int
    
    class MyConfig(Config):
        y: int

    # Define a pure kernel
    def add(a, b):
        return a + b

    # Define an extractor
    def extractor(state: MyState, config: MyConfig):
        return (state.x, config.y), {}

    # Create the selector
    selector = Selector(extractor)
    
    # Wrap the kernel
    wrapped_add = selector(add)

    # Test
    state = MyState(x=10)
    config = MyConfig(y=5)
    
    assert wrapped_add(state, config) == 15
    assert wrapped_add.raw(2, 3) == 5
    
    # Check if wrapped kernel is an Equinox Module
    assert isinstance(wrapped_add, eqx.Module)
    # Check if State is an Equinox Module
    assert isinstance(state, eqx.Module)

def test_field_selector_functionality():
    class MyState(State):
        x: int
        nested: dict

    class MyConfig(Config):
        y: int

    # Define a pure kernel
    def calculate(val_x, val_y, val_z):
        return val_x + val_y + val_z

    # Use FieldSelector
    # We map val_x -> state.x, val_y -> config.y, val_z -> state.nested.z
    selector = FieldSelector(
        val_x="state.x",
        val_y="config.y",
        val_z="state.nested.z"
    )

    wrapped_calc = selector(calculate)

    state = MyState(x=10, nested={"z": 100})
    config = MyConfig(y=5)

    assert wrapped_calc(state, config) == 115
    assert wrapped_calc.raw(1, 2, 3) == 6

def test_field_selector_decorator_usage():
    class MyState(State):
        val: int
    
    class MyConfig(Config):
        factor: int

    # Use as decorator
    @FieldSelector(a="state.val", b="config.factor")
    def multiply(a, b):
        return a * b

    state = MyState(val=10)
    config = MyConfig(factor=2)

    # Calling the decorated function with state and config
    result = multiply(state, config)
    assert result == 20

    # Calling the original function
    assert multiply.raw(3, 4) == 12

def test_jit_compatibility():
    class MyState(State):
        x: jnp.ndarray

    class MyConfig(Config):
        factor: int

    # Define kernel
    @FieldSelector(val="state.x", scale="config.factor")
    def compute(val, scale):
        return val * scale

    state = MyState(x=jnp.array([1.0, 2.0]))
    config = MyConfig(factor=3)

    # 1. Test JIT on the raw function
    # Note: raw function takes arrays/scalars directly
    jitted_raw = jax.jit(compute.raw)
    res_raw = jitted_raw(state.x, config.factor)
    assert jnp.array_equal(res_raw, jnp.array([3.0, 6.0]))

    # 2. Test JIT on the decorated kernel via eqx.filter_jit
    @eqx.filter_jit
    def step(k, s, c):
        return k(s, c)

    res_step = step(compute, state, config)
    assert jnp.array_equal(res_step, jnp.array([3.0, 6.0]))

def test_nested_state_selector():
    class SubState(State):
        val: int

    class ComplexState(State):
        x: int
        sub: SubState
        d: dict
        t: tuple

    class EmptyConfig(Config):
        pass

    # Use FieldSelector to access nested fields
    # "state.t.0" accesses the first element of the tuple
    # "state.d.k" accesses key 'k' in dict
    # "state.sub.val" accesses field 'val' in SubState
    @FieldSelector(
        val_x="state.x",
        val_sub="state.sub.val",
        val_dict="state.d.k",
        val_tuple="state.t.0"
    )
    def aggregate(val_x, val_sub, val_dict, val_tuple):
        return val_x + val_sub + val_dict + val_tuple

    sub = SubState(val=20)
    # Using tuple and dict with mixed types
    state = ComplexState(
        x=10, 
        sub=sub, 
        d={"k": 30}, 
        t=(40, "ignored")
    )
    config = EmptyConfig()

    result = aggregate(state, config)
    assert result == 10 + 20 + 30 + 40
