import jax.numpy as jnp

from researchlab.design.core import Config, State
from researchlab.design.utils import (
    flatten_config,
    flatten_pytree,
    unflatten_config,
    unflatten_pytree,
)


def test_flatten_unflatten_state():
    class SubState(State):
        val: jnp.ndarray

    class ComplexState(State):
        x: int
        sub: SubState
        d: dict
        t: tuple

    sub = SubState(val=jnp.array([1, 2]))
    state = ComplexState(
        x=10, sub=sub, d={"a": 3, "b": jnp.array([4])}, t=(5, jnp.array([6]))
    )

    # Flatten
    flat = flatten_pytree(state)

    # Check keys
    assert flat["x"] == 10
    assert jnp.array_equal(flat["sub.val"], jnp.array([1, 2]))
    assert flat["d.a"] == 3
    assert jnp.array_equal(flat["d.b"], jnp.array([4]))
    assert flat["t.0"] == 5
    assert jnp.array_equal(flat["t.1"], jnp.array([6]))

    # Unflatten
    # We need a structure for unflattening (with placeholders)
    structure_sub = SubState(val=jnp.zeros(2))
    structure = ComplexState(
        x=0, sub=structure_sub, d={"a": 0, "b": jnp.zeros(1)}, t=(0, jnp.zeros(1))
    )

    restored = unflatten_pytree(flat, structure)

    assert restored.x == 10
    assert jnp.array_equal(restored.sub.val, state.sub.val)
    assert restored.d["a"] == 3
    assert jnp.array_equal(restored.d["b"], state.d["b"])
    assert restored.t[0] == 5
    assert jnp.array_equal(restored.t[1], state.t[1])


def test_flatten_unflatten_config():
    class NestedConfig(Config):
        val: int

    class MyConfig(Config):
        x: int
        nested: NestedConfig
        lst: list[int]

    config = MyConfig(x=10, nested=NestedConfig(val=20), lst=[1, 2, 3])

    # Flatten
    flat = flatten_config(config)

    assert flat["x"] == 10
    assert flat["nested.val"] == 20
    assert flat["lst"] == [1, 2, 3]  # Lists are not flattened by default in our impl

    # Unflatten
    restored = unflatten_config(flat, MyConfig)

    assert restored == config
