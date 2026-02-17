from unittest.mock import patch

import jax.numpy as jnp
import pytest

from researchlab.design.infra import Config, EquinoxPersister, MLFlowTelemetry, State


class MyState(State):
    x: jnp.ndarray
    y: jnp.ndarray


class MyConfig(Config):
    name: str = "test"


def test_equinox_persister_save_load(tmp_path):
    persister = EquinoxPersister()

    # Create state and config
    state = MyState(x=jnp.array([1.0, 2.0]), y=jnp.array(10))
    config = MyConfig(name="test")

    # Save
    save_path = tmp_path / "checkpoint.eqx"
    persister.save(state, config, step=10, path=save_path)

    assert save_path.exists()

    # Load
    structure = MyState(x=jnp.zeros_like(state.x), y=jnp.zeros_like(state.y))

    loaded_state, loaded_config, step = persister.load(save_path, structure, config)

    assert jnp.array_equal(loaded_state.x, state.x)
    assert jnp.array_equal(loaded_state.y, state.y)
    assert loaded_config == config
    assert step == 0


def test_equinox_persister_nested_state(tmp_path):
    class SubState(State):
        val: jnp.ndarray

    class ComplexState(State):
        sub: SubState
        d: dict
        t: tuple

    persister = EquinoxPersister()

    # Create nested state
    sub = SubState(val=jnp.array([1, 2, 3]))
    state = ComplexState(sub=sub, d={"a": jnp.array([4, 5])}, t=(jnp.array([6]), "static_string"))

    config = MyConfig(name="nested")
    save_path = tmp_path / "nested.eqx"

    persister.save(state, config, step=0, path=save_path)

    # Load
    structure_sub = SubState(val=jnp.zeros(3, dtype=jnp.int32))
    structure = ComplexState(
        sub=structure_sub,
        d={"a": jnp.zeros(2, dtype=jnp.int32)},
        t=(jnp.zeros(1, dtype=jnp.int32), "static_string"),
    )

    loaded_state, _, _ = persister.load(save_path, structure, config)

    assert jnp.array_equal(loaded_state.sub.val, state.sub.val)
    assert jnp.array_equal(loaded_state.d["a"], state.d["a"])
    assert jnp.array_equal(loaded_state.t[0], state.t[0])
    assert loaded_state.t[1] == "static_string"


def test_mlflow_telemetry():
    telemetry = MLFlowTelemetry()

    # Test log_metrics
    with patch("mlflow.log_metrics") as mock_log_metrics:
        metrics = {"loss": 0.5, "accuracy": jnp.array(0.95)}  # JAX scalar
        telemetry.log_metrics(metrics, step=100)

        args, kwargs = mock_log_metrics.call_args
        logged_metrics = args[0]
        step = kwargs["step"]

        assert step == 100
        assert logged_metrics["loss"] == 0.5
        assert logged_metrics["accuracy"] == pytest.approx(0.95)
        assert isinstance(logged_metrics["accuracy"], float)

    # Test log_params
    with patch("mlflow.log_param") as mock_log_param:
        params = {"lr": 0.01, "net": {"layers": 2, "dims": [64, 64]}}
        telemetry.log_params(params)

        # Verify calls
        calls = mock_log_param.call_args_list
        # Should be called 3 times: lr, net.layers, net.dims
        logged = {c[0][0]: c[0][1] for c in calls}

        assert logged["lr"] == 0.01
        assert logged["net.layers"] == 2
        assert logged["net.dims"] == [64, 64]
