# Copyright 2025 Scaleway
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import pytest

import pennylane as qml
from pennylane.exceptions import PennyLaneDeprecationWarning

# Credentials
SCW_PROJECT_ID = os.environ["SCW_PROJECT_ID"]
SCW_SECRET_KEY = os.environ["SCW_SECRET_KEY"]
SCW_API_URL = os.getenv("SCW_API_URL")

if SCW_SECRET_KEY in ["fake-token", ""]:
    TEST_CASES = [
        ("scaleway.aer", "EMU-AER-LOCAL"),
        ("scaleway.aqt", "EMU-IBEX-12PQ-LOCAL"),
        # ("scaleway.aqt", "EMU-IBEX-12PQ"),
    ]
else:
    TEST_CASES = [
        ("scaleway.aer", "EMU-AER-16C-128M"),
        ("scaleway.aqt", "EMU-IBEX-12PQ-L4"),
    ]

SHOTS = 4096


# Fixtures
@pytest.fixture(scope="module")
def device_kwargs():
    """Module-scoped fixture for device keyword arguments."""
    return {
        "project_id": SCW_PROJECT_ID,
        "secret_key": SCW_SECRET_KEY,
        "url": SCW_API_URL,
    }


@pytest.mark.parametrize("device_name, backend_name", TEST_CASES)
def test_device_instantiation(device_name, backend_name, device_kwargs):
    """Test basic device loading and session start/stop."""

    with qml.device(device_name, wires=2, backend=backend_name, **device_kwargs) as dev:
        assert dev.name == device_name
        # assert dev.num_wires == 2
        assert dev._session_id is not None
        assert dev._platform.name == backend_name

    # After 'with' block, session should be stopped
    assert dev._session_id is None

    # Test that calling stop() again raises an error
    with pytest.raises(RuntimeError, match="No session running"):
        dev.stop()


@pytest.mark.parametrize("device_name, backend_name", TEST_CASES)
def test_invalid_device_manipulation(device_name, backend_name, device_kwargs):
    """Test invalid device manipulation."""

    device_kwargs_copy = device_kwargs.copy()

    device_kwargs_copy["useless_key"] = "useless_value"
    with pytest.warns(
        UserWarning, match="The following keyword arguments are not supported by "
    ):
        device = qml.device(
            device_name, backend=backend_name, wires=2, **device_kwargs_copy
        )
        # max_qubits = device._platform.num_qubits
    device_kwargs_copy.pop("useless_key")

    # with pytest.warns(UserWarning, match="Number of wires "):
    #     qml.device(
    #         device_name,
    #         backend=backend_name,
    #         wires=max_qubits + 1,
    #         **device_kwargs_copy,
    #     )

    device = qml.device(
        device_name, backend=backend_name, wires=2, **device_kwargs_copy
    )
    with pytest.raises(RuntimeError, match="No session running."):
        device.stop()

    with pytest.raises(ValueError, match="Platform "):
        qml.device(
            device_name, backend="invalid_backend", wires=2, **device_kwargs_copy
        )


@pytest.mark.parametrize("device_name, backend_name", TEST_CASES)
def test_tracker(device_name, backend_name, device_kwargs):
    """Test that the device tracker works correctly."""
    with qml.device(device_name, backend=backend_name, wires=1, **device_kwargs) as dev:

        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=0)
            return qml.probs(wires=0)

        # Check proper initialization, without tracking out of context
        assert not dev.tracker.active
        circuit(4.2)
        assert dev.tracker.history == dev.tracker.latest == dev.tracker.totals == {}

        with dev.tracker:
            assert dev.tracker.active
            assert dev.tracker.history == dev.tracker.latest == dev.tracker.totals == {}

            # Checks simple execution tracking
            circuit(2.1)
            assert dev.tracker.history["executions"] and (
                len(dev.tracker.history["executions"]) == 1
            )
            assert dev.tracker.totals["executions"] and (
                dev.tracker.totals["executions"] == 1
            )
            assert dev.tracker.latest["executions"] and (
                dev.tracker.latest["executions"] == 1
            )

        with dev.tracker:
            # Checks multiple executions at once, as well as persistent tracking between contexts
            circuit([0.0, 0.5, 1.0])
            assert len(dev.tracker.history["executions"]) == 4
            assert dev.tracker.totals["executions"] == 4
            assert dev.tracker.latest["executions"] == 1

            history = dev.tracker.history.copy()
            totals = dev.tracker.totals.copy()
            latest = dev.tracker.latest.copy()

        # Checks that out of context execution is not tracked
        circuit(3.14)
        assert history == dev.tracker.history
        assert totals == dev.tracker.totals
        assert latest == dev.tracker.latest

    # Checks that stopping the device resets its tracker
    assert dev.tracker.history == dev.tracker.latest == dev.tracker.totals == {}


@pytest.mark.parametrize("device_name, backend_name", TEST_CASES)
def test_bell_state_probs(device_name, backend_name, device_kwargs):
    """Tests probs() for a Bell state."""

    with qml.device(device_name, wires=2, backend=backend_name, **device_kwargs) as dev:

        @qml.set_shots(SHOTS)
        @qml.qnode(dev)
        def circuit_probs():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        probs = circuit_probs()

    # Result should be |00> and |11>
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (4,)
    assert np.isclose(
        probs[0], 0.5, atol=0.1
    )  # Allow 10% tolerance to account for statistical noise
    assert np.isclose(probs[1], 0.0, atol=0.1)
    assert np.isclose(probs[2], 0.0, atol=0.1)
    assert np.isclose(probs[3], 0.5, atol=0.1)
    assert np.isclose(np.sum(probs), 1.0)


@pytest.mark.parametrize("device_name, backend_name", TEST_CASES)
def test_hadamard_counts(device_name, backend_name, device_kwargs):
    """Tests counts() measurement."""

    with pytest.warns(
        PennyLaneDeprecationWarning, match="Setting shots on device is deprecated"
    ):
        # Use a specific shot count for this test
        with qml.device(
            device_name, backend=backend_name, wires=1, shots=100, **device_kwargs
        ) as dev:

            @qml.qnode(dev)
            def circuit_counts():
                qml.Hadamard(wires=0)
                qml.Hadamard(wires=0)  # H.H = I, state is |0>
                return qml.counts(wires=0)

            counts = circuit_counts()

            assert isinstance(counts, dict)
            assert counts == {"0": 100}


@pytest.mark.parametrize("device_name, backend_name", TEST_CASES)
def test_hadamard_samples(device_name, backend_name, device_kwargs):
    """Tests sample() measurement."""

    with pytest.warns(
        PennyLaneDeprecationWarning, match="Setting shots on device is deprecated"
    ):
        with qml.device(
            device_name, backend=backend_name, wires=1, shots=50, **device_kwargs
        ) as dev:

            @qml.qnode(dev)
            def circuit_samples():
                qml.Hadamard(wires=0)
                qml.Hadamard(wires=0)  # State is |0>
                return qml.sample(wires=0)

            samples = circuit_samples()

            assert isinstance(samples, np.ndarray)
            assert samples.shape == (50, 1)
            assert np.all(samples == 0)


@pytest.mark.parametrize("device_name, backend_name", TEST_CASES)
def test_random_circuit(device_name, backend_name, device_kwargs):
    """Test that a large, complex circuit can be executed."""

    n_qubits = 10
    n_operations = 200

    with qml.device(
        device_name, backend=backend_name, wires=n_qubits, **device_kwargs
    ) as dev:

        @qml.set_shots(SHOTS)
        @qml.qnode(dev)
        def circuit():
            for i in range(n_qubits):
                qml.Hadamard(i)

            for _ in range(n_operations):
                qml.RX(np.random.rand(), wires=np.random.randint(0, n_qubits))
                qml.RY(np.random.rand(), wires=np.random.randint(0, n_qubits))
                qml.RZ(np.random.rand(), wires=np.random.randint(0, n_qubits))

                if np.random.rand() > 0.5:
                    wires = np.random.choice(n_qubits, size=2, replace=False)
                    qml.CNOT(wires=wires)

            return qml.expval(qml.PauliZ(0))

        result = circuit()

        assert isinstance(result, (float, np.float64, np.float32))
        assert np.isfinite(result)


@pytest.mark.parametrize("device_name, backend_name", TEST_CASES)
def test_variational_circuit(device_name, backend_name, device_kwargs):
    """Test a simple variational circuit to maximize P(|1>)."""

    from scipy.optimize import minimize

    with qml.device(device_name, backend=backend_name, wires=1, **device_kwargs) as dev:

        @qml.set_shots(100)
        @qml.set_shots(SHOTS)
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.probs(wires=0)

        # Objective: Find x to maximize P(|1>), which is circuit(x)[1]
        def objective_fn(x: np.ndarray) -> float:
            probs = circuit(x)
            probs = probs.squeeze()
            return -probs[1]

        result = minimize(objective_fn, x0=np.array([0.0]), method="COBYLA")
        final_probs = circuit(result.x).squeeze()

        assert result.success

        # Check if the parameter is close to pi (modulo 2*pi)
        optimized_param = result.x[0]
        assert np.isclose(
            np.abs(optimized_param) % (2 * np.pi), np.pi, atol=0.5
        ), f"Expected optimized parameter near pi, got {optimized_param}"

        # Check if the final probability is close to 1.0
        # We use a large tolerance (0.2) because of stochasticity
        final_prob_1 = final_probs[1]
        assert np.allclose(
            final_prob_1, 1.0, atol=0.2
        ), f"Expected P(|1>) ~ 1.0, got {final_prob_1}"

