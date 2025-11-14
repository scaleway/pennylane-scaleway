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

import pytest
import os
import numpy as np

from pennylane.exceptions import PennyLaneDeprecationWarning
import pennylane as qml

# Credentials
SCW_PROJECT_ID = os.environ["SCW_PROJECT_ID"]
SCW_SECRET_KEY = os.environ["SCW_SECRET_KEY"]
SCW_BACKEND_NAME = os.getenv("SCW_BACKEND_NAME", "aer_simulation_pop_c16m128")
SCW_API_URL = os.getenv("SCW_API_URL")

SHOTS = 4096


@pytest.fixture(scope="module")
def device_kwargs():
    """Module-scoped fixture for device keyword arguments."""
    return {
        "project_id": SCW_PROJECT_ID,
        "secret_key": SCW_SECRET_KEY,
        "url": SCW_API_URL,
        "backend": SCW_BACKEND_NAME,
    }


@pytest.fixture
def device_2wires(device_kwargs):
    """Fixture for a 2-wire device, auto-stopped after test."""
    with qml.device("scaleway.aer", wires=2, **device_kwargs) as dev:
        yield dev


def test_device_instantiation(device_kwargs):
    """Test basic device loading and session start/stop."""
    print(device_kwargs)
    with qml.device("scaleway.aer", wires=2, **device_kwargs) as dev:
        assert dev.name == "scaleway.aer"
        assert dev.num_wires == 2
        assert dev._session_id is not None
        assert dev._platform.name == SCW_BACKEND_NAME

    # After 'with' block, session should be stopped
    assert dev._session_id is None

    # Test that calling stop() again raises an error
    with pytest.raises(RuntimeError, match="No session running"):
        dev.stop()


def test_invalid_device_manipulation(device_kwargs):
    """Test invalid device manipulation."""

    device_kwargs_copy = device_kwargs.copy()

    device_kwargs_copy["useless_key"] = "useless_value"
    with pytest.warns(
        UserWarning, match="The following keyword arguments are not supported by "
    ):
        device = qml.device("scaleway.aer", wires=2, **device_kwargs_copy)
        max_qubits = device._platform.num_qubits
    device_kwargs_copy.pop("useless_key")

    with pytest.warns(UserWarning, match="Number of wires "):
        qml.device("scaleway.aer", wires=max_qubits + 1, **device_kwargs_copy)

    device = qml.device("scaleway.aer", wires=2, **device_kwargs_copy)
    with pytest.raises(RuntimeError, match="No session running."):
        device.stop()

    device_kwargs_copy["backend"] = "invalid_backend"
    with pytest.raises(ValueError, match="Platform "):
        qml.device("scaleway.aer", wires=2, **device_kwargs_copy)


def test_bell_state_probs(device_2wires):
    """Tests probs() for a Bell state."""

    @qml.set_shots(SHOTS)
    @qml.qnode(device_2wires)
    def circuit_probs():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    probs = circuit_probs()

    # Result should be |00> and |11>
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (4,)
    assert np.isclose(probs[0], 0.5, atol=0.05)  # Allow 5% tolerance for shots
    assert np.isclose(probs[1], 0.0, atol=0.05)
    assert np.isclose(probs[2], 0.0, atol=0.05)
    assert np.isclose(probs[3], 0.5, atol=0.05)
    assert np.isclose(np.sum(probs), 1.0)


def test_bell_state_expval_analytic(device_2wires):
    """Tests expval() with shots=None (analytic)."""

    @qml.qnode(device_2wires)
    def circuit_expval():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        # <Z(0)> for |00>+|11> is 0
        return qml.expval(qml.PauliZ(0))

    with pytest.warns(UserWarning, match="analytic calculation"):
        expval = circuit_expval()

    assert np.isclose(expval, 0.0, atol=0.1)


def test_bell_state_expval_shots(device_2wires):
    """Tests expval() with shots (Estimator)."""

    @qml.set_shots(SHOTS)
    @qml.qnode(device_2wires)
    def circuit_expval():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        # <Z(0) @ Z(1)> for |00>+|11> is 1
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    expval = circuit_expval()

    # Aer simulator should return the exact value even with shots
    assert np.isclose(expval, 1.0, atol=1e-8)


def test_hadamard_counts(device_kwargs):
    """Tests counts() measurement."""

    with pytest.warns(
        PennyLaneDeprecationWarning, match="Setting shots on device is deprecated"
    ):
        # Use a specific shot count for this test
        with qml.device("scaleway.aer", wires=1, shots=100, **device_kwargs) as dev:

            @qml.qnode(dev)
            def circuit_counts():
                qml.Hadamard(wires=0)
                qml.Hadamard(wires=0)  # H.H = I, state is |0>
                return qml.counts(wires=0)

            counts = circuit_counts()

            assert isinstance(counts, dict)
            assert counts == {"0": 100}


def test_hadamard_samples(device_kwargs):
    """Tests sample() measurement."""

    with pytest.warns(
        PennyLaneDeprecationWarning, match="Setting shots on device is deprecated"
    ):
        with qml.device("scaleway.aer", wires=1, shots=50, **device_kwargs) as dev:

            @qml.qnode(dev)
            def circuit_samples():
                qml.Hadamard(wires=0)
                qml.Hadamard(wires=0)  # State is |0>
                return qml.sample(wires=0)

            samples = circuit_samples()

            assert isinstance(samples, np.ndarray)
            assert samples.shape == (50, 1)
            assert np.all(samples == 0)


def test_pauli_z_variance_analytic(device_2wires):
    """Tests var() with shots=None (analytic)."""

    @qml.qnode(device_2wires)
    def circuit_var():
        qml.Hadamard(wires=0)  # State is |+>
        # Var(Z) for |+> = <Z^2> - <Z>^2 = <I> - 0^2 = 1
        return qml.var(qml.PauliZ(0))

    var = circuit_var()

    # Estimated variance should be close to 1
    assert np.isclose(var, 1.0, atol=0.01)


def test_pauli_z_variance_shots(device_2wires):
    """Tests var() with shots."""

    @qml.set_shots(SHOTS)
    @qml.qnode(device_2wires)
    def circuit_var():
        qml.Hadamard(wires=0)  # State is |+>
        # Var(Z) for |+> = <Z^2> - <Z>^2 = <I> - 0^2 = 1
        return qml.var(qml.PauliZ(0))

    var = circuit_var()

    # Estimated variance should be close to 1
    assert np.isclose(var, 1.0, atol=0.1)


def test_shot_vector_error(device_kwargs):
    """Tests that a shot vector raises a ValueError."""

    with pytest.warns(
        PennyLaneDeprecationWarning, match="Setting shots on device is deprecated"
    ):
        with qml.device("scaleway.aer", wires=1, shots=10, **device_kwargs) as dev:

            @qml.qnode(dev)
            def circuit():
                qml.Hadamard(wires=0)
                return qml.expval(qml.PauliZ(0))

            circuit()

    with pytest.raises(
        ValueError, match="Only integer number of shots is supported on this device"
    ):
        with qml.device(
            "scaleway.aer", wires=1, shots=[10, 20], **device_kwargs
        ) as dev:

            @qml.qnode(dev)
            def circuit():
                qml.Hadamard(wires=0)
                return qml.expval(qml.PauliZ(0))

            circuit()


def test_mixed_measurement_bell_state(device_2wires):
    """
    Tests the execution of a Bell state circuit returning both
    expectation value (Estimator path) and probabilities (Sampler path).
    """

    @qml.set_shots(SHOTS)
    @qml.qnode(device_2wires)
    def circuit():
        qml.Hadamard(0)
        qml.CNOT([0, 1])
        # Requesting two different types of results
        return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

    result_expval, result_probs = circuit()

    # Theoretical expectation value of Z on the first qubit of a
    # |Phi+> state (1/sqrt(2)(|00> + |11>)) is 0.0
    expected_expval = 0.0

    # Theoretical probabilities are 0.5 for |00> and 0.5 for |11>
    expected_probs = np.array([0.5, 0.0, 0.0, 0.5])

    # 1 / sqrt(2048) approx 0.022. We use 0.1 as a safe buffer.
    epsilon = 0.1

    assert isinstance(result_expval, (float, np.float64, np.float32))
    assert np.isclose(
        result_expval, expected_expval, atol=epsilon
    ), f"Expected expval ~{expected_expval}, but got {result_expval}"

    assert isinstance(result_probs, np.ndarray)
    assert result_probs.shape == (4,), "Probs array has incorrect shape"
    assert np.isclose(np.sum(result_probs), 1.0), "Probabilities must sum to 1.0"
    assert np.allclose(
        result_probs, expected_probs, atol=epsilon
    ), f"Expected probs ~{expected_probs}, but got {result_probs}"


def test_random_circuit(device_kwargs):
    """Test that a large, complex circuit can be executed."""

    n_qubits = 10
    n_operations = 200

    with qml.device("scaleway.aer", wires=n_qubits, **device_kwargs) as dev:

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


def test_variational_circuit(device_kwargs):
    """Test a simple variational circuit to maximize P(|1>)."""

    from scipy.optimize import minimize

    with qml.device("scaleway.aer", wires=1, **device_kwargs) as dev:

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
            np.abs(optimized_param) % (2 * np.pi), np.pi, atol=0.2
        ), f"Expected optimized parameter near pi, got {optimized_param}"

        # Check if the final probability is close to 1.0
        # We use a large tolerance (0.2) because of stochasticity
        final_prob_1 = final_probs[1]
        assert np.allclose(
            final_prob_1, 1.0, atol=0.2
        ), f"Expected P(|1>) ~ 1.0, got {final_prob_1}"
