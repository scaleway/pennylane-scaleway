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
    SCW_BACKEND_NAME = "EMU-AER-LOCAL"
else:
    SCW_BACKEND_NAME = "EMU-AER-16C-128M"

SHOTS = 4096
EPSILON = 0.2  # High because we test noisy devices too.


# Fixtures
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

    assert np.isclose(expval, 0.0, atol=EPSILON)


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
    assert np.isclose(expval, 1.0, atol=EPSILON)


def test_pauli_z_variance_analytic(device_2wires):
    """Tests var() with shots=None (analytic)."""

    @qml.qnode(device_2wires)
    def circuit_var():
        qml.Hadamard(wires=0)  # State is |+>
        # Var(Z) for |+> = <Z^2> - <Z>^2 = <I> - 0^2 = 1
        return qml.var(qml.PauliZ(0))

    var = circuit_var()

    # Estimated variance should be close to 1
    assert np.isclose(var, 1.0, atol=EPSILON)


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
    assert np.isclose(var, 1.0, atol=EPSILON)


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

    assert isinstance(result_expval, (float, np.float64, np.float32))
    assert np.isclose(
        result_expval, expected_expval, atol=EPSILON
    ), f"Expected expval ~{expected_expval}, but got {result_expval}"

    assert isinstance(result_probs, np.ndarray)
    assert result_probs.shape == (4,), "Probs array has incorrect shape"
    assert np.isclose(np.sum(result_probs), 1.0), "Probabilities must sum to 1.0"
    assert np.allclose(
        result_probs, expected_probs, atol=EPSILON
    ), f"Expected probs ~{expected_probs}, but got {result_probs}"
