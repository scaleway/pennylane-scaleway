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

import os
import pennylane as qml
from pennylane import numpy as np
import pytest


SCW_PROJECT_ID = os.environ.get("SCW_PROJECT_ID")
SCW_SECRET_KEY = os.environ.get("SCW_SECRET_KEY")
SCW_API_URL = os.getenv("SCW_API_URL")

TEST_CASES = [("scaleway.iqm", "QPU-SIRIUS-24PQ")]

SHOTS = 100
EPSILON = 0.2


@pytest.fixture(scope="module")
def device_kwargs():
    return {
        "project_id": SCW_PROJECT_ID,
        "secret_key": SCW_SECRET_KEY,
        "url": SCW_API_URL,
    }


@pytest.mark.parametrize("device_name, backend_name", TEST_CASES)
def test_device_instantiation(device_name, backend_name, device_kwargs):
    """Test connection and session creation with the IQM backend."""
    with qml.device(device_name, wires=5, backend=backend_name, **device_kwargs) as dev:
        assert dev.name == device_name
        assert dev._platform.name == backend_name
        assert dev._session_id is not None

    assert dev._session_id is None


@pytest.mark.parametrize("device_name, backend_name", TEST_CASES)
def test_bell_state_consistency(device_name, backend_name, device_kwargs):
    """Tests coherence of a Bell state on IQM emulator."""
    with qml.device(device_name, wires=2, backend=backend_name, **device_kwargs) as dev:

        @qml.set_shots(SHOTS)
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        probs = circuit()

    assert np.isclose(probs[0], 0.5, atol=EPSILON)  # |00>
    assert np.isclose(probs[3], 0.5, atol=EPSILON)  # |11>
    assert np.sum(probs[1:3]) < 0.2


# @pytest.mark.parametrize("device_name, backend_name", TEST_CASES)
# def test_multi_measurement(device_name, backend_name, device_kwargs):
#     """Test returning multiple types of measurements (Counts and Expectation)."""

#     with qml.device(device_name, wires=2, backend=backend_name, **device_kwargs) as dev:

#         @qml.set_shots(SHOTS)
#         @qml.qnode(dev)
#         def circuit():
#             qml.PauliX(wires=0)
#             qml.Hadamard(wires=1)
#             # Return counts for wire 0 (deterministic) and expval for wire 1 (random)
#             return qml.counts(wires=0), qml.expval(qml.PauliZ(1))

#         counts_0, expval_1 = circuit()

#         # Wire 0 is X|0> = |1>, so counts should be all '1' in the ideal case
#         nb_1 = counts_0["1"]
#         assert nb_1 / SHOTS > 0.90

#         # Wire 1 is H|0> = |+>, so <Z> should be close to 0
#         assert np.isclose(expval_1, 0.0, atol=EPSILON)


@pytest.mark.parametrize("device_name, backend_name", TEST_CASES)
def test_complex_circuit_execution(device_name, backend_name, device_kwargs):
    """
    Test a slightly deeper circuit to ensure the backend
    compilation/transpilation pipeline holds up.
    """
    with qml.device(device_name, wires=3, backend=backend_name, **device_kwargs) as dev:

        @qml.set_shots(SHOTS)
        @qml.qnode(dev)
        def circuit():
            # GHZ State creation
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            return qml.probs(wires=[0, 1, 2])

        probs = circuit()

        # GHZ state |000> + |111>
        # Index 0 is |000>, Index 7 is |111>
        assert probs[0] > EPSILON
        assert probs[7] > EPSILON
        assert np.sum(probs) > 1.0 - EPSILON
