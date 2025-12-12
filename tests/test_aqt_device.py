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

if SCW_SECRET_KEY in ["fake-token", "", None]:
    TEST_CASES = [("scaleway.aqt", "EMU-IBEX-12PQ-LOCAL")]
else:
    TEST_CASES = [("scaleway.aqt", "EMU-IBEX-12PQ-L4")]

SHOTS = 1000


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
    """Test connection and session creation with the AQT backend."""
    with qml.device(device_name, wires=2, backend=backend_name, **device_kwargs) as dev:
        assert dev.name == device_name
        assert dev._session_id is not None
        assert dev._platform.name == backend_name

    # Ensure session is cleaned up
    assert dev._session_id is None


@pytest.mark.parametrize("device_name, backend_name", TEST_CASES)
def test_bell_state_probs(device_name, backend_name, device_kwargs):
    """Tests probs() for a Bell state on AQT."""
    with qml.device(device_name, wires=2, backend=backend_name, **device_kwargs) as dev:

        @qml.set_shots(SHOTS)
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        probs = circuit()

    # Expecting |00> and |11>
    assert np.isclose(probs[0], 0.5, atol=0.15)  # |00>
    assert np.isclose(probs[1], 0.0, atol=0.15)  # |01>
    assert np.isclose(probs[2], 0.0, atol=0.15)  # |10>
    assert np.isclose(probs[3], 0.5, atol=0.15)  # |11>


@pytest.mark.parametrize("device_name, backend_name", TEST_CASES)
def test_tracker_integration(device_name, backend_name, device_kwargs):
    """Test that the device tracker records AQT usage correctly."""
    with qml.device(device_name, backend=backend_name, wires=1, **device_kwargs) as dev:

        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.counts()

        with dev.tracker:
            circuit()
            assert dev.tracker.totals["executions"] == 1


@pytest.mark.parametrize("device_name, backend_name", TEST_CASES)
def test_batch_execution(device_name, backend_name, device_kwargs):
    """Test executing multiple circuits in a batch (list of tapes)."""
    with qml.device(device_name, backend=backend_name, wires=2, **device_kwargs) as dev:

        tape1 = qml.tape.QuantumTape(
            [qml.Hadamard(wires=0)], [qml.expval(qml.PauliZ(0))], shots=SHOTS
        )
        tape2 = qml.tape.QuantumTape(
            [qml.PauliX(wires=0)], [qml.expval(qml.PauliZ(0))], shots=SHOTS
        )

        # Execute batch
        results = dev.execute([tape1, tape2])

        # H|0> -> Z expval should be 0
        # X|0> -> |1> -> Z expval should be -1
        assert len(results) == 2
        assert np.isclose(results[0], 0.0, atol=0.2)
        assert np.isclose(results[1], -1.0, atol=0.2)


@pytest.mark.parametrize("device_name, backend_name", TEST_CASES)
def test_variational_optimization(device_name, backend_name, device_kwargs):
    """Test a simple optimization loop to ensure gradients (parameter-shift) work over API."""

    with qml.device(device_name, wires=1, backend=backend_name, **device_kwargs) as dev:

        @qml.set_shots(100)
        @qml.qnode(dev)
        def circuit(theta):
            qml.Hadamard(wires=0)
            qml.RY(theta, wires=0)
            return qml.expval(qml.PauliZ(0))

        # Optimize to flip the qubit from |+> towrds |1> (Target -1.0)
        opt = qml.GradientDescentOptimizer(stepsize=0.4)
        theta = np.array(0.0, requires_grad=True)

        for _ in range(3):
            theta = opt.step(circuit, theta)

        final_val = circuit(theta)

        assert final_val < 0.0
