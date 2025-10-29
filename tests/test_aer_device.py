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
from pennylane.devices import Device
from pennylane.tape import QuantumScript

from pennylane_scaleway import AerDevice
from qiskit_scaleway import ScalewayProvider

SCW_PROJECT_ID = os.environ["SCW_PROJECT_ID"]
SCW_SECRET_KEY = os.environ["SCW_SECRET_KEY"]
SCW_API_URL = os.getenv("SCW_API_URL")
SCW_INSTANCE_TYPE = os.getenv("SCW_BACKEND_NAME", "aer_simulation_local")


def _bell_state_circuit(device: Device) -> QuantumScript:
    @qml.qnode(device)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs()
    return circuit


def test_context_manager():
    
    with AerDevice(
            wires=2,
            project_id=SCW_PROJECT_ID,
            secret_key=SCW_SECRET_KEY,
            url=SCW_API_URL,
            instance=SCW_INSTANCE_TYPE,
        ) as device:

        circuit = _bell_state_circuit(device)
        
        result = circuit()

        print(result)

        assert result.shape == (4,)
        assert result[1] == result[2] == 0
    
    raised = False
    try:
        device.stop()
    except RuntimeError:
        raised = True

    assert raised, "Device should be stopped."


def test_instanciation():

    device = qml.device(
        name="scaleway.aer",
        wires=2,
        project_id=SCW_PROJECT_ID,
        secret_key=SCW_SECRET_KEY,
        url=SCW_API_URL,
        instance=SCW_INSTANCE_TYPE,
    )

    circuit = _bell_state_circuit(device)

    result = circuit()

    assert result.shape == (4,)
    assert result[1] == result[2] == 0

    device.stop()

    raised = False
    try:
        device.stop()
    except RuntimeError:
        raised = True

    assert raised, "Device should be stopped."


def test_minimal_method_call():

    from pennylane.devices import ExecutionConfig

    with AerDevice(
            wires=2,
            project_id=SCW_PROJECT_ID,
            secret_key=SCW_SECRET_KEY,
            url=SCW_API_URL,
            instance=SCW_INSTANCE_TYPE,
        ) as device:

        circuit = _bell_state_circuit(device)

        initial_config = ExecutionConfig()
        initial_circuit_batch = [circuit]

        execution_config = device.setup_execution_config(initial_config)
        transform_program = device.preprocess_transforms(execution_config)
        circuit_batch, postprocessing = transform_program(initial_circuit_batch)
        results = device.execute(circuit_batch, execution_config)
        final_results = postprocessing(results)

    print(final_results)

    assert False, "Test not implemented yet."
