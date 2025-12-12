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

from pennylane.devices.modifiers import simulator_tracking, single_tape_support

from qiskit_scaleway.backends import AerBackend

from pennylane_scaleway.scw_device import ScalewayDevice


@simulator_tracking  # update device.tracker with some relevant information
@single_tape_support  # add support for device.execute(tape) in addition to device.execute((tape,))
class AerDevice(ScalewayDevice):
    """
    Scaleway's device to run Pennylane circuits on Aer emulators.

    This device:
        * Supports any operations with explicit PennyLane to Qiskit gate conversions defined in the plugin.
        * Supports both CPU and GPU Aer backends.
        * Approximates analytic calculations by sampling.
        * Does not support Shots vector.
        * Does not support state vector simulation.
        * Does not intrinsically support parameter broadcasting.
    """

    name = "scaleway.aer"
    backend_types = (AerBackend,)

    # operations = set(QISKIT_OPERATION_MAP.keys())
    # observables = {
    #     "PauliX",
    #     "PauliY",
    #     "PauliZ",
    #     "Identity",
    #     "Hadamard",
    #     "Hermitian",
    #     "Projector",
    #     "Prod",
    #     "Sum",
    #     "LinearCombination",
    #     "SProd",
    # }

    def __init__(self, wires, shots=None, seed=None, **kwargs):
        """
        Params:

            wires (int, Iterable): Number of subsystems represented by the device,
                or iterable that contains a unique label for each subsystem.
            shots (int, ~pennylane.measurements.Shots): DEPRECATED, use the qml.set_shot() decorator for each QNode instead. Number of circuit evaluations to run.
            seed (int): Random seed used to initialize the pseudo-random number generator.
            **kwargs:
                - project_id (str): The Scaleway Quantum Project ID.
                - secret_key (str): The API token for authentication with Scaleway.
                - backend (str): The specific quantum backend to run on Scaleway.
                - url (str): The Scaleway API URL (optional).
                - session_name (str): Name of the session (optional).
                - deduplication_id (str): Unique deduplication identifier for session (optional).
                - max_duration (str): Maximum uptime session duration (e.g., "1h", "30m") (optional).
                - max_idle_duration (str): Maximum idle session duration (e.g., "1h", "5m") (optional).
                - Any options supported by qiskit's BackendSamplerV2 or BackendEstimatorV2 primitives (optional).

        Example:
            ```python
            import pennylane as qml

            with qml.device("scaleway.aer",
                wires=2,
                project_id=<your-project-id>,
                secret_key=<your-secret-key>,
                backend="EMU-AER-16C-128M"
            ) as dev:
                @qml.set_shots(512)
                @qml.qnode(dev)
                def circuit():
                    qml.Hadamard(wires=0)
                    qml.CNOT(wires=[0, 1])
                    return qml.counts()
                print(circuit())
            ```
        """

        super().__init__(wires=wires, kwargs=kwargs, shots=shots)
