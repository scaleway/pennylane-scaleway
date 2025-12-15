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

from qiskit_scaleway.backends import IqmBackend

from pennylane_scaleway.scw_device import ScalewayDevice


@simulator_tracking  # update device.tracker with some relevant information
@single_tape_support  # add support for device.execute(tape) in addition to device.execute((tape,))
class IqmDevice(ScalewayDevice):
    """
    Scaleway's device to run Pennylane circuits on IQM platforms.

    This device is based on superconducting qubits.
    """

    name = "scaleway.iqm"
    backend_type = IqmBackend

    operations = None
    observables = None

    def __init__(self, wires=None, shots=None, seed=None, **kwargs):
        """
        Params:

            shots (int): number of circuit evaluations/random samples used by default. This is override by the circuit's own shots, which is the proper way to declare shots.
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
                - run_options (dict): Any options supported by qiskit's Backend and Job submission logic (noise_model, memory, etc.) (optional).

        Example:
            ```python
            import pennylane as qml

            with qml.device("scaleway.iqm",
                project_id=<your-project-id>,
                secret_key=<your-secret-key>,
                backend="QPU-SIRIUS-24PQ"
            ) as dev:
                @qml.set_shots(100)
                @qml.qnode(dev)
                def circuit():
                    qml.Hadamard(wires=0)
                    qml.CNOT(wires=[0, 1])
                    return qml.counts()
                print(circuit())
            ```
        """

        super().__init__(wires=wires, kwargs=kwargs, shots=shots, seed=seed)
