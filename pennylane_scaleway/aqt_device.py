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

from typing import Callable, Sequence, Tuple
import warnings

from pennylane.devices import ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.transforms.core import TransformProgram

from qiskit_scaleway.backends import AqtBackend, AerBackend

from pennylane_scaleway.scw_device import ScalewayDevice


@transform
def limit_aqt_shots(
    tape: QuantumTape, default_shots=None
) -> Tuple[Sequence[QuantumTape], Callable]:
    """
    A transform that checks if the total shots in the tape exceed 2000.
    If so, it warns the user and caps the shots at 2000.
    """

    if tape.shots and tape.shots.total_shots and tape.shots.total_shots > 2000:
        warnings.warn(
            "The number of shots exceeds the limit of 2000 for AQT devices. "
            "Execution will proceed with the maximum allowed shots of 2000.",
            UserWarning,
        )
        new_tape = tape.copy(shots=2000)

        def processing_fn(results: Sequence) -> any:
            return results[0]

        return [new_tape], processing_fn

    return [tape], lambda results: results[0]


@simulator_tracking  # update device.tracker with some relevant information
@single_tape_support  # add support for device.execute(tape) in addition to device.execute((tape,))
class AqtDevice(ScalewayDevice):
    """
    Scaleway's device to run Pennylane circuits on AQT platforms.

    This device:
        * Has 12 qubits available.
        * Supports up to 2000 shots maximum.
        * Does not support more than 2000 operations AFTER decomposition, due to hardware limitation. This translates to roughly 12 qubits and ~20 layers deep for a pennylane circuit.
        * Follows the same constraints as AerDevice, as it uses qiskit as a common interface with AQT emulators.
    """

    name = "scaleway.aqt"
    backend_types = (AqtBackend, AerBackend)

    # operations = {
    #     # native PennyLane operations also native to AQT
    #     "RX",
    #     "RY",
    #     "RZ",
    #     # additional operations not native to PennyLane but present in AQT
    #     "R",
    #     "MS",
    #     # operations not natively implemented in AQT
    #     "BasisState",
    #     "PauliX",
    #     "PauliY",
    #     "PauliZ",
    #     "Hadamard",
    #     "S",
    #     "CNOT",
    #     # adjoint versions of operators are also allowed
    #     "Adjoint(RX)",
    #     "Adjoint(RY)",
    #     "Adjoint(RZ)",
    #     "Adjoint(PauliX)",
    #     "Adjoint(PauliY)",
    #     "Adjoint(PauliZ)",
    #     "Adjoint(Hadamard)",
    #     "Adjoint(S)",
    #     "Adjoint(CNOT)",
    #     "Adjoint(R)",
    #     "Adjoint(MS)",
    # }
    # observables = {
    #     "PauliX",
    #     "PauliY",
    #     "PauliZ",
    #     "Identity",
    #     "Hadamard",
    # }

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

            with qml.device("scaleway.aqt",
                project_id=<your-project-id>,
                secret_key=<your-secret-key>,
                backend="EMU-IBEX-12PQ-L4"
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

        super().__init__(wires=wires, kwargs=kwargs, shots=shots, seed=seed)

    def preprocess(
        self,
        execution_config: ExecutionConfig | None = None,
    ) -> tuple[TransformProgram, ExecutionConfig]:
        transform_program, config = super().preprocess(execution_config)
        transform_program.add_transform(
            limit_aqt_shots, default_shots=self._default_shots
        )
        return transform_program, config
