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
import warnings

from inspect import signature
from typing import List, Union
from tenacity import retry, stop_after_attempt, stop_after_delay

from pennylane.devices import ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.tape import QuantumScriptOrBatch, QuantumScript
from pennylane.transforms.core import TransformProgram

from qiskit.result import Result

from qiskit_scaleway.backends import IqmBackend

from pennylane_scaleway.utils import (
    circuit_to_qiskit,
)
from pennylane_scaleway.scw_device import ScalewayDevice


@simulator_tracking  # update device.tracker with some relevant information
@single_tape_support  # add support for device.execute(tape) in addition to device.execute((tape,))
class IqmDevice(ScalewayDevice):
    """
    This is Scaleway's AQT device.
    It allows to run quantum circuits on Scaleway's AQT emulation backends.

    This device:
        * Follows the same constraints as AerDevice, as it uses qiskit as a common interface with AQT emulators.
        * Has 12 wires available.
        * Supports up to 2000 shots maximum.
        * Does not support more than 2000 operations AFTER decomposition, due to hardware limitation. This translates to roughly 12 wires and ~20 layers deep for a pennylane circuit.
    """

    name = "scaleway.iqm"
    backend_types = (IqmBackend,)

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

        self._default_shots = None
        if isinstance(shots, int):
            self._default_shots = shots

        self._handle_kwargs(**kwargs)

    def _handle_kwargs(self, **kwargs):
        ### Extract runner-specific arguments
        self._run_options = {
            k: v
            for k, v in kwargs.items()
            if k in signature(self._platform.run).parameters.keys()
        }
        [kwargs.pop(k) for k in self._run_options.keys()]
        self._run_options.update(
            {
                "session_name": self._session_options.get("name"),
                "session_max_duration": self._session_options.get("max_duration"),
                "session_max_idle_duration": self._session_options.get(
                    "max_idle_duration"
                ),
            }
        )

        if len(kwargs) > 0:
            warnings.warn(
                f"The following keyword arguments are not supported by '{self.name}' device: {list(kwargs.keys())}",
                UserWarning,
            )

    def preprocess(
        self,
        execution_config: ExecutionConfig | None = None,
    ) -> tuple[TransformProgram, ExecutionConfig]:
        transform_program, config = super().preprocess(execution_config)
        return transform_program, config

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> List:
        if not self._session_id:
            raise RuntimeError(
                "No active session. Please instanciate the device using a context manager, or call start() first. You can also attach to an existing deduplication_id."
            )

        if isinstance(circuits, QuantumScript):
            circuits = [circuits]

        qiskit_circuits = []
        for circuit in circuits:
            qiskit_circuits.append(
                circuit_to_qiskit(
                    circuit, self.num_wires, diagonalize=True, measure=True
                )
            )

        shots = self._default_shots
        if circuits[0].shots and circuits[0].shots.total_shots:
            shots = circuits[0].shots.total_shots

        @retry(stop=stop_after_attempt(3) | stop_after_delay(3 * 60), reraise=True)
        def run() -> Union[Result, List[Result]]:
            return self._platform.run(
                qiskit_circuits,
                session_id=self._session_id,
                shots=shots,
                **self._run_options,
            ).result()

        results = run()
        if isinstance(results, Result):
            results = [results]

        counts = []
        for result in results:
            if isinstance(result.get_counts(), dict):
                counts.append(result.get_counts())
            else:
                counts.extend([count for count in result.get_counts()])

        all_results = []
        for original_circuit, qcirc, count in zip(circuits, qiskit_circuits, counts):
            # Reconstruct the list of samples from the counts dictionary
            samples_list = []
            for key, value in count.items():
                samples_list.extend([key] * value)

            if not samples_list:
                # Handle case with no samples (e.g., 0 shots)
                num_clbits = len(qcirc.clbits)
                samples = np.empty((0, num_clbits), dtype=int)
            else:
                # Convert bitstrings to numpy array of ints, reversing for convention
                samples = np.vstack(
                    [np.array([int(i) for i in s[::-1]]) for s in samples_list]
                )

            # Process the samples according to the measurements in the original circuit
            res = [
                mp.process_samples(samples, wire_order=self.wires)
                for mp in original_circuit.measurements
            ]

            single_measurement = len(original_circuit.measurements) == 1
            res_tuple = res[0] if single_measurement else tuple(res)
            all_results.append(res_tuple)

        return all_results
