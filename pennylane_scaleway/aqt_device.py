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
from inspect import signature
from typing import Callable, List, Sequence, Tuple
import warnings

import pennylane as qml
from pennylane.devices import ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import (
    decompose,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from pennylane.tape import QuantumScriptOrBatch, QuantumScript
from pennylane.transforms import broadcast_expand, split_non_commuting, transform
from pennylane.transforms.core import TransformProgram
from pennylane.measurements import Shots

from qiskit.result import Result

from qiskit_scaleway.backends import AqtBackend

from pennylane_scaleway.aer_utils import circuit_to_qiskit, split_execution_types, accepted_sample_measurement
from pennylane_scaleway.scw_device import ScalewayDevice
from pennylane_scaleway.utils import analytic_warning



@simulator_tracking  # update device.tracker with some relevant information
@single_tape_support  # add support for device.execute(tape) in addition to device.execute((tape,))
class AqtDevice(ScalewayDevice):

    name = "scaleway.aqt"
    backend_types = (AqtBackend,)
    operations = {
        # native PennyLane operations also native to AQT
        "RX",
        "RY",
        "RZ",
        # additional operations not native to PennyLane but present in AQT
        "R",
        "MS",
        # operations not natively implemented in AQT
        "BasisState",
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "S",
        "CNOT",
        # adjoint versions of operators are also allowed
        "Adjoint(RX)",
        "Adjoint(RY)",
        "Adjoint(RZ)",
        "Adjoint(PauliX)",
        "Adjoint(PauliY)",
        "Adjoint(PauliZ)",
        "Adjoint(Hadamard)",
        "Adjoint(S)",
        "Adjoint(CNOT)",
        "Adjoint(R)",
        "Adjoint(MS)",
    }
    observables = {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Identity",
        "Hadamard",
    }

    def __init__(self, shots=None, seed=None, **kwargs):
        # self._rng = np.random.default_rng(seed)

        super().__init__(wires=12, kwargs=kwargs, shots=shots)
        self._handle_kwargs(**kwargs)

    def _handle_kwargs(self, **kwargs):

        ### Extract runner-specific arguments
        self._run_options = {
            k: v
            for k, v in kwargs.items()
            if k in signature(self._platform.run).parameters.keys()
        }
        [
            kwargs.pop(k) for k in self._run_options.keys()
        ]
        self._run_options.update(
            {
                "session_name": self._session_options["name"],
                "session_deduplication_id": self._session_options[
                    "deduplication_id"
                ],
                "session_max_duration": self._session_options["max_duration"],
                "session_max_idle_duration": self._session_options[
                    "max_idle_duration"
                ],
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

        transform_program = TransformProgram()
        config = execution_config or ExecutionConfig()
        # config = replace(config, use_device_gradient=False)

        transform_program.add_transform(analytic_warning)
        # transform_program.add_transform(validate_shots)
        transform_program.add_transform(
            validate_device_wires, self.wires, name=self.name
        )
        transform_program.add_transform(
            decompose,
            stopping_condition=lambda x: x.name in self.operations,
            name=self.name,
            skip_initial_state_prep=False,
        )

        transform_program.add_transform(
            validate_measurements,
            sample_measurements=accepted_sample_measurement,
            name=self.name,
        )
        transform_program.add_transform(
            validate_observables,
            stopping_condition=lambda x: x.name in self.observables,
            name=self.name,
        )

        transform_program.add_transform(broadcast_expand)
        transform_program.add_transform(split_non_commuting)

        return transform_program, config

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> List:

        if not self._session_id:
            self.start()

        if isinstance(circuits, QuantumScript):
            circuits = [circuits]

        qiskit_circuits = []
        for circuit in circuits:
            qiskit_circuits.append(circuit_to_qiskit(circuit, self.num_wires, diagonalize=True, measure=True))

        results = self._platform.run(qiskit_circuits, session_id=self._session_id, **self._run_options).result()
        if isinstance(results, Result):
            results = [results]

        all_results = []
        for original_circuit, qcirc, result in zip(circuits, qiskit_circuits, results):

            counts = result.get_counts()

            # Reconstruct the list of samples from the counts dictionary
            samples_list = []
            for key, value in counts.items():
                samples_list.extend([key] * value)

            if not samples_list:
                # Handle case with no samples (e.g., 0 shots)
                # Create an empty array with the correct number of columns
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
            res_tuple = (res[0],) if single_measurement else tuple(res)
            # res_tuple = res[0] if single_measurement else tuple(res)
6           all_results.append(res_tuple)

        # return all_results[0] if len(all_results) == 1 else all_results
        return all_results


if __name__ == "__main__":

    import os

    with AqtDevice(
        project_id=os.environ["SCW_PROJECT_ID"],
        secret_key=os.environ["SCW_SECRET_KEY"],
        url=os.getenv("SCW_API_URL"),
        backend=os.getenv("SCW_BACKEND_NAME", "aqt_ibex_simulation_local"),
        shots=100,
        seed=42,
        max_duration="42m",
        abelian_grouping=True,
    ) as device:

        ### Simple bell state circuit execution
        @qml.qnode(device)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            # return qml.expval(qml.PauliZ(0))
            return qml.probs(wires=[0, 1]), qml.counts(wires=[0, 1])

        result = circuit()
        print(result)
