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

from dataclasses import replace, fields
import numpy as np
from typing import Iterable, List, Tuple
import warnings

import pennylane as qml
from pennylane.devices import ExecutionConfig
from pennylane.devices.preprocess import (
    decompose,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from pennylane.measurements import ExpectationMP, VarianceMP, MeasurementProcess
from pennylane.tape import QuantumScript, QuantumScriptOrBatch, QuantumTape
from pennylane.transforms import split_non_commuting, broadcast_expand
from pennylane.transforms.core import TransformProgram

from qiskit.primitives.containers import PrimitiveResult, PubResult
from qiskit.primitives.backend_estimator_v2 import Options as EstimatorOptions
from qiskit.primitives.backend_sampler_v2 import Options as SamplerOptions

from qiskit_scaleway.primitives import Estimator, Sampler

from pennylane_scaleway.scw_device import ScalewayDevice
from pennylane_scaleway.aer_utils import (
    QISKIT_OPERATION_MAP,
    accepted_sample_measurement,
    circuit_to_qiskit,
    mp_to_pauli,
    split_execution_types,
)


@qml.transform
def analytic_warning(tape: QuantumTape):
    if not tape.shots:
        warnings.warn(
            "The analytic calculation of results is not supported on "
            "this device. All statistics obtained from this device are estimates based "
            "on samples. A default number of shots will be selected by the Qiskit backend."
            "(Shots were not set for this circuit).",
            UserWarning,
        )
    return (tape,), lambda results: results[0]


class AerDevice(ScalewayDevice):
    """
    This is Scaleway's device to run Pennylane's circuits on Aer emulators.

    This device:
        * Supports any operations with explicit PennyLane to Qiskit gate conversions defined in the plugin.
        * Supports both CPU and GPU Aer backends.
        * Approximates analytic calculations by sampling.
        * Does not support Shots vector.
        * Does not support state vector simulation.
        * Does not intrinsically support parameter broadcasting.
    """

    name = "scaleway.aer"

    operations = set(QISKIT_OPERATION_MAP.keys())
    observables = {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Identity",
        "Hadamard",
        "Hermitian",
        "Projector",
        "Prod",
        "Sum",
        "LinearCombination",
        "SProd",
    }

    def __init__(self, wires=None, shots=None, seed=None, **kwargs):
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
                backend="aer_simulation_pop_c16m128"
            ) as dev:
                @qml.qnode(dev)
                def circuit():
                    qml.Hadamard(wires=0)
                    qml.CNOT(wires=[0, 1])
                    return qml.sample()
                print(circuit())
            ```
        """

        if shots and not isinstance(shots, int):
            raise ValueError(
                "Only integer number of shots is supported on this device (vectors are not supported either). The set 'shots' value will be ignored."
            )

        super().__init__(wires=wires, kwargs=kwargs, shots=shots)

        if isinstance(seed, int):
            kwargs.update({"seed_simulator": seed})
        self._rng = np.random.default_rng(seed)

        self._handle_kwargs(**kwargs)

        if self.num_wires > self._platform.num_qubits:
            warnings.warn(
                f"Number of wires ({self.num_wires}) exceeds the theoretical limit of qubits in the platform ({self._platform.num_qubits})."
                "This may lead to unexpected behavior and crash.",
                UserWarning,
            )

    def _handle_kwargs(self, **kwargs):

        ### Extract Estimator/Sampler-specific options
        self._sampler_options = {
            k: v
            for k, v in kwargs.items()
            if k in (field.name for field in fields(SamplerOptions))
        }
        self._estimator_options = {
            k: v
            for k, v in kwargs.items()
            if k in (field.name for field in fields(EstimatorOptions))
        }
        [
            kwargs.pop(k)
            for k in (self._sampler_options.keys() | self._estimator_options.keys())
        ]

        if len(kwargs) > 0:
            warnings.warn(
                f"The following keyword arguments are not supported by '{self.name}' device: {list(kwargs.keys())}",
                UserWarning,
            )

    def preprocess(
        self,
        execution_config: ExecutionConfig | None = None,
    ) -> tuple[TransformProgram, ExecutionConfig]:

        config = execution_config or ExecutionConfig()
        config = replace(config, use_device_gradient=False)

        transform_program = TransformProgram()

        transform_program.add_transform(analytic_warning)
        transform_program.add_transform(
            validate_device_wires, self.wires, name=self.name
        )
        transform_program.add_transform(
            decompose,
            stopping_condition=self._stopping_condition,
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
            stopping_condition=self._observable_stopping_condition,
            name=self.name,
        )
        transform_program.add_transform(broadcast_expand)
        transform_program.add_transform(split_non_commuting)
        transform_program.add_transform(split_execution_types)

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

        estimator_indices = []
        estimator_circuits = []
        sampler_circuits = []
        for i, circuit in enumerate(circuits):
            if circuit.shots and len(circuit.shots.shot_vector) > 1:
                raise ValueError(
                    f"Setting shot vector {circuit.shots.shot_vector} is not supported for {self.name}."
                    "Please use a single integer instead when specifying the number of shots."
                )

            if isinstance(
                circuit.measurements[0], (ExpectationMP, VarianceMP)
            ) and getattr(circuit.measurements[0].obs, "pauli_rep", None):
                estimator_indices.append(i)
                estimator_circuits.append(circuit)
            else:
                sampler_circuits.append(circuit)

        if sampler_circuits:
            sampler_results = self._run_sampler(sampler_circuits)
        if estimator_circuits:
            estimator_results = self._run_estimator(estimator_circuits)
        results = []
        s, e = 0, 0
        for i, circuit in enumerate(circuits):
            if i in estimator_indices:
                results.append(estimator_results[e])
                e += 1
            else:
                results.append(sampler_results[s])
                s += 1

        return results

    def _run_estimator(self, circuits: Iterable[QuantumScript]) -> List[Tuple]:

        qcircs = [
            circuit_to_qiskit(circuit, self.num_wires, diagonalize=False, measure=False)
            for circuit in circuits
        ]

        estimator = Estimator(
            backend=self._platform,
            session_id=self._session_id,
            options=self._estimator_options,
        )

        circ_and_obs = []
        for qcirc, circuit in zip(qcircs, circuits):
            pauli_observables = [
                mp_to_pauli(mp, self.num_wires) for mp in circuit.measurements
            ]
            compiled_observables = [
                op.apply_layout(qcirc.layout) for op in pauli_observables
            ]
            circ_and_obs.append((qcirc, compiled_observables))

        precision = (
            np.sqrt(1 / circuits[0].shots.total_shots)
            if circuits[0].shots.total_shots
            else None
        )

        results = estimator.run(circ_and_obs, precision=precision).result()

        processed_results = []
        for i, circuit in enumerate(circuits):
            processed_result = self._process_estimator_job(
                circuit.measurements, results[i]
            )
            processed_results.append(processed_result)

        return processed_results

    def _run_sampler(self, circuits: Iterable[QuantumScript]) -> List[Tuple]:

        qcircs = [
            circuit_to_qiskit(circuit, self.num_wires, diagonalize=True, measure=True)
            for circuit in circuits
        ]

        sampler = Sampler(self._platform, self._session_id, self._sampler_options)

        results = sampler.run(
            qcircs,
            shots=(
                circuits[0].shots.total_shots if circuits[0].shots.total_shots else None
            ),
        ).result()

        all_results = []
        for original_circuit, qcirc, result in zip(circuits, qcircs, results):

            # Extract counts from the classical register
            # Assumes one classical register per circuit, which circuit_to_qiskit sets up
            c = getattr(result.data, qcirc.cregs[0].name)
            counts = c.get_counts()
            if not isinstance(counts, dict):
                # Handle cases where get_counts() might return a list
                counts = c.get_counts()[0]

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

            # Format the final result tuple for this circuit
            single_measurement = len(original_circuit.measurements) == 1
            res_tuple = (res[0],) if single_measurement else tuple(res)
            all_results.append(res_tuple)

        return all_results

    @staticmethod
    def _process_estimator_job(
        measurements: List[MeasurementProcess], job_result: PrimitiveResult[PubResult]
    ):

        expvals = job_result.data.evs
        variances = (
            job_result.data.stds / job_result.metadata["target_precision"]
        ) ** 2

        result = []
        for i, mp in enumerate(measurements):
            if isinstance(mp, ExpectationMP):
                result.append(expvals[i])
            elif isinstance(mp, VarianceMP):
                result.append(variances[i])

        single_measurement = len(measurements) == 1
        result = (result[0],) if single_measurement else tuple(result)

        return result

    def _stopping_condition(self, op: qml.operation.Operator) -> bool:
        """Specifies whether or not an Operator is accepted by QiskitDevice2."""
        return op.name in self.operations

    def _observable_stopping_condition(self, obs: qml.operation.Operator) -> bool:
        """Specifies whether or not an observable is accepted by QiskitDevice2."""
        return obs.name in self.observables


if __name__ == "__main__":

    import os

    with AerDevice(
        wires=2,
        project_id=os.environ["SCW_PROJECT_ID"],
        secret_key=os.environ["SCW_SECRET_KEY"],
        url=os.getenv("SCW_API_URL"),
        backend=os.getenv("SCW_BACKEND_NAME", "aer_simulation_local"),
        shots=100,
        seed=42,
        max_duration="42m",
        abelian_grouping=True,
        test="useless",
    ) as device:

        ### Simple bell state circuit execution
        @qml.qnode(device)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        result = circuit()
        print(result)
