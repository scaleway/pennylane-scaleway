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

from dataclasses import replace, fields
from tenacity import retry, stop_after_attempt, stop_after_delay
from typing import Callable, Iterable, List, Sequence, Tuple

import pennylane as qml
from pennylane.devices import ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import (
    decompose,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from pennylane.measurements import ExpectationMP, VarianceMP, MeasurementProcess
from pennylane.tape import QuantumScript, QuantumScriptOrBatch
from pennylane.transforms import split_non_commuting, broadcast_expand, transform
from pennylane.transforms.core import TransformProgram

from qiskit.primitives.containers import PrimitiveResult, PubResult
from qiskit.primitives.backend_estimator_v2 import Options as EstimatorOptions
from qiskit.primitives.backend_sampler_v2 import Options as SamplerOptions
from qiskit.quantum_info import SparsePauliOp

from qiskit_scaleway.primitives import Estimator, Sampler
from qiskit_scaleway.backends import AerBackend

from pennylane_scaleway.scw_device import ScalewayDevice
from pennylane_scaleway.utils import (
    QISKIT_OPERATION_MAP,
    accepted_sample_measurement,
    circuit_to_qiskit,
)
from pennylane_scaleway.utils import analytic_warning


@simulator_tracking  # update device.tracker with some relevant information
@single_tape_support  # add support for device.execute(tape) in addition to device.execute((tape,))
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
    backend_types = (AerBackend,)

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

        if shots and not isinstance(shots, int):
            raise ValueError(
                "Only integer number of shots is supported on this device (vectors are not supported either). The set 'shots' value will be ignored."
            )

        if isinstance(seed, int):
            kwargs.update({"seed_simulator": seed})
        # self._rng = np.random.default_rng(seed)

        super().__init__(wires=wires, kwargs=kwargs, shots=shots)

        self._handle_kwargs(**kwargs)

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
            stopping_condition=lambda op: op.name in self.operations,
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
            stopping_condition=lambda obs: obs.name in self.observables,
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
            raise RuntimeError("No active session. Please instanciate the device using a context manager, or call start() first. You can also attach to an existing deduplication_id.")

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
            pauli_observables = [self.mp_to_pauli(mp) for mp in circuit.measurements]
            compiled_observables = [
                op.apply_layout(qcirc.layout) for op in pauli_observables
            ]
            circ_and_obs.append((qcirc, compiled_observables))

        precision = (
            np.sqrt(1 / circuits[0].shots.total_shots)
            if circuits[0].shots.total_shots
            else None
        )

        @retry(stop=stop_after_attempt(3) | stop_after_delay(3 * 60), reraise=True)
        def run():
            return estimator.run(circ_and_obs, precision=precision).result()

        results = run()

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

        @retry(stop=stop_after_attempt(3) | stop_after_delay(3 * 60), reraise=True)
        def run():
            return sampler.run(
                qcircs,
                shots=(
                    circuits[0].shots.total_shots
                    if circuits[0].shots.total_shots
                    else None
                ),
            ).result()

        results = run()

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

    def mp_to_pauli(self, mp):
        """Convert a Pauli observable to a SparsePauliOp for measurement via Estimator

        Args:
            mp(Union[ExpectationMP, VarianceMP]): MeasurementProcess to be converted to a SparsePauliOp

        Returns:
            SparsePauliOp: the ``SparsePauliOp`` of the given Pauli observable
        """
        op = mp.obs

        if op.pauli_rep:
            pauli_strings = [
                "".join(
                    [
                        "I" if i not in pauli_term.wires else pauli_term[i]
                        for i in range(self.num_wires)
                    ][
                        ::-1
                    ]  ## Qiskit follows opposite wire order convention
                )
                for pauli_term in op.pauli_rep.keys()
            ]
            coeffs = list(op.pauli_rep.values())
        else:
            raise ValueError(
                f"The operator {op} does not have a representation for SparsePauliOp"
            )

        return SparsePauliOp(data=pauli_strings, coeffs=coeffs).simplify()


@transform
def split_execution_types(
    tape: qml.tape.QuantumTape,
) -> tuple[Sequence[qml.tape.QuantumTape], Callable]:
    """Split into separate tapes based on measurement type. Counts and sample-based measurements
    will use the Qiskit Sampler. ExpectationValue and Variance will use the Estimator, except
    when the measured observable does not have a `pauli_rep`. In that case, the Sampler will be
    used, and the raw samples will be processed to give an expectation value."""
    estimator = []
    sampler = []

    for i, mp in enumerate(tape.measurements):
        if isinstance(mp, (ExpectationMP, VarianceMP)):
            if mp.obs.pauli_rep:
                estimator.append((mp, i))
            else:
                warnings.warn(
                    f"The observable measured {mp.obs} does not have a `pauli_rep` "
                    "and will be run without using the Estimator primitive. Instead, "
                    "raw samples from the Sampler will be used."
                )
                sampler.append((mp, i))
        else:
            sampler.append((mp, i))

    order_indices = [[i for mp, i in group] for group in [estimator, sampler]]

    tapes = []
    if estimator:
        tapes.extend(
            [
                qml.tape.QuantumScript(
                    tape.operations,
                    measurements=[mp for mp, i in estimator],
                    shots=tape.shots,
                )
            ]
        )
    if sampler:
        tapes.extend(
            [
                qml.tape.QuantumScript(
                    tape.operations,
                    measurements=[mp for mp, i in sampler],
                    shots=tape.shots,
                )
            ]
        )

    def reorder_fn(res):
        """re-order the output to the original shape and order"""

        flattened_indices = [i for group in order_indices for i in group]
        flattened_results = [r for group in res for r in group]

        if len(flattened_indices) != len(flattened_results):
            raise ValueError(
                "The lengths of flattened_indices and flattened_results do not match."
            )  # pragma: no cover

        result = dict(zip(flattened_indices, flattened_results))

        result = tuple(result[i] for i in sorted(result.keys()))

        return result[0] if len(result) == 1 else result

    return tapes, reorder_fn
