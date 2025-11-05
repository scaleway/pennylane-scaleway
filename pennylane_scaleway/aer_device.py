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
from pennylane.devices import Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import (
    decompose,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from pennylane.measurements import ExpectationMP, VarianceMP, MeasurementProcess
from pennylane.measurements.shots import Shots
from pennylane.tape import QuantumScript, QuantumScriptOrBatch, QuantumTape
from pennylane.transforms import split_non_commuting, broadcast_expand
from pennylane.transforms.core import TransformProgram

from qiskit.primitives.containers import PrimitiveResult, PubResult
from qiskit.primitives.backend_estimator_v2 import Options as EstimatorOptions
from qiskit.primitives.backend_sampler_v2 import Options as SamplerOptions

from qiskit_scaleway import ScalewayProvider
from qiskit_scaleway.backends import AerBackend
from qiskit_scaleway.primitives import Estimator, Sampler

try:
    from .utils import (
        QISKIT_OPERATION_MAP,
        accepted_sample_measurement,
        circuit_to_qiskit,
        mp_to_pauli,
        split_execution_types,
    )
except ImportError:
    from utils import (
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


@simulator_tracking  # update device.tracker with some relevant information
@single_tape_support  # add support for device.execute(tape) in addition to device.execute((tape,))
class AerDevice(Device):
    """
    The way to call it:
        device = qml.device("scaleway.aer", wires=XXX, project_id=XXX, secret_key=XXX)
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

        if shots and not isinstance(shots, int):
            raise ValueError(
                "Only integer number of shots is supported on this device (vectors are not supported either). The set 'shots' value will be ignored."
            )

        super().__init__(wires=wires, shots=shots)

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

        self._session_id = None

    def _handle_kwargs(self, **kwargs):

        ### Setup Scaleway API and backend
        backend = kwargs.pop("backend", None)

        self._provider = ScalewayProvider(
            project_id=kwargs.pop("project_id", None),
            secret_key=kwargs.pop("secret_key", None),
            url=kwargs.pop("url", None),
        )

        platforms = [
            platform
            for platform in self._provider.backends()
            if isinstance(platform, AerBackend)
        ]
        if backend not in [platform.name for platform in platforms]:
            raise ValueError(
                f"Platform '{backend}' not found. Available platforms are {[platform.name for platform in platforms]}."
            )

        self._platform = self._provider.get_backend(backend)
        if self._platform.availability != "available":
            raise RuntimeError(
                f"Platform '{backend}' is not available. Please try again later, or check availability at https://console.scaleway.com/qaas/sessions/create."
            )

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

        ### Extract Scaleway's session-specific arguments
        self._session_options = {
            "name": kwargs.pop("session_name", None),
            "deduplication_id": kwargs.pop("deduplication_id", None),
            "max_duration": kwargs.pop("max_duration", None),
            "max_idle_duration": kwargs.pop("max_idle_duration", None),
        }

        self._kwargs = kwargs

    def preprocess(
        self,
        execution_config: ExecutionConfig | None = None,
    ) -> tuple[TransformProgram, ExecutionConfig]:
        """This function defines the device transform program to be applied and an updated device configuration.

        Args:
            execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): A data structure describing the
                parameters needed to fully describe the execution.

        Returns:
            TransformProgram, ExecutionConfig: A transform program that when called returns QuantumTapes that the device
            can natively execute as well as a postprocessing function to be called after execution, and a configuration with
            unset specifications filled in.

        This device:

        * Supports any operations with explicit PennyLane to Qiskit gate conversions defined in the plugin
        * Does not intrinsically support parameter broadcasting
        """
        config = execution_config or ExecutionConfig()
        config = replace(config, use_device_gradient=False)

        transform_program = TransformProgram()

        transform_program.add_transform(analytic_warning)
        transform_program.add_transform(
            validate_device_wires, self.wires, name=self.name
        )
        transform_program.add_transform(
            decompose,
            stopping_condition=self.stopping_condition,
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
            stopping_condition=self.observable_stopping_condition,
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
    ):

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

        qcircs = [circuit_to_qiskit(
            circuit, self.num_wires, diagonalize=False, measure=False
        ) for circuit in circuits]

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

        precision = np.sqrt(1 / circuits[0].shots.total_shots) if circuits[0].shots.total_shots else None

        results = estimator.run(
            circ_and_obs,
            precision=precision
        ).result()

        processed_results = []
        for i, circuit in enumerate(circuits):
            processed_result = self._process_estimator_job(circuit.measurements, results[i])
            processed_results.append(processed_result)

        return processed_results

    def _run_sampler(self, circuits: Iterable[QuantumScript]) -> List[Tuple]:

        qcircs = [circuit_to_qiskit(
            circuit, self.num_wires, diagonalize=True, measure=True
        )for circuit in circuits]

        sampler = Sampler(self._platform, self._session_id, self._sampler_options)

        results = sampler.run(
            qcircs,
            shots=circuits[0].shots.total_shots if circuits[0].shots.total_shots else None,
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

    def start(self) -> str:
        """
        Starts a session on the specified Scaleway platform. If a session is already running, it returns the existing session ID.
        """
        if not self._session_id:
            self._session_id = self._platform.start_session(**self._session_options)
        return self._session_id

    def stop(self):
        """
        Stops the currently running session on the Scaleway platform. Raises an error if no session is running.
        """
        if self._session_id:
            self._platform.stop_session(self._session_id)
            self._session_id = None
        else:
            raise RuntimeError("No session running.")

    @property
    def num_wires(self):
        return len(self.wires)

    @property
    def session_id(self):
        return self._session_id

    def stopping_condition(self, op: qml.operation.Operator) -> bool:
        """Specifies whether or not an Operator is accepted by QiskitDevice2."""
        return op.name in self.operations

    def observable_stopping_condition(self, obs: qml.operation.Operator) -> bool:
        """Specifies whether or not an observable is accepted by QiskitDevice2."""
        return obs.name in self.observables

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


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
        abelian_grouping=False,
        max_duration="42m",
    ) as device:

        ### Simple bell state circuit execution

        @qml.qnode(device)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        result = circuit()
        print(result)

        ### Testing different output types for the same circuit

        # Counts
        @qml.set_shots(10)
        @qml.qnode(device)
        def circuit() -> QuantumScript:
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.counts(wires=0)

        counts = circuit()
        assert counts == {"0": 10}, "Expected {'0': 10}, got " + str(counts)

        # Expectation value
        @qml.set_shots(1024)
        @qml.qnode(device)
        def circuit() -> QuantumScript:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        epsilon = 0.1
        expval = circuit()
        assert np.isclose(expval, 0.0, atol=epsilon), f"Expected ~0.0, got {expval}"

        # Probabilities
        @qml.set_shots(1024)
        @qml.qnode(device)
        def circuit() -> QuantumScript:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        probs = circuit()
        assert np.allclose(
            [0.5, 0.0, 0.0, 0.5], probs, atol=epsilon
        ), f"Expected ~[0.5, 0.0, 0.0, 0.5], got {probs}"

        # Samples
        @qml.set_shots(10)
        @qml.qnode(device)
        def circuit() -> QuantumScript:
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.sample(wires=0)

        samples = circuit()
        assert np.array_equal(
            [np.array([0])] * 10, samples
        ), f"Expected [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], got {samples}"

        print("Passed!")
