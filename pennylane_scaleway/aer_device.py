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

from dataclasses import replace
import numpy as np
from typing import List, Tuple
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
from pennylane.tape import QuantumScript, QuantumScriptOrBatch, QuantumTape
from pennylane.transforms import split_non_commuting, broadcast_expand
from pennylane.transforms.core import TransformProgram

from qiskit.primitives.containers import PrimitiveResult, PubResult

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
        update_options,
    )
except ImportError:
    from utils import (
        QISKIT_OPERATION_MAP,
        accepted_sample_measurement,
        circuit_to_qiskit,
        mp_to_pauli,
        split_execution_types,
        update_options,
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

@simulator_tracking     # update device.tracker with some relevant information
@single_tape_support    # add support for device.execute(tape) in addition to device.execute((tape,))
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

    def __init__(
            self,
            wires=None,
            shots=None,
            seed=None,
            **kwargs
        ):

        super().__init__(wires=wires, shots=shots)

        self._rng = np.random.default_rng(seed)

        try:
            self._handle_kwargs(**kwargs)
        except Exception as e:
            print(f"Error handling kwargs: {e}")

            backend = kwargs.get("backend")
            self._provider = ScalewayProvider(
                project_id=kwargs.get("project_id"),
                secret_key=kwargs.get("secret_key"),
                url=kwargs.get("url"),
            )

            platforms = [platform for platform in self._provider.backends() if isinstance(platform, AerBackend)]
            if backend not in [platform.name for platform in platforms]:
                raise ValueError(f"Platform \'{backend}\' not found. Available platforms are {[platform.name for platform in platforms]}.")

            self._platform = self._provider.get_backend(backend)
            if self._platform.availability != "available":
                raise RuntimeError(f"Platform {backend} is not available. Please try again later, or check availability at https://console.scaleway.com/qaas/sessions/create.")

        if self.num_wires > self._platform.num_qubits:
            warnings.warn(
                f"Number of wires ({self.num_wires}) exceeds the theoretical limit of qubits in the platform ({self._platform.num_qubits})."
                "This may lead to unexpected behavior.",
                UserWarning
            )

        try:
            self._scaleway_session_setup(**kwargs)
        except Exception as e:
            print(f"Error setting up Scaleway session: {e}")
            self._session_id = self._platform.start_session(name="all-you-need-is-love", deduplication_id="all-you-need-is-love")

    def _handle_kwargs(self, **kwargs):
        self._kwargs = kwargs
        raise NotImplementedError("This method has to be implemented.")

    def _scaleway_session_setup(self, **kwargs):
        raise NotImplementedError("This method has to be implemented.")

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
        transform_program.add_transform(validate_device_wires, self.wires, name=self.name)
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
        execution_config: ExecutionConfig | None = None
    ):
        if isinstance(circuits, QuantumScript):
            circuits = [circuits]

        results = []

        for circuit in circuits:
            if circuit.shots and len(circuit.shots.shot_vector) > 1:
                raise ValueError(
                    f"Setting shot vector {circuit.shots.shot_vector} is not supported for {self.name}."
                    "Please use a single integer instead when specifying the number of shots."
                )

            if isinstance(circuit.measurements[0], (ExpectationMP, VarianceMP)) and getattr(
                circuit.measurements[0].obs, "pauli_rep", None
            ):
                results.append(self._run_estimator(circuit))
            else:
                results.append(self._run_sampler(circuit))

        return results

    def _run_estimator(self, circuit: QuantumScript) -> Tuple:

        qcirc = circuit_to_qiskit(circuit, self.num_wires, diagonalize=False, measure=False)

        estimator = Estimator(backend=self._platform, session_id=self._session_id)

        pauli_observables = [mp_to_pauli(mp, self.num_wires) for mp in circuit.measurements]
        compiled_observables = [
            op.apply_layout(qcirc.layout) for op in pauli_observables
        ]

        update_options(estimator, self._kwargs)

        circ_and_obs = [(qcirc, compiled_observables)]

        result = estimator.run(
            circ_and_obs,
            precision=np.sqrt(1 / circuit.shots.total_shots) if circuit.shots else None,
        ).result()
        result = self._process_estimator_job(circuit.measurements, result)

        return result

    def _run_sampler(self, circuit: QuantumScript) -> Tuple:

        qcirc = circuit_to_qiskit(circuit, self.num_wires, diagonalize=True, measure=True)

        sampler = Sampler(self._platform, self._session_id)

        update_options(sampler, self._kwargs)

        result = sampler.run(
            [qcirc],
            shots=circuit.shots.total_shots if circuit.shots.total_shots else None,
        ).result()[0]

        c = getattr(result.data, qcirc.cregs[0].name)
        counts = c.get_counts()
        if not isinstance(counts, dict):
            counts = c.get_counts()[0]

        samples = []
        for key, value in counts.items():
            samples.extend([key] * value)
        samples = np.vstack([np.array([int(i) for i in s[::-1]]) for s in samples])

        res = [
            mp.process_samples(samples, wire_order=self.wires) for mp in circuit.measurements
        ]

        single_measurement = len(circuit.measurements) == 1
        res = (res[0],) if single_measurement else tuple(res)

        return res

    @staticmethod
    def _process_estimator_job(measurements: List[MeasurementProcess], job_result: PrimitiveResult[PubResult]):

        expvals = job_result[0].data.evs
        variances = (job_result[0].data.stds / job_result[0].metadata["target_precision"]) ** 2

        result = []
        for i, mp in enumerate(measurements):
            if isinstance(mp, ExpectationMP):
                result.append(expvals[i])
            elif isinstance(mp, VarianceMP):
                result.append(variances[i])

        single_measurement = len(measurements) == 1
        result = (result[0],) if single_measurement else tuple(result)

        return result

    def stop(self):
        if self._session_id:
            self._platform.stop_session(self._session_id)
            self._session_id = None
        else:
            raise RuntimeError("No session running.")

    @property
    def num_wires(self):
        return len(self.wires)

    def stopping_condition(self, op: qml.operation.Operator) -> bool:
        """Specifies whether or not an Operator is accepted by QiskitDevice2."""
        return op.name in self.operations

    def observable_stopping_condition(self, obs: qml.operation.Operator) -> bool:
        """Specifies whether or not an observable is accepted by QiskitDevice2."""
        return obs.name in self.observables

    def __enter__(self):
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
        assert np.allclose([0.5, 0.0, 0.0,  0.5], probs, atol=epsilon), f"Expected ~[0.5, 0.0, 0.0, 0.5], got {probs}"

        # Counts
        @qml.set_shots(10)
        @qml.qnode(device)
        def circuit() -> QuantumScript:
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.counts(wires=0)

        counts = circuit()
        assert counts == {"0": 10}, "Expected {'0': 10}, got " + str(counts)

        # Samples
        @qml.set_shots(10)
        @qml.qnode(device)
        def circuit() -> QuantumScript:
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.sample(wires=0)

        samples = circuit()
        assert np.array_equal([np.array([0])] * 10, samples), f"Expected [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], got {samples}"

        print("Passed!")
