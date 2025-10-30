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
import warnings
import numpy as np

import pennylane as qml
from pennylane.measurements import ExpectationMP, VarianceMP
from pennylane.devices import Device, ExecutionConfig
from pennylane.tape import QuantumScript, QuantumScriptOrBatch, QuantumTape
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.transforms import split_non_commuting, broadcast_expand
from pennylane.transforms.core import TransformProgram
from pennylane.devices.preprocess import (
    decompose,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)

from qiskit_scaleway import ScalewayProvider
from qiskit_scaleway.primitives import Estimator, Sampler
from qiskit_scaleway.backends import AerBackend

try:
    from .utils import circuit_to_qiskit, mp_to_pauli, QISKIT_OPERATION_MAP
except ImportError:
    from utils import circuit_to_qiskit, mp_to_pauli, QISKIT_OPERATION_MAP


@qml.transform
def analytic_warning(tape: QuantumTape):
    """Transform that adds a warning for circuits without shots set."""
    if not tape.shots:
        warnings.warn(
            "Expected an integer number of shots, but received shots=None. A default "
            "number of shots will be selected by the Qiskit backend. The analytic calculation of results is not supported on "
            "this device. All statistics obtained from this device are estimates based "
            "on samples.",
            UserWarning,
        )
    return (tape,), lambda results: results[0]

def accepted_sample_measurement(m: qml.measurements.MeasurementProcess) -> bool:
    """Specifies whether or not a measurement is accepted when sampling."""

    return isinstance(
        m,
        (
            qml.measurements.SampleMeasurement,
            qml.measurements.ClassicalShadowMP,
            qml.measurements.ShadowExpvalMP,
        ),
    )

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

        # seed and rng not necessary for a device, but part of recommended
        # numpy practices to use a local random number generator
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        instance = kwargs.get("instance", "aer_simulation_pop_c16m128")
        self._provider = ScalewayProvider(
            project_id=kwargs.get("project_id"),
            secret_key=kwargs.get("secret_key"),
            url=kwargs.get("url"),
        )

        backends = [backend for backend in self._provider.backends() if isinstance(backend, AerBackend)]
        if instance not in [backend.name for backend in backends]:
            raise ValueError(f"Backend {instance} not found. Available backends are {[backend.name for backend in backends]}.")

        self._backend = self._provider.get_backend(instance)
        if self._backend.availability != "available":
            raise RuntimeError(f"Backend {instance} is not available. Please try again later, or check availability at https://console.scaleway.com/qaas/sessions/create.")

        if self.num_wires > self._backend.num_qubits:
            warnings.warn(
                f"Number of wires ({self.num_wires}) exceeds the theoretical number of qubits in the backend ({self._backend.num_qubits})."
                "This may lead to unexpected behavior.",
                UserWarning
            )

        self._session_id = self._backend.start_session(name="all-you-need-is-love", deduplication_id="all-you-need-is-love")

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

        # transform_program.add_transform(split_execution_types)

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
            # elif not circuit.shots:
            #     circuit._shots = self.default_shots

            if isinstance(circuit.measurements[0], (ExpectationMP, VarianceMP)) and getattr(
                circuit.measurements[0].obs, "pauli_rep", None
            ):
                results.append(self._execute_estimator(circuit))
            else:
                results.append(self._execute_sampler(circuit))

        if len(results) == 1:
            results = results[0]

        return results

    def _execute_estimator(self, circuit: QuantumScript):
        """Returns the result of the execution of the circuit using the EstimatorV2 Primitive.
        Note that this result has been processed respective to the MeasurementProcess given.
        E.g. `qml.expval` returns an expectation value whereas `qml.var` will return the variance.

        Args:
            circuits (list[QuantumCircuit]): the circuits to be executed via EstimatorV2
            session (Session): the session that the execution will be performed with

        Returns:
            result (tuple): the processed result from EstimatorV2
        """
        # the Estimator primitive takes care of diagonalization and measurements itself,
        # so diagonalizing gates and measurements are not included in the circuit
        qcirc = circuit_to_qiskit(circuit, self.num_wires, diagonalize=False, measure=False)
        estimator = Estimator(backend=self._backend, session_id=self._session_id)

        pauli_observables = [mp_to_pauli(mp, self.num_wires) for mp in circuit.measurements]
        compiled_observables = [
            op.apply_layout(qcirc.layout) for op in pauli_observables
        ]
        # estimator.options.update(**self._kwargs)
        circ_and_obs = [(qcirc, compiled_observables)]
        result = estimator.run(
            circ_and_obs,
            precision=np.sqrt(1 / circuit.shots.total_shots) if circuit.shots else None,
        ).result()
        result = self._process_estimator_job(circuit.measurements, result)

        return result

    def _execute_sampler(self, circuit: QuantumScript):
        """Returns the result of the execution of the circuit using the SamplerV2 Primitive.
        Note that this result has been processed respective to the MeasurementProcess given.
        E.g. `qml.expval` returns an expectation value whereas `qml.sample()` will return the raw samples.

        Args:
            circuits (list[QuantumCircuit]): the circuits to be executed via SamplerV2
            session (Session): the session that the execution will be performed with

        Returns:
            result (tuple): the processed result from SamplerV2
        """
        qcirc = circuit_to_qiskit(circuit, self.num_wires, diagonalize=True, measure=True)
        sampler = Sampler(self._backend, self._session_id)
        # sampler.options.update(**self._kwargs)

        result = sampler.run(
            [qcirc],
            shots=circuit.shots.total_shots if circuit.shots.total_shots else None,
        ).result()[0]

        # needs processing function to convert to the correct format for states, and
        # also handle instances where wires were specified in probs, and for multiple probs measurements

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
    def _process_estimator_job(measurements, job_result):
        """Estimator returns the expectation value and standard error for each observable measured,
        along with some metadata that contains the precision. Extracts the relevant number for each
        measurement process and return the requested results from the Estimator executions.

        Note that for variance, we calculate the variance by using the standard error and the
        precision value.

        Args:
            measurements (list[MeasurementProcess]): the measurements in the circuit
            job_result (Any): the result from EstimatorV2

        Returns:
            result (tuple): the processed result from EstimatorV2
        """
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
        self._backend.stop_session(self._session_id)

    @property
    def num_wires(self):
        """Get the number of wires.

        Returns:
            int: The number of wires.
        """
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
        instance=os.getenv("SCW_BACKEND_NAME", "aer_simulation_local"),
    ) as device:

    # device = qml.device("default.qubit", wires=2)

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
