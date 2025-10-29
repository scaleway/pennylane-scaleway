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

from logging import warning
from typing import Optional
import numpy as np

import pennylane as qml
from pennylane.measurements import ExpectationMP, VarianceMP
from pennylane.devices import Device, ExecutionConfig
from pennylane.tape import QuantumScript, QuantumScriptOrBatch
from pennylane.devices.modifiers import simulator_tracking, single_tape_support

from qiskit_scaleway import ScalewayProvider
from qiskit_scaleway.primitives import Estimator
from qiskit_scaleway.backends import AerBackend

from utils import circuit_to_qiskit, mp_to_pauli


# def _apply_measurements(tape: QuantumScript, samples: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
#     if tape.shots:
#         # Shot vector support
#         results = []
#         for lower, upper in tape.shots.bins():
#             sub_samples = samples[lower:upper]
#             results.append(
#                 tuple(mp.process_samples(sub_samples, tape.wires) for mp in tape.measurements)
#             )
#         if len(tape.measurements) == 1:
#             results = [res[0] for res in results]
#         if not tape.shots.has_partitioned_shots:
#             results = results[0]
#         else:
#             results = tuple(results)
#     else:
#         state = qml.math.sum(samples, axis=0) / tape.shots.total_shots
#         results = tuple(mp.process_state(state, tape.wires) for mp in tape.measurements)
#         if len(tape.measurements) == 1:
#             results = results[0]

#     return results


@simulator_tracking     # update device.tracker with some relevant information
@single_tape_support    # add support for device.execute(tape) in addition to device.execute((tape,))
class AerDevice(Device):
    """
    The way to call it:
        device = qml.device("scaleway.aer", wires=XXX, project_id=XXX, secret_key=XXX)
    """

    name = "scaleway.aer"

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
            warning.warn(
                f"Number of wires ({self.num_wires}) exceeds the theoretical number of qubits in the backend ({self._backend.num_qubits})."
                "This may lead to unexpected behavior."
            )

        self._session_id = self._backend.start_session(name="all-you-need-is-love", deduplication_id="all-you-need-is-love")

    def preprocess_transforms(self, execution_config = None):
        return super().preprocess_transforms(execution_config)

    def setup_execution_config(self, config = None, circuit = None):
        return super().setup_execution_config(config, circuit)


#####
    # def execute(
    #     self,
    #     circuits: QuantumTape_or_Batch,
    #     execution_config: ExecutionConfig | None = None,
    # ) -> Result_or_ResultBatch:
    #     """Execute a circuit or a batch of circuits and turn it into results."""
    #     session = self._session

    #     results = []

    #     if isinstance(circuits, QuantumScript):
    #         circuits = [circuits]

    #     for circ in circuits:
    #         if circ.shots and len(circ.shots.shot_vector) > 1:
    #             raise ValueError(
    #                 f"Setting shot vector {circ.shots.shot_vector} is not supported for {self.name}."
    #                 "Please use a single integer instead when specifying the number of shots."
    #             )
    #         if isinstance(circ.measurements[0], (ExpectationMP, VarianceMP)) and getattr(
    #             circ.measurements[0].obs, "pauli_rep", None
    #         ):
    #             execute_fn = self._execute_estimator
    #         else:
    #             execute_fn = self._execute_sampler
    #         results.append(execute_fn(circ, session))
    #     return results
#####


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

        # qiskit_circuits = list(map(
        #     lambda circuit: qasm2.loads(qml.to_openqasm(circuit, measure_all=True)),
        #     circuits
        # ))

        # qiskit_circuits = [circuit_to_qiskit(circuit, circuit.num_wires) for circuit in circuits]

        # job = self._backend.run(qiskit_circuits, session_id=self._session_id)
        # results = job.result().results

        # results = [
        #     np.expand_dims(
        #         np.array(list(result.data.counts.values())),
        #         axis=1
        #     ) for result in results
        # ]

        # self._samples = self.generate_samples(0)
        # res = [
        #     mp.process_samples(self._samples, wire_order=self.wires) for mp in circuit.measurements
        # ]

        # results = tuple(map(
        #     _apply_measurements,
        #     circuits,
        #     results,
        #     [self._seed] * len(results)
        # ))

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
        @qml.set_shots(1000)
        @qml.qnode(device)
        def circuit() -> QuantumScript:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        expval = circuit()
        assert np.isclose(expval, 0.0), f"Expected 0.0, got {expval}"

        # Probabilities
        @qml.set_shots(10)
        @qml.qnode(device)
        def circuit() -> QuantumScript:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=0)

        probs = circuit()
        assert np.allclose([0.5, 0.5], probs), f"Expected [0.5, 0.5], got {probs}"

        # Counts
        @qml.set_shots(10)
        @qml.qnode(device)
        def circuit() -> QuantumScript:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.counts(wires=0)

        counts = circuit()
        assert counts() == {"0": 5}, f"Expected {'0': 5}, got {counts}"

        # Samples
        @qml.set_shots(10)
        @qml.qnode(device)
        def circuit() -> QuantumScript:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.sample(wires=0)

        samples = circuit()
        assert np.array_equal([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], samples), f"Expected [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], got {samples}"
