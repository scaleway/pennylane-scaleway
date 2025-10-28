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

from typing import Optional
import numpy as np
import pennylane as qml
from pennylane.devices import Device, ExecutionConfig
from pennylane.devices.reference_qubit import sample_state
from pennylane.tape import QuantumScript, QuantumScriptOrBatch
from pennylane.devices.modifiers import simulator_tracking, single_tape_support

from qiskit import qasm2
from qiskit_scaleway import ScalewayProvider
from qiskit_scaleway.backends import AerBackend


def _apply_measurements(tape: QuantumScript, samples: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    if tape.shots:
        # Shot vector support
        results = []
        for lower, upper in tape.shots.bins():
            sub_samples = samples[lower:upper]
            results.append(
                tuple(mp.process_samples(sub_samples, tape.wires) for mp in tape.measurements)
            )
        if len(tape.measurements) == 1:
            results = [res[0] for res in results]
        if not tape.shots.has_partitioned_shots:
            results = results[0]
        else:
            results = tuple(results)
    else:
        state = qml.math.sum(samples, axis=0) / tape.shots.total_shots
        results = tuple(mp.process_state(state, tape.wires) for mp in tape.measurements)
        if len(tape.measurements) == 1:
            results = results[0]

    return results


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

        self._instance = kwargs.get("instance", "aer_simulation_pop_c16m128")
        self._provider = ScalewayProvider(
            project_id=kwargs.get("project_id"),
            secret_key=kwargs.get("secret_key"),
            url=kwargs.get("url"),
        )

        backends = [backend for backend in self._provider.backends() if isinstance(backend, AerBackend)]
        if self._instance not in [backend.name for backend in backends]:
            raise ValueError(f"Backend {self._instance} not found. Available backends are {[backend.name for backend in backends]}.")

        self._backend = self._provider.get_backend(self._instance)
        if self._backend.availability != "available":
            raise RuntimeError(f"Backend {self._instance} is not available. Please try again later, or check availability at https://console.scaleway.com/qaas/sessions/create.")

        self._session_id = self._backend.start_session(name="all-you-need-is-love", deduplication_id="all-you-need-is-love")

    def preprocess_transforms(self, execution_config = None):
        transform = super().preprocess_transforms(execution_config)
        transform.add_transform(set_shots)

    def setup_execution_config(self, config = None, circuit = None):
        return super().setup_execution_config(config, circuit)

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None
    ):
        if isinstance(circuits, qml.tape.QuantumScript):
            circuits = [circuits]

        for circuit in circuits:
            if circuit.shots and len(circuit.shots.shot_vector) > 1:
                raise ValueError(
                    f"Setting shot vector {circuit.shots.shot_vector} is not supported for {self.name}."
                    "Please use a single integer instead when specifying the number of shots."
                )
            elif circuit.shots and (circuit.shots.total_shots is None):
                circuit.shots.total_shots = self.shots                

        qiskit_circuits = list(map(
            lambda circuit: qasm2.loads(qml.to_openqasm(circuit, measure_all=True)),
            circuits
        ))
        job = self._backend.run(qiskit_circuits, session_id=self._session_id)
        results = job.result().results

        results = tuple(tuple(result.data.counts.values()) for result in results)
        results = tuple(map(
            _apply_measurements,
            circuits,
            results,
            [self._seed] * len(results)
        ))

        if len(results) == 1:
            results = results[0]

        return results

    def stop(self):
        self._backend.stop_session(self._session_id)

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

        ### Simple bell state circuit execution
        @qml.qnode(device)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs()

        result = circuit()
        print(result)

        ### Testing different output types for the same circuit (a single untouched qubit)
        def example_value(m):
            tape = qml.tape.QuantumScript((), (m,), shots=10)
            return device.execute(tape)

        # Probabilities
        probs = example_value(qml.probs(wires=0))
        assert np.array_equal([1.0, 0.0], probs)

        # Expectation value
        assert example_value(qml.expval(qml.Z(0))) == 1.0

        # Counts
        assert example_value(qml.counts(wires=0)) == {"0": 10}

        # Samples
        samples = example_value(qml.sample(wires=0))
        assert np.array_equal([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], samples)
