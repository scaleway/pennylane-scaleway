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
import pennylane as qml
from pennylane.devices import Device, ExecutionConfig
from pennylane.tape import QuantumScript, QuantumScriptOrBatch

from qiskit_scaleway import ScalewayProvider
from qiskit_scaleway.backends import AerBackend


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
        return super().preprocess_transforms(execution_config)

    def setup_execution_config(self, config = None, circuit = None):
        return super().setup_execution_config(config, circuit)

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None
    ):
        return 0.0 if isinstance(circuits, qml.tape.QuantumScript) else tuple(0.0 for c in circuits)

    def stop(self):
        self._backend.stop_session(self._session_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


if __name__ == "__main__":
    import os

    with AerDevice(
            project_id=os.environ["SCW_PROJECT_ID"],
            secret_key=os.environ["SCW_SECRET_KEY"],
            url=os.getenv("SCW_API_URL"),
            instance=os.getenv("SCW_AER_INSTANCE", "aer_simulation_local"),
        ) as dev:

        ###

        @qml.qnode(dev)
        def circuit():
            return qml.state()

        print(circuit())

        ###

        initial_config = ExecutionConfig()
        initial_circuit_batch = [circuit]

        execution_config = dev.setup_execution_config(initial_config)
        transform_program = dev.preprocess_transforms(execution_config)
        circuit_batch, postprocessing = transform_program(initial_circuit_batch)
        results = dev.execute(circuit_batch, execution_config)
        final_results = postprocessing(results)
        print(final_results)

        ###
