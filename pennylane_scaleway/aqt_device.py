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
from inspect import signature
from typing import List, Type
import warnings

# from qiskit_aqt_provider import AQTProvider

import pennylane as qml
from pennylane.devices import ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.tape import QuantumScriptOrBatch, QuantumScript
from pennylane.transforms.core import TransformProgram

from qiskit_scaleway import ScalewayProvider
from qiskit_scaleway.backends import AqtBackend, AerBackend

from pennylane_scaleway.scw_device import ScalewayDevice
from pennylane_scaleway.aer_device import AerDevice
from pennylane.transforms import broadcast_expand, split_non_commuting

from pennylane_scaleway.aer_utils import circuit_to_qiskit
from pennylane_scaleway.utils import analytic_warning


@simulator_tracking  # update device.tracker with some relevant information
@single_tape_support  # add support for device.execute(tape) in addition to device.execute((tape,))
class AqtDevice(ScalewayDevice):

    name = "scaleway.aqt"
    backend_types = (AerBackend, AqtBackend)

    def __init__(self, wires, shots, seed, **kwargs):
        device_type = self._check_backend_type(kwargs)
        if device_type == AerDevice:
            # Uses an AerDevice to run AQT emulation.
            self._aerdevice = AerDevice(wires=wires, shots=shots, seed=seed, **kwargs)
            # Acts as a proxy for the AerDevice's attributes, ugly but does the job without headaches nor overhead.
            for attr in dir(self._aerdevice):
                if attr.startswith("_") and not attr.startswith("__"):
                    setattr(self, attr, getattr(self._aerdevice, attr))
        else:
            self._aerdevice = None

        super().__init__(wires=wires, kwargs=kwargs, shots=shots)

    def _handle_kwargs(self, **kwargs):

        ### Extract runner-specific arguments
        self._run_options = {
            k: v
            for k, v in kwargs.items()
            if k in signature(self._platform.run).parameters.keys()
        }
        [kwargs.pop(k) for k in self._run_options.keys()]

        if len(kwargs) > 0:
            warnings.warn(
                f"The following keyword arguments are not supported by '{self.name}' device: {list(kwargs.keys())}",
                UserWarning,
            )

    def preprocess(
        self,
        execution_config: ExecutionConfig | None = None,
    ) -> tuple[TransformProgram, ExecutionConfig]:

        if self._aerdevice:
            print("AER Preprocessing!")
            return self._aerdevice.preprocess(execution_config)
        else:
            print("AQT Preprocessing!")

            transform_program = TransformProgram()
            config = execution_config or ExecutionConfig()
            config = replace(config, use_device_gradient=False)

            transform_program.add_transform(analytic_warning)
            transform_program.add_transform(broadcast_expand)
            transform_program.add_transform(split_non_commuting)

            return transform_program, config

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> List:
        if self._aerdevice:
            print("AER Execute!")
            return self._aerdevice.execute(circuits, execution_config)
        else:
            print("AQT Execute!")

            if not self._session_id:
                self.start()

            if isinstance(circuits, QuantumScript):
                circuits = [circuits]

            qiskit_circuits = []
            for circuit in circuits:
                if circuit.shots and len(circuit.shots.shot_vector) > 1:
                    raise ValueError(
                        f"Setting shot vector {circuit.shots.shot_vector} is not supported for {self.name}."
                        "Please use a single integer instead when specifying the number of shots."
                    )
                qiskit_circuits.append(circuit_to_qiskit(circuit))

            self._run_options.update(
                {
                    "session_id": self._session_id,
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
            results = self._platform.run(qiskit_circuits, **self._run_options).result()

            return results

    def _check_backend_type(cls, kwargs) -> Type[ScalewayDevice]:
        backend = kwargs.get("backend", None)
        provider = ScalewayProvider(
            project_id=kwargs.get("project_id", None),
            secret_key=kwargs.get("secret_key", None),
            url=kwargs.get("url", None),
        )

        platforms = provider.backends()
        for platform in platforms:
            if backend == platform.name:
                if type(platform) == AerBackend:
                    return AerDevice
                elif type(platform) == AqtBackend:
                    return AqtDevice

        raise ValueError(
            f"Platform '{backend}' not found. Available platforms are {[platform.name for platform in platforms if (platform.availability == 'available' and type(platform) in cls.backend_types)]}."
        )


if __name__ == "__main__":

    import os

    with AqtDevice(
        wires=2,
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
            return qml.expval(qml.PauliZ(0))

        result = circuit()
        print(result)
