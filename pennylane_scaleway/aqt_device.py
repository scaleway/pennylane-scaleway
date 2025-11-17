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

from typing import List, Type

# from qiskit_aqt_provider import AQTProvider

import pennylane as qml
from pennylane.devices import Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.tape import QuantumScriptOrBatch
from pennylane.transforms.core import TransformProgram

from qiskit_scaleway import ScalewayProvider
from qiskit_scaleway.backends import AqtBackend, AerBackend, BaseBackend

from pennylane_scaleway.scw_device import ScalewayDevice
from pennylane_scaleway.aer_device import AerDevice


class AqtDevice(ScalewayDevice):
    """Device to run quantum computations on AQT hardware via Qiskit."""

    name = "scaleway.aqt"
    backend_types = (AerBackend, AqtBackend)

    def __init__(self, wires, shots, seed, **kwargs):
        device_type = self._check_backend_type(kwargs)
        if device_type == AerDevice:
            self._aerdevice = AerDevice(wires=wires, shots=shots, seed=seed, **kwargs)
            for attr in dir(self._aerdevice):
                if attr.startswith("_") and not attr.startswith("__"):
                    setattr(self, attr, getattr(self._aerdevice, attr))
        else:
            self._aerdevice = None

        super().__init__(wires=wires, kwargs=kwargs, shots=shots)

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
            return [0.42]

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
