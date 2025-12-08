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

from inspect import signature
import warnings

from pennylane.devices import ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.transforms.core import TransformProgram

from qiskit_scaleway.backends import IqmBackend

from pennylane_scaleway.scw_device import ScalewayDevice


@simulator_tracking  # update device.tracker with some relevant information
@single_tape_support  # add support for device.execute(tape) in addition to device.execute((tape,))
class IqmDevice(ScalewayDevice):
    """
    Scaleway's device to run Pennylane circuits on IQM platforms.
    """

    name = "scaleway.iqm"
    backend_types = (IqmBackend,)

    def __init__(self, wires=None, shots=None, seed=None, **kwargs):

        super().__init__(wires=wires, kwargs=kwargs, shots=shots, seed=seed)

        self._default_shots = None
        if isinstance(shots, int):
            self._default_shots = shots

        self._handle_kwargs(**kwargs)

    def _handle_kwargs(self, **kwargs):
        ### Extract runner-specific arguments
        self._run_options = {
            k: v
            for k, v in kwargs.items()
            if k in signature(self._platform.run).parameters.keys()
        }
        [kwargs.pop(k) for k in self._run_options.keys()]
        self._run_options.update(
            {
                "session_name": self._session_options.get("name"),
                "session_max_duration": self._session_options.get("max_duration"),
                "session_max_idle_duration": self._session_options.get(
                    "max_idle_duration"
                ),
            }
        )

        if len(kwargs) > 0:
            warnings.warn(
                f"The following keyword arguments are not supported by '{self.name}' device: {list(kwargs.keys())}",
                UserWarning,
            )

    def preprocess(
        self,
        execution_config: ExecutionConfig | None = None,
    ) -> tuple[TransformProgram, ExecutionConfig]:
        transform_program, config = super().preprocess(execution_config)
        return transform_program, config
