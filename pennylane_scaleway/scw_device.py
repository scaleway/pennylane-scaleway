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
import os
import numpy as np
import warnings

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from pennylane.devices import Device, ExecutionConfig
from pennylane.tape import QuantumScriptOrBatch
from pennylane.transforms.core import TransformProgram

from qiskit_scaleway import ScalewayProvider
from qiskit_scaleway.backends import BaseBackend


class ScalewayDevice(Device, ABC):
    """A Base PennyLane device that runs on Scaleway. Used as interface for all platforms."""

    def __init__(
        self,
        wires: Union[None, int, Iterable[int]],
        kwargs: Dict[str, Any],
        shots: Union[None, int, Sequence[int]] = None,
        seed: Optional[int] = None,
    ):
        """
        Params:

            wires (int, Iterable): Number of subsystems represented by the device,
                or iterable that contains a unique label for each subsystem.
            shots (int, Sequence[int], ~pennylane.measurements.Shots): Number of circuit evaluations to run.
            **kwargs:
                - project_id (str): The Scaleway Quantum Project ID.
                - secret_key (str): The API token for authentication with Scaleway.
                - backend (str): The specific quantum backend to run on Scaleway.
                - url (str): The Scaleway API URL (optional).
                - session_name (str): Name of the session (optional).
                - deduplication_id (str): Unique deduplication identifier for session (optional).
                - max_duration (str): Maximum uptime session duration (e.g., "1h", "30m") (optional).
                - max_idle_duration (str): Maximum idle session duration (e.g., "1h", "5m") (optional).
        """

        super().__init__(wires=wires, shots=shots)
        self.tracker.persistent = True
        self._rng = np.random.default_rng(seed)

        ### Setup Scaleway API and backend
        backend = kwargs.pop("backend", None)

        self._provider = ScalewayProvider(
            project_id=kwargs.pop("project_id", os.getenv("SCW_PROJECT_ID", None)),
            secret_key=kwargs.pop("secret_key", os.getenv("SCW_SECRET_KEY", None)),
            url=kwargs.pop("url", None),
        )

        platforms = self._provider.backends()
        if backend not in [platform.name for platform in platforms]:
            raise ValueError(
                f"Platform '{backend}' not found. Available platforms are {[platform.name for platform in platforms if (platform.availability == 'available' and type(platform) in self.backend_types)]}."
            )

        self._platform = self._provider.get_backend(backend)
        if self._platform.availability != "available":
            raise RuntimeError(
                f"Platform '{backend}' is not available. Please try again later, or check availability at https://console.scaleway.com/qaas/sessions/create."
            )

        if self.num_wires > self._platform.num_qubits:
            warnings.warn(
                f"Number of wires ({self.num_wires}) exceeds the theoretical limit of qubits in the platform ({self._platform.num_qubits})."
                "This may lead to unexpected behavior and crash.",
                UserWarning,
            )

        ### Extract Scaleway's session-specific arguments
        self._session_options = {
            "name": kwargs.pop("session_name", None),
            "deduplication_id": kwargs.pop("deduplication_id", None),
            "max_duration": kwargs.pop("max_duration", None),
            "max_idle_duration": kwargs.pop("max_idle_duration", None),
        }

        self._session_id = None

    @abstractmethod
    def preprocess(
        self,
        execution_config: ExecutionConfig | None = None,
    ) -> tuple[TransformProgram, ExecutionConfig]:
        pass

    @abstractmethod
    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> List:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def backend_types(self) -> Tuple[BaseBackend]:
        pass

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
            self.tracker.reset()
            self._session_id = None
        else:
            raise RuntimeError("No session running.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def session_id(self):
        return self._session_id

    @property
    def num_wires(self):
        return len(self.wires)
