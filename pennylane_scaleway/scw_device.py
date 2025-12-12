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

from abc import ABC, abstractmethod
from inspect import signature
import coolname
from dataclasses import replace
import numpy as np
import os
from tenacity import retry, stop_after_attempt, stop_after_delay
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import warnings

from pennylane.devices import Device, ExecutionConfig
from pennylane.devices.preprocess import (
    decompose,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from pennylane.tape import QuantumScriptOrBatch, QuantumScript
from pennylane.transforms import split_non_commuting, broadcast_expand
from pennylane.transforms.core import TransformProgram

from pennylane_scaleway.utils import (
    accepted_sample_measurement,
    analytic_warning,
    circuit_to_qiskit,
)

from qiskit.result import Result

from qiskit_scaleway import ScalewayProvider
from qiskit_scaleway.backends import BaseBackend


class ScalewayDevice(Device, ABC):
    """A Base PennyLane device that runs on Scaleway. Used as interface for all platforms."""

    operations = None
    observables = None

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

        self._default_shots = None
        if shots and not isinstance(shots, int):
            raise ValueError(
                "Only integer number of shots is supported on this device (vectors are not supported either). The set 'shots' value will be ignored."
            )
        elif isinstance(shots, int):
            self._default_shots = shots

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
                f"Number of wires ({self.num_wires}) exceeds the limit of qubits in the platform ({self._platform.num_qubits}). "
                "This will probably lead to unexpected behavior and crash.",
                UserWarning,
            )

        ### Extract Scaleway's session-specific arguments
        self._session_options = {
            "name": kwargs.pop("session_name", None),
            "deduplication_id": kwargs.pop(
                "deduplication_id", coolname.generate_slug(2)
            ),
            "max_duration": kwargs.pop("max_duration", None),
            "max_idle_duration": kwargs.pop("max_idle_duration", None),
        }

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

        self._session_id = None

    def preprocess(
        self,
        execution_config: ExecutionConfig | None = None,
    ) -> tuple[TransformProgram, ExecutionConfig]:
        transform_program = TransformProgram()

        config = replace(execution_config, use_device_gradient=False)

        transform_program.add_transform(analytic_warning)
        transform_program.add_transform(
            validate_device_wires, self.wires, name=self.name
        )
        transform_program.add_transform(
            decompose,
            stopping_condition=self.stop_validating_operations,
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
            stopping_condition=self.stop_validating_observables,
            name=self.name,
        )

        transform_program.add_transform(broadcast_expand)
        transform_program.add_transform(split_non_commuting)

        return transform_program, config

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> List:
        if not self._session_id:
            raise RuntimeError(
                "No active session. Please instanciate the device using a context manager, or call start() first. You can also attach to an existing deduplication_id."
            )

        if isinstance(circuits, QuantumScript):
            circuits = [circuits]

        qiskit_circuits = []
        for circuit in circuits:
            qiskit_circuits.append(
                circuit_to_qiskit(
                    circuit, self.num_wires, diagonalize=True, measure=True
                )
            )

        shots = self._default_shots
        if circuits[0].shots and circuits[0].shots.total_shots:
            shots = circuits[0].shots.total_shots

        if shots is None:
            raise ValueError(
                "Number of shots must be specified, "
                "either through a default value when instanciating the device, "
                "or preferably using the set_shots() decorator on the circuit. "
                "If you run an analytic measurement, you still need to set a shots value. "
                "Setting higher shots will result in better precision."
            )

        @retry(stop=stop_after_attempt(3) | stop_after_delay(3 * 60), reraise=True)
        def run() -> Union[Result, List[Result]]:
            return self._platform.run(
                qiskit_circuits,
                session_id=self._session_id,
                shots=shots,
                **self._run_options,
            ).result()

        results = run()
        if isinstance(results, Result):
            results = [results]

        counts = []
        for result in results:
            if isinstance(result.get_counts(), dict):
                counts.append(result.get_counts())
            else:
                counts.extend([count for count in result.get_counts()])

        all_results = []
        for original_circuit, qcirc, count in zip(circuits, qiskit_circuits, counts):
            # Reconstruct the list of samples from the counts dictionary
            samples_list = []
            for key, value in count.items():
                samples_list.extend([key] * value)

            if not samples_list:
                # Handle case with no samples (e.g., 0 shots)
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

            single_measurement = len(original_circuit.measurements) == 1
            res_tuple = res[0] if single_measurement else tuple(res)
            all_results.append(res_tuple)

        return all_results

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def backend_types(self) -> Tuple[BaseBackend]:
        pass

    def stop_validating_observables(self, obs):
        if not self.observables:
            return True
        return obs.name in self.observables

    def stop_validating_operations(self, op):
        if not self.operations:
            return True
        return op.name in self.operations

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
