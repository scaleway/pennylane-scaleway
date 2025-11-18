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

import warnings

import pennylane as qml
from pennylane.tape import QuantumTape


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
