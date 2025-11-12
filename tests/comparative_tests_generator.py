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
import random
from inspect import signature
import math
from typing import Set
import numpy as np

import pennylane as qml

# Credentials
SCW_PROJECT_ID = os.environ["SCW_PROJECT_ID"]
SCW_SECRET_KEY = os.environ["SCW_SECRET_KEY"]
SCW_BACKEND_NAME = os.getenv("SCW_BACKEND_NAME", "aer_simulation_pop_c16m128")
SCW_API_URL = os.getenv("SCW_API_URL")


ESTIMATOR_MEASUREMENTS = [
    qml.expval,
    qml.var,
]
SAMPLER_MEASUREMENTS = [
    qml.sample,
    qml.probs,
    qml.counts,
]
MEASUREMENTS = ESTIMATOR_MEASUREMENTS + SAMPLER_MEASUREMENTS

OPERATIONS = [
    qml.PauliX,
    qml.PauliY,
    qml.PauliZ,
    qml.Hadamard,
    qml.CNOT,
    qml.CZ,
    qml.SWAP,
    qml.ISWAP,
    qml.RX,
    qml.RY,
    qml.RZ,
    qml.Identity,
    qml.CSWAP,
    qml.CRX,
    qml.CRY,
    qml.CRZ,
    qml.PhaseShift,
    qml.Toffoli,
    qml.U1,
    qml.U2,
    qml.U3,
    qml.IsingZZ,
    qml.IsingYY,
    qml.IsingXX,
    qml.S,
    qml.T,
    qml.SX,
    qml.CY,
    qml.CH,
    qml.CPhase,
    qml.CCZ,
    qml.ECR,
    qml.Barrier,
]


def randint_except(n: int, exclude: Set[int]):
    choice = set(range(n)) - exclude
    return random.choice(list(choice))

def random_circuit_generator(device, n_wires, n_layers, shots=1024, seed=None):

    random.seed(seed)

    @qml.set_shots(shots)
    @qml.qnode(device)
    def circuit():
        for i in range(n_layers):
            for j in range(n_wires):

                op = random.choice(OPERATIONS)
                if op.num_wires == 2:
                    c = randint_except(n_wires, {j})
                    wires = [j, c]
                elif op.num_wires == 3:
                    c =  randint_except(n_wires, {j})
                    cc = randint_except(n_wires, {j,c})
                    wires = [j, c, cc]
                elif op.num_wires == n_wires:
                    wires = list(range(n_wires))
                else:
                    wires = j

                args = {'wires': wires}

                parameters = dict(signature(op).parameters)
                if 'phi' in parameters:
                    args['phi'] = random.random() * 2 * math.pi
                if 'theta' in parameters:
                    args['theta'] = random.random() * 2 * math.pi
                if 'delta' in parameters:
                    args['delta'] = random.random() * 2 * math.pi

                op(**args)

        measurement = random.choice(MEASUREMENTS)
        if measurement in ESTIMATOR_MEASUREMENTS:
            basis = random.choice([qml.PauliX, qml.PauliY, qml.PauliZ])
            return measurement(basis(wires=random.randint(0, n_wires)))
        else:
            return measurement(wires=random.sample(range(n_wires), random.randint(1, n_wires)))

    return circuit


def statistical_match(result_a, result_b, measurement_type, tolerance=0.1):
    if measurement_type == "expectation":
        return abs(result_a - result_b) < tolerance * (abs(result_a) + abs(result_b)) / 2
    elif measurement_type == "variance":
        return abs(result_a - result_b) < tolerance * (abs(result_a) + abs(result_b)) / 2
    else:
        raise ValueError(f"Unknown measurement type: {measurement_type}")


if __name__ == "__main__":

    n_wires = 10
    n_layers = 20
    seed = 42
    shots = 4096

    default_device = qml.device("default.qubit", wires=n_wires)
    scw_device = qml.device("scaleway.aer", wires=n_wires, project_id=SCW_PROJECT_ID, secret_key=SCW_SECRET_KEY, backend=SCW_BACKEND_NAME, url=SCW_API_URL)

    try:

        for _ in range(3):

            default_circuit = random_circuit_generator(default_device, n_wires=n_wires, n_layers=n_layers, seed=seed, shots=shots)

            print("="*50)
            print("\nDEFAULT CIRCUIT:\n")
            print(qml.draw(default_circuit)())
            print("\nDEFAULT RESULTS:\n")
            default_result = default_circuit()
            print(default_result)

            scw_circuit = random_circuit_generator(scw_device, n_wires=n_wires, n_layers=n_layers, seed=seed, shots=shots)

            print("="*50)
            print("\nSCALEWAY CIRCUIT:\n")
            print(qml.draw(scw_circuit)())
            print("\nSCALEWAY RESULTS:\n")
            scw_result = scw_circuit()
            print(scw_result)

            # assert statistical_match(default_result, scw_result, measurement_type), f"Results do not match!"

            seed = random.randint(0, 1000000000)
            print(f"Seed: {seed}")
    
    # except Exception as e:
    #     print(f"EXCEPTION RAISED: {e}")
    #     print(qml.to_openqasm(scw_circuit)())
    #     raise Exception(e) from e

    finally:
        scw_device.stop()

    print("Done!")
