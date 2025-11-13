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
from typing import Dict, Set, Tuple
import numpy as np

from pennylane.measurements import SampleMP, CountsMP
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
                    c = randint_except(n_wires, {j})
                    cc = randint_except(n_wires, {j, c})
                    wires = [j, c, cc]
                elif op.num_wires == n_wires:
                    wires = list(range(n_wires))
                else:
                    wires = j

                args = {"wires": wires}

                parameters = dict(signature(op).parameters)
                if "phi" in parameters:
                    args["phi"] = random.random() * 2 * math.pi
                if "theta" in parameters:
                    args["theta"] = random.random() * 2 * math.pi
                if "delta" in parameters:
                    args["delta"] = random.random() * 2 * math.pi

                op(**args)

        measurement = random.choice(MEASUREMENTS)
        if measurement in ESTIMATOR_MEASUREMENTS:
            basis = random.choice([qml.PauliX, qml.PauliY, qml.PauliZ])
            return measurement(basis(wires=random.randint(0, n_wires - 1)))
        else:
            return measurement(
                wires=random.sample(range(n_wires), random.randint(1, n_wires))
            )

    return circuit


def dict_to_ndarray(a: Dict, b: Dict) -> Tuple[np.ndarray, np.ndarray]:

    for k in a.keys():
        if k not in b:
            b[k] = 0
    for k in b.keys():
        if k not in a:
            a[k] = 0

    # Make sure the dictionary is sorted by keys
    a = dict(sorted(a.items()))
    b = dict(sorted(b.items()))

    a = np.array(list(a.values()))
    b = np.array(list(b.values()))

    shots = np.sum(a)
    if shots != np.sum(b):
        raise ValueError("Number of shots must be equal")

    return a / shots, b / shots


def samples_to_probs(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    shots = a.shape[0]
    a = np.sum(a, axis=0) / shots
    b = np.sum(b, axis=0) / shots
    return a, b


def statistical_match(result_a, result_b, circuit, tolerance=0.1):

    if type(result_b) != type(result_a):
        return False

    if isinstance(result_a, (int, float)):
        return abs(result_a - result_b) < tolerance  # Direct scalar comparison

    try:
        tape = circuit._tape
    except Exception as e:
        raise ValueError(
            f"Could not construct tape from circuit. "
            f"Ensure circuit can be called with no arguments. Error: {e}"
        )

    if not tape.measurements:
        raise ValueError("Circuit has no measurements to compare.")

    measurement = tape.measurements[0]
    m_type = type(measurement)

    if m_type == CountsMP and isinstance(result_a, dict):
        result_a, result_b = dict_to_ndarray(result_a, result_b)

    elif m_type == SampleMP and isinstance(result_a, np.ndarray):
        result_a, result_b = samples_to_probs(result_a, result_b)

    return np.allclose(result_a, result_b, atol=tolerance)


def main():

    n_wires = 20
    n_layers = 20
    shots = 4096
    seed = None  # Setup the seed you want to test, in order to reproduce the same circuit, otherwise leave to None

    print_ascii_art()

    default_device = qml.device("default.qubit", wires=n_wires)
    scw_device = qml.device(
        "scaleway.aer",
        wires=n_wires,
        project_id=SCW_PROJECT_ID,
        secret_key=SCW_SECRET_KEY,
        backend=SCW_BACKEND_NAME,
        url=SCW_API_URL,
    )

    try:
        for i in range(100):
            if not seed:
                seed = random.randint(0, 1000000000)

            default_circuit = random_circuit_generator(
                default_device,
                n_wires=n_wires,
                n_layers=n_layers,
                seed=seed,
                shots=shots,
            )
            default_result = default_circuit()

            scw_circuit = random_circuit_generator(
                scw_device,
                n_wires=n_wires,
                n_layers=n_layers,
                seed=seed,
                shots=shots,
            )
            scw_result = scw_circuit()

            assert statistical_match(
                default_result, scw_result, default_circuit
            ), f"ERROR: Results do not match!"

            print(f"#{i:03d} - Seed: {seed} - ✓")

            seed = None

    except Exception as e:
        print(f"EXCEPTION RAISED: {e}")
        print(f"Seed: {seed}")

        random.seed(seed)

        print("=" * 50)
        print("\nCIRCUIT:\n")
        print(qml.draw(default_circuit)())

        print("=" * 50)
        print("\nDEFAULT RESULTS:\n")
        print(default_result)

        print("=" * 50)
        print("\nSCALEWAY RESULTS:\n")
        print(scw_result)

    finally:
        scw_device.stop()

    print("Done!")


def print_ascii_art():
    art = r"""
  █████   ███▄    █  ▒█████  ▓█████▄ ▓█████         
▒██▓  ██▒ ██ ▀█   █ ▒██▒  ██▒▒██▀ ██▌▓█   ▀         
▒██▒  ██░▓██  ▀█ ██▒▒██░  ██▒░██   █▌▒███           
░██  █▀ ░▓██▒  ▐▌██▒▒██   ██░░▓█▄   ▌▒▓█  ▄         
░▒███▒█▄ ▒██░   ▓██░░ ████▓▒░░▒████▓ ░▒████▒        
░░ ▒▒░ ▒ ░ ▒░   ▒ ▒ ░ ▒░▒░▒░  ▒▒▓  ▒ ░░ ▒░ ░        
 ░ ▒░  ░ ░ ░░   ░ ▒░  ░ ▒ ▒░  ░ ▒  ▒  ░ ░  ░        
   ░   ░    ░   ░ ░ ░ ░ ░ ▒   ░ ░  ░    ░           
    ░             ░     ░ ░     ░       ░  ░        
                              ░                     
  ██████ ▄▄▄█████▓ ██▀███  ▓█████   ██████   ██████ 
▒██    ▒ ▓  ██▒ ▓▒▓██ ▒ ██▒▓█   ▀ ▒██    ▒ ▒██    ▒ 
░ ▓██▄   ▒ ▓██░ ▒░▓██ ░▄█ ▒▒███   ░ ▓██▄   ░ ▓██▄   
  ▒   ██▒░ ▓██▓ ░ ▒██▀▀█▄  ▒▓█  ▄   ▒   ██▒  ▒   ██▒
▒██████▒▒  ▒██▒ ░ ░██▓ ▒██▒░▒████▒▒██████▒▒▒██████▒▒
▒ ▒▓▒ ▒ ░  ▒ ░░   ░ ▒▓ ░▒▓░░░ ▒░ ░▒ ▒▓▒ ▒ ░▒ ▒▓▒ ▒ ░
░ ░▒  ░ ░    ░      ░▒ ░ ▒░ ░ ░  ░░ ░▒  ░ ░░ ░▒  ░ ░
░  ░  ░    ░        ░░   ░    ░   ░  ░  ░  ░  ░  ░  
      ░              ░        ░  ░      ░        ░  
                                                    
▄▄▄█████▓▓█████   ██████ ▄▄▄█████▓                  
▓  ██▒ ▓▒▓█   ▀ ▒██    ▒ ▓  ██▒ ▓▒                  
▒ ▓██░ ▒░▒███   ░ ▓██▄   ▒ ▓██░ ▒░                  
░ ▓██▓ ░ ▒▓█  ▄   ▒   ██▒░ ▓██▓ ░                   
  ▒██▒ ░ ░▒████▒▒██████▒▒  ▒██▒ ░                   
  ▒ ░░   ░░ ▒░ ░▒ ▒▓▒ ▒ ░  ▒ ░░                     
    ░     ░ ░  ░░ ░▒  ░ ░    ░                      
  ░         ░   ░  ░  ░    ░                        
            ░  ░      ░                             
    """
    print(art)


if __name__ == "__main__":
    main()
