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
from pennylane.tape.tape import rotations_and_diagonal_measurements

from qiskit.circuit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import library as lib
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import SparsePauliOp


QISKIT_OPERATION_MAP = {
    # native PennyLane operations also native to qiskit
    "PauliX": lib.XGate,
    "PauliY": lib.YGate,
    "PauliZ": lib.ZGate,
    "Hadamard": lib.HGate,
    "CNOT": lib.CXGate,
    "CZ": lib.CZGate,
    "SWAP": lib.SwapGate,
    "ISWAP": lib.iSwapGate,
    "RX": lib.RXGate,
    "RY": lib.RYGate,
    "RZ": lib.RZGate,
    "Identity": lib.IGate,
    "CSWAP": lib.CSwapGate,
    "CRX": lib.CRXGate,
    "CRY": lib.CRYGate,
    "CRZ": lib.CRZGate,
    "PhaseShift": lib.PhaseGate,
    "StatePrep": lib.Initialize,
    "Toffoli": lib.CCXGate,
    "QubitUnitary": lib.UnitaryGate,
    "U1": lib.U1Gate,
    "U2": lib.U2Gate,
    "U3": lib.U3Gate,
    "IsingZZ": lib.RZZGate,
    "IsingYY": lib.RYYGate,
    "IsingXX": lib.RXXGate,
    "S": lib.SGate,
    "T": lib.TGate,
    "SX": lib.SXGate,
    "Adjoint(S)": lib.SdgGate,
    "Adjoint(T)": lib.TdgGate,
    "Adjoint(SX)": lib.SXdgGate,
    "CY": lib.CYGate,
    "CH": lib.CHGate,
    "CPhase": lib.CPhaseGate,
    "CCZ": lib.CCZGate,
    "ECR": lib.ECRGate,
    "Barrier": lib.Barrier,
    "Adjoint(GlobalPhase)": lib.GlobalPhaseGate,
}


def accepted_sample_measurement(m: qml.measurements.MeasurementProcess) -> bool:
    """Specifies whether or not a measurement is accepted when sampling."""

    return isinstance(
        m,
        (
            qml.measurements.SampleMeasurement,
            qml.measurements.ClassicalShadowMP,
            qml.measurements.ShadowExpvalMP,
        ),
    )

# diagonalize is currently only used if measuring
# maybe always diagonalize when measuring, and never when not?
# will this be used for a user-facing function to convert from PL to Qiskit as well?
def circuit_to_qiskit(circuit, register_size, diagonalize=True, measure=True):
    """Builds the circuit objects based on the operations and measurements
    specified to apply.

    Args:
        circuit (QuantumTape): the circuit applied
            to the device
        register_size (int): the total number of qubits on the device the circuit is
            executed on; this must include any qubits not used in the given
            circuit to ensure correct indexing of the returned samples

    Keyword args:
        diagonalize (bool): whether or not to apply diagonalizing gates before the
            measurements
        measure (bool): whether or not to apply measurements at the end of the circuit;
            a full circuit is represented either as a Qiskit circuit with operations
            and measurements (measure=True), or a Qiskit circuit with only operations,
            paired with a Qiskit Estimator defining the measurement process.

    Returns:
        QuantumCircuit: the qiskit equivalent of the given circuit
    """

    reg = QuantumRegister(register_size)

    if not measure:
        qc = QuantumCircuit(reg, name="temp")

        for op in circuit.operations:
            qc &= operation_to_qiskit(op, reg)

        return qc

    creg = ClassicalRegister(register_size)
    qc = QuantumCircuit(reg, creg, name="temp")

    for op in circuit.operations:
        qc &= operation_to_qiskit(op, reg, creg)

    # rotate the state for measurement in the computational basis
    # ToDo: check this in cases with multiple different bases
    if diagonalize:
        rotations, measurements = rotations_and_diagonal_measurements(circuit)
        for _, m in enumerate(measurements):
            if m.obs is not None:
                rotations.extend(m.obs.diagonalizing_gates())

        for rot in rotations:
            qc &= operation_to_qiskit(rot, reg, creg)

    # barrier ensures we first do all operations, then do all measurements
    qc.barrier(reg)
    # we always measure the full register
    qc.measure(reg, creg)

    return qc


def operation_to_qiskit(operation, reg, creg=None):
    """Take a Pennylane operator and convert to a Qiskit circuit

    Args:
        operation (List[pennylane.Operation]): operation to be converted
        reg (Quantum Register): the total number of qubits on the device
        creg (Classical Register): classical register

    Returns:
        QuantumCircuit: a quantum circuit objects containing the translated operation
    """
    op_wires = operation.wires
    par = operation.parameters

    for idx, p in enumerate(par):
        if isinstance(p, np.ndarray):
            # Convert arrays so that Qiskit accepts the parameter
            par[idx] = p.tolist()

    operation = operation.name

    mapped_operation = QISKIT_OPERATION_MAP[operation]

    qregs = [reg[i] for i in op_wires.labels]

    # Need to revert the order of the quantum registers used in
    # Qiskit such that it matches the PennyLane ordering
    if operation in ("QubitUnitary", "StatePrep"):
        qregs = list(reversed(qregs))

    if creg:
        dag = circuit_to_dag(QuantumCircuit(reg, creg, name=""))
    else:
        dag = circuit_to_dag(QuantumCircuit(reg, name=""))
    gate = mapped_operation(*par)

    dag.apply_operation_back(gate, qargs=qregs)
    circuit = dag_to_circuit(dag)

    return circuit


def mp_to_pauli(mp, register_size):
    """Convert a Pauli observable to a SparsePauliOp for measurement via Estimator

    Args:
        mp(Union[ExpectationMP, VarianceMP]): MeasurementProcess to be converted to a SparsePauliOp
        register_size(int): total size of the qubit register being measured

    Returns:
        SparsePauliOp: the ``SparsePauliOp`` of the given Pauli observable
    """
    op = mp.obs

    if op.pauli_rep:
        pauli_strings = [
            "".join(
                ["I" if i not in pauli_term.wires else pauli_term[i] for i in range(register_size)][
                    ::-1
                ]  ## Qiskit follows opposite wire order convention
            )
            for pauli_term in op.pauli_rep.keys()
        ]
        coeffs = list(op.pauli_rep.values())
    else:
        raise ValueError(f"The operator {op} does not have a representation for SparsePauliOp")

    return SparsePauliOp(data=pauli_strings, coeffs=coeffs).simplify()
