import os
import pennylane as qml

# You do not need to import pennylane_scaleway BUT you need to have it installed.

project_id = os.environ["SCW_PROJECT_ID"]
secret_key = os.environ["SCW_SECRET_KEY"]
backend = os.getenv("SCW_BACKEND_NAME")

with qml.device(
    "scaleway.aer",
    wires=2,
    project_id=project_id,
    secret_key=secret_key,
    backend=backend,
) as dev:

    @qml.set_shots(100)
    @qml.qnode(dev)
    def bell_state_circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    result = bell_state_circuit()

print(result)
