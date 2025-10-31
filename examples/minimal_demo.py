import os
import pennylane as qml

# You do not need to import pennylane_scaleway BUT you need to have it installed ('pip install .' at the moment).

project_id=os.environ["SCW_PROJECT_ID"]
secret_key=os.environ["SCW_SECRET_KEY"]
url=os.getenv("SCW_API_URL")
instance=os.getenv("SCW_BACKEND_NAME")

with qml.device("scaleway.aer", wires=2, project_id=project_id, secret_key=secret_key, url=url, instance=instance) as dev:

    @qml.set_shots(1000)
    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    result = circuit()

print(result)
