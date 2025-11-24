# Pennylane's integration with Scaleway Quantum-as-a-Service
Scaleway provider implementation for Pennylane SDK (Quantum Machine Learning Framework).

**Pennylane-Scaleway** is a Python package to run pennylane's QML circuits on [Scaleway](https://www.scaleway.com/) infrastructure, providing access to:
- [Aer](https://github.com/Qiskit/qiskit-aer) state vector and tensor network multi-GPU emulators
- [AQT](https://www.aqt.eu/) ion-trapped quantum computers
- *Coming Soon!* [IQM](https://meetiqm.com/) superconducting quantum computers

More info on the [Quantum service web page](https://www.scaleway.com/en/quantum-as-a-service/).

## Installation

We encourage installing Scaleway provider via the pip tool (a Python package manager):

```bash
pip install pennylane-scaleway
```

## Getting started

To run your pennylane's circuits on Scaleway's quantum backends, simply change your device's name and add your project_id and secret_key:
```python
import pennylane as qml

device = qml.device("scaleway.aer",
        wires=2,
        project_id=<your-project-id>,
        secret_key=<your-secret-key>,
        backend="EMU-AER-16C-128M"
    )

@qml.set_shots(512)
@qml.qnode(device)
def my_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.counts()

print(my_circuit())

device.stop() # Don't forget to close the session when you're done!
```

You can also use the device as a context manager so your session is automatically closed when exiting the context:
```python
import pennylane as qml

with qml.device("scaleway.aer",
    wires=2,
    project_id=<your-project-id>,
    secret_key=<your-secret-key>,
    backend="EMU-AER-16C-128M"
) as dev:

    @qml.set_shots(512)
    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.counts()

    print(circuit())
```


Define a quantum circuit and run it

```python
# Define a quantum circuit that produces a 4-qubit GHZ state.
qc = QuantumCircuit(4)
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)
qc.cx(0, 3)
qc.measure_all()

## Development
This repository is in a very early stage and is still in active development. If you are looking for a way to contribute please read [CONTRIBUTING.md](CONTRIBUTING.md).

## Reach us
We love feedback. Feel free to reach us on [Scaleway Slack community](https://slack.scaleway.com/), we are waiting for you on [#opensource](https://scaleway-community.slack.com/app_redirect?channel=opensource)..

## License
[License Apache 2.0](LICENSE)
