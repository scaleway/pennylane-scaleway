
<table align="center" bgcolor="black">
    <tr>
        <td bgcolor="black" align="center" width="300"><img src="assets/pennylane-logo.png" width="300" alt="Pennylane Logo"></td>
        <td valign="middle" style="font-size: 24px; font-weight: bold; padding: 0 20px;">×</td>
        <td bgcolor="black" align="center" width="300"><img src="assets/scaleway-logo.png" width="200" alt="Scaleway Logo"></td>
    </tr>
</table>

# Pennylane devices running on Scaleway's Quantum-as-a-Service
**[Pennylane](https://pennylane.ai/)** is an open-source framework for quantum machine learning, automatic differentiation, and optimization of hybrid quantum-classical computations.

**Pennylane-Scaleway** is a Python package to run pennylane's QML circuits on **[Scaleway](https://www.scaleway.com/)** infrastructure, providing access to:
- [Aer](https://github.com/Qiskit/qiskit-aer) state vector and tensor network multi-GPU emulators
- [AQT](https://www.aqt.eu/) trapped-ions quantum computers
- *Coming Soon!* - [IQM](https://meetiqm.com/) superconducting quantum computers

More info on the **[Quantum service web page](https://www.scaleway.com/en/quantum-as-a-service/)**.

## Installation

### Prerequisites

Get your `project-id` as well as `secret-key` credentials from your Scaleway account.
You can create and find them in the **[Scaleway console](https://console.scaleway.com/)**. Here how to create an [API key](https://www.scaleway.com/en/docs/iam/how-to/create-api-keys/).

For more information about the device you want to use, its pricing and capabilities, you can visit [this page](https://www.scaleway.com/fr/quantum-as-a-service/). Use the `backend` parameter to select the device you want to use (for instance `backend="EMU-IBEX-12PQ-L4"` if you want to try AQT emulation using a L4 GPU).

### Supported device names

The following device names are supported:
 - `scaleway.aer` - Aer emulation, offers flexibility, noiseless by default but can handle given Aer's noise models, large choice of backends.
 - `scaleway.aqt` - AQT (Alpine Quantum Technologies), noisy trapped-ions based quantum computers.
 - `scaleway.iqm` - __*Coming soon*__ IQM, superconducting quantum computers.

### Install the package
We encourage installing Scaleway provider via pip:

```bash
pip install pennylane-scaleway
```

## Getting started

To run your pennylane's circuits on Scaleway's quantum backends, simply change your device's name and add your `project_id` and `secret_key` (OR set the environment variables `SCW_PROJECT_ID` and `SCW_SECRET_KEY`):
```python
import pennylane as qml # No need to import pennylane-scaleway as long as it is installed in your current environment.

device = qml.device("scaleway.aer",
        wires=2,
        project_id=<your-project-id>,   # Or set SCW_PROJECT_ID environment variable
        secret_key=<your-secret-key>,   # Or set SCW_SECRET_KEY environment variable
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

with qml.device("scaleway.aqt",
    project_id=<your-project-id>,
    secret_key=<your-secret-key>,
    backend="EMU-IBEX-12PQ-L4",
) as dev:

    @qml.set_shots(512)
    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.counts()

    print(circuit())
```

> **Friendly reminder** to avoid writing your credentials directly in your code. Use environment variables instead, load from a .env file or any secret management technique of your choice.

## Session management
A QPU session is automatically created when you instantiate a device. You can manage it manually by calling `device.start()` and `device.stop()`, but it is recommended to use the context manager approach instead. You may also attach to an existing session, handle maximum session duration and idle duration by setting these as keyword arguments when instantiating the device. For example:

```python
import pennylane as qml

with qml.device("scaleway.aer",
    wires=2,
    project_id=<your-project-id>,
    secret_key=<your-secret-key>,
    backend="EMU-AER-16C-128M",
    max_duration="1h",
    max_idle_duration="5m"
) as dev:
...
```

You can visualize your sessions on the **[Scaleway Console](https://console.scaleway.com/)** under the Labs/Quantum section.

## Documentation
Documentation is available at **[Scaleway Docs](https://www.scaleway.com/en/docs/)**.

You can find examples under the [examples folder](doc/examples/) of this repository.

## Development
This repository is in a very early stage and is still in active development. If you are looking for a way to contribute please read [CONTRIBUTING.md](CONTRIBUTING.md).

## Reach us
We love feedback. Feel free to reach us on [Scaleway Slack community](https://slack.scaleway.com/), we are waiting for you on [#opensource](https://scaleway-community.slack.com/app_redirect?channel=opensource)..

## License
[License Apache 2.0](LICENSE)
