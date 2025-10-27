import numpy as np
import pennylane as qml
from pennylane.devices import Device, ExecutionConfig
from pennylane.tape import QuantumScript, QuantumScriptOrBatch

from qiskit_scaleway import ScalewayProvider


class AerDevice(Device):
    """
    The way to call it:
        device = qml.device("scaleway.aer", wires=XXX, project_id=XXX, secret_key=XXX)
    """

    name = "scaleway.aer"

    def __init__(
            self,
            wires=None,
            shots=None,
            seed=None,
            **kwargs
        ):
        super().__init__(wires=wires, shots=shots)

        # seed and rng not necessary for a device, but part of recommended
        # numpy practices to use a local random number generator
        self._rng = np.random.default_rng(seed)

        self._instance = kwargs.get("instance", "aer_simulation_pop_c16m128")
        self._provider = ScalewayProvider(
            project_id=kwargs.get("project_id"),
            secret_key=kwargs.get("secret_key"),
            api_url=kwargs.get("api_url"),
        )

        self._backend = self._provider.get_backend(self._instance)
        if self._backend.availability != "available":
            raise RuntimeError(f"Backend {self._instance} is not available")

        self._session_id = self._backend.start_session(name="all-you-need-is-love")

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None
    ):
        return 0.0 if isinstance(circuits, qml.tape.QuantumScript) else tuple(0.0 for c in circuits)

    def __del__(self):
        self._backend.stop_session(self._session_id)


if __name__ == "__main__":
    dev = AerDevice()

    @qml.qnode(dev)
    def circuit():
        return qml.state()

    print(circuit())
