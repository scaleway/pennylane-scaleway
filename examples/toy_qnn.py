import pennylane as qml
from pennylane import numpy as np

MAX_ITER = 100
EPSILON = 1e-9


with qml.device("scaleway.aer", wires=1, backend="EMU-AER-16C-128M") as dev:

    # Circuit definition, with a parameterized rotation
    @qml.set_shots(100)
    @qml.qnode(dev)
    def circuit(params):

        qml.RX(params, wires=0)

        # Output is 1 for state |0> and -1 for state |1>
        return qml.expval(qml.PauliZ(0))

    # Loss function
    def cost(params):
        prediction = circuit(params)
        target = -1.0  # TARGET -> We want the qubit to output -1 (the |1> state)
        return np.abs(prediction - target)

    opt = qml.GradientDescentOptimizer(stepsize=0.5)
    params = np.array(0.0, requires_grad=True)

    print(f"Initial rotation: {params:.4f} rad")

    for i in range(MAX_ITER):

        old_params = params.copy()
        params = opt.step(cost, params)
        delta_params = np.abs(params - old_params)

        cost_t = cost(params)

        print(f"Step {i}: Delta params = {delta_params:.4f} rad, Cost = {cost_t:.6f}")

        if cost_t < EPSILON and delta_params < EPSILON:
            print("Converged!")
            break

print(f"\nFinal rotation: {params % (2*np.pi):.4f} rad (Target: Pi approx 3.14)")
