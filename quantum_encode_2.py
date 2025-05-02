import pennylane as qml
from pennylane import numpy as np

# Step 1: Load the second captured identity vector
identity_vector = np.load("identity_vector_2.npy")

# Step 2: Reduce to first 4 features for quantum encoding
# (You can change 4 to 8 if your system can handle more qubits)
reduced_vector = identity_vector[:4] * np.pi  # scale for RY rotations

# Step 3: Set up quantum simulator with matching number of qubits
n_qubits = len(reduced_vector)
dev = qml.device("default.qubit", wires=n_qubits)

# Step 4: Define quantum circuit with angle encoding
@qml.qnode(dev)
def encode_identity(x):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    return qml.state()

# Step 5: Run circuit and get state vector
quantum_state = encode_identity(reduced_vector)

# Step 6: Print and save
print("Second identity quantum state encoded.")
print(quantum_state)

np.save("encoded_quantum_state_2.npy", quantum_state)
