import pennylane as qml
from pennylane import numpy as np

# Load your saved 128D identity vector
identity_vector = np.load("identity_vector.npy")

# Optionally compress to 4 or 8 features for limited qubits
# Here we'll take the first 4 values for simplicity
reduced_vector = identity_vector[:4] * np.pi  # Scale for rotation

# Set up quantum device (simulator with 4 qubits)
n_qubits = len(reduced_vector)
dev = qml.device("default.qubit", wires=n_qubits)

# Define a quantum circuit with angle encoding
@qml.qnode(dev)
def encode_identity(x):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    return qml.state()

# Run the circuit
quantum_state = encode_identity(reduced_vector)

# Display quantum state vector
print("Encoded quantum state:")
print(quantum_state)

# Optional: Save it for verification step
np.save("encoded_quantum_state.npy", quantum_state)
