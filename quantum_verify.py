import pennylane as qml
from pennylane import numpy as np

# Step 1: Load both quantum-encoded identity states
reference_state = np.load("encoded_quantum_state.npy")       # registered face
prover_state = np.load("encoded_quantum_state_2.npy")        # new face to verify

# Step 2: Compute fidelity (similarity between quantum state vectors)
fidelity_score = np.abs(np.vdot(reference_state, prover_state)) ** 2

# Step 3: Display result
print(f"Fidelity between reference and prover: {fidelity_score:.6f}")

if fidelity_score >= 0.95:
    print("✅ Identity Verified")
else:
    print("❌ Identity NOT Verified")
