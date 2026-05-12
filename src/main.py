from grover import build_grover_circuit
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

def main():
    # Parameters
    n_qubits = 3
    target = 1   # Search for state |001>
    shots = 4096

    # Build circuit
    circuit = build_grover_circuit(n_qubits, target)

    # Use AerSimulator
    backend = AerSimulator()

    # Transpile for the backend (optional but good practice)
    transpiled_circuit = transpile(circuit, backend)

    # Run simulation
    result = backend.run(transpiled_circuit, shots=shots).result()
    counts = result.get_counts()

    print("Measurement results (counts):")
    print(counts)

    # Plot histogram
    plot_histogram(counts)
    plt.title(f"Grover search: {n_qubits} qubits, target={target}")
    plt.show()


if __name__ == "__main__":
    main()