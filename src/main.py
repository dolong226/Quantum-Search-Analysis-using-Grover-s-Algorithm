from grover import build_grover_circuit
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

def main():
    n_qubits = 3
    target = 1  
    shots = 4096

    circuit = build_grover_circuit(n_qubits, target)

    backend = AerSimulator()

    transpiled_circuit = transpile(circuit, backend)

    result = backend.run(transpiled_circuit, shots=shots).result()
    counts = result.get_counts()

    print("Measurement results (counts):")
    print(counts)

    plot_histogram(counts)
    plt.title(f"Grover search: {n_qubits} qubits, target={target}")
    plt.show()


if __name__ == "__main__":
    main()