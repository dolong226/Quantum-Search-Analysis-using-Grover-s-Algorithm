import math
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import Aer, AerSimulator


# ORACLE builder

def build_phase_oracle(n_qubits: int, target_index: int) -> QuantumCircuit:
    """ Đánh dấu trạng thái đích = đảo pha """

    # check target_index
    if not (0 <= target_index < 2 ** n_qubits) :
        raise ValueError(
            f"target_index={target_index} phải nằm trong [0, {2**n_qubits})."
        )
    
    # Init quantum circuit
    oracle_circuit = QuantumCircuit(n_qubits, name="Oracle")

    binary_target = format(target_index, f"0{n_qubits}")

    # B1: Add X
    flip_positions = []
    for qubit_idx in range(n_qubits):
        bit_position = n_qubits - 1 - qubit_idx
        if binary_target[bit_position] == "0":
            oracle_circuit.x(qubit_idx)
            flip_positions.append(qubit_idx)

    oracle_circuit.barrier() 

    # B2: MCZ
    control_qubits = list(range(n_qubits - 1))
    target_qubit = n_qubits - 1
    oracle_circuit.h(target_qubit)
    oracle_circuit.mcx(control_qubits, target_qubit)
    oracle_circuit.h(target_qubit)

    oracle_circuit.barrier()

    # B3: Uncompute
    for qubit_idx in flip_positions:
        oracle_circuit.x(qubit_idx)

    return oracle_circuit


# DIFFUSION OPERATOR

def build_diffusion_operator(n_qubits: int) -> QuantumCircuit:
    diffusion = QuantumCircuit(n_qubits, name="Diffusion")

    # B1: Hadamard -> 0000..00
    diffusion.h(range(n_qubits))

    diffusion.barrier()

    # B2: Phase flip -> 111.11
    diffusion.x(range(n_qubits))

    control_qubits = list(range(n_qubits - 1))
    target_qubit = n_qubits - 1
    diffusion.h(target_qubit)
    diffusion.mcx(control_qubits, target_qubit)
    diffusion.h(target_qubit)

    diffusion.x(range(n_qubits))

    diffusion.barrier()

    diffusion.h(range(n_qubits))

    return diffusion


# GROVER circuit builder

def build_grover_circuit(n_qubits: int, target_index: int, n_iterations: Optional[int] = None) -> QuantumCircuit:
    """|0...0⟩ → [H^⊗n] → [Oracle + Diffusion] × k → [Measure]

        Số vòng lặp tối ưu:
        k_opt = round(π/4 * √(N/M))   (M là số phần tử thỏa mãn điều kiện)
    """
    # tính số vòng lặp tối ưu
    n_states = 2 ** n_qubits
    k_optimal = round(math.pi / 4 * math.sqrt(n_states))
    k_optimal = max(1, k_optimal)

    if n_iterations is None: 
        n_iterations = k_optimal

    # init
    qreg = QuantumRegister(n_qubits, name="q")
    creg = ClassicalRegister(n_qubits, name="c")
    circuit = QuantumCircuit(qreg, creg)

    # B1: Superposition
    circuit.h(qreg)
    circuit.barrier(label="Init")

    # B2: Oracle + Diffusion
    oracle = build_phase_oracle(n_qubits, target_index)
    diffusion = build_diffusion_operator(n_qubits)

    # B3: Grover Iterations
    for iteration in range(n_iterations):
        circuit.append(oracle, qreg)
        circuit.append(diffusion, qreg)
        circuit.barrier(label=f"Iter {iteration + 1}")

    # B4: Measurement
    circuit.measure(qreg, creg)

    return circuit


# SIMULATOR & RUNNER

def run_grover_simulation(n_qubits: int, target_index: int, n_iterations: int, n_shots: int = 1024, noise_model=None) -> dict:
    circuit = build_grover_circuit(n_qubits, target_index, n_iterations)

    # Init simulator
    if noise_model is not None:
        simulator = AerSimulator(noise_model=noise_model)
    else:
        simulator = AerSimulator()

    transpiled = transpile(circuit, simulator)
    job = simulator.run(transpiled, shots=n_shots)
    result = job.result()

    counts = result.get_counts()

    return counts

def calculate_success_probability(counts: dict, target_index: int, n_qubits: int) -> float:
    total_shots = sum(counts.values)

    if total_shots == 0:
        return 0.0
    
    target_bitstring = format(target_index, f"0{n_qubits}b")

    target_count = counts.get(target_bitstring, 0)

    success_prob = target_count / total_shots

    return success_prob

def get_theoretical_success_probability(n_qubits: int, n_iterations: int) -> float:
    n_states = 2 ** n_qubits
    theta = math.asin(1.0 / math.sqrt(n_states))
    angle = (2 * n_iterations + 1) * theta
    prob = math.sin(angle) ** 2

    return prob

