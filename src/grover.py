import math
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit.circuit.library import GroverOperator
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import Statevector


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

    binary_target = format(target_index, f"0{n_qubits}b")

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
    total_shots = sum(counts.values())

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


# STATEVECTOR ANALYSIS

def get_statevector_probabilities(n_qubits: int, target_index: int, n_iterations: int) -> np.ndarray:
    circuit_no_measure = build_grover_circuit(n_qubits, target_index, n_iterations)
    circuit_no_measure.remove_final_measurements(inplace=False)

    circuit_sv = QuantumCircuit(n_qubits)
    circuit_sv.h(range(n_qubits))

    oracle = build_phase_oracle(n_qubits, target_index)
    diffusion = build_diffusion_operator(n_qubits)

    for _ in range(n_iterations):
        circuit_sv.append(oracle, range(n_qubits))
        circuit_sv.append(diffusion, range(n_qubits))

    sv = Statevector.from_instruction(circuit_sv)

    prob = np.abs(sv.data) ** 2

    return prob

# OPTIMAL ITERATION ANALYSIS

# Check xem grover có chính xác không (so sánh mô phỏng mạch và lý thuyết)
def analyze_iteration_sweep(n_qubits: int, target_index: int, max_iterations: Optional[int] = None) -> dict:
    """Phân tích xs thành công theo số vòng lặp Grover."""

    n_states = 2 ** n_qubits

    # Tính số vòng lặp tối đa (2 chu kỳ)
    if max_iterations is None:
        max_iterations = int(2 * math.pi / 2 * math.sqrt(n_states)) + 1
        max_iterations = min(max_iterations, 50) 

    k_optimal = round(math.pi / 4 * math.sqrt(n_states))

    iterations_list = list(range(max_iterations + 1))
    theoratical_prob = []
    simulated_probs = []

    for k in iterations_list:
        # Xác suất lý thuyết
        theo_prob = get_theoretical_success_probability(n_qubits, k)
        theoratical_prob.append(theo_prob)

        # Xác suất mô phỏng
        if k == 0:
            # trạng thái uniform superposition
            sim_prob = 1.0 / n_states
        else:
            probs = get_statevector_probabilities(n_qubits, target_index, k)
            sim_prob = float(probs[target_index])
        simulated_probs.append(sim_prob)

    return {
        "iterations": iterations_list,
        "theoretical_probs": theoratical_prob,
        "simulated_probs": simulated_probs,
        "optimal_k": k_optimal,
        "n_states": n_states,
        "n_qubits": n_qubits,
        "target_index": target_index,
    }

# Classical search baseline

def classical_search_expected_queries(n_states: int, n_targets: int = 1) -> float:
    return (n_states + 1) / (n_targets + 1)

def quantum_seach_queries(n_states: int) -> float:
    return math.pi / 4 * math.sqrt(n_states)

# main demo
if __name__ == "__main__":
    print("Demo 3 qubits")

    N_QUBITS = 3
    TARGET = 5
    N_SHOTS = 2048

    n_states = 2 ** N_QUBITS
    k_opt = round(math.pi / 4 * math.sqrt(n_states))
    print(f"  Vòng lặp tối ưu: {k_opt}")

    # XS Lý thuyết
    theo_prob = get_theoretical_success_probability(N_QUBITS, k_opt)
    print(f"\nXác suất lý thuyết (k={k_opt}): {theo_prob:.4f} ({theo_prob*100:.2f}%)")

    counts = run_grover_simulation(N_QUBITS, TARGET, k_opt, N_SHOTS)
    success_prob = calculate_success_probability(counts, TARGET, N_QUBITS)
    print(f"Xác suất thực nghiệm         : {success_prob:.4f} ({success_prob*100:.2f}%)")

     # Top 3 kết quả
    print(f"\nTop 3 trạng thái đo được:")
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for bitstring, count in sorted_counts[:3]:
        idx = int(bitstring, 2)
        prob = count / N_SHOTS
        marker = " <- TARGET" if idx == TARGET else ""
        print(f"  |{bitstring}⟩ (={idx}): {count:4d} shots ({prob:.3f}){marker}")
 
