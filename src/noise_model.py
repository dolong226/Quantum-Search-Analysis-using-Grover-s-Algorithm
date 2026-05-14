from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import math
import numpy as np
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    pauli_error,
    amplitude_damping_error,
    thermal_relaxation_error,
    ReadoutError,
    phase_damping_error,
    kraus_error,
)
from qiskit_aer import AerSimulator
from qiskit import transpile

from src.grover import *

@dataclass
class NoiseConfig:
    depolarizing_1q: float = 0.001     
    depolarizing_2q: float = 0.01      
 
    bit_flip: float = 0.0              
    phase_flip: float = 0.0            
 
    amplitude_damping: float = 0.0    

    thermal_t1: Optional[float] = None   
    thermal_t2: Optional[float] = None  
    gate_time_1q: float = 50.0           
    gate_time_2q: float = 300.0          #
 
   
    readout_error_p0: float = 0.0     
    readout_error_p1: float = 0.0   

    name: str = "custom_noise"

    def __post_init__(self):
        # check tham số
        prob_params = {
            "depolarizing_1q": self.depolarizing_1q,
            "depolarizing_2q": self.depolarizing_2q,
            "bit_flip": self.bit_flip,
            "phase_flip": self.phase_flip,
            "amplitude_damping": self.amplitude_damping,
            "readout_error_p0": self.readout_error_p0,
            "readout_error_p1": self.readout_error_p1,
        }
        for param_name, value in prob_params.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Tham số '{param_name}' = {value} phải trong khoảng [0, 1]."
                )
            
        # Kiểm tra điều kiện T2 ≤ 2*T1
        if self.thermal_t1 is not None and self.thermal_t2 is not None:
            if self.thermal_t2 > 2 * self.thermal_t1:
                raise ValueError("Lỗi tham số thermal.")
            
# Preset

NOISE_PRESETS = {
    "ideal": NoiseConfig(
        depolarizing_1q=0.0,
        depolarizing_2q=0.0,
        name="ideal"
    ),

    "low_noise": NoiseConfig(
        depolarizing_1q=0.001,   # 0.1%
        depolarizing_2q=0.005,   # 0.5%
        readout_error_p0=0.005,
        readout_error_p1=0.005,
        name="low_noise"
    ),

    "medium_noise": NoiseConfig(
        depolarizing_1q=0.005,   # 0.5%
        depolarizing_2q=0.02,    # 2%
        readout_error_p0=0.01,
        readout_error_p1=0.02,
        name="medium_noise"
    ),

    "high_noise": NoiseConfig(
        depolarizing_1q=0.02,    # 2%
        depolarizing_2q=0.08,    # 8%
        readout_error_p0=0.05,
        readout_error_p1=0.08,
        name="high_noise"
    ),

    # Chỉ có readout error - để phân tích riêng tác động của SPAM
    "readout_only": NoiseConfig(
        depolarizing_1q=0.0,
        depolarizing_2q=0.0,
        readout_error_p0=0.03,
        readout_error_p1=0.03,
        name="readout_only"
    ),

    "gate_heavy": NoiseConfig(
        depolarizing_1q=0.002,
        depolarizing_2q=0.05,   # 5% - cổng 2-qubit chủ yếu là nguồn nhiễu
        name="gate_heavy"
    ),
}

# Noise model builder

def build_noise_model(config: NoiseConfig) -> NoiseModel:
    noise_model = NoiseModel()

    one_qubit_gates = ["h", "x", "y", "z", "s", "t", "sdg", "tdg",
                       "rx", "ry", "rz", "u1", "u2", "u3", "sx"]
    two_qubit_gates = ["cx", "cz", "cy", "swap", "rzz", "ecr", "rxx"]

    # 1. Thermal Relaxation Noise
    if config.thermal_t1 is not None and config.thermal_t2 is not None:
        t1_ns = config.thermal_t1 * 1000 
        t2_ns = config.thermal_t2 * 1000  
 
        # Tạo thermal relaxation error cho cổng 1-qubit
        thermal_error_1q = thermal_relaxation_error(
            t1=t1_ns,
            t2=t2_ns,
            time=config.gate_time_1q 
        )
 
        # Tạo thermal relaxation error cho cổng 2-qubit
        thermal_error_2q = thermal_relaxation_error(
            t1=t1_ns,
            t2=t2_ns,
            time=config.gate_time_2q
        ).expand(thermal_relaxation_error(t1=t1_ns, t2=t2_ns, time=config.gate_time_2q))
 
        noise_model.add_all_qubit_quantum_error(thermal_error_1q, one_qubit_gates)
        noise_model.add_all_qubit_quantum_error(thermal_error_2q, two_qubit_gates)

    # 2. Depolarizing Noise
    if config.depolarizing_1q > 0:
        dep_error_1q = depolarizing_error(config.depolarizing_1q, 1)
        # Thêm vào tất cả qubit cho tất cả cổng 1-qubit
        noise_model.add_all_qubit_quantum_error(dep_error_1q, one_qubit_gates)
 
    if config.depolarizing_2q > 0:
        dep_error_2q = depolarizing_error(config.depolarizing_2q, 2)
        noise_model.add_all_qubit_quantum_error(dep_error_2q, two_qubit_gates)
 
    # 3. Bit Flip
    if config.bit_flip > 0:
        bit_flip_error = pauli_error([
            ("X", config.bit_flip),                   
            ("I", 1.0 - config.bit_flip)              
        ])
        noise_model.add_all_qubit_quantum_error(bit_flip_error, one_qubit_gates)
 
    # 4. Phase-Flip Noise
    if config.phase_flip > 0:
        phase_flip_error = pauli_error([
            ("Z", config.phase_flip),                 
            ("I", 1.0 - config.phase_flip)           
        ])
        noise_model.add_all_qubit_quantum_error(phase_flip_error, one_qubit_gates)
 
    # 5. Amplitude Damping
    if config.amplitude_damping > 0:
        amp_damp_error = amplitude_damping_error(config.amplitude_damping)
        noise_model.add_all_qubit_quantum_error(amp_damp_error, one_qubit_gates)
 
    # 6. Readout Error (SPAM) 
    if config.readout_error_p0 > 0 or config.readout_error_p1 > 0:
        p0 = config.readout_error_p0 
        p1 = config.readout_error_p1 

        readout_matrix = [
            [1.0 - p0, p0   ],  
            [p1,       1-p1 ]   
        ]
        readout_err = ReadoutError(readout_matrix)
        noise_model.add_all_qubit_readout_error(readout_err)
 
    return noise_model
 
def build_noise_model_from_preset(preset_name: str) -> NoiseModel:
    if preset_name not in NOISE_PRESETS:
        valid_names = list(NOISE_PRESETS.keys())
        raise KeyError(f"Preset '{preset_name}' không tồn tại.")
 
    config = NOISE_PRESETS[preset_name]
    return build_noise_model(config)

# Noise sweep analysis
# Phân tích tác động của cường độ nhiễu lên Grover
# Mô phỏng Grover với các mức nhiễu khác nhau
def sweep_noise_levels(n_qubits: int, target_index: int, noise_levels: List[float], n_iterations: Optional[int] = None, n_shots: int = 2048, noise_type: str = "depolarizing") -> dict:
    n_states = 2 ** n_qubits
    if n_iterations is None:
        n_iterations = round(math.pi / 4 * math.sqrt(n_states))
        n_iterations = max(1, n_iterations)

    # Xác suất lý tưởng
    ideal_prob = get_theoretical_success_probability(n_qubits, n_iterations)

    # classical baseline
    classical_random_prob = 1.0 / n_states

    success_probs = []

    for noise_level in noise_levels:
        # Tạo noise config dựa trên loại nhiễu
        if noise_type == "depolarizing":
            config = NoiseConfig(
                depolarizing_1q=noise_level,
                depolarizing_2q=min(noise_level * 5, 0.999),  # 2-qubit thường cao hơn
                name=f"dep_{noise_level:.4f}"
            )
        elif noise_type == "bit_flip":
            config = NoiseConfig(
                depolarizing_1q=0.0,
                depolarizing_2q=0.0,
                bit_flip=noise_level,
                name=f"bitflip_{noise_level:.4f}"
            )
        elif noise_type == "phase_flip":
            config = NoiseConfig(
                depolarizing_1q=0.0,
                depolarizing_2q=0.0,
                phase_flip=noise_level,
                name=f"phaseflip_{noise_level:.4f}"
            )
        elif noise_type == "amplitude_damping":
            config = NoiseConfig(
                depolarizing_1q=0.0,
                depolarizing_2q=0.0,
                amplitude_damping=noise_level,
                name=f"ampdamp_{noise_level:.4f}"
            )
        else:
            raise ValueError(f"noise_type='{noise_type}' không hợp lệ.")
        
        noise_model = build_noise_model(config)

        counts = run_grover_simulation(
            n_qubits=n_qubits,
            target_index=target_index,
            n_iterations=n_iterations,
            n_shots=n_shots,
            noise_model=noise_model
            )
        
        prob = calculate_success_probability(counts=counts, target_index=target_index, n_qubits=n_qubits)
        success_probs.append(prob)
        
        # Lấy độ lệch 10%
        quantum_advantage = [p > classical_random_prob * 1.1 for p in success_probs]

    return {
    "noise_levels": noise_levels,
    "success_probs": success_probs,
    "ideal_prob": ideal_prob,
    "classical_random_prob": classical_random_prob,
    "quantum_advantage": quantum_advantage,
    "noise_type": noise_type,
    "n_qubits": n_qubits,
    "n_iterations": n_iterations,
}

# So sánh tác động của nhiều loại nhiễu khác nhau ở cùng mức độ
def compare_noise_types(n_qubits: int, target_index: int, noise_level: float = 0.01, n_shots: int = 2048) -> dict:
    n_states = 2 ** n_qubits
    n_iterations = max(1, round(math.pi / 4 * math.sqrt(n_states)))
    ideal_prob = get_theoretical_success_probability(n_qubits, n_iterations)
 
    # Định nghĩa các cấu hình nhiễu cần so sánh
    configs_to_compare = {
        "Lý tưởng (Không nhiễu)": NoiseConfig(
            depolarizing_1q=0.0, depolarizing_2q=0.0, name="ideal"
        ),
        "Depolarizing": NoiseConfig(
            depolarizing_1q=noise_level,
            depolarizing_2q=min(noise_level * 5, 0.999),
            name="depolarizing"
        ),
        "Bit-Flip": NoiseConfig(
            depolarizing_1q=0.0, depolarizing_2q=0.0,
            bit_flip=noise_level, name="bit_flip"
        ),
        "Phase-Flip": NoiseConfig(
            depolarizing_1q=0.0, depolarizing_2q=0.0,
            phase_flip=noise_level, name="phase_flip"
        ),
        "Amplitude Damping": NoiseConfig(
            depolarizing_1q=0.0, depolarizing_2q=0.0,
            amplitude_damping=noise_level, name="amp_damp"
        ),
        "Readout Error": NoiseConfig(
            depolarizing_1q=0.0, depolarizing_2q=0.0,
            readout_error_p0=noise_level, readout_error_p1=noise_level,
            name="readout"
        ),
    }

    results = {}

    for noise_name, config in configs_to_compare.items():
        if config.name == "ideal":
            nm = None
        else:
            nm = build_noise_model(config=config)

        # Simulated
        counts = run_grover_simulation(n_qubits, target_index, n_iterations, n_shots, nm)

        prob = calculate_success_probability(counts, target_index, n_qubits)

        degradation = ideal_prob - prob

        results[noise_name] = {
            "probability": prob,
            "degradation_absolute": abs(degradation),
            "degradation_relative":degradation / ideal_prob if ideal_prob > 0 else 0,
        }

    results["_meta"] = {
        "noise_level": noise_level,
        "ideal_prob": ideal_prob,
        "n_qubits": n_qubits,
        "n_iterations": n_iterations,
        "n_shots": n_shots
    } 

    return results


# Log

 
def describe_noise_model(config: NoiseConfig) -> str:
    """
    Tạo mô tả văn bản của noise model để in ra hoặc log.
    """
    lines = [
        f"{'─' * 50}",
        f"  NOISE MODEL: {config.name.upper()}",
        f"{'─' * 50}",
    ]
 
    # Depolarizing
    if config.depolarizing_1q > 0 or config.depolarizing_2q > 0:
        lines.append(f"  Depolarizing (1-qubit) : {config.depolarizing_1q * 100:.3f}%")
        lines.append(f"  Depolarizing (2-qubit) : {config.depolarizing_2q * 100:.3f}%")
 
    # Pauli errors
    if config.bit_flip > 0:
        lines.append(f"  Bit-Flip (X error)     : {config.bit_flip * 100:.3f}%")
    if config.phase_flip > 0:
        lines.append(f"  Phase-Flip (Z error)   : {config.phase_flip * 100:.3f}%")
 
    # Amplitude damping
    if config.amplitude_damping > 0:
        lines.append(f"  Amplitude Damping (γ)  : {config.amplitude_damping:.4f}")
 
    # Thermal
    if config.thermal_t1 is not None:
        lines.append(f"  T1 relaxation          : {config.thermal_t1:.1f} μs")
        lines.append(f"  T2 coherence           : {config.thermal_t2:.1f} μs")
        lines.append(f"  Gate time (1Q/2Q)      : {config.gate_time_1q:.0f}/{config.gate_time_2q:.0f} ns")
 
    # Readout
    if config.readout_error_p0 > 0 or config.readout_error_p1 > 0:
        lines.append(f"  Readout P(1|0)         : {config.readout_error_p0 * 100:.3f}%")
        lines.append(f"  Readout P(0|1)         : {config.readout_error_p1 * 100:.3f}%")
 
    if len(lines) == 3:
        lines.append("  (Không có nhiễu - mô phỏng lý tưởng)")
 
    lines.append(f"{'─' * 50}")
    return "\n".join(lines)
 

# Main demo

if __name__ == "__main__":
    """
    Demo: So sánh các mức nhiễu khác nhau trên hệ thống 3 qubit.
    """
    print("=" * 60)
    print("  MÔ HÌNH NHIỄU LƯỢNG TỬ - DEMO")
    print("=" * 60)
 
    # Hiển thị các preset có sẵn
    print("\nCác Noise Preset có sẵn:")
    for name, config in NOISE_PRESETS.items():
        print(f"\n{describe_noise_model(config)}")
 
    # Demo tạo noise model tùy chỉnh
    print("\n\nTạo Noise Model tùy chỉnh (thermal relaxation):")
    custom_config = NoiseConfig(
        thermal_t1=100.0,       # T1 = 100 μs
        thermal_t2=80.0,        # T2 = 80 μs (≤ 2*T1)
        gate_time_1q=50.0,      # Cổng 1-qubit: 50 ns
        gate_time_2q=300.0,     # Cổng 2-qubit: 300 ns
        readout_error_p0=0.01,  # 1% readout error
        readout_error_p1=0.02,  # 2% readout error
        depolarizing_1q=0.0,    # Tắt depolarizing để tách biệt thermal
        depolarizing_2q=0.0,
        name="thermal_realistic"
    )
    print(describe_noise_model(custom_config))
    nm = build_noise_model(custom_config)
    print(f"\nNoise model '{custom_config.name}' đã được tạo thành công!")
    print(f"  Basis gates bị ảnh hưởng: {nm.basis_gates}")
 