from dataclasses import dataclass, field
from typing import Optional, List, Tuple

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

NOISE_PRESET = {
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

