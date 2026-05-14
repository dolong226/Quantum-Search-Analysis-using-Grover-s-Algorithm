import math

from src.noise_model import *
from src.grover import *

def test_noise_config_validation():
    print("\n[TEST] NoiseConfig validation")

    try:
        NoiseConfig(depolarizing_1q=1.5)
        print("FAILED: invalid probability accepted")
    except ValueError:
        print("PASSED: invalid probability rejected")

    try:
        NoiseConfig(
            thermal_t1=50,
            thermal_t2=200
        )
        print("FAILED: invalid thermal relation accepted")
    except ValueError:
        print("PASSED: invalid thermal relation rejected")


def test_build_noise_model():
    print("\n[TEST] Build Noise Model")

    config = NoiseConfig(
        depolarizing_1q=0.001,
        depolarizing_2q=0.01,
        readout_error_p0=0.01,
        readout_error_p1=0.02,
        name="test_model"
    )

    noise_model = build_noise_model(config)

    print("Noise model created successfully")
    print("Basis gates:", noise_model.basis_gates)


def test_grover_with_noise():
    print("\n[TEST] Grover Simulation with Noise")

    n_qubits = 3
    target_index = 5
    n_states = 2 ** n_qubits

    n_iterations = max(
        1,
        round(math.pi / 4 * math.sqrt(n_states))
    )

    config = NOISE_PRESETS["medium_noise"]

    noise_model = build_noise_model(config)

    counts = run_grover_simulation(
        n_qubits=n_qubits,
        target_index=target_index,
        n_iterations=n_iterations,
        n_shots=2048,
        noise_model=noise_model
    )

    prob = calculate_success_probability(
        counts=counts,
        target_index=target_index,
        n_qubits=n_qubits
    )

    print("Counts:")
    print(counts)

    print(f"Success probability: {prob:.4f}")

    assert prob >= 0.0
    assert prob <= 1.0

    print("PASSED")


def test_noise_sweep():
    print("\n[TEST] Noise Sweep Analysis")

    results = sweep_noise_levels(
        n_qubits=3,
        target_index=5,
        noise_levels=[
            0.0,
            0.001,
            0.005,
            0.01,
            0.02,
        ],
        noise_type="depolarizing",
        n_shots=1024
    )

    print("\nSweep Results:")

    for level, prob in zip(
        results["noise_levels"],
        results["success_probs"]
    ):
        print(
            f"Noise={level:.4f} "
            f"-> Success={prob:.4f}"
        )

    assert len(results["noise_levels"]) == len(results["success_probs"])

    print("PASSED")


def test_compare_noise_types():
    print("\n[TEST] Compare Noise Types")

    results = compare_noise_types(
        n_qubits=3,
        target_index=5,
        noise_level=0.01,
        n_shots=1024
    )

    for noise_name, data in results.items():

        if noise_name == "_meta":
            continue

        print(f"\n{noise_name}")

        print(
            f"Probability: "
            f"{data['probability']:.4f}"
        )

        print(
            f"Absolute degradation: "
            f"{data['degradation_absolute']:.4f}"
        )

        print(
            f"Relative degradation: "
            f"{data['degradation_relative']:.4f}"
        )

    print("PASSED")


def test_describe_noise_model():
    print("\n[TEST] Describe Noise Model")

    config = NoiseConfig(
        depolarizing_1q=0.002,
        depolarizing_2q=0.01,
        readout_error_p0=0.01,
        readout_error_p1=0.01,
        name="demo"
    )

    description = describe_noise_model(config)

    print(description)

    assert isinstance(description, str)

    print("PASSED")


if __name__ == "__main__":

    print("=" * 60)
    print("RUNNING NOISE MODEL TEST SUITE")
    print("=" * 60)

    test_noise_config_validation()

    test_build_noise_model()

    test_grover_with_noise()

    test_noise_sweep()

    test_compare_noise_types()

    test_describe_noise_model()

    print("\n" + "=" * 60)
    print("ALL TESTS FINISHED")
    print("=" * 60)