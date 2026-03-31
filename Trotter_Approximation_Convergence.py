import matplotlib.pyplot as plt

from Final_Script import (create_initial_final_states, compute_fidelity_curve, compute_fidelity_curve_gate,)

ns = [25, 50, 100, 150,200]


def first_or_zero(arr):
    return arr[0] if len(arr) > 0 else 0.0


if __name__ == "__main__":
    threshold = 0.5
    total_time = 1.5
    J = -1.0
    s = 3

    initial_state = create_initial_final_states(s, True)
    final_state = create_initial_final_states(s, False)

    # Exact reference
    _, _, peak_times_exact, peak_values_exact, _ = compute_fidelity_curve(
        total_chain_length=s,
        total_time=total_time,
        initial_state=initial_state,
        final_state=final_state,
        threshold=threshold,
        J_in=J,
        method="exact",
        coupling_type="NN",
    )

    exact_peak = first_or_zero(peak_values_exact)
    exact_peak_time = first_or_zero(peak_times_exact)

    print(f"Exact peak fidelity: {exact_peak}")
    print(f"Exact peak time: {exact_peak_time}")

    gate_peak_fidelities = []
    gate_peak_times = []

    trotter_peak_fidelities = []
    trotter_peak_times = []

    for n in ns:
        print(f"\nRunning n = {n}")

        # Trotter
        _, _, peak_times_T, peak_values_T, _ = compute_fidelity_curve(
            total_chain_length=s,
            total_time=total_time,
            initial_state=initial_state,
            final_state=final_state,
            threshold=threshold,
            J_in=J,
            trotter_steps=n,
            method="trotter",
            coupling_type="NN",
        )

        trotter_peak_fidelities.append(first_or_zero(peak_values_T))
        trotter_peak_times.append(first_or_zero(peak_times_T))

        # Gate
        _, _, peak_times_G, peak_values_G, _ = compute_fidelity_curve_gate(
            total_chain_length=s,
            total_time=total_time,
            dt=total_time/n,
            threshold=threshold,
            J_in=J,
            trotter_steps=n,
            coupling_type="NN",
            shots=400,
            mode="statevector",
        )

        gate_peak_fidelities.append(first_or_zero(peak_values_G))
        gate_peak_times.append(first_or_zero(peak_times_G))

    print("\nResults")
    print("n =", ns)
    print("Trotter first peak fidelities:", trotter_peak_fidelities)
    print("Gate first peak fidelities:   ", gate_peak_fidelities)
    print("Trotter first peak times:", trotter_peak_times)
    print("Gate first peak times:   ", gate_peak_times)

    # Plot fidelity convergence
    plt.figure(figsize=(9, 5))
    plt.plot(ns, gate_peak_fidelities, marker="o", label="Gate Peak Fidelity")
    plt.plot(ns, trotter_peak_fidelities, marker="o", label="Trotter Peak Fidelity")
    plt.axhline(
        exact_peak,
        color="green",
        linestyle="--",
        label=f"Exact Peak = {exact_peak:.4f}",
    )
    plt.xlabel("Number of Steps (n)")
    plt.ylabel("Peak Fidelity")
    plt.title(f"Fidelity Convergence with Number of Steps, N={s}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot peak time convergence
    plt.figure(figsize=(9, 5))
    plt.plot(ns, gate_peak_times, marker="o", label="Gate Peak Time")
    plt.plot(ns, trotter_peak_times, marker="o", label="Trotter Peak Time")
    plt.axhline(
        exact_peak_time,
        color="green",
        linestyle="--",
        label=f"Exact Peak Time = {exact_peak_time:.4f}",
    )
    plt.xlabel("Number of Steps (n)")
    plt.ylabel("Peak Time")
    plt.title(f"Peak Time Convergence with Number of Steps, N={s}")
    plt.legend()
    plt.grid(True)
    plt.show()