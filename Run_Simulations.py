import pickle

from Final_Script import (create_initial_final_states, compute_fidelity_curve, compute_fidelity_curve_gate)

ns = [5,20,50,100]
Ls = [4,5]


def store_results():
    """
    Function that runs each method for each coupling and stores the results for a certain system size.
    """
    dt = 0.001
    threshold = 0.666

    total_time = 6
    J_in = -1.0
    alpha = 5

    for s in Ls:
        print(f"System size L = {s}")

        initial_state = create_initial_final_states(s, True)
        final_state = create_initial_final_states(s, False)

        results = {
            "meta": {
                "L": s,
                "dt": dt,
                "threshold": threshold,
                "total_time": total_time,
                "J_in": J_in,
                "alpha": alpha,
                "ns": list(ns),
            },
            "exact": {
                "NN": {},
                "LRI": {},
                "PST": {},
            },
            "trotter": {
                "NN": {},
                "LRI": {},
            },
            "gate": {
                "NN": {},
                "LRI": {},
            },
        }

        # exact NN
        times, F, peak_times, peak_values, evolved_statevector = compute_fidelity_curve(
            total_chain_length=s,
            total_time=total_time,
            initial_state=initial_state,
            final_state=final_state,
            dt=dt,
            threshold=threshold,
            J_in=J_in,
            method="exact",
            coupling_type="NN",
        )
        results["exact"]["NN"] = {
            "times": times,
            "F": F,
            "peak_times": peak_times,
            "peak_values": peak_values,
            "evolved_statevector": evolved_statevector,
        }
        print(f"Exact NN: peak fidelity {peak_values} at {peak_times}")

        # exact LRI
        times, F, peak_times, peak_values, evolved_statevector = compute_fidelity_curve(
            total_chain_length=s,
            total_time=total_time,
            initial_state=initial_state,
            final_state=final_state,
            dt=dt,
            threshold=threshold,
            alpha=alpha,
            J_in=J_in,
            method="exact",
            coupling_type="LRI",
        )
        results["exact"]["LRI"] = {
            "times": times,
            "F": F,
            "peak_times": peak_times,
            "peak_values": peak_values,
            "evolved_statevector": evolved_statevector,
        }
        print(f"Exact LRI: peak fidelity {peak_values} at {peak_times}")

        # exact PST
        times, F, peak_times, peak_values, evolved_statevector = compute_fidelity_curve(
            total_chain_length=s,
            total_time=total_time,
            initial_state=initial_state,
            final_state=final_state,
            dt=dt,
            threshold=threshold,
            J_in=J_in,
            method="exact",
            coupling_type="PST",
        )
        results["exact"]["PST"] = {
            "times": times,
            "F": F,
            "peak_times": peak_times,
            "peak_values": peak_values,
            "evolved_statevector": evolved_statevector,
        }
        print(f"Exact PST: peak fidelity {peak_values} at {peak_times}")


        for n in ns:
            print(f"\nRunning Trotter/Gate simulations for n = {n}")

            # Trotter NN
            times_T_NN, F_T_NN, peak_times_T_NN, peak_values_T_NN, evolved_T_NN = compute_fidelity_curve(
                total_chain_length=s,
                total_time=total_time,
                initial_state=initial_state,
                final_state=final_state,
                dt=dt,
                threshold=threshold,
                J_in=J_in,
                trotter_steps=n,
                method="trotter",
                coupling_type="NN"
            )
            results["trotter"]["NN"][n] = {
                "times": times_T_NN,
                "F": F_T_NN,
                "peak_times": peak_times_T_NN,
                "peak_values": peak_values_T_NN,
                "evolved_statevector": evolved_T_NN,
            }

            # Trotter LRI
            times_T_LRI, F_T_LRI, peak_times_T_LRI, peak_values_T_LRI, evolved_T_LRI = compute_fidelity_curve(
                total_chain_length=s,
                total_time=total_time,
                initial_state=initial_state,
                final_state=final_state,
                dt=dt,
                threshold=threshold,
                alpha=alpha,
                J_in=J_in,
                trotter_steps=n,
                method="trotter",
                coupling_type="LRI"
            )
            results["trotter"]["LRI"][n] = {
                "times": times_T_LRI,
                "F": F_T_LRI,
                "peak_times": peak_times_T_LRI,
                "peak_values": peak_values_T_LRI,
                "evolved_statevector": evolved_T_LRI,
            }

            # Gate NN
            times_G_NN, F_G_NN, peak_times_G_NN, peak_values_G_NN, evolved_G_NN = compute_fidelity_curve_gate(
            total_chain_length=s,
                total_time=total_time,
                dt=0.01,
                threshold=threshold,
                J_in=J_in,
                trotter_steps=n,
                coupling_type="NN",
                shots=400,
                mode="statevector"
            )
            results["gate"]["NN"][n] = {
                "times": times_G_NN,
                "F": F_G_NN,
                "peak_times": peak_times_G_NN,
                "peak_values": peak_values_G_NN,
                "evolved_statevector": evolved_G_NN,
            }

            # Gate LRI
            times_G_LRI, F_G_LRI, peak_times_G_LRI, peak_values_G_LRI, evolved_G_LRI = compute_fidelity_curve_gate(
            total_chain_length=s,
                total_time=total_time,
                dt=0.01,
                threshold=threshold,
                J_in=J_in,
                trotter_steps=n,
                coupling_type="LRI",
                shots=400,
                mode="statevector"
            )
            results["gate"]["LRI"][n] = {
                "times": times_G_LRI,
                "F": F_G_LRI,
                "peak_times": peak_times_G_LRI,
                "peak_values": peak_values_G_LRI,
                "evolved_statevector": evolved_G_LRI,
            }

            print(f"Trotter NN (n={n}): peak {peak_values_T_NN} at {peak_times_T_NN}")
            print(f"Trotter LRI (n={n}): peak {peak_values_T_LRI} at {peak_times_T_LRI}")
            print(f"Gate NN (n={n}): peak {peak_values_G_NN} at {peak_times_G_NN}")
            print(f"Gate LRI (n={n}): peak {peak_values_G_LRI} at {peak_times_G_LRI}")

        ns_str = "-".join(str(n) for n in ns)
        outname = f"simulation_L{s}_T{total_time}_J{J_in}_a{alpha}_ns{ns_str}.pkl"

        with open(outname, "wb") as f:
            pickle.dump(results, f)

        print(f"Saved results to {outname}")


if __name__ == "__main__":
    store_results()