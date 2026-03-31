import matplotlib.pyplot as plt

from Final_Script import (create_initial_final_states, compute_fidelity_curve)

Ls = [4]

if __name__ == "__main__":
    total_time = 6 # total simulation time
    dt = 0.001 # chosen time discretisation for exact method to plot the fidelity over time
    threshold = 2/3 # peaks only identified above the classical limit
    J = -1.0 # J_in parameter

    for s in Ls:
        print(f"system size = {s}")
        # Initial and final states
        initial_state = create_initial_final_states(s, True)
        final_state = create_initial_final_states(s, False)

        times_NN, F_NN, peak_times_NN, peak_values_NN, _ = compute_fidelity_curve(
            total_chain_length=s,
            total_time=total_time,
            initial_state=initial_state,
            final_state=final_state,
            dt=dt,
            threshold=threshold,
            J_in=J,
            method="exact",
            coupling_type="NN")
        
        times_PST, F_PST, peak_times_PST, peak_values_PST, _ = compute_fidelity_curve(
            total_chain_length=s,
            total_time=total_time,
            initial_state=initial_state,
            final_state=final_state,
            dt=dt,
            threshold=threshold,
            J_in=J/s,
            method="exact",
            coupling_type="PST")


        plt.figure()

        plt.plot(times_NN, F_NN, label="NN")
        plt.plot(times_PST, F_PST, label="PST")

        print(f"Method: NN, Peak Fidelity: {peak_values_NN} at {peak_times_NN}")
        print(f"Method: PST, Peak Fidelity: {peak_values_PST} at {peak_times_PST}")

        if len(peak_times_NN) > 0:
            plt.scatter(peak_times_NN, peak_values_NN, color="blue", zorder=2)
            plt.annotate(
                f"NN: {peak_values_NN[0]:.4f}",
                xy=(peak_times_NN[0], peak_values_NN[0]),
                xytext=(-40,4),
                textcoords="offset points",
                fontsize=7)

            
        if len(peak_times_PST) > 0:
            plt.scatter(peak_times_PST, peak_values_PST, color="red", zorder=2)
            plt.annotate(
                f"PST: {peak_values_PST[0]:.4f}",
                xy=(peak_times_PST[0], peak_values_PST[0]),
                xytext=(5,4),
                textcoords="offset points",
                fontsize=7)
            
        plt.xlim(0,4)
        plt.xlabel("Time / s")
        plt.ylabel("Fidelity")
        plt.title(f"Fidelity Over Time for a {s} Qubit System Optimising State Transfer")
        plt.legend(fontsize=7)
        plt.grid(True, alpha=0.3)
        plt.show()