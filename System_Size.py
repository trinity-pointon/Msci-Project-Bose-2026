import matplotlib.pyplot as plt

from Final_Script import (create_initial_final_states, compute_fidelity_curve)


# Compare NN for an s qubit system

dt = 0.001 # chosen time discretisation for exact method to plot the fidelity over time
threshold = 2/3 # peaks only identified above the classical limit
alpha = 5 # decay strength
total_time = 5 # total simulation time
J = -1.0 # J_in parameter

system_size = [3,4,5,6,7,8]

plt.figure(figsize=(9, 5))


for s in system_size:
    print(f"Simulating for system size {s}")
    # Initial and final states
    initial_state = create_initial_final_states(s, True)
    final_state = create_initial_final_states(s, False)


    # NN evolution, exact method
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


    line, = plt.plot(times_NN, F_NN, label=f"N = {s}")
    current_colour = line.get_color()


    # Plot peaks if any were found
    if len(peak_times_NN) > 0:
        plt.scatter(peak_times_NN[0], peak_values_NN[0], color=current_colour, zorder=3)


    # Peak annotation
    if len(peak_values_NN) > 0:
        text = f"{peak_values_NN[0]:.4g}"
        plt.annotate(
            text,
            xy=(peak_times_NN[0], peak_values_NN[0]),
            xytext=(5,4),
            textcoords="offset points",
            fontsize=8)


plt.xlabel("Time /s")
plt.ylabel("Fidelity")
plt.title(f"Fidelity Over Time for Various NN System Sizes evolved Exactly")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)

plt.show()

print(f"NN peak values = {peak_values_NN}")
print(f"NN peak times = {peak_times_NN}")

