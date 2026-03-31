import matplotlib.pyplot as plt

from Final_Script import (create_initial_final_states, compute_fidelity_curve)


# Compare NN and LRI for an s qubit system
s = 3
print(f"Simulating for system size {s}")

dt = 0.001 # chosen time discretisation for exact method to plot the fidelity over time
threshold = 2/3 # peaks only identified above the classical limit
alpha = 5 # decay strength
total_time = 2.5 # total simulation time
J = -1.0 # J_in parameter

# Initial and final states
initial_state = create_initial_final_states(s, True)
final_state = create_initial_final_states(s, False)

# NN evolution
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

# LRI evolution with alpha = 5
times_LRI, F_LRI, peak_times_LRI, peak_values_LRI, _ = compute_fidelity_curve(
    total_chain_length=s,
    total_time=total_time,
    initial_state=initial_state,
    final_state=final_state,
    dt=dt,
    threshold=threshold,
    alpha=alpha,
    J_in=J,
    method="exact",
    coupling_type="LRI")

# Plot both curves
plt.figure()
plt.plot(times_NN, F_NN, label="Nearest-Neighbour")
plt.plot(times_LRI, F_LRI, label=f"Long-Range, α = {alpha}")

# Plot peaks if any were found
if len(peak_times_NN) > 0:
    plt.scatter(peak_times_NN, peak_values_NN, color="red", zorder=3)

if len(peak_times_LRI) > 0:
    plt.scatter(peak_times_LRI, peak_values_LRI, color="blue", zorder=3)

# Peak annotation
if len(peak_values_NN) > 0 and len(peak_values_LRI) > 0:
    text = f"NN: {peak_values_NN[0]:.4g}, LRI: {peak_values_LRI[0]:.4g}"
    plt.annotate(
        text,
        xy=(peak_times_LRI[0], peak_values_LRI[0]),
        xytext=(5,4),
        textcoords="offset points",
        fontsize=8)

plt.xlabel("Time / s")
plt.ylabel("Fidelity")
plt.title(f"Fidelity Over Time for a {s} Qubit System")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"NN peak values = {peak_values_NN}")
print(f"LRI peak values = {peak_values_LRI}")
print(f"NN peak times = {peak_times_NN}")
print(f"LRI peak times = {peak_times_LRI}")

# Vary the decay strength for the system
# Simulate and plot a range of alpha values
alphas = [1, 1.5, 2, 2.5, 3, 3.5]

for a in alphas:
    print(f"For a system with α = {a}")

    times_LRI, F_LRI, peak_times_LRI, peak_values_LRI, _ = compute_fidelity_curve(
        total_chain_length=s,
        total_time=total_time,
        initial_state=initial_state,
        final_state=final_state,
        dt=dt,
        threshold=threshold,
        alpha=a,
        J_in=J,
        method="exact",
        coupling_type="LRI")

    # Plot LRI curve
    plt.plot(times_LRI, F_LRI, label=f"α = {a}")

    # Optional peak markers
    # if len(peak_times_LRI) > 0:
    #     plt.scatter(peak_times_LRI, peak_values_LRI, zorder=3)

    print(f"LRI peak values = {peak_values_LRI}")
    print(f"LRI peak times = {peak_times_LRI}")


# NN comparison curve
plt.plot(times_NN, F_NN, label="NN")

# NN peaks
# if len(peak_times_NN) > 0:
#     plt.scatter(peak_times_NN, peak_values_NN, color='black', zorder=3)

plt.xlabel("Time / s")
plt.ylabel("Fidelity")
plt.title(f"Fidelity Over Time for a {s} Qubit System Varying Decay Strength (α)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()