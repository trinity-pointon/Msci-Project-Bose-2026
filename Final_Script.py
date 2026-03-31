import numpy as np
from scipy.linalg import expm
from scipy.signal import find_peaks
from braket.circuits import Circuit
from braket.devices import LocalSimulator
import matplotlib.pyplot as plt


# Pauli matrices
Sx = np.array([[0, 1], [1, 0]])
Sy = np.array([[0, -1j], [1j, 0]])
Sz = np.array([[1, 0], [0, -1]]) # not required but defined here for completeness

# up and down states
up = np.array([[1], 
               [0]])
down = np.array([[0],
                 [1]])

# Identity
qubit_identity = np.identity(2)

def tensor_product_initialiser(number_particles, state, iterated_state, first):
    """
    Creates a tensor product over a certain number of particles, with an initial state, and an iterated 
    state that attaches however many times. If first is true, the initial state remains as the first state,
    otherwise it moves to last.
    """
    states_created = 1
    while states_created < number_particles:
        if first == True: # If the initial state is the first state
            state = np.kron(state, iterated_state) # Matrix product of the initial state (normally up) and the rest are iterated
        else:
            state = np.kron(iterated_state, state) # Otherwise put the initial state at the end to generate the desired communication for fidelity calculations
        states_created += 1 # Iterate through to get the chain of correct length
    return state

def create_initial_final_states(chain_length, initial):
    """Function to create inital or final states to be time evolved"""
    return tensor_product_initialiser(chain_length, down, up, initial)

def chain_operator_constructor(chain_length, chain_position, qubit_operator):
    """
    Takes an operator acting on a single qubit, in a given position of the chain. Outputs the operator on the entire 
    chain.
    """
    operator_left = tensor_product_initialiser(chain_position, qubit_operator, qubit_identity, False) # qubit operator placed at right of chain
    # So we have created identity up until the chain position and then placed the operator there
    # Currently I I I ... I Operator
    # Then creates the whole operator by placing the left at the initial position and putting identity for the rest of the chain
    # I I ... I Operator I ... I I
    operator = tensor_product_initialiser(chain_length - chain_position + 1, operator_left, qubit_identity, True)
    return operator

def Spin_List_Creator(chain_length):
    """
    Creates list of size chain_length of the Spin operators for each site, where S = [Sx, Sy, Sz]. Indexed from 0.
    """
    Spin_List = []
    for qubit_chain_position in range(chain_length):
        Qubit_Spin_Array = np.array([chain_operator_constructor(chain_length, qubit_chain_position + 1, Sx), 
                                      chain_operator_constructor(chain_length, qubit_chain_position + 1, Sy),
                                      chain_operator_constructor(chain_length, qubit_chain_position + 1, Sz)])
        Spin_List.append(Qubit_Spin_Array)
    return Spin_List

def dot_product_spin_operators_XY(chain_length, Spin_Operator_List, qubit_position_1, qubit_position_2):
    """
    Perform dot product of 2 spin operators (Sx, Sy) in 2 postions
    """
    dot_product = np.zeros((2**chain_length, 2**chain_length))
    dot_product = dot_product.astype('complex128')
    for Spin_Operator in range(2):
            Spin_Operator_1 = Spin_Operator_List[qubit_position_1][Spin_Operator]
            Spin_Operator_2 = Spin_Operator_List[qubit_position_2][Spin_Operator]
            dot_product += np.matmul(Spin_Operator_1, Spin_Operator_2)
    return dot_product

def time_evolution(hamiltonian_matrix, time):
    """
    Evolve the Hamiltonian for specific time interval
    """
    time_scaled_matrix = -1j * time * hamiltonian_matrix
    time_evol = expm(time_scaled_matrix)
    return time_evol


def Heisenberg_Hamiltonian_Constructor(chain_length, alpha=None, J_in=1.0,
                                       method="exact", coupling_type="LRI"):
    """
    method:
        "exact"
        "trotter"

    coupling_type:
        "NN"
        "LRI"
        "PST"
    """
    if method not in ("exact", "trotter"):
        raise ValueError("method must be 'exact' or 'trotter'")

    if coupling_type == "LRI" and alpha is None:
        raise ValueError("alpha must be provided for coupling_type='LRI'")

    Spin_Operator_List = Spin_List_Creator(chain_length)
    Heisenberg = np.zeros((2**chain_length, 2**chain_length), dtype="complex128")
    H_terms = []

    J_pst = None
    if coupling_type == "PST":
        J_pst = np.zeros(chain_length - 1, dtype=float)
        for n in range(1, chain_length):
            J_pst[n - 1] = J_in *  np.sqrt(n * (chain_length - n))

    if coupling_type == "LRI":
        for i in range(chain_length):
            for j in range(i + 1, chain_length):
                J_ij = J_in / abs(i - j)**alpha
                H_ij = J_ij * dot_product_spin_operators_XY(chain_length, Spin_Operator_List, i, j)
                Heisenberg += H_ij
                H_terms.append(H_ij)

    elif coupling_type == "NN":
        for i in range(chain_length - 1):
            j = i + 1
            J_ij = J_in
            H_ij = J_ij * dot_product_spin_operators_XY(chain_length, Spin_Operator_List, i, j)
            Heisenberg += H_ij
            H_terms.append(H_ij)

    elif coupling_type == "PST":
        for i in range(chain_length - 1):
            j = i + 1
            J_ij = 2*J_pst[i]
            H_ij = J_ij * dot_product_spin_operators_XY(chain_length, Spin_Operator_List, i, j)
            Heisenberg += H_ij
            H_terms.append(H_ij)

    else:
        raise ValueError("coupling_type must be 'LRI', 'NN', or 'PST'")

    return Heisenberg, H_terms


def time_evolution_trotter(H_terms, time, steps):
    dt = time / steps
    dim = H_terms[0].shape[0]
    time_evol = np.eye(dim, dtype=complex)

    for H in H_terms:
        time_evol_ij = expm(-1j * dt * H)
        time_evol = time_evol @ time_evol_ij

    return np.linalg.matrix_power(time_evol, steps)


def Calculate_Fidelity(total_chain_length, time=None, initial_state=None, final_state=None,
                       alpha=None, J_in=1.0, trotter_steps=50,
                       method="exact", coupling_type="LRI"):
    """
    Exact/trotter only.
    """
    if method not in ("exact", "trotter"):
        raise ValueError("method must be 'exact' or 'trotter'")

    Temp_Hamiltonian, Temp_terms = Heisenberg_Hamiltonian_Constructor(
        total_chain_length, alpha, J_in, method=method, coupling_type=coupling_type
    )

    if method == "exact":
        time_evolved_matrix = time_evolution(Temp_Hamiltonian, time)
    else:
        time_evolved_matrix = time_evolution_trotter(Temp_terms, time, trotter_steps)

    evolved_statevector = time_evolved_matrix @ initial_state
    fidelity = np.vdot(final_state, evolved_statevector)
    probability = np.abs(fidelity)**2

    return probability, evolved_statevector


def compute_fidelity_curve(total_chain_length, total_time, initial_state, final_state,
                           dt=0.001, threshold=0.99, alpha=None, J_in=1.0,
                           trotter_steps=50, method="exact", coupling_type="LRI"):
    """
    Exact/trotter fidelity curve only.

    method = "exact", "trotter"
    coupling_type = "LRI", "NN", "PST"
    """
    if method not in ("exact", "trotter"):
        raise ValueError("compute_fidelity_curve only supports method='exact' or method='trotter'")

    print(
        f"Coupling: {coupling_type}"
        + (f", α={alpha}" if alpha is not None and coupling_type == "LRI" else "")
        + f" | Computing for {method} method"
        + (f" with steps: {trotter_steps}" if method == "trotter" else "")
    )

    times = np.arange(0, total_time+dt, dt)
    F = np.zeros(times.size)

    for idx, t in enumerate(times):
        F[idx], evolved_statevector = Calculate_Fidelity(
            total_chain_length=total_chain_length,
            time=t,
            initial_state=initial_state,
            final_state=final_state,
            alpha=alpha,
            J_in=J_in,
            trotter_steps=trotter_steps,
            method=method,
            coupling_type=coupling_type
        )

    peaks, _ = find_peaks(F, height=threshold)
    peak_times = times[peaks]
    peak_values = F[peaks]

    return times, F, peak_times, peak_values, evolved_statevector


# Gate only functions
def xy_interaction(circuit, q1, q2, theta):
    circuit.xx(q1, q2, theta)
    circuit.rz(q1, np.pi / 2)
    circuit.rz(q2, np.pi / 2)
    circuit.xx(q1, q2, theta)
    circuit.rz(q1, -np.pi / 2)
    circuit.rz(q2, -np.pi / 2)


def initial_circuit(chain_length, first=True):
    """
    first=True  -> excitation on first qubit
    first=False -> excitation on last qubit
    """
    circuit = Circuit()
    if first:
        circuit.x(0)
    else:
        circuit.x(chain_length - 1)
    return circuit


def xy_trotter_circuit_gate(chain_length, J_in, total_time, trotter_steps, coupling_type="NN"):
    """
    Gate-based XY evolution using LocalSimulator.

    coupling_type = "LRI", "NN", "PST"
    """
    trotter_dt = total_time / trotter_steps
    circuit = Circuit()

    # Initial state: excitation at last qubit, matching your pasted code
    circuit.x(chain_length - 1)

    if coupling_type == "NN":
        theta_nn = 2*J_in * trotter_dt
        for _ in range(trotter_steps):
            for i in range(chain_length - 1):
                xy_interaction(circuit, i, i + 1, theta_nn)

    elif coupling_type == "PST":
        J_pst = np.zeros(chain_length - 1)
        for n in range(1, chain_length):
            J_pst[n - 1] = J_in * np.sqrt(n * (chain_length - n))

        for _ in range(trotter_steps):
            for i in range(chain_length - 1):
                theta = 2*J_pst[i] * trotter_dt
                xy_interaction(circuit, i, i + 1, theta)

    else:
        raise ValueError("coupling_type must be 'NN' or 'PST'")

    return circuit


def compute_fidelity_curve_gate(total_chain_length, total_time, dt=0.001,threshold=0.67,
                                J_in=1.0, trotter_steps=50,
                                coupling_type="NN", shots=400,
                                mode="statevector"):
    """
    Calculates fidelity curve using LocalSimulator.
    mode =
        "statevector" -> exact fidelity (no sampling noise)
        "shots"       -> sampled fidelity (hardware-like)
    """

    if coupling_type not in ("NN", "PST"):
        raise ValueError("compute_fidelity_curve_gate only supports 'NN' or 'PST'")

    if mode not in ("statevector", "shots"):
        raise ValueError("mode must be 'statevector' or 'shots'")

    print(
        f"Coupling: {coupling_type}"
        f" | Gate method"
        f" | mode = {mode}"
        f" | steps = {trotter_steps}"
        + (f" | shots = {shots}" if mode == "shots" else "")
    )

    device = LocalSimulator()

    times = np.arange(0, total_time+dt, dt)
    F = np.zeros(times.size)

    target_state = "1" + "0" * (total_chain_length - 1)
    target_index = int(target_state, 2)

    last_result = None

    for idx, t in enumerate(times):
        circuit = xy_trotter_circuit_gate(
            chain_length=total_chain_length,
            J_in=J_in,
            total_time=t,
            trotter_steps=trotter_steps,
            coupling_type=coupling_type
        )

        if mode == "shots":
            # ---- measurement-based ----
            for q in range(total_chain_length):
                circuit.measure(q)

            task = device.run(circuit, shots=shots)
            result = task.result()
            counts = result.measurement_counts
            last_result = counts

            total_shots = sum(counts.values())
            fidelity = counts.get(target_state, 0) / total_shots if total_shots > 0 else 0.0

        else:
            # ---- statevector-based ----
            result = device.run(circuit.state_vector(), shots=0).result()
            psi = result.values[0]
            last_result = psi

            fidelity = np.abs(psi[target_index])**2

        F[idx] = fidelity

    peaks, _ = find_peaks(F, height=threshold)
    # peak_times = times[peaks]
    # peak_values = F[peaks]

    if len(peaks) == 0:
        peak_times = np.array([])
        peak_values = np.array([])
    else:
        best_peak = peaks[np.argmax(F[peaks])]
        peak_times = np.array([times[best_peak]])
        peak_values = np.array([F[best_peak]])

    return times, F, peak_times, peak_values, last_result

if __name__ == "__main__":
    # Testing the simulations and understanding the limits of the approximation
    dt = 0.001 # chosen time discretisation for exact method to plot the fidelity over time
    threshold = 2/3 # peaks only identified above the classical limit
    total_time = 3 # total simulation time
    J = -1.0 # J_in parameter
    s = 3 # system size
    alpha = 5 # decay strength for LRI couplings
    trotter_steps = [5,10,20,50]
    coupling = 'NN'

    # Initialise states
    initial_state = create_initial_final_states(s, True)
    final_state = create_initial_final_states(s, False)

    plt.figure()
    times_exact, F_exact, peak_times_exact, peak_values_exact, evolved_statevector_exact = compute_fidelity_curve(
        total_chain_length=s,
        total_time=total_time,
        initial_state=initial_state,
        final_state=final_state,
        dt=dt,
        threshold=threshold,
        J_in=J,
        method="exact",
        coupling_type=coupling
    )

    times_trotter, F_trotter, peak_times_trotter, peak_values_trotter, evolved_statevector_trotter = compute_fidelity_curve(
        total_chain_length=s,
        total_time=total_time,
        initial_state=initial_state,
        final_state=final_state,
        trotter_steps=50,
        threshold=threshold,
        J_in=J,
        method="trotter",
        coupling_type=coupling
    )

    
    plt.plot(times_exact, F_exact, label='Exact Time Evolution')
    plt.plot(times_trotter, F_trotter, label=f'Trotter (n = 50)')


    times_gate, F_gate, peak_times_gate, peak_values_gate, counts_gate = compute_fidelity_curve_gate(
        total_chain_length=s,
        total_time=total_time,
        dt=0.1,
        threshold=threshold,
        J_in=J,
        trotter_steps=50,
        coupling_type=coupling,
        shots=400,
        mode="statevector"
    )

    plt.plot(times_gate, F_gate, label=f'Gate (n = 50)')


    print(f"Method: Gate")
    print(peak_values_gate)
    print(peak_times_gate)
    print(f"Method: Exact")
    print(peak_values_exact)
    print(peak_times_exact)
    print(f"Method: Trotter")
    print(peak_values_trotter)
    print(peak_times_trotter)


    plt.xlabel("Time /s")
    plt.ylabel("Fidelity")
    plt.title(f"Fidelity Over Time for a {s} Qubit {coupling} Coupled System")
    plt.legend(fontsize=8,loc='best')
    plt.grid(True)
    plt.show()



    plt.figure()
    plt.plot(times_exact, F_exact, label='Exact Time Evolution')

    for n in trotter_steps:
        times_trotter, F_trotter, peak_times_trotter, peak_values_trotter, evolved_statevector_trotter = compute_fidelity_curve(
            total_chain_length=s,
            total_time=total_time,
            initial_state=initial_state,
            final_state=final_state,
            trotter_steps=n,
            threshold=threshold,
            J_in=J,
            method="trotter",
            coupling_type=coupling
        )

        plt.plot(times_trotter, F_trotter, label=f'Trotter (n = {n})')

        times_gate, F_gate, peak_times_gate, peak_values_gate, counts_gate = compute_fidelity_curve_gate(
            total_chain_length=s,
            total_time=total_time,
            dt=0.01,
            threshold=threshold,
            J_in=J,
            trotter_steps=n,
            coupling_type=coupling,
            shots=400,
            mode="statevector"
        )

        plt.plot(times_gate, F_gate, label=f'Gate (n = {n})')



    plt.xlabel("Time /s")
    plt.ylabel("Fidelity")
    plt.title(f"Fidelity Over Time for a {s} Qubit {coupling} Coupled System")
    plt.legend(fontsize=7,loc='best')
    plt.grid(True)
    plt.show()






