import pennylane as qml
from pennylane import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from scipy.linalg import sqrtm
from scipy.stats import entropy # For KL divergence

def partial_trace(state_vector, num_qubits, subsystem_wires):
    """Calculates the reduced density matrix using PennyLane's function."""
    # Ensure state_vector is a NumPy array for qml.density_matrix
    if TORCH_AVAILABLE and isinstance(state_vector, torch.Tensor):
        state_vector_np = state_vector.detach().cpu().numpy()
    else:
        state_vector_np = np.array(state_vector)

    # Check if state_vector_np is already a density matrix (e.g., shape [2**N, 2**N])
    # or a state vector (shape [2**N])
    rho = state_vector_np
    if len(state_vector_np.shape) == 1:
        # Convert state vector to density matrix
        # Note: qml.math.dm_from_state_vector doesn't accept wires parameter
        rho = qml.math.dm_from_state_vector(state_vector_np)
    elif state_vector_np.shape != (2**num_qubits, 2**num_qubits):
        raise ValueError("Input must be a state vector or density matrix.")

    # Wires to trace out
    all_wires = list(range(num_qubits))
    keep_wires = subsystem_wires  # The wires to keep (not trace out)
    trace_wires = [w for w in all_wires if w not in keep_wires]  # The wires to trace out
    
    # Use PennyLane's partial_trace with correct parameters
    reduced_dm = qml.math.partial_trace(rho, indices=trace_wires, c_dtype=rho.dtype)
    
    return reduced_dm


def calculate_purity(density_matrix):
    """Calculates the purity Tr(rho^2) of a density matrix."""
    # Ensure input is NumPy array
    if TORCH_AVAILABLE and isinstance(density_matrix, torch.Tensor):
        density_matrix = density_matrix.detach().cpu().numpy()

    # Calculate rho^2
    rho_squared = np.dot(density_matrix, density_matrix)
    # Calculate trace
    purity = np.trace(rho_squared)
    # Return the real part (should be real anyway)
    return np.real(purity)

def calculate_meyer_wallach(state_vector, num_qubits):
    """
    Calculate the Meyer-Wallach entanglement measure (Q) for a quantum state.
    
    Q = 0 for a completely separable state
    Q = 1 for a maximally entangled state (averaged over bipartitions)
    
    Args:
        state_vector: The quantum state vector
        num_qubits: Number of qubits in the system
        
    Returns:
        float: The Meyer-Wallach entanglement measure (0 to 1)
    """
    # Convert to numpy if it's a torch tensor
    if TORCH_AVAILABLE and isinstance(state_vector, torch.Tensor):
        state_vector = state_vector.detach().cpu().numpy()
    
    # Ensure state is complex
    if not np.iscomplexobj(state_vector):
        state_vector = state_vector.astype(complex)
    
    # Normalize state vector if needed
    norm = np.linalg.norm(state_vector)
    if abs(norm - 1.0) > 1e-6:
        state_vector = state_vector / norm
    
    # Create density matrix representation of state
    rho = np.outer(state_vector, np.conj(state_vector))
    
    # Calculate the average subsystem purity
    avg_purity = 0
    dim = 2**num_qubits
    
    for target_qubit in range(num_qubits):
        # Calculate the reduced density matrix by tracing out all qubits except target_qubit
        reduced_rho = np.zeros((2, 2), dtype=complex)
        
        for i in range(2):
            for j in range(2):
                for k in range(2**(num_qubits-1)):
                    # Convert k to binary representation of length num_qubits-1
                    k_bits = format(k, f'0{num_qubits-1}b')
                    
                    # Insert bit i at position target_qubit
                    idx1 = int(k_bits[:target_qubit] + str(i) + k_bits[target_qubit:], 2)
                    
                    # Insert bit j at position target_qubit
                    idx2 = int(k_bits[:target_qubit] + str(j) + k_bits[target_qubit:], 2)
                    
                    # Add to the reduced density matrix
                    reduced_rho[i, j] += rho[idx1, idx2]
        
        # Calculate the purity of the reduced density matrix
        purity_val = np.abs(np.trace(reduced_rho @ reduced_rho))
        avg_purity += purity_val
    
    # Calculate the Meyer-Wallach measure
    avg_purity = avg_purity / num_qubits
    meyer_wallach = 2 * (1 - avg_purity)
    
    return meyer_wallach

def purity(state_vector):
    """
    Calculate the purity of a quantum state.
    
    Purity = Tr(ρ²), where ρ is the density matrix.
    For a pure state, purity = 1.
    For a completely mixed state, purity = 1/d where d is the dimension.
    
    Args:
        state_vector: The quantum state vector
        
    Returns:
        float: The purity of the state
    """
    # Convert to numpy if it's a torch tensor
    if TORCH_AVAILABLE and isinstance(state_vector, torch.Tensor):
        state_vector = state_vector.detach().cpu().numpy()
    
    # Ensure state is complex
    if not np.iscomplexobj(state_vector):
        state_vector = state_vector.astype(complex)
    
    # Normalize state vector if needed
    norm = np.linalg.norm(state_vector)
    if abs(norm - 1.0) > 1e-6:
        state_vector = state_vector / norm
    
    # Reshape into density matrix form
    state_vector = state_vector.reshape(-1, 1)
    rho = state_vector @ state_vector.conj().T
    
    # Calculate purity
    return np.real(np.trace(rho @ rho))

def linear_entropy(state_vector):
    """
    Calculate the linear entropy of a quantum state.
    
    Linear entropy = 1 - Tr(ρ²)
    For a pure state, linear entropy = 0.
    For a completely mixed state, linear entropy = 1 - 1/d where d is the dimension.
    
    Args:
        state_vector: The quantum state vector
        
    Returns:
        float: The linear entropy of the state
    """
    return 1 - purity(state_vector)

def von_neumann_entropy(state_vector):
    """
    Calculate the von Neumann entropy of a quantum state.
    
    S(ρ) = -Tr(ρ log₂ ρ) = -∑ᵢ λᵢ log₂ λᵢ
    where λᵢ are the eigenvalues of ρ.
    
    Args:
        state_vector: The quantum state vector
        
    Returns:
        float: The von Neumann entropy of the state
    """
    # Convert to numpy if it's a torch tensor
    if TORCH_AVAILABLE and isinstance(state_vector, torch.Tensor):
        state_vector = state_vector.detach().cpu().numpy()
    
    # Ensure state is complex
    if not np.iscomplexobj(state_vector):
        state_vector = state_vector.astype(complex)
    
    # Normalize state vector if needed
    norm = np.linalg.norm(state_vector)
    if abs(norm - 1.0) > 1e-6:
        state_vector = state_vector / norm
    
    # Reshape into density matrix form
    state_vector = state_vector.reshape(-1, 1)
    rho = state_vector @ state_vector.conj().T
    
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvalsh(rho)
    
    # Filter out very small eigenvalues (numerical errors)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    # Calculate entropy
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    
    return np.real(entropy)

def schmidt_rank(state_vector, num_qubits, partition=None):
    """
    Calculate the Schmidt rank of a quantum state with respect to a bipartition.
    
    The Schmidt rank is the number of non-zero Schmidt coefficients in the 
    Schmidt decomposition of the state with respect to a bipartition.
    
    Args:
        state_vector: The quantum state vector
        num_qubits: Number of qubits in the system
        partition: Qubits in the first subsystem, defaults to [0] (first qubit)
        
    Returns:
        int: The Schmidt rank
    """
    # Default partition
    if partition is None:
        partition = [0]  # First qubit
    
    # Convert to numpy if it's a torch tensor
    if TORCH_AVAILABLE and isinstance(state_vector, torch.Tensor):
        state_vector = state_vector.detach().cpu().numpy()
    
    # Ensure state is complex
    if not np.iscomplexobj(state_vector):
        state_vector = state_vector.astype(complex)
    
    # Normalize state vector if needed
    norm = np.linalg.norm(state_vector)
    if abs(norm - 1.0) > 1e-6:
        state_vector = state_vector / norm
    
    # Calculate dimensions
    dim_A = 2 ** len(partition)
    dim_B = 2 ** (num_qubits - len(partition))
    
    # Reshape state vector into a matrix based on the partition
    state_matrix = state_vector.reshape(dim_A, dim_B)
    
    # Perform singular value decomposition
    _, singular_values, _ = np.linalg.svd(state_matrix)
    
    # Count singular values above threshold
    rank = np.sum(singular_values > 1e-10)
    
    return int(rank)

# --- KL Divergence for Expressibility (Placeholder) ---

def sample_pqc_states(qnode_func, param_shape, num_samples=1000):
    """Samples output states from a PQC with random parameters."""
    # This requires executing the qnode repeatedly with random parameters
    # Implementation depends heavily on the qnode structure and parameter distribution
    # Placeholder: Returns random complex vectors for demonstration
    print("[Warning] sample_pqc_states is a placeholder.")
    num_qubits = qnode_func.device.num_wires
    output_dim = 2**num_qubits
    sampled_states = []
    # Example of how it might work (needs actual qnode execution):
    # for _ in range(num_samples):
    #     # Generate random parameters (e.g., uniform from 0 to 2pi)
    #     random_params = torch.rand(param_shape) * 2 * np.pi
    #     # Execute the qnode (assuming dummy input needed)
    #     dummy_input = torch.rand(output_dim)
    #     dummy_input = dummy_input / torch.norm(dummy_input)
    #     state = qnode_func(dummy_input, random_params).detach().cpu().numpy()
    #     sampled_states.append(state)

    # Using random states as placeholder output
    for _ in range(num_samples):
        state = np.random.randn(output_dim) + 1j * np.random.randn(output_dim)
        state /= np.linalg.norm(state)
        sampled_states.append(state)
    return np.array(sampled_states)

def sample_haar_random_states(num_qubits, num_samples=1000):
    """Samples states from the Haar random distribution."""
    # Ginibre ensemble method: Sample complex Gaussian matrix Z, compute state |psi> = Z |0> / || Z |0> ||
    output_dim = 2**num_qubits
    haar_states = []
    for _ in range(num_samples):
        # Sample complex Gaussian vector
        psi = np.random.randn(output_dim) + 1j * np.random.randn(output_dim)
        # Normalize
        psi /= np.linalg.norm(psi)
        haar_states.append(psi)
    return np.array(haar_states)

def fidelity(state1, state2):
    """Calculates the fidelity |<state1|state2>|^2."""
    return np.abs(np.vdot(state1, state2))**2

def calculate_kl_expressibility(qnode_func, param_shape, num_samples_pqc=1000, num_samples_haar=10000, num_bins=50):
    """
    Estimates the expressibility of a PQC using KL divergence between PQC
    state fidelities and Haar random state fidelities.

    NOTE: This is computationally very expensive and serves as a structural placeholder.
          A proper implementation requires careful sampling and binning.

    Args:
        qnode_func: The QNode function (or a callable wrapper).
        param_shape: The shape of the parameters for the qnode.
        num_samples_pqc: Number of samples from the PQC.
        num_samples_haar: Number of samples from Haar distribution.
        num_bins: Number of bins for histogramming fidelities.

    Returns:
        float: Estimated KL divergence.
    """
    print("[Warning] calculate_kl_expressibility is a placeholder and computationally expensive.")
    num_qubits = qnode_func.device.num_wires

    # 1. Sample states from PQC
    pqc_states = sample_pqc_states(qnode_func, param_shape, num_samples_pqc)

    # 2. Sample states from Haar distribution
    haar_states = sample_haar_random_states(num_qubits, num_samples_haar)

    # 3. Calculate pairwise fidelities for PQC states
    fidelities_pqc = []
    for i in range(num_samples_pqc):
        for j in range(i + 1, num_samples_pqc):
            fidelities_pqc.append(fidelity(pqc_states[i], pqc_states[j]))

    # 4. Calculate pairwise fidelities for Haar states
    fidelities_haar = []
    for i in range(num_samples_haar):
        for j in range(i + 1, num_samples_haar):
            fidelities_haar.append(fidelity(haar_states[i], haar_states[j]))

    if not fidelities_pqc or not fidelities_haar:
        print("[Error] Could not compute fidelities. Skipping KL divergence calculation.")
        return np.nan

    # 5. Create histograms (probability distributions)
    # Ensure range is [0, 1] for fidelity
    hist_range = (0.0, 1.0)
    pqc_hist, bin_edges = np.histogram(fidelities_pqc, bins=num_bins, range=hist_range, density=True)
    haar_hist, _ = np.histogram(fidelities_haar, bins=bin_edges, range=hist_range, density=True)

    # Add small epsilon to avoid log(0) and division by zero
    epsilon = 1e-10
    pqc_hist += epsilon
    haar_hist += epsilon

    # Normalize again after adding epsilon? Or handle zeros in entropy calculation.
    # scipy.stats.entropy handles zeros appropriately if using pk=pqc_hist, qk=haar_hist

    # 6. Calculate KL Divergence D_KL(PQC || Haar)
    # Measures how much the PQC distribution diverges from the Haar distribution.
    # Lower KL means better expressibility (closer to Haar random).
    kl_divergence = entropy(pk=pqc_hist, qk=haar_hist)

    # Note: Could also calculate D_KL(Haar || PQC)
    # kl_divergence_rev = entropy(pk=haar_hist, qk=pqc_hist)

    print(f"Calculated KL Divergence (placeholder): {kl_divergence}")
    return kl_divergence 