"""
Test entanglement metrics with known highly entangled states.
"""
import pennylane as qml
from pennylane import numpy as np  # Use pennylane's numpy
from core.metrics import calculate_meyer_wallach, purity, von_neumann_entropy

# Define linear entropy function locally for convenience
def linear_entropy(state_vector):
    """Calculate linear entropy as 1 - purity."""
    return 1.0 - purity(state_vector)

def test_bell_state():
    """
    Test the entanglement metrics on a Bell state.
    A Bell state should have maximum entanglement.
    """
    print("\nTesting Bell state with 2 qubits")
    dev = qml.device("default.qubit", wires=2)
    
    @qml.qnode(dev)
    def bell_circuit():
        # Create Bell state (|00> + |11>)/sqrt(2)
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()
    
    # Get Bell state
    bell_state = bell_circuit()
    
    # Calculate entanglement metrics
    mw_entanglement = calculate_meyer_wallach(bell_state, 2)
    state_purity = purity(bell_state)
    entropy = linear_entropy(bell_state)
    vn_entropy = von_neumann_entropy(bell_state)
    
    print(f"Bell state: (|00> + |11>)/sqrt(2)")
    print(f"Meyer-Wallach entanglement: {mw_entanglement:.6f}")
    print(f"State purity: {state_purity:.6f}")
    print(f"Linear entropy: {entropy:.6f}")
    print(f"von Neumann entropy: {vn_entropy:.6f}")
    
    # For Bell states, Meyer-Wallach should be 1.0
    assert mw_entanglement > 0.95, f"Bell state should have maximum entanglement, got {mw_entanglement:.6f}"
    return mw_entanglement

def test_ghz_state(num_qubits=4):
    """
    Test the entanglement metrics on a GHZ state.
    A GHZ state should have high entanglement.
    """
    print(f"\nTesting GHZ state with {num_qubits} qubits")
    dev = qml.device("default.qubit", wires=num_qubits)
    
    @qml.qnode(dev)
    def ghz_circuit():
        # Start with |0> state
        # Apply Hadamard to first qubit
        qml.Hadamard(wires=0)
        # Apply CNOTs to create GHZ state
        for i in range(num_qubits-1):
            qml.CNOT(wires=[i, i+1])
        return qml.state()
    
    # Get GHZ state
    ghz_state = ghz_circuit()
    
    # Calculate entanglement metrics
    mw_entanglement = calculate_meyer_wallach(ghz_state, num_qubits)
    state_purity = purity(ghz_state)
    entropy = linear_entropy(ghz_state)
    vn_entropy = von_neumann_entropy(ghz_state)
    
    print(f"GHZ state: |0>^⊗n + |1>^⊗n")
    print(f"Meyer-Wallach entanglement: {mw_entanglement:.6f}")
    print(f"State purity: {state_purity:.6f}")
    print(f"Linear entropy: {entropy:.6f}")
    print(f"von Neumann entropy: {vn_entropy:.6f}")
    
    # For GHZ states, Meyer-Wallach should be 1.0
    assert mw_entanglement > 0.9, f"GHZ state should have high entanglement, got {mw_entanglement:.6f}"
    return mw_entanglement

def test_w_state(num_qubits=4):
    """
    Test the entanglement metrics on a W state.
    A W state should have high entanglement.
    """
    print(f"\nTesting W state with {num_qubits} qubits")
    dev = qml.device("default.qubit", wires=num_qubits)
    
    @qml.qnode(dev)
    def w_circuit():
        # Create a simpler W state implementation without using ctrl
        # W state = (|100...0> + |010...0> + ... + |000...1>)/sqrt(n)
        
        # Start in |0>^⊗n state
        
        # Apply a Hadamard to the first qubit to create superposition
        qml.Hadamard(wires=0)
        
        # For a 2-qubit system, create (|10> + |01>)/sqrt(2)
        if num_qubits == 2:
            qml.PauliX(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=0)
            return qml.state()
            
        # For 3+ qubits, build a specific circuit for W state
        if num_qubits == 3:
            # For 3 qubits, we can use a specific approach
            # Start by applying rotation to put qubit 0 in a specific superposition
            qml.RY(2 * np.arccos(1/np.sqrt(3)), wires=0)
            
            # Apply CNOT to entangle qubit 0 and 1
            qml.CNOT(wires=[0, 1])
            
            # Apply X to qubit 0
            qml.PauliX(wires=0)
            
            # Apply Toffoli (CCX) to create the remaining part
            qml.Toffoli(wires=[0, 1, 2])
            
            # Apply X to qubit 1
            qml.PauliX(wires=1)
            
        elif num_qubits == 4:
            # For 4 qubits, use another fixed approach
            # Prepare first qubit in specific superposition
            qml.RY(2 * np.arccos(1/np.sqrt(4)), wires=0)
            
            # Entangle with qubit 1
            qml.CNOT(wires=[0, 1])
            
            # Rotate qubit 0 for next entanglement
            qml.PauliX(wires=0)
            qml.RY(2 * np.arccos(1/np.sqrt(3)), wires=1)
            
            # Entangle with qubit 2
            qml.CNOT(wires=[1, 2])
            
            # Rotate again
            qml.PauliX(wires=1)
            qml.RY(2 * np.arccos(1/np.sqrt(2)), wires=2)
            
            # Final entanglement
            qml.CNOT(wires=[2, 3])
            qml.PauliX(wires=2)
        
        return qml.state()
    
    # Get W state
    w_state = w_circuit()
    
    # Calculate entanglement metrics
    mw_entanglement = calculate_meyer_wallach(w_state, num_qubits)
    state_purity = purity(w_state)
    entropy = linear_entropy(w_state)
    vn_entropy = von_neumann_entropy(w_state)
    
    print(f"W state: |100...0> + |010...0> + ... + |000...1>")
    print(f"Meyer-Wallach entanglement: {mw_entanglement:.6f}")
    print(f"State purity: {state_purity:.6f}")
    print(f"Linear entropy: {entropy:.6f}")
    print(f"von Neumann entropy: {vn_entropy:.6f}")
    
    # For W states, Meyer-Wallach should be high
    assert mw_entanglement > 0.5, f"W state should have high entanglement, got {mw_entanglement:.6f}"
    return mw_entanglement

def test_cluster_state(num_qubits=4):
    """
    Test the entanglement metrics on a cluster state.
    A cluster state should have high entanglement.
    """
    print(f"\nTesting cluster state with {num_qubits} qubits")
    dev = qml.device("default.qubit", wires=num_qubits)
    
    @qml.qnode(dev)
    def cluster_circuit():
        # Create |+> states
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        # Apply CZ gates between neighbors
        for i in range(num_qubits-1):
            qml.CZ(wires=[i, i+1])
        # Make it a ring
        if num_qubits > 2:
            qml.CZ(wires=[num_qubits-1, 0])
        return qml.state()
    
    # Get cluster state
    cluster_state = cluster_circuit()
    
    # Calculate entanglement metrics
    mw_entanglement = calculate_meyer_wallach(cluster_state, num_qubits)
    state_purity = purity(cluster_state)
    entropy = linear_entropy(cluster_state)
    vn_entropy = von_neumann_entropy(cluster_state)
    
    print(f"Cluster state: CZ_neighbor_links |+>^⊗n")
    print(f"Meyer-Wallach entanglement: {mw_entanglement:.6f}")
    print(f"State purity: {state_purity:.6f}")
    print(f"Linear entropy: {entropy:.6f}")
    print(f"von Neumann entropy: {vn_entropy:.6f}")
    
    # For cluster states, Meyer-Wallach should be high
    assert mw_entanglement > 0.5, f"Cluster state should have high entanglement, got {mw_entanglement:.6f}"
    return mw_entanglement

if __name__ == "__main__":
    # Test with known highly entangled states
    print("Starting benchmark tests...")
    try:
        bell_ent = test_bell_state()
        print(f"Bell state entanglement: {bell_ent}")
    except Exception as e:
        print(f"Bell state test failed: {e}")
    
    try:
        ghz_ent = test_ghz_state(4)
        print(f"GHZ state entanglement: {ghz_ent}")
    except Exception as e:
        print(f"GHZ state test failed: {e}")
    
    try:
        w_ent = test_w_state(4)
        print(f"W state entanglement: {w_ent}")
    except Exception as e:
        print(f"W state test failed: {e}")
    
    try:
        cluster_ent = test_cluster_state(4)
        print(f"Cluster state entanglement: {cluster_ent}")
    except Exception as e:
        print(f"Cluster state test failed: {e}")
    
    print("\nBenchmark tests completed!") 