"""
Test the entanglement measures with known quantum states.
"""
import numpy as np
import pennylane as qml
from core.metrics import calculate_meyer_wallach

def test_bell_state():
    """Test Meyer-Wallach measure on a maximally entangled Bell state."""
    print("Testing Bell state entanglement...")
    
    # Create a Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    dev = qml.device("default.qubit", wires=2)
    
    @qml.qnode(dev)
    def bell_circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()
    
    state = bell_circuit()
    entanglement = calculate_meyer_wallach(state, 2)
    
    print(f"Bell state: {state}")
    print(f"Entanglement: {entanglement:.4f}")
    
    # Bell state should have maximum (1.0) entanglement
    assert entanglement > 0.99, f"Bell state entanglement should be ~1.0, got {entanglement:.4f}"
    print("✅ Bell state test passed!")

def test_ghz_state():
    """Test Meyer-Wallach measure on a GHZ state."""
    print("\nTesting GHZ state entanglement...")
    
    # Create a GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2
    dev = qml.device("default.qubit", wires=3)
    
    @qml.qnode(dev)
    def ghz_circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        return qml.state()
    
    state = ghz_circuit()
    entanglement = calculate_meyer_wallach(state, 3)
    
    print(f"GHZ state entanglement: {entanglement:.4f}")
    
    # GHZ state should also have high entanglement
    assert entanglement > 0.6, f"GHZ state entanglement should be high, got {entanglement:.4f}"
    print("✅ GHZ state test passed!")

def test_w_state():
    """Test Meyer-Wallach measure on a W state."""
    print("\nTesting W state entanglement...")
    
    # Create a W state |W⟩ = (|100⟩ + |010⟩ + |001⟩)/√3
    dev = qml.device("default.qubit", wires=3)
    
    @qml.qnode(dev)
    def w_circuit():
        # This is an approximation of the W state preparation
        qml.RY(2 * np.arccos(1/np.sqrt(3)), wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(np.arccos(1/np.sqrt(2)), wires=0)
        qml.CNOT(wires=[0, 2])
        qml.PauliX(wires=0)
        return qml.state()
    
    state = w_circuit()
    entanglement = calculate_meyer_wallach(state, 3)
    
    print(f"W state entanglement: {entanglement:.4f}")
    
    # W state should have high entanglement
    assert entanglement > 0.6, f"W state entanglement should be high, got {entanglement:.4f}"
    print("✅ W state test passed!")

def test_product_state():
    """Test Meyer-Wallach measure on a product (separable) state."""
    print("\nTesting product state entanglement...")
    
    # Create a product state |+⟩⊗|+⟩⊗|+⟩
    dev = qml.device("default.qubit", wires=3)
    
    @qml.qnode(dev)
    def product_circuit():
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)
        return qml.state()
    
    state = product_circuit()
    entanglement = calculate_meyer_wallach(state, 3)
    
    print(f"Product state entanglement: {entanglement:.4f}")
    
    # Product state should have zero entanglement
    assert entanglement < 0.01, f"Product state entanglement should be ~0, got {entanglement:.4f}"
    print("✅ Product state test passed!")

if __name__ == "__main__":
    print("Testing entanglement measures with known quantum states...")
    test_bell_state()
    test_ghz_state()
    test_w_state()
    test_product_state()
    print("\n✅ All entanglement measure tests passed!") 