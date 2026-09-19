import torch
import numpy as np
import unittest
import sys
import os

# Add the current directory to the path to import the modules
sys.path.append(os.path.abspath('.'))

# Import the needed functions
from core.metrics import calculate_meyer_wallach
from core.fusion import HybridModel
from core.quantum_models import ansatz1

class TestEntanglementFix(unittest.TestCase):
    """Tests the fix for NaN entanglement values"""
    
    def test_valid_state_has_valid_entanglement(self):
        """Test that a valid state returns a non-NaN entanglement value"""
        # Create a valid state vector
        num_qubits = 3
        dim = 2**num_qubits
        
        # Random normalized state
        state = np.random.rand(dim) + 1j * np.random.rand(dim)
        state = state / np.linalg.norm(state)
        
        # Calculate entanglement
        entanglement = calculate_meyer_wallach(state, num_qubits)
        
        # Check that it's not NaN
        self.assertFalse(np.isnan(entanglement), "Entanglement should not be NaN for valid state")
        
        # Also check it's in the valid range [0,1]
        self.assertGreaterEqual(entanglement, 0.0, "Entanglement should be ≥ 0")
        self.assertLessEqual(entanglement, 1.0, "Entanglement should be ≤ 1")
    
    def test_nan_input_returns_nan(self):
        """Test that a state with NaN returns NaN entanglement but doesn't crash"""
        num_qubits = 3
        dim = 2**num_qubits
        
        # Create a state with NaN
        state = np.random.rand(dim) + 1j * np.random.rand(dim)
        state[0] = np.nan  # Insert a NaN
        
        # Calculate entanglement - should return NaN but not crash
        entanglement = calculate_meyer_wallach(state, num_qubits)
        
        # Check that it returns NaN
        self.assertTrue(np.isnan(entanglement), "Entanglement should be NaN for input with NaN")
    
    def test_wrong_size_input_returns_nan(self):
        """Test that a state with wrong size returns NaN entanglement but doesn't crash"""
        num_qubits = 3
        dim = 2**num_qubits
        
        # Create a state with wrong size
        state = np.random.rand(dim//2) + 1j * np.random.rand(dim//2)  # Half the expected size
        state = state / np.linalg.norm(state)
        
        # Calculate entanglement - should return NaN but not crash
        entanglement = calculate_meyer_wallach(state, num_qubits)
        
        # Check that it returns NaN
        self.assertTrue(np.isnan(entanglement), "Entanglement should be NaN for wrong size input")
    
    def test_ghz_state_high_entanglement(self):
        """Test that a GHZ state has high entanglement"""
        num_qubits = 3
        dim = 2**num_qubits
        
        # Create GHZ state |000> + |111> / sqrt(2)
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0 / np.sqrt(2)   # |000>
        state[-1] = 1.0 / np.sqrt(2)  # |111>
        
        # Calculate entanglement
        entanglement = calculate_meyer_wallach(state, num_qubits)
        
        # Check that it's high (close to 1)
        self.assertGreaterEqual(entanglement, 0.9, "GHZ state should have high entanglement")
    
    def test_separable_state_zero_entanglement(self):
        """Test that a separable state has zero entanglement"""
        num_qubits = 3
        dim = 2**num_qubits
        
        # Create a separable state |000>
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        
        # Calculate entanglement
        entanglement = calculate_meyer_wallach(state, num_qubits)
        
        # Check that it's zero or very close to zero
        self.assertLessEqual(entanglement, 1e-10, "Separable state should have zero entanglement")

def test_quantum_state_storage():
    """Test that the model correctly stores the quantum state for entanglement calculation."""
    print("\nTesting quantum state storage for entanglement calculation...\n")
    
    # 1. Create a test model with classification output head
    config = {
        'classical_backbone_type': 'none',
        'num_qubits': 4,  # 4 qubits = 16-dimensional state vector
        'num_layers': 2,
        'ansatz_func': ansatz1,
        'use_gpu': False,
        'observation_type': 'state vector',
        'num_classes': 2  # Binary classification
    }
    
    model = HybridModel(config)
    print("Model created with classification head (num_classes=2)")
    
    # 2. Create a test input
    batch_size = 2
    input_size = 2**config['num_qubits']  # 16
    test_input = torch.rand(batch_size, input_size)
    
    # 3. Run the model
    print(f"Running model with input shape: {test_input.shape}")
    output = model(test_input)
    
    # 4. Check the output shape
    print(f"Model output shape: {output.shape}")
    # For classification, should be [batch_size, num_classes]
    assert output.shape == (batch_size, config['num_classes']), \
        f"Expected output shape {(batch_size, config['num_classes'])}, got {output.shape}"
    
    # 5. Check that last_quantum_output was stored
    print("Checking last_quantum_output...")
    assert model.last_quantum_output is not None, "last_quantum_output should be set"
    
    # 6. Check that last_quantum_output has the correct shape
    expected_quantum_shape = (batch_size, 2**config['num_qubits'])
    assert model.last_quantum_output.shape == expected_quantum_shape, \
        f"Expected quantum output shape {expected_quantum_shape}, got {model.last_quantum_output.shape}"
    
    # 7. Verify that entanglement can be calculated correctly
    print("Calculating entanglement for the first sample...")
    quantum_state = model.last_quantum_output[0]
    entanglement = calculate_meyer_wallach(quantum_state, config['num_qubits'])
    print(f"Entanglement value: {entanglement}")
    
    # 8. Check that entanglement is a valid value
    assert not np.isnan(entanglement), "Entanglement should not be NaN"
    assert 0 <= entanglement <= 1, "Entanglement should be between 0 and 1"
    
    print("\n✅ All tests passed! The model correctly stores quantum state for entanglement calculation.")
    return True

if __name__ == "__main__":
    test_quantum_state_storage() 