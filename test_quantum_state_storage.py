"""
Test the storage and retrieval of quantum state for entanglement calculation.
"""
import torch
import numpy as np
import sys
import os

# Add the current directory to the path for imports
sys.path.append(os.path.abspath('.'))

# Import the functions we need to test
from core.fusion import HybridModel
from core.metrics import calculate_meyer_wallach
from core.quantum_models import ansatz1

def test_quantum_state_storage():
    """Test that the model properly stores the quantum state output before applying the output head."""
    print("\nTesting quantum state storage in HybridModel...")
    
    # Create a minimal configuration for a hybrid model
    config = {
        'classical_backbone_type': 'none',  # No classical backbone for testing
        'num_qubits': 4,                    # 4 qubits = 16-dimensional state vector
        'num_layers': 2,                    # Small circuit for quick test
        'ansatz_func': ansatz1,             # Use default ansatz
        'use_gpu': False,                   # Use CPU for testing
        'observation_type': 'state vector', # Use state vector output
        'num_classes': 2                    # Binary classification
    }
    
    # Create the model
    print("Creating model...")
    model = HybridModel(config)
    
    # Initialize last_quantum_output
    assert hasattr(model, 'last_quantum_output'), "Model should have last_quantum_output attribute"
    assert model.last_quantum_output is None, "last_quantum_output should be initialized as None"
    
    # Create input - for 'none' backbone, input size should be 2^num_qubits = 16
    input_data = torch.rand(2, 16)  # Batch of 2 samples
    input_data = input_data / torch.norm(input_data, dim=1, keepdim=True)  # Normalize
    
    # Run forward pass
    print("Running forward pass...")
    output = model(input_data)
    
    # Check that last_quantum_output was saved
    print("Checking last_quantum_output...")
    assert model.last_quantum_output is not None, "last_quantum_output should be set after forward pass"
    assert model.last_quantum_output.shape[0] == 2, "Batch dimension should be preserved"
    assert model.last_quantum_output.shape[1] == 16, "State vector dimension should be 2^num_qubits = 16"
    
    # Check that output is different from last_quantum_output (proving it went through output head)
    assert output.shape != model.last_quantum_output.shape, "Output shape should differ from quantum output shape"
    
    # Make sure model output has correct shape for classification
    assert output.shape == (2, 2), f"Expected output shape (2, 2), got {output.shape}"
    
    # For a 2-class model, the output dimension should be different from quantum state
    assert output.shape[1] == 2, "Classification output should have num_classes dimensions"
    assert model.last_quantum_output.shape[1] == 2**config['num_qubits'], "Quantum output should have 2^num_qubits dimensions"
    
    # Try calculating entanglement using the stored quantum state
    print("Calculating entanglement from stored quantum state...")
    quantum_state = model.last_quantum_output[0].detach().cpu()  # First sample
    entanglement = calculate_meyer_wallach(quantum_state, config['num_qubits'])
    print(f"Entanglement value: {entanglement}")
    
    # Entanglement should be a valid value
    assert not np.isnan(entanglement), "Entanglement should not be NaN"
    assert 0 <= entanglement <= 1, "Entanglement should be between 0 and 1"
    
    print("✅ All tests passed! The model correctly stores quantum state before applying output head.")
    return True

if __name__ == "__main__":
    test_quantum_state_storage() 