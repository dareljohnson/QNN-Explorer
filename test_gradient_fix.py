"""
Test that the quantum model parameters support gradients and can be trained.
"""
import torch
import numpy as np
import sys
import os

# Add the current directory to the path for imports
sys.path.append(os.path.abspath('.'))

# Import the functions we need to test
from core.fusion import HybridModel
from core.quantum_models import ansatz1

def test_quantum_gradient_flow():
    """Test that the quantum model parameters support gradients and can be trained."""
    print("\nTesting quantum gradient flow in HybridModel...")
    
    # Create a minimal configuration for a hybrid model
    config = {
        'classical_backbone_type': 'none',  # No classical backbone for testing
        'num_qubits': 2,                   # Small number of qubits for faster test
        'num_layers': 1,                   # Small circuit for quick test
        'ansatz_func': ansatz1,            # Use default ansatz
        'use_gpu': False,                  # Use CPU for testing
        'observation_type': 'state vector', # Use state vector output
        'num_classes': 2                   # Binary classification
    }
    
    # Create the model
    print("Creating model...")
    model = HybridModel(config)
    
    # Verify parameters require gradients
    print("Checking parameter gradients...")
    assert model.quantum_params.requires_grad, "Quantum parameters should require gradients"
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Create a simple input and target
    input_size = 2**config['num_qubits']   # 4 for 2 qubits
    batch_size = 2
    
    # Create random input data
    x = torch.rand(batch_size, input_size)
    
    # Create random target labels
    y = torch.randint(0, 2, (batch_size,), dtype=torch.long)
    
    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Run a simple training step
    print("Running forward and backward pass...")
    
    # Save initial parameters
    initial_params = model.quantum_params.clone().detach()
    
    # Training loop
    for i in range(5):
        # Forward pass
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        
        # Print diagnostics
        print(f"Step {i+1}, Loss: {loss.item()}")
        
        # Backward pass
        loss.backward()
        
        # Check if gradients exist
        assert model.quantum_params.grad is not None, "Quantum parameters should have gradients after backward pass"
        print(f"Gradient norm: {model.quantum_params.grad.norm().item()}")
        
        # Optimizer step
        optimizer.step()
    
    # Verify parameters changed
    param_diff = (model.quantum_params - initial_params).abs().sum().item()
    print(f"Parameter change: {param_diff}")
    assert param_diff > 1e-6, "Parameters should change after optimization"
    
    print("✅ All tests passed! Quantum parameters can be trained with gradients.")
    return True

if __name__ == "__main__":
    test_quantum_gradient_flow() 