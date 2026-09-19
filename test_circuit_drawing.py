"""
Test that circuit drawing works with the new model structure.
"""
import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pennylane as qml

# Add the current directory to the path for imports
sys.path.append(os.path.abspath('.'))

# Import the functions we need to test
from core.fusion import HybridModel
from core.quantum_models import ansatz1, create_qnn

def test_circuit_drawing():
    """Test that we can draw the circuit with the new model structure."""
    print("\nTesting circuit drawing with the new model structure...")
    
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
    
    # Verify the model has the expected attributes
    print("Checking model attributes...")
    assert hasattr(model, 'quantum_circuit_fn'), "Model should have quantum_circuit_fn attribute"
    assert hasattr(model, 'quantum_params'), "Model should have quantum_params attribute"
    assert hasattr(model, 'num_quantum_params'), "Model should have num_quantum_params attribute"
    
    # Create a QNode for visualization - this simulates what we do in app.py
    print("Creating QNode for visualization...")
    qnode_vis, _ = create_qnn(
        num_qubits=config['num_qubits'],
        num_layers=config['num_layers'],
        ansatz_func=config['ansatz_func'],
        use_gpu=config['use_gpu'],
        observation_type=config['observation_type']
    )
    
    # Prepare dummy inputs for circuit drawing
    print("Preparing dummy inputs...")
    n_qubits = config['num_qubits']
    dummy_input = torch.rand(2**n_qubits)
    dummy_input = dummy_input / torch.norm(dummy_input)
    dummy_params = model.quantum_params.detach().cpu()
    
    # Try drawing the circuit
    print("Drawing circuit...")
    try:
        fig, ax = qml.draw_mpl(qnode_vis)(dummy_input, dummy_params)
        print("✅ Circuit drawing successful!")
        
        # Save the figure to a file (optional)
        plt.savefig('test_circuit.png')
        print("Saved circuit diagram to test_circuit.png")
        
        # Clean up
        plt.close(fig)
        
        return True
    except Exception as e:
        print(f"❌ Circuit drawing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
if __name__ == "__main__":
    test_circuit_drawing() 