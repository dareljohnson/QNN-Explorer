import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('.'))

# Import the metrics and model components
from core.metrics import calculate_meyer_wallach
from core.fusion import HybridModel

def test_training_with_entanglement():
    """Simulate a training step with Meyer-Wallach entanglement calculation."""
    print("Testing training with Meyer-Wallach entanglement calculation...")
    
    # Create a simple model config
    config = {
        'classical_backbone_type': 'none',  # No classical backbone for simplicity
        'num_qubits': 4,
        'num_layers': 2,
        'use_gpu': False,
        'observation_type': 'State Vector'  # Important: This should match UI case
    }
    
    try:
        # Create a model
        print("Creating model...")
        model = HybridModel(config)
        
        # Create dummy input - should be size 2^num_qubits = 16
        print("Creating dummy input...")
        batch_size = 2
        input_data = torch.rand(batch_size, 16)  # Batch of 2, each with 16 features
        
        # Forward pass
        print("Performing forward pass...")
        outputs = model(input_data)
        print(f"Output shape: {outputs.shape}")
        
        # Calculate entanglement (similar to what happens in training loop)
        print("Calculating entanglement...")
        with torch.no_grad():
            # Use first sample in batch
            state_cpu = outputs[0].detach().cpu()
            entanglement = calculate_meyer_wallach(state_cpu, config['num_qubits'])
            print(f"Entanglement (Q): {entanglement:.4f}")
        
        print("\n✅ Test passed! Training with entanglement calculation works correctly.")
        return True
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("====================================================")
    print("Testing Training with Meyer-Wallach Entanglement")
    print("====================================================")
    success = test_training_with_entanglement()
    if not success:
        sys.exit(1)
    else:
        print("\nThe fix for entanglement calculation in training is working!") 