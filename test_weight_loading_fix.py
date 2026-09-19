"""
Test loading weights between models with different configurations.
"""
import torch
import numpy as np
import sys
import os
import shutil

# Add the current directory to the path for imports
sys.path.append(os.path.abspath('.'))

# Import the necessary functions
from core.fusion import HybridModel
from core.quantum_models import ansatz1

def test_model_weight_loading():
    """Test loading weights between models with different configurations."""
    print("\nTesting model weight loading with different configurations...")
    
    # Create test directory
    test_dir = "test_weights"
    os.makedirs(test_dir, exist_ok=True)
    
    # 1. Create a model with output head (classification)
    config_with_head = {
        'classical_backbone_type': 'none',
        'num_qubits': 4,
        'num_layers': 2,
        'ansatz_func': ansatz1,
        'use_gpu': False,
        'observation_type': 'state vector',
        'num_classes': 2  # This adds an output head
    }
    model_with_head = HybridModel(config_with_head)
    
    # Save weights
    weights_with_head_path = os.path.join(test_dir, 'weights_with_head.pt')
    torch.save(model_with_head.state_dict(), weights_with_head_path)
    print(f"Saved model weights with output head to {weights_with_head_path}")
    
    # 2. Create a model without output head
    config_no_head = {
        'classical_backbone_type': 'none',
        'num_qubits': 4,
        'num_layers': 2,
        'ansatz_func': ansatz1,
        'use_gpu': False,
        'observation_type': 'state vector'
        # No num_classes, so no output head
    }
    model_no_head = HybridModel(config_no_head)
    
    # Save weights
    weights_no_head_path = os.path.join(test_dir, 'weights_no_head.pt')
    torch.save(model_no_head.state_dict(), weights_no_head_path)
    print(f"Saved model weights without output head to {weights_no_head_path}")
    
    # 3. Test loading weights with head into model without head
    print("\nTest 1: Loading weights WITH head into model WITHOUT head")
    try:
        # This should fail with strict=True
        model_no_head.load_state_dict(torch.load(weights_with_head_path), strict=True)
        print("❌ ERROR: Should have failed with strict=True!")
        assert False
    except Exception as e:
        print(f"✅ Expected error with strict=True: {e}")
    
    try:
        # This should succeed with strict=False
        model_no_head.load_state_dict(torch.load(weights_with_head_path), strict=False)
        print("✅ Successfully loaded weights with strict=False")
    except Exception as e:
        print(f"❌ ERROR: Should have succeeded with strict=False: {e}")
        assert False
    
    # 4. Test loading weights without head into model with head
    print("\nTest 2: Loading weights WITHOUT head into model WITH head")
    model_with_head_2 = HybridModel(config_with_head)  # New instance
    
    try:
        # This should fail with strict=True
        model_with_head_2.load_state_dict(torch.load(weights_no_head_path), strict=True)
        print("❌ ERROR: Should have failed with strict=True!")
        assert False
    except Exception as e:
        print(f"✅ Expected error with strict=True: {e}")
    
    try:
        # This should succeed with strict=False
        model_with_head_2.load_state_dict(torch.load(weights_no_head_path), strict=False)
        print("✅ Successfully loaded weights with strict=False")
    except Exception as e:
        print(f"❌ ERROR: Should have succeeded with strict=False: {e}")
        assert False
    
    # Clean up
    shutil.rmtree(test_dir)
    
    print("\n✅ All weight loading tests passed!")
    return True

if __name__ == "__main__":
    test_model_weight_loading() 