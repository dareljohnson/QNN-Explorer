"""
Test loading the weights_last_train file with different model configurations.
"""
import torch
import sys
import os

# Add the current directory to the path for imports
sys.path.append(os.path.abspath('.'))

# Import necessary functions
from core.fusion import HybridModel
from core.quantum_models import ansatz1

def test_load_last_train_weights():
    """Test loading the real weights_last_train file with different model configs."""
    print("\nTesting loading weights_last_train file...")
    
    # Path to the last trained weights
    weights_path = os.path.join("saved_models", "weights", "weights_last_train")
    
    # Check if the weights file exists
    if not os.path.exists(weights_path):
        print(f"⚠️ Weights file not found at {weights_path}")
        return False
        
    # First examine the structure of the weights file
    print(f"Examining weights file: {weights_path}")
    try:
        # Load the weights file
        weights = torch.load(weights_path, map_location='cpu')
        
        # Get the weights keys (parameters)
        weight_keys = list(weights.keys())
        print(f"Weight file contains {len(weight_keys)} parameters")
        
        # Check if it has output head parameters
        has_output_head = any('output_head' in k for k in weight_keys)
        print(f"Weight file has output head parameters: {has_output_head}")
        
        # Print key names for output head parameters
        if has_output_head:
            output_head_keys = [k for k in weight_keys if 'output_head' in k]
            print(f"Output head parameters: {output_head_keys}")
            
            # Determine number of classes from output head weights
            if 'output_head.weight' in weights:
                output_shape = weights['output_head.weight'].shape
                num_classes = output_shape[0]
                input_size = output_shape[1]
                print(f"Output head shape: ({num_classes} classes, {input_size} input features)")
            
    except Exception as e:
        print(f"❌ Error examining weights file: {e}")
        return False
    
    # Now try to create a compatible model and load the weights
    if has_output_head:
        print("\nCreating model with output head to match weights file...")
        try:
            # Extract parameters from the weights themselves
            config = {
                'classical_backbone_type': 'none',
                'num_qubits': 4,  # Default, might need adjustment
                'num_layers': 2,   # Default, might need adjustment
                'ansatz_func': ansatz1,
                'use_gpu': False,
                'observation_type': 'state vector',
                'num_classes': num_classes  # From the output head shape
            }
            
            # Create the model
            model = HybridModel(config)
            
            # Try to load the weights
            model.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)
            print("✅ Successfully loaded weights with strict=False")
            
            # Check what parameters were actually loaded
            loaded_params = set(model.state_dict().keys())
            saved_params = set(weights.keys())
            
            # Calculate differences
            missing_in_model = saved_params - loaded_params
            missing_in_weights = loaded_params - saved_params
            
            # Report differences
            if missing_in_model:
                print(f"Parameters in weights but not in model: {missing_in_model}")
            if missing_in_weights:
                print(f"Parameters in model but not in weights: {missing_in_weights}")
                
            print("✅ Weight loading test successful!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            return False
    else:
        print("⚠️ Weights file doesn't have output head parameters - skipping load test")
        return True

if __name__ == "__main__":
    test_load_last_train_weights() 