import sys
import os
import torch
import unittest
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from core.fusion import HybridModel

class TestCSVIncompatible(unittest.TestCase):
    def test_csv_incompatible_with_cnn(self):
        """Test that appropriate error message is shown when CSV data can't be reshaped to image format."""
        print("\nTesting incompatible CSV data with CNN backbone...")
        
        # Create a model config with CNN backbone
        config = {
            'classical_backbone_type': 'cnn',
            'classical_model_name': 'resnet18',
            'num_qubits': 4,  # 2^4 = 16 quantum states
            'num_layers': 1,
            'use_gpu': False,
            'observation_type': 'state'
        }
        
        # Create a mock input with 400 features (which isn't cleanly divisible by 3 for RGB channels)
        batch_size = 30
        features = 400  # This is the exact shape from the error message
        
        # Create a mock CSV input tensor
        csv_input = torch.rand(batch_size, features)
        print(f"Input shape: {csv_input.shape}")
        
        try:
            # Create the model
            model = HybridModel(config)
            model.eval()
            
            # Process the input - this should raise a ValueError
            with torch.no_grad():
                output = model(csv_input)
            
            # If we get here, the test has failed
            self.fail("Expected ValueError but no exception was raised")
            return False
        except ValueError as e:
            # Check that the error message contains our expected text
            print(f"Got expected ValueError: {e}")
            self.assertIn("cannot be reshaped", str(e), "Error message should indicate reshaping issue")
            self.assertIn("use a 'None' backbone", str(e), "Error message should suggest using None backbone")
            print("✅ Test passed! Appropriate error raised for incompatible CSV data")
            return True
        except Exception as e:
            # Any other exception type is unexpected
            print(f"❌ Test failed with unexpected error type: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Unexpected error type: {e}")
            return False

if __name__ == "__main__":
    unittest.main() 