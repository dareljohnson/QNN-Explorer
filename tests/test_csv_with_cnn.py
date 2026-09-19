import sys
import os
import torch
import unittest
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from core.fusion import HybridModel

class TestCSVWithCNN(unittest.TestCase):
    def test_csv_data_with_cnn_backbone(self):
        """Test that CSV data (2D tensor) can be processed by a HybridModel with CNN backbone."""
        print("\nTesting CSV data with CNN backbone...")
        
        # Create a model config with CNN backbone
        config = {
            'classical_backbone_type': 'cnn',
            'classical_model_name': 'resnet18',
            'num_qubits': 4,  # 2^4 = 16 quantum states
            'num_layers': 1,
            'use_gpu': False,
            'observation_type': 'state'
        }
        
        # Create a mock input that resembles CSV data
        # Use a feature count that can be reshaped to an image tensor
        # 3 channels * 20 height * 20 width = 1200 features
        batch_size = 30
        features = 1200
        
        # Create a mock CSV input tensor
        csv_input = torch.rand(batch_size, features)
        print(f"Input shape: {csv_input.shape}")
        
        try:
            # Create the model
            model = HybridModel(config)
            model.eval()
            
            # Process the input
            with torch.no_grad():
                output = model(csv_input)
            
            # Check that we got valid output
            print(f"Output shape: {output.shape}")
            self.assertEqual(output.shape[0], batch_size, "Output batch size should match input")
            self.assertEqual(output.shape[1], 2**config['num_qubits'], "Output feature dimension should be 2^num_qubits")
            
            print("✅ Test passed! CSV data was successfully processed by CNN backbone")
            return True
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Processing CSV data with CNN backbone failed: {e}")
            return False

if __name__ == "__main__":
    unittest.main() 