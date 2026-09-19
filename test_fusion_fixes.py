"""
Test the fixes for dimension mismatch and data type issues in the fusion layer.
"""
import os
import sys
import torch
import torch.nn as nn
import unittest

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fusion import HybridModel
from core.quantum_models import ansatz1

class TestFusionFixes(unittest.TestCase):
    """Test the fixes for the fusion layer."""
    
    def setUp(self):
        """Set up common test resources."""
        # Create a minimal model configuration
        self.config = {
            'num_qubits': 5,        # 2^5 = 32
            'num_layers': 2,
            'encoding_method': 'Amplitude Encoding',
            'ansatz_name': 'Ansatz 1 (Rot+CNOT Chain)',
            'ansatz_func': ansatz1,
            'observation_type': 'State Vector', 
            'classical_backbone_type': 'None',  # Direct quantum input
            'use_gpu': False,
            'quantum_input_size': 32, # 2^5 = 32
            'num_classes': 2,        # Binary classification
        }
        
    def test_dimension_and_dtype_fixes(self):
        """Test that both dimension mismatch and data type issues are fixed."""
        model = HybridModel(self.config)
        model.eval()
        
        # Create input with different feature dimensions to trigger fusion layer creation
        feature_dim = 2  # This will mismatch with quantum_input_size=32
        test_input = torch.rand(64, feature_dim)
        
        # Forward pass should work without errors
        try:
            output = model(test_input)
            
            # Check output shape and dtype
            self.assertEqual(output.shape, (64, self.config['num_classes']))
            self.assertEqual(output.dtype, torch.float32)
            
            print("✅ Test passed: Both dimension mismatch and data type issues are fixed!")
            return True
        except Exception as e:
            self.fail(f"Forward pass failed with error: {e}")
            return False
        
    def test_training_with_fixes(self):
        """Test that training works with the fixes in place."""
        model = HybridModel(self.config)
        model.train()
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Create synthetic data
        feature_dim = 2  # Mismatched with quantum_input_size=32
        batch_size = 16  # Smaller batch for faster test
        
        # Generate fake data
        X = torch.rand(batch_size, feature_dim)
        y = torch.randint(0, 2, (batch_size,))
        
        # Run training for 2 iterations
        success = True
        try:
            for i in range(2):  # Just 2 iterations for testing
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(X)
                loss = loss_fn(outputs, y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                print(f"Iteration {i+1} - Loss: {loss.item():.4f}")
                
            print("✅ Test passed: Training works with both fixes in place!")
        except Exception as e:
            self.fail(f"Training failed with error: {e}")
            success = False
            
        return success
    
    def test_all_fixes(self):
        """Run all tests and summarize results."""
        dim_dtype_fixed = self.test_dimension_and_dtype_fixes()
        training_works = self.test_training_with_fixes()
        transformer_works = self.test_transformer_pooling()
        
        if dim_dtype_fixed and training_works and transformer_works:
            print("\n🎉 All tests passed! The fixes for dimension mismatch, data type issues, and transformer pooling are working correctly.")
        else:
            print("\n❌ Some tests failed. Please check the error messages above.")
            
    def test_transformer_pooling(self):
        """Test that the transformer_pooling method works correctly."""
        model = HybridModel(self.config)
        
        # Create a mock transformer output tensor (batch_size, seq_len, hidden_size)
        mock_output = torch.rand(3, 10, 768)
        
        try:
            # Call the transformer_pooling method
            pooled = model.transformer_pooling(mock_output)
            
            # Check shape and content
            self.assertEqual(pooled.shape, (3, 768))
            self.assertTrue(torch.all(torch.eq(pooled, mock_output[:, 0])))
            
            print("✅ Test passed: transformer_pooling method works correctly!")
            return True
        except Exception as e:
            self.fail(f"transformer_pooling test failed with error: {e}")
            return False

if __name__ == "__main__":
    tester = TestFusionFixes()
    tester.setUp()
    tester.test_all_fixes() 