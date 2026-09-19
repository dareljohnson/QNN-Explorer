"""
Test script to verify we handle dtype consistency properly.
"""

import sys
import os
import unittest

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if requirements are available
try:
    import torch
    from utils.tensor_utils import ensure_real
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Required imports not available: {e}")
    IMPORTS_AVAILABLE = False

@unittest.skipUnless(IMPORTS_AVAILABLE, "Required imports not available")
class TestDtypeConsistency(unittest.TestCase):
    """Test case for handling dtype consistency."""
    
    def test_double_to_float_conversion(self):
        """Test conversion from double to float."""
        # Create a double precision tensor
        double_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        
        # Convert to float32
        result = ensure_real(double_tensor, target_dtype=torch.float32)
        
        # Check dtype
        self.assertEqual(result.dtype, torch.float32, "Result should be float32")
        
        # Check values
        self.assertTrue(torch.allclose(
            result, 
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        ))
    
    def test_matrix_multiplication_dtypes(self):
        """Test matrix multiplication with different dtypes."""
        # Create tensors with different dtypes
        double_matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        float_matrix = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
        
        # Without ensure_real, this would raise a dtype mismatch error
        with self.assertRaises(RuntimeError) as context:
            # This should fail with dtype mismatch
            result = torch.matmul(double_matrix, float_matrix)
        
        # Check error message mentions dtype (using a more generic check)
        error_msg = str(context.exception)
        self.assertTrue(
            "dtype" in error_msg.lower() and "double" in error_msg and "float" in error_msg,
            f"Error message should mention dtype mismatch, got: {error_msg}"
        )
        
        # Use ensure_real to make dtypes consistent
        consistent_double_matrix = ensure_real(double_matrix, target_dtype=torch.float32)
        
        # This should work now
        result = torch.matmul(consistent_double_matrix, float_matrix)
        
        # Check result dtype and shape
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.shape, (2, 2))
        
        # Check result values
        expected = torch.tensor([
            [1.0 * 5.0 + 2.0 * 7.0, 1.0 * 6.0 + 2.0 * 8.0],
            [3.0 * 5.0 + 4.0 * 7.0, 3.0 * 6.0 + 4.0 * 8.0]
        ], dtype=torch.float32)
        self.assertTrue(torch.allclose(result, expected))
    
    def test_loss_function_dtype_consistency(self):
        """Test loss function with different dtypes."""
        # Create prediction and target with different dtypes
        predictions = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
        targets = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        
        # Create MSE loss
        mse_loss = torch.nn.MSELoss()
        
        # Try direct computation (might work with automatic conversion in newer PyTorch)
        try:
            # In some PyTorch versions, this might work with automatic dtype conversion
            direct_loss = mse_loss(predictions, targets)
            print(f"PyTorch automatically handled dtype conversion for loss function")
        except RuntimeError as e:
            # If it fails, check the error message
            self.assertIn("expected scalar type", str(e))
        
        # Use ensure_real to make dtypes consistent (should always work)
        consistent_predictions = ensure_real(predictions, target_dtype=torch.float32)
        
        # This should work now
        loss = mse_loss(consistent_predictions, targets)
        
        # Check loss is a scalar and has expected dtype
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(loss.numel(), 1, "Loss should be a scalar")
        
        # Check loss value
        expected_loss = ((0.1 - 0.0)**2 + (0.2 - 0.0)**2 + (0.3 - 1.0)**2) / 3.0
        self.assertTrue(torch.allclose(loss, torch.tensor(expected_loss, dtype=torch.float32)))

if __name__ == '__main__':
    unittest.main() 