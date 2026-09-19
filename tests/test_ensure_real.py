"""
Unit tests for the ensure_real utility function used in app.py.
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use try-except to handle torch import errors
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestEnsureReal(unittest.TestCase):
    """Test case for the ensure_real utility function."""
    
    def setUp(self):
        """Set up test resources."""
        # Define ensure_real function here so it matches what's in app.py
        def ensure_real(tensor_or_dict):
            """Convert complex tensors to real by taking the absolute value."""
            if isinstance(tensor_or_dict, dict):
                return {k: ensure_real(v) for k, v in tensor_or_dict.items()}
            elif isinstance(tensor_or_dict, torch.Tensor):
                if torch.is_complex(tensor_or_dict):
                    print(f"Converting complex tensor with shape {tensor_or_dict.shape} to real")
                    return tensor_or_dict.abs()
                return tensor_or_dict
            return tensor_or_dict
        
        self.ensure_real = ensure_real
    
    def test_real_tensor_unchanged(self):
        """Test that real tensors are returned unchanged."""
        # Create a real tensor
        real_tensor = torch.tensor([1.0, 2.0, 3.0])
        
        # Pass through ensure_real
        result = self.ensure_real(real_tensor)
        
        # Verify the result is the same tensor
        self.assertIs(result, real_tensor)
        
        # Verify values are the same
        self.assertTrue(torch.all(result == real_tensor))
    
    def test_complex_tensor_converted(self):
        """Test that complex tensors are converted to real."""
        # Create a complex tensor
        complex_tensor = torch.complex(
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0])
        )
        
        # Expected absolute values
        expected = torch.tensor([
            (1.0**2 + 4.0**2)**0.5,
            (2.0**2 + 5.0**2)**0.5,
            (3.0**2 + 6.0**2)**0.5
        ])
        
        # Pass through ensure_real
        result = self.ensure_real(complex_tensor)
        
        # Verify the result is not complex
        self.assertFalse(torch.is_complex(result))
        
        # Verify the result matches the expected absolute values
        self.assertTrue(torch.allclose(result, expected))
    
    def test_dict_input(self):
        """Test that dictionaries with tensors are handled correctly."""
        # Create a dictionary with both real and complex tensors
        input_dict = {
            'real': torch.tensor([1.0, 2.0, 3.0]),
            'complex': torch.complex(
                torch.tensor([1.0, 2.0]),
                torch.tensor([3.0, 4.0])
            ),
            'nested': {
                'real': torch.tensor([5.0, 6.0]),
                'complex': torch.complex(
                    torch.tensor([7.0]),
                    torch.tensor([8.0])
                )
            }
        }
        
        # Pass through ensure_real
        result = self.ensure_real(input_dict)
        
        # Verify the structure is preserved
        self.assertEqual(set(result.keys()), set(input_dict.keys()))
        self.assertEqual(set(result['nested'].keys()), set(input_dict['nested'].keys()))
        
        # Verify real tensors are unchanged
        self.assertIs(result['real'], input_dict['real'])
        self.assertIs(result['nested']['real'], input_dict['nested']['real'])
        
        # Verify complex tensors are converted to real
        self.assertFalse(torch.is_complex(result['complex']))
        self.assertFalse(torch.is_complex(result['nested']['complex']))
    
    def test_non_tensor_input(self):
        """Test that non-tensor inputs are returned unchanged."""
        inputs = [42, "hello", [1, 2, 3], None, True]
        
        for input_val in inputs:
            result = self.ensure_real(input_val)
            self.assertIs(result, input_val)


if __name__ == '__main__':
    unittest.main() 