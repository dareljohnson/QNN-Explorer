"""
Unit tests for the ComplexModuleWrapper class.
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
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestComplexModuleWrapper(unittest.TestCase):
    """Test case for the ComplexModuleWrapper class."""
    
    def setUp(self):
        """Set up test resources."""
        try:
            from core.fusion import ComplexModuleWrapper
            
            # Create a simple linear layer
            self.linear_layer = nn.Linear(10, 5)
            
            # Create a wrapper around it
            self.wrapped_layer = ComplexModuleWrapper(self.linear_layer)
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def test_real_input_passthrough(self):
        """Test that real inputs pass through the wrapper unchanged."""
        # Create a real input tensor
        real_input = torch.rand(3, 10)
        
        # Process through the wrapped layer
        with patch.object(self.linear_layer, 'forward', wraps=self.linear_layer.forward) as mock_forward:
            output = self.wrapped_layer(real_input)
            
            # Verify that the wrapped module's forward was called with the same input
            mock_forward.assert_called_once()
            self.assertTrue(torch.all(mock_forward.call_args[0][0].eq(real_input)))
        
        # Verify output shape
        self.assertEqual(output.shape, (3, 5))
        
        # Verify output is real
        self.assertFalse(torch.is_complex(output))
    
    def test_complex_input_conversion(self):
        """Test that complex inputs are converted to real before passing to the wrapped module."""
        # Create a complex input tensor
        complex_input = torch.complex(
            torch.rand(3, 10),
            torch.rand(3, 10)
        )
        
        # Expected real input after conversion
        expected_real = complex_input.abs()
        
        # Process through the wrapped layer
        with patch.object(self.linear_layer, 'forward', wraps=self.linear_layer.forward) as mock_forward:
            output = self.wrapped_layer(complex_input)
            
            # Verify that the wrapped module's forward was called with the real version of the input
            mock_forward.assert_called_once()
            input_to_module = mock_forward.call_args[0][0]
            self.assertFalse(torch.is_complex(input_to_module))
            self.assertTrue(torch.allclose(input_to_module, expected_real))
        
        # Verify output shape
        self.assertEqual(output.shape, (3, 5))
        
        # Verify output is real
        self.assertFalse(torch.is_complex(output))


if __name__ == '__main__':
    unittest.main() 