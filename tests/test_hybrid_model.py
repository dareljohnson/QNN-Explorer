"""
Unit tests for the HybridModel class.
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use try-except to handle torch import error
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create a mock torch module for testing
    torch = MagicMock()
    torch.rand = lambda *args: MagicMock()
    torch.ones = lambda *args, **kwargs: MagicMock()
    torch.Tensor = MagicMock

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestHybridModel(unittest.TestCase):
    """Test case for the HybridModel class in the fusion module."""
    
    def setUp(self):
        """Set up test resources."""
        try:
            from core.fusion import HybridModel
            
            # Mock dependencies to avoid actually running quantum circuits
            self.mock_quantum_circuit = MagicMock()
            self.mock_quantum_circuit.return_value = torch.rand(2, 4)  # Mock output
            
            # Import ansatz1 from quantum_models
            from core.quantum_models import ansatz1
            
            # Create a configuration dictionary for the model
            self.config = {
                'num_qubits': 2,
                'num_layers': 2,
                'classical_backbone_type': 'None',
                'encoding_method': 'Amplitude Encoding',
                'observation_type': 'State Vector',
                'ansatz_func': ansatz1  # Use a real ansatz function
            }
            
            # Create a minimal HybridModel for testing
            self.model = HybridModel(self.config)
        except ImportError:
            self.skipTest("Required modules not available")
    
    def test_get_batch_size_tensor(self):
        """Test get_batch_size with tensor input."""
        # Test with a tensor with batch size of 5
        tensor_input = torch.rand(5, 10)  # Shape already set by creation
        batch_size = self.model.get_batch_size(tensor_input)
        self.assertEqual(batch_size, 5)
        
        # Test with a single sample (keeping batch dimension)
        single_input = torch.rand(1, 10)  # Shape already set by creation
        batch_size = self.model.get_batch_size(single_input)
        self.assertEqual(batch_size, 1)
    
    def test_get_batch_size_dict(self):
        """Test get_batch_size with dictionary input (transformer case)."""
        # Test with a dictionary containing input_ids (transformer case)
        dict_input = {
            'input_ids': torch.ones(3, 128, dtype=torch.long),  # Shape already set by creation
            'attention_mask': torch.ones(3, 128, dtype=torch.long)
        }
        batch_size = self.model.get_batch_size(dict_input)
        self.assertEqual(batch_size, 3)
        
        # Test with a dictionary that doesn't have input_ids
        invalid_dict = {'other_key': torch.ones(4, 10)}  # Shape already set by creation
        batch_size = self.model.get_batch_size(invalid_dict)
        # Should try len() on the dict which will be 1
        self.assertEqual(batch_size, 1)
    
    def test_get_batch_size_list(self):
        """Test get_batch_size with list input."""
        # Test with a list
        list_input = [1, 2, 3, 4]  # Length of 4
        batch_size = self.model.get_batch_size(list_input)
        self.assertEqual(batch_size, 4)
        
        # Test with an empty list
        empty_list = []
        batch_size = self.model.get_batch_size(empty_list)
        self.assertEqual(batch_size, 0)
    
    def test_get_batch_size_unsupported(self):
        """Test get_batch_size with unsupported input type."""
        # Test with input that doesn't support len()
        unsupported_input = 42  # Integer has no len()
        batch_size = self.model.get_batch_size(unsupported_input)
        self.assertEqual(batch_size, 1)  # Default to 1

if __name__ == '__main__':
    unittest.main() 