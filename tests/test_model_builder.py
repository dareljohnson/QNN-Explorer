"""
Test script for the model_builder module.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if requirements are available
try:
    import torch
    import torch.nn as nn
    from core.model_builder import build_model
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Required imports not available: {e}")
    IMPORTS_AVAILABLE = False

@unittest.skipUnless(IMPORTS_AVAILABLE, "Required imports not available")
class TestModelBuilder(unittest.TestCase):
    """Test case for the model_builder module."""
    
    def test_build_hybrid_model(self):
        """Test building a hybrid model."""
        # Create a minimal config for a hybrid model
        config = {
            "model_type": "hybrid",
            "num_qubits": 2,
            "num_layers": 1,
            "ansatz": "ansatz1",
            "observation_type": "state",
            "encoding_method": "amplitude encoding",
            "output_size": 1
        }
        
        # Build the model
        with patch('core.model_builder.HybridModel') as mock_hybrid_model:
            # Mock the HybridModel class to avoid actual instantiation
            mock_instance = MagicMock()
            mock_hybrid_model.return_value = mock_instance
            
            # Call build_model
            model, updated_config = build_model(config)
            
            # Check that HybridModel was called with the config
            mock_hybrid_model.assert_called_once_with(config)
            
            # Check that the config was updated with the device
            self.assertIn('device', updated_config)
    
    def test_build_classical_model_mlp(self):
        """Test building a classical MLP model."""
        # Create a config for a classical MLP model
        config = {
            "model_type": "classical",
            "classical_backbone_type": "mlp",
            "input_size": 10,
            "output_size": 2,
            "hidden_layers": [64, 32]
        }
        
        # Build the model
        with patch('core.model_builder.get_regression_model') as mock_get_model:
            # Mock the get_regression_model function
            mock_instance = MagicMock()
            mock_get_model.return_value = mock_instance
            
            # Call build_model
            model, updated_config = build_model(config)
            
            # Check that get_regression_model was called with the right args
            mock_get_model.assert_called_once_with(
                input_size=10,
                output_size=2,
                hidden_layers=[64, 32],
                dropout=0.1
            )
            
            # Check that the config was updated with the device
            self.assertIn('device', updated_config)
    
    def test_unsupported_model_type(self):
        """Test error handling for unsupported model types."""
        # Create a config with an unsupported model type
        config = {
            "model_type": "unsupported_type"
        }
        
        # Check that it raises a ValueError
        with self.assertRaises(ValueError) as context:
            model, _ = build_model(config)
        
        # Check the error message
        self.assertIn("Unsupported model type", str(context.exception))
    
    def test_unsupported_classical_type(self):
        """Test error handling for unsupported classical model types."""
        # Create a config with an unsupported classical model type
        config = {
            "model_type": "classical",
            "classical_backbone_type": "unsupported_type"
        }
        
        # Check that it raises a ValueError
        with self.assertRaises(ValueError) as context:
            model, _ = build_model(config)
        
        # Check the error message
        self.assertIn("Unsupported classical model type", str(context.exception))

if __name__ == '__main__':
    unittest.main() 