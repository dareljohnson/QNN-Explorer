"""
Unit tests for complex-to-real conversion in the HybridModel.
"""

import unittest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use try-except to handle torch/qiskit import errors
try:
    import torch
    import pennylane as qml
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required imports not available")
class TestComplexConversion(unittest.TestCase):
    """Test case for the complex-to-real conversion in HybridModel."""
    
    def setUp(self):
        """Set up test resources."""
        try:
            from core.fusion import HybridModel
            from core.quantum_models import ansatz1
            
            # Create a mock quantum circuit function that returns complex values
            def mock_complex_circuit(x, params):
                # Create a complex tensor with same batch size as input
                batch_size = x.shape[0]
                # 4 is 2^2 qubits for a state vector output
                return torch.complex(
                    torch.rand(batch_size, 4),
                    torch.rand(batch_size, 4)
                )
            
            # Create a configuration dictionary for the model
            self.config = {
                'num_qubits': 2,
                'num_layers': 2,
                'classical_backbone_type': 'None',
                'encoding_method': 'Amplitude Encoding',
                'observation_type': 'State Vector',
                'ansatz_func': ansatz1
            }
            
            # Create the model and patch the quantum circuit function
            self.model = HybridModel(self.config)
            # Replace the real quantum circuit with our mock
            self.model.quantum_circuit_fn = mock_complex_circuit
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def test_complex_conversion_single_sample(self):
        """Test that complex output is correctly converted to real for a single sample."""
        # Create input tensor (single sample)
        input_tensor = torch.rand(1, 4)
        
        # Process through the model
        output = self.model(input_tensor)
        
        # Verify output is real (not complex)
        self.assertFalse(torch.is_complex(output), "Output should be real, not complex")
        
        # Verify output shape is correct
        self.assertEqual(output.shape, (1, 4), "Output shape should match input shape")
    
    def test_complex_conversion_batch(self):
        """Test that complex output is correctly converted to real for a batch."""
        # Create input tensor (batch of 5)
        input_tensor = torch.rand(5, 4)
        
        # Process through the model
        output = self.model(input_tensor)
        
        # Verify output is real (not complex)
        self.assertFalse(torch.is_complex(output), "Output should be real, not complex")
        
        # Verify output shape is correct
        self.assertEqual(output.shape, (5, 4), "Output shape should match input shape")


if __name__ == '__main__':
    unittest.main() 