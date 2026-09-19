"""
Unit tests for complex value handling in the fusion layer.
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
class TestComplexFusion(unittest.TestCase):
    """Test case for complex value handling in the fusion layer."""
    
    def setUp(self):
        """Set up test resources."""
        try:
            from core.fusion import HybridModel
            import torch.nn as nn
            
            # Create a mock quantum circuit function that returns complex values
            def mock_complex_circuit(x, params):
                # Create a complex tensor with same batch size as input
                batch_size = x.shape[0]
                # 4 is 2^2 qubits for a state vector output
                return torch.complex(
                    torch.rand(batch_size, 4),
                    torch.rand(batch_size, 4)
                )
            
            # Create a mock complex classical backbone
            class ComplexBackbone(nn.Module):
                def forward(self, x):
                    # Return complex values
                    batch_size = x.shape[0]
                    return torch.complex(
                        torch.rand(batch_size, 8),
                        torch.rand(batch_size, 8)
                    )
            
            # Create a configuration dictionary for the model
            self.config = {
                'num_qubits': 2,
                'num_layers': 2,
                'classical_backbone_type': 'None',
                'encoding_method': 'Amplitude Encoding',
                'observation_type': 'State Vector',
                'ansatz_func': None  # Will be replaced with mock
            }
            
            # Create configs for both models
            self.direct_config = self.config.copy()
            self.backbone_config = self.config.copy()
            self.backbone_config['classical_backbone_type'] = 'CNN'  # Arbitrary type
            self.backbone_config['classical_model_name'] = 'resnet18'  # Required for CNN backbone
            
            # Create models
            self.direct_model = HybridModel(self.direct_config)
            self.backbone_model = HybridModel(self.backbone_config)
            
            # Replace quantum circuits with mocks
            self.direct_model.quantum_circuit_fn = mock_complex_circuit
            self.backbone_model.quantum_circuit_fn = mock_complex_circuit
            
            # Replace classical backbone with mock
            self.backbone_model.classical_backbone = ComplexBackbone()
            
            # Create fusion layer
            self.backbone_model.fusion_layer = nn.Linear(8, 4)
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def test_complex_input_handling(self):
        """Test that complex input tensor is properly converted to real."""
        # Create complex input tensor
        input_tensor = torch.complex(
            torch.rand(3, 4),
            torch.rand(3, 4)
        )
        
        # Process through the model
        output = self.direct_model(input_tensor)
        
        # Verify output is real (not complex)
        self.assertFalse(torch.is_complex(output), "Output should be real, not complex")
        
        # Verify output shape is correct
        self.assertEqual(output.shape, (3, 4), "Output shape should match expected shape")
    
    def test_complex_backbone_handling(self):
        """Test that complex output from classical backbone is properly converted."""
        # Create real input tensor
        input_tensor = torch.rand(3, 10)  # Shape doesn't matter as our mock backbone ignores it
        
        # Process through the model with complex backbone
        output = self.backbone_model(input_tensor)
        
        # Verify output is real (not complex)
        self.assertFalse(torch.is_complex(output), "Output should be real, not complex")
        
        # Verify output shape is correct (should be 4 for our 2-qubit system)
        self.assertEqual(output.shape, (3, 4), "Output shape should match expected shape")


if __name__ == '__main__':
    unittest.main() 