"""
Unit tests for universal complex value handling in the HybridModel.
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use try-except to handle torch/qiskit import errors
try:
    import torch
    import torch.nn as nn
    import pennylane as qml
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class ComplexOutputLayer(nn.Module):
    """Layer that deliberately returns complex values for testing."""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
    def forward(self, x):
        """Return a complex-valued tensor, regardless of input."""
        batch_size = x.shape[0]
        return torch.complex(
            torch.rand(batch_size, self.output_size),
            torch.rand(batch_size, self.output_size)
        )


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required imports not available")
class TestUniversalComplexFix(unittest.TestCase):
    """Test case for universal complex value handling in HybridModel."""
    
    def setUp(self):
        """Set up test resources."""
        try:
            from core.fusion import HybridModel
            from core.quantum_models import ansatz1
            
            # Base config
            self.config = {
                'num_qubits': 2,
                'num_layers': 2,
                'observation_type': 'State Vector',
                'encoding_method': 'Amplitude Encoding',
                'ansatz_func': ansatz1
            }
            
            # Mock function that returns complex values
            def complex_quantum_circuit(x, params):
                batch_size = x.shape[0]
                return torch.complex(
                    torch.rand(batch_size, 4),  # 2^num_qubits = 4
                    torch.rand(batch_size, 4)
                )
            
            # Three test scenarios:
            # 1. Direct quantum processing (no backbone)
            direct_config = self.config.copy()
            direct_config['classical_backbone_type'] = 'None'
            
            # 2. With classical backbone that returns complex values
            backbone_config = self.config.copy()
            backbone_config['classical_backbone_type'] = 'CNN'
            backbone_config['classical_model_name'] = 'resnet18'
            
            # 3. With complex quantum processing and output head
            output_config = self.config.copy()
            output_config['classical_backbone_type'] = 'None'
            output_config['num_classes'] = 2
            
            # Create models
            self.direct_model = HybridModel(direct_config)
            self.backbone_model = HybridModel(backbone_config)
            self.output_model = HybridModel(output_config)
            
            # Replace quantum circuits with complex output function
            self.direct_model.quantum_circuit_fn = complex_quantum_circuit
            self.backbone_model.quantum_circuit_fn = complex_quantum_circuit
            self.output_model.quantum_circuit_fn = complex_quantum_circuit
            
            # Create a complex backbone
            self.backbone_model.classical_backbone = ComplexOutputLayer(10, 512)  # Match ResNet18's output size of 512
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
    
    def test_complex_input_tensor(self):
        """Test handling of complex input tensor."""
        # Create complex input tensor
        input_tensor = torch.complex(
            torch.rand(3, 4),
            torch.rand(3, 4)
        )
        
        # Process through the direct model
        output = self.direct_model(input_tensor)
        
        # Verify output is real (not complex)
        self.assertFalse(torch.is_complex(output), "Output should be real, not complex")
    
    def test_complex_backbone_output(self):
        """Test handling of complex values from classical backbone."""
        # Input shape doesn't matter as our mock backbone ignores it
        input_tensor = torch.rand(3, 10)
        
        # Process through the model with complex backbone
        output = self.backbone_model(input_tensor)
        
        # Verify output is real (not complex)
        self.assertFalse(torch.is_complex(output), "Output should be real, not complex")
    
    def test_end_to_end_complex_handling(self):
        """Test end-to-end handling of complex values through the entire model."""
        # Create complex input
        input_tensor = torch.complex(
            torch.rand(5, 4),
            torch.rand(5, 4)
        )
        
        # Process through model with output head
        output = self.output_model(input_tensor)
        
        # Verify output is real (not complex)
        self.assertFalse(torch.is_complex(output), "Output should be real, not complex")
        
        # Verify output shape matches expected (batch_size, num_classes)
        self.assertEqual(output.shape, (5, 2), "Output shape should be (batch_size, num_classes)")


if __name__ == '__main__':
    unittest.main() 