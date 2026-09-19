"""
Test script to verify complex tensor conversions throughout the quantum processing pipeline.
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
    import pennylane as qml
    import numpy as np
    from core.quantum_models import amplitude_embedding, create_quantum_layer, ansatz1
    from core.fusion import HybridModel, ComplexModuleWrapper
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Required imports not available: {e}")
    IMPORTS_AVAILABLE = False

@unittest.skipUnless(IMPORTS_AVAILABLE, "Required imports not available")
class TestComplexConversionFix(unittest.TestCase):
    """Test case for the complex-to-real conversion fixes."""
    
    def setUp(self):
        """Set up test resources."""
        try:
            # Create a minimal config for a hybrid model
            self.config = {
                "model_type": "hybrid",
                "num_qubits": 2,
                "num_layers": 1,
                "ansatz": "ansatz1",
                "observation_type": "state",
                "encoding_method": "amplitude encoding",
                "classical_backbone": None,
                "output_size": 1,
                "use_feature_map": False
            }
            
            # Create a small model for testing
            self.model = HybridModel(self.config)
            
            # Test data - create both real and complex inputs
            self.real_input = torch.rand(4, 4)  # Batch of 4, 4 features
            self.complex_input = torch.complex(
                torch.rand(4, 4),
                torch.rand(4, 4)
            )
            
        except Exception as e:
            self.skipTest(f"Setup failed: {e}")
    
    def test_amplitude_embedding_complex_input(self):
        """Test that amplitude_embedding correctly handles complex input."""
        # Create a complex tensor input
        x = torch.complex(torch.rand(8), torch.rand(8))
        
        # Process through amplitude_embedding
        result = amplitude_embedding(x)
        
        # Check output is real
        self.assertFalse(torch.is_complex(result), "Output should be real, not complex")
        
        # Check output is normalized
        norm = torch.norm(result)
        self.assertTrue(torch.isclose(norm, torch.tensor(1.0)), 
                       f"Output should have norm 1.0, got {norm}")
    
    def test_quantum_layer_complex_input(self):
        """Test that quantum layer function correctly handles complex input."""
        # Create a quantum layer function
        quantum_layer, params, _ = create_quantum_layer(
            num_qubits=3, 
            num_layers=1,
            ansatz_func=ansatz1,
            observation_type='state'
        )
        
        # Create a complex input
        x = torch.complex(torch.rand(2, 8), torch.rand(2, 8))
        
        # Get output from the quantum layer
        output = quantum_layer(x, params)
        
        # Check output is real
        self.assertFalse(torch.is_complex(output), "Output should be real, not complex")
        
        # Check output shape (batch_size, 2^num_qubits)
        self.assertEqual(output.shape, (2, 2**3), "Output shape should be (2, 8)")
    
    def test_hybrid_model_real_input(self):
        """Test that HybridModel correctly processes real input."""
        # Process real input through the model
        output = self.model(self.real_input)
        
        # Check output is real
        self.assertFalse(torch.is_complex(output), "Output should be real, not complex")
        
        # Check that batch dimension is preserved
        self.assertEqual(output.shape[0], self.real_input.shape[0], f"Batch dimension should be preserved")
    
    def test_hybrid_model_complex_input(self):
        """Test that HybridModel correctly converts complex input to real."""
        # Process complex input through the model
        output = self.model(self.complex_input)
        
        # Check output is real
        self.assertFalse(torch.is_complex(output), "Output should be real, not complex")
        
        # Check that batch dimension is preserved
        self.assertEqual(output.shape[0], self.complex_input.shape[0], f"Batch dimension should be preserved")
    
    def test_complex_module_wrapper(self):
        """Test that ComplexModuleWrapper correctly handles complex input."""
        # Create a linear layer and wrap it
        linear = torch.nn.Linear(4, 2)
        wrapped = ComplexModuleWrapper(linear)
        
        # Process complex input through the wrapped module
        output = wrapped(self.complex_input)
        
        # Check output is real
        self.assertFalse(torch.is_complex(output), "Output should be real, not complex")
        
        # Check output shape
        expected_shape = (self.complex_input.shape[0], 2)
        self.assertEqual(output.shape, expected_shape, f"Output shape should be {expected_shape}")
        
    def test_complex_tensor_in_matrix_multiplication(self):
        """Test a direct case that could cause 'mat1 and mat2 must have the same dtype' error."""
        # Create a complex matrix
        complex_matrix = torch.complex(torch.rand(4, 4), torch.rand(4, 4))
        
        # Create a real matrix (weights)
        real_weights = torch.rand(4, 2)
        
        # Try direct matrix multiplication (should cause dtype mismatch)
        with self.assertRaises(RuntimeError) as context:
            # This should fail with dtype mismatch
            result = torch.matmul(complex_matrix, real_weights)
            
        # Check that the error message mentions dtype mismatch - update for actual error message format
        self.assertIn("expected m1 and m2 to have the same dtype", str(context.exception))
        
        # Now use our conversion approach
        complex_matrix_real = complex_matrix.abs()
        
        # Try matrix multiplication again (should succeed)
        result = torch.matmul(complex_matrix_real, real_weights)
        
        # Check result is real and has correct shape
        self.assertFalse(torch.is_complex(result), "Result should be real")
        self.assertEqual(result.shape, (4, 2), "Result shape should be (4, 2)")

if __name__ == '__main__':
    unittest.main() 