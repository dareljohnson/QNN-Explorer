"""
Test the adaptive fusion layer fix for dimension mismatch.
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

class TestAdaptiveFusionFix(unittest.TestCase):
    """Test the adaptive fusion layer fix for dimension mismatch."""
    
    def test_adaptive_fusion_layer_creation(self):
        """Test that the fusion layer adapts to input dimensions."""
        # Create a minimal model configuration
        config = {
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
        
        # Create the model
        model = HybridModel(config)
        model.eval()
        
        # Initially the fusion layer should be an Identity if no classical backbone
        self.assertIsInstance(model.fusion_layer, nn.Identity)
        
        # Create input with different feature dimensions to trigger fusion layer creation
        feature_dim = 2  # This will mismatch with quantum_input_size=32
        test_input = torch.rand(64, feature_dim)
        
        # Forward pass should create a new Linear fusion layer
        output = model(test_input)
        
        # Check that the fusion layer is now a Linear layer with correct dimensions
        self.assertIsInstance(model.fusion_layer, nn.Linear)
        self.assertEqual(model.fusion_layer.in_features, feature_dim)
        self.assertEqual(model.fusion_layer.out_features, config['quantum_input_size'])
        
        # Check that the output has the expected shape: [batch_size, num_classes]
        self.assertEqual(output.shape, (64, config['num_classes']))
        
        print("✅ Test passed: Fusion layer correctly adapts to input dimensions!")
        
    def test_training_with_dimension_mismatch(self):
        """Test training works with feature dimensions mismatching quantum input size."""
        # Create a model configuration
        config = {
            'num_qubits': 5,        # 2^5 = 32
            'num_layers': 2,
            'encoding_method': 'Amplitude Encoding',
            'ansatz_name': 'Ansatz 1 (Rot+CNOT Chain)',
            'ansatz_func': ansatz1,
            'observation_type': 'State Vector', 
            'classical_backbone_type': 'None',
            'use_gpu': False,
            'quantum_input_size': 32,
            'num_classes': 2,
        }
        
        # Create the model
        model = HybridModel(config)
        model.train()
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Create synthetic data with mismatched feature dimension
        feature_dim = 2  # Mismatched with quantum_input_size=32
        batch_size = 64
        
        # Generate fake data
        X = torch.rand(batch_size, feature_dim)
        y = torch.randint(0, 2, (batch_size,))
        
        # Training iterations
        for i in range(2):  # Just 2 iterations for testing
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X)
            loss = loss_fn(outputs, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            print(f"Iteration {i+1} - Loss: {loss.item():.4f}")
        
        # Check that the model now has a Linear fusion layer
        self.assertIsInstance(model.fusion_layer, nn.Linear)
        self.assertEqual(model.fusion_layer.in_features, feature_dim)
        self.assertEqual(model.fusion_layer.out_features, config['quantum_input_size'])
        
        print("✅ Test passed: Successfully trained with mismatched dimensions!")

if __name__ == "__main__":
    unittest.main() 