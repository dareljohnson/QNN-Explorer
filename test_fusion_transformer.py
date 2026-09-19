"""
Test the transformer_pooling method to ensure it properly handles transformer outputs.
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

class TestTransformerPooling(unittest.TestCase):
    """Test the transformer_pooling method in HybridModel."""
    
    def setUp(self):
        """Set up common test resources."""
        # Create a basic model for testing
        self.config = {
            'num_qubits': 4,        # 2^4 = 16
            'num_layers': 2,
            'ansatz_func': ansatz1,
            'observation_type': 'State Vector', 
            'classical_backbone_type': 'None',  # Direct quantum input for simplicity
            'use_gpu': False,
            'quantum_input_size': 16, # 2^4 = 16
            'num_classes': 2,
        }
        
        # Create the model
        self.model = HybridModel(self.config)
        
    def test_transformer_pooling(self):
        """Test that transformer_pooling properly extracts fixed-size representations."""
        # Create a mock transformer output tensor
        # Shape: [batch_size, sequence_length, hidden_size]
        batch_size = 3
        seq_length = 10
        hidden_size = 768
        
        mock_transformer_output = torch.rand(batch_size, seq_length, hidden_size)
        
        # Call transformer_pooling
        pooled_output = self.model.transformer_pooling(mock_transformer_output)
        
        # Check that the output has the expected shape [batch_size, hidden_size]
        self.assertEqual(pooled_output.shape, (batch_size, hidden_size))
        
        # Check that it correctly extracts the CLS token (first token)
        expected_cls_tokens = mock_transformer_output[:, 0]
        self.assertTrue(torch.all(torch.eq(pooled_output, expected_cls_tokens)))
        
        print("✅ Test passed: transformer_pooling correctly extracts CLS token!")
        
    def test_forward_with_transformer(self):
        """Test forward pass with simulated transformer output."""
        # Create a model with transformer backbone type
        config = self.config.copy()
        config['classical_backbone_type'] = 'Transformer'
        config['classical_model_name'] = 'bert-base-uncased'  # Just for config completeness
        
        # We'll mock the transformer backbone behavior rather than actually loading it
        model = HybridModel(config)
        
        # Mock the classical_backbone to return a tensor of the right shape
        class MockTransformer(nn.Module):
            def forward(self, **kwargs):
                # Return a tensor with shape [batch_size, seq_len, hidden_size]
                batch_size = kwargs['input_ids'].shape[0]
                return torch.rand(batch_size, 10, 768)
        
        model.classical_backbone = MockTransformer()
        
        # Create dummy input for transformer (typically a dict with input_ids and attention_mask)
        dummy_input = {
            'input_ids': torch.ones(2, 10, dtype=torch.long),  # [batch_size, seq_len]
            'attention_mask': torch.ones(2, 10, dtype=torch.long)  # [batch_size, seq_len]
        }
        
        try:
            output = model(dummy_input)
            print(f"✅ Test passed: Forward pass with transformer input works! Output shape: {output.shape}")
        except Exception as e:
            self.fail(f"Forward pass with transformer input failed: {e}")
        
    def test_all(self):
        """Run all tests."""
        print("\nTesting transformer_pooling method...")
        self.test_transformer_pooling()
        self.test_forward_with_transformer()
        print("\n🎉 All transformer_pooling tests passed!")

if __name__ == "__main__":
    tester = TestTransformerPooling()
    tester.setUp()
    tester.test_all() 