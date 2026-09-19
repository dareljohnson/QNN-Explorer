import sys
import os
import torch
import pytest
import unittest.mock as mock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from core.fusion import HybridModel
from core.quantum_models import amplitude_embedding

class MockTransformerOutput:
    """Mock for transformer output with pooler_output and last_hidden_state"""
    def __init__(self, batch_size=1, hidden_size=768):
        self.pooler_output = torch.randn(batch_size, hidden_size)
        self.last_hidden_state = torch.randn(batch_size, 10, hidden_size)  # [batch, seq_len, hidden]

def test_transformer_input_handling():
    """Test that the HybridModel correctly handles transformer input (BatchEncoding objects)"""
    print("\nTesting transformer input handling...")
    
    # Create mock objects
    mock_transformer = mock.MagicMock()
    
    # For batch processing test
    def mock_transformer_side_effect(**kwargs):
        # Extract the batch size from input_ids
        batch_size = kwargs['input_ids'].shape[0]
        return MockTransformerOutput(batch_size=batch_size)
    
    mock_transformer.side_effect = mock_transformer_side_effect
    
    mock_qnn = mock.MagicMock()
    mock_qnn.return_value = torch.ones(16)  # Simulated quantum output
    
    mock_torch_layer = mock.MagicMock()
    mock_torch_layer.return_value = torch.ones(16)
    
    # Create config for hybrid model
    config = {
        'classical_backbone_type': 'transformer',
        'classical_model_name': 'bert-base-uncased',
        'num_qubits': 4,  # 2^4 = 16 quantum input size
        'num_layers': 1,
        'use_gpu': False,
        'observation_type': 'state'
    }
    
    # Create a model with mocked components
    with mock.patch('core.fusion.get_transformer_model', return_value=(mock_transformer, 768)), \
         mock.patch('core.fusion.create_qnn', return_value=(mock_qnn, 8)), \
         mock.patch('pennylane.qnn.TorchLayer', return_value=mock_torch_layer):
        
        model = HybridModel(config)
        
        # Test Case 1: Single sample
        print("\n🔍 Test Case 1: Single sample BatchEncoding")
        batch_encoding_single = {
            'input_ids': torch.ones(1, 10, dtype=torch.long),  # [batch=1, seq_len]
            'attention_mask': torch.ones(1, 10, dtype=torch.long)  # [batch=1, seq_len]
        }
        
        # Test the forward pass with single sample
        try:
            output_single = model(batch_encoding_single)
            print(f"✅ Model processed single sample, output shape: {output_single.shape}")
            assert output_single is not None, "Output should not be None"
            
            # Test Case 2: Batch processing
            print("\n🔍 Test Case 2: Batch processing with BatchEncoding")
            batch_encoding_multiple = {
                'input_ids': torch.ones(3, 10, dtype=torch.long),  # [batch=3, seq_len]
                'attention_mask': torch.ones(3, 10, dtype=torch.long)  # [batch=3, seq_len]
            }
            
            # Test the forward pass with batch
            output_batch = model(batch_encoding_multiple)
            print(f"✅ Model processed batch of 3 samples, output shape: {output_batch.shape}")
            assert output_batch.shape[0] == 3, "Output should have batch dimension of 3"
            
            return True
        except Exception as e:
            print(f"❌ Error processing BatchEncoding: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    # Run the test
    success = test_transformer_input_handling()
    assert success, "Test failed"
    print("All tests passed!") 