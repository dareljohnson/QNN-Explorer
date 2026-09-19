"""Test that demonstrates the fix for the BatchEncoding issue in the prediction workflow."""
import sys
import os
import torch
import unittest.mock as mock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.metrics import calculate_meyer_wallach
from core.fusion import HybridModel

class MockTransformerOutput:
    """Mock for transformer output with pooler_output and last_hidden_state"""
    def __init__(self, batch_size=1, hidden_size=768):
        self.pooler_output = torch.randn(batch_size, hidden_size)
        self.last_hidden_state = torch.randn(batch_size, 10, hidden_size)

def process_batch_encoding():
    """Test function that simulates how the app processes BatchEncoding objects."""
    print("\nTesting BatchEncoding processing fix...")
    
    # Create a mock BatchEncoding object similar to transformers.BatchEncoding
    batch_encoding = {
        'input_ids': torch.ones(1, 10, dtype=torch.long),
        'attention_mask': torch.ones(1, 10, dtype=torch.long)
    }
    
    # Create mock objects for testing
    mock_transformer = mock.MagicMock()
    mock_transformer.return_value = MockTransformerOutput(batch_size=1)
    
    mock_qnn = mock.MagicMock()
    # Return a normalized state vector for a 4-qubit system
    state_vector = torch.randn(16)
    mock_qnn.return_value = state_vector / torch.norm(state_vector)
    
    mock_torch_layer = mock.MagicMock()
    mock_torch_layer.return_value = mock_qnn.return_value
    
    # Model config for a transformer model
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
        
        try:
            # Process the BatchEncoding through the model
            output = model(batch_encoding)
            print(f"✅ Model successfully processed BatchEncoding, output shape: {output.shape}")
            
            # Now try to calculate entanglement on the output
            # This was previously failing with "Input must be a state vector or density matrix"
            entanglement = calculate_meyer_wallach(output[0], 4)  # Use the first item from batch
            print(f"✅ Successfully calculated entanglement: {entanglement:.6f}")
            
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = process_batch_encoding()
    assert success, "BatchEncoding processing test failed"
    print("BatchEncoding processing test passed!") 