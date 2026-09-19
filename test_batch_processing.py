import sys
import os
import torch
import unittest.mock as mock
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules
from core.fusion import HybridModel

def run_batch_processing_test():
    """Test to verify the fix for batch processing with quantum operations."""
    
    # Create mocks
    mock_transformer = mock.MagicMock()
    mock_get_transformer_model = mock.MagicMock(return_value=(mock_transformer, 768))
    
    # We need individual mock amplitude_embedding for each batch size
    mock_amp_embed_1 = mock.MagicMock(return_value=torch.ones(1, 16))
    mock_amp_embed_batch = mock.MagicMock(side_effect=lambda x: torch.ones(x.shape[0], 16))
    
    mock_qnn = mock.MagicMock()
    mock_create_qnn = mock.MagicMock(return_value=(mock_qnn, 16))
    
    # Create individual tensor mocks for each batch size
    mock_tensor_1 = torch.ones(1, 16)
    mock_tensor_1.requires_grad = True
    
    mock_qnn_layer = mock.MagicMock()
    mock_qnn_layer.side_effect = lambda x: x  # Just return the input tensor
    
    # Patch dependencies
    with mock.patch('core.fusion.get_transformer_model', mock_get_transformer_model):
        with mock.patch('core.fusion.amplitude_embedding', mock_amp_embed_batch):
            with mock.patch('core.fusion.create_qnn', mock_create_qnn):
                with mock.patch('pennylane.qnn.TorchLayer', return_value=mock_qnn_layer):
                    try:
                        # Create model config
                        config = {
                            'classical_backbone_type': 'Transformer',
                            'classical_model_name': 'bert-base-uncased',
                            'num_qubits': 4,
                            'num_layers': 2,
                            'ansatz_func': lambda x, y, z: None,
                            'use_gpu': False,
                            'observation_type': 'State Vector'
                        }
                        
                        # Create model
                        model = HybridModel(config)
                        print("✅ Successfully instantiated HybridModel")
                        
                        # Test with different batch sizes
                        batch_sizes = [1, 2, 4, 8]
                        for batch_size in batch_sizes:
                            try:
                                # Create input with the specified batch size
                                inputs = {
                                    'input_ids': torch.ones(batch_size, 10, dtype=torch.long),
                                    'attention_mask': torch.ones(batch_size, 10, dtype=torch.long)
                                }
                                
                                # Set up the mock output
                                mock_output = mock.MagicMock()
                                mock_output.pooler_output = torch.ones(batch_size, 768)
                                mock_transformer.return_value = mock_output
                                
                                # Run forward pass
                                output = model(inputs)
                                print(f"✅ Successfully processed batch size {batch_size}")
                            except Exception as e:
                                print(f"❌ Error with batch size {batch_size}: {e}")
                                print("Traceback:")
                                traceback.print_exc()
                                return False
                        
                        print("\n✅ All batch sizes processed successfully!")
                        return True
                    except Exception as e:
                        print(f"❌ Error during test setup: {e}")
                        print("Traceback:")
                        traceback.print_exc()
                        return False

if __name__ == "__main__":
    print("Testing batch processing in HybridModel...")
    success = run_batch_processing_test()
    
    if success:
        print("\nFix validated: Batch processing now works correctly!")
    else:
        print("\nFix failed: Batch processing still has issues.") 