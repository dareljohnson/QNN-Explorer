import sys
import os
import torch
import unittest.mock as mock
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules
from core.fusion import HybridModel
from core.quantum_models import amplitude_embedding

def test_amplitude_embedding():
    """Test that amplitude_embedding can handle different input shapes."""
    # Test single vector (no batch)
    input1 = torch.randn(16)
    output1 = amplitude_embedding(input1)
    assert len(output1.shape) == 1, f"Expected 1D tensor, got shape {output1.shape}"
    
    # Test single sample with batch dimension
    input2 = torch.randn(1, 16)
    output2 = amplitude_embedding(input2)
    assert len(output2.shape) == 1, f"Expected 1D tensor, got shape {output2.shape}"
    
    # Test flattening of multi-dimensional input
    input3 = torch.randn(2, 8)  # Should be flattened to 16
    output3 = amplitude_embedding(input3)
    assert len(output3.shape) == 1, f"Expected 1D tensor, got shape {output3.shape}"
    assert output3.shape[0] == 16, f"Expected 16 elements, got {output3.shape[0]}"
    
    print("✅ All amplitude_embedding tests passed!")
    return True

def test_hybrid_model_quantum_processing():
    """Test the quantum processing part of the HybridModel with different batch sizes."""
    
    # Mock dependencies for isolated testing
    mock_transformer = mock.MagicMock()
    mock_get_transformer = mock.MagicMock(return_value=(mock_transformer, 768))
    
    mock_qnn = mock.MagicMock()
    mock_create_qnn = mock.MagicMock(return_value=(mock_qnn, 16))
    
    # Track if batch processing is properly called
    batch_processing_called = [False]
    single_processing_called = [False]
    
    # Original amplitude_embedding function to use for verification
    original_amplitude_embedding = amplitude_embedding
    
    # Use a wrapper to verify what's called
    def mock_amplitude_embedding(x):
        print(f"Mock amplitude_embedding called with shape: {x.shape}")
        return original_amplitude_embedding(x)
    
    # Create mock quantum layer that returns fixed size output
    def mock_quantum_fn(x):
        # For state vector output - always return 16 values 
        # (simulate a 4-qubit quantum system)
        return torch.ones(16)
    
    # Create the mock for TorchLayer
    mock_torchLayer = mock.MagicMock(return_value=mock_quantum_fn)
    
    # Patch dependencies
    with mock.patch('core.fusion.get_transformer_model', mock_get_transformer):
        with mock.patch('core.fusion.create_qnn', mock_create_qnn):
            with mock.patch('core.fusion.amplitude_embedding', mock_amplitude_embedding):
                with mock.patch('pennylane.qnn.TorchLayer', mock_torchLayer):
                    try:
                        # Create a simple model for testing
                        config = {
                            'classical_backbone_type': 'Transformer',
                            'classical_model_name': 'bert-base-uncased',
                            'num_qubits': 4,  # 2^4 = 16 amplitudes
                            'num_layers': 1,
                            'ansatz_func': lambda x, y, z: None,
                            'use_gpu': False,
                            'observation_type': 'state'
                        }
                        
                        # Create the model
                        model = HybridModel(config)
                        print("✅ Model instantiated successfully")
                        
                        # Test with batch size 1
                        try:
                            # Mock transformer output
                            mock_output1 = mock.MagicMock()
                            mock_output1.pooler_output = torch.randn(1, 768)
                            mock_transformer.return_value = mock_output1
                            
                            # Create input
                            input1 = {
                                'input_ids': torch.ones(1, 10, dtype=torch.long),
                                'attention_mask': torch.ones(1, 10, dtype=torch.long)
                            }
                            
                            # Process
                            output1 = model(input1)
                            print(f"✅ Processed batch size 1, output shape: {output1.shape}")
                            single_processing_called[0] = True
                        except Exception as e:
                            print(f"❌ Error processing batch size 1: {e}")
                            traceback.print_exc()
                            return False
                        
                        # Test with batch size 4
                        try:
                            # Mock transformer output for batch size 4
                            mock_output4 = mock.MagicMock()
                            mock_output4.pooler_output = torch.randn(4, 768)
                            mock_transformer.return_value = mock_output4
                            
                            # Create input
                            input4 = {
                                'input_ids': torch.ones(4, 10, dtype=torch.long),
                                'attention_mask': torch.ones(4, 10, dtype=torch.long)
                            }
                            
                            # Process
                            output4 = model(input4)
                            print(f"✅ Processed batch size 4, output shape: {output4.shape}")
                            batch_processing_called[0] = True
                        except Exception as e:
                            print(f"❌ Error processing batch size 4: {e}")
                            traceback.print_exc()
                            return False
                        
                        # Verify both paths were executed
                        if not single_processing_called[0]:
                            print("❌ Single sample processing path was not executed!")
                            return False
                            
                        if not batch_processing_called[0]:
                            print("❌ Batch processing path was not executed!")
                            return False
                        
                        print("✅ All quantum processing tests passed!")
                        return True
                    except Exception as e:
                        print(f"❌ Error during test setup: {e}")
                        traceback.print_exc()
                        return False

if __name__ == "__main__":
    print("Testing amplitude_embedding function...")
    if not test_amplitude_embedding():
        print("❌ amplitude_embedding tests failed")
        sys.exit(1)
        
    print("\nTesting HybridModel quantum processing...")
    if not test_hybrid_model_quantum_processing():
        print("❌ HybridModel quantum processing tests failed")
        sys.exit(1)
    
    print("\n✅ All tests passed successfully!")
    print("The fixes for batch processing in quantum operations are working correctly.") 