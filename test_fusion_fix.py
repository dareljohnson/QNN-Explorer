import sys
import os
import torch
import unittest.mock as mock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the HybridModel class
from core.fusion import HybridModel

def run_fusion_test():
    # Create mocks
    mock_transformer = mock.MagicMock()
    mock_get_transformer_model = mock.MagicMock(return_value=(mock_transformer, 768))
    mock_amplitude_embedding = mock.MagicMock(return_value=torch.ones(16))
    mock_qnn = mock.MagicMock()
    mock_create_qnn = mock.MagicMock(return_value=(mock_qnn, 16))
    mock_tensor = torch.ones(1, 16)  # Real tensor instead of MagicMock
    mock_qnn_layer = mock.MagicMock()
    mock_qnn_layer.return_value = mock_tensor
    
    # Patch the TorchLayer and other dependencies
    with mock.patch('core.fusion.get_transformer_model', mock_get_transformer_model):
        with mock.patch('core.fusion.amplitude_embedding', mock_amplitude_embedding):
            with mock.patch('core.fusion.create_qnn', mock_create_qnn):
                with mock.patch('pennylane.qnn.TorchLayer', return_value=mock_qnn_layer):
                    # Test different case variations for observation_type
                    variations = [
                        'state',
                        'State',
                        'STATE',
                        'State Vector',
                        'STATE VECTOR',
                        'expval',
                        'EXPVAL',
                        'Expectation Value',
                        'Expectation Value (PauliZ)'
                    ]
                    
                    for obs_type in variations:
                        try:
                            print(f"Testing fusion with observation_type: '{obs_type}'")
                            
                            # Create config with the observation type
                            config = {
                                'classical_backbone_type': 'Transformer',
                                'classical_model_name': 'bert-base-uncased',
                                'num_qubits': 4,
                                'num_layers': 2,
                                'ansatz_func': lambda x, y, z: None,
                                'use_gpu': False,
                                'observation_type': obs_type
                            }
                            
                            # Create model instance
                            model = HybridModel(config)
                            print(f"✅ Successfully instantiated HybridModel with observation_type: '{obs_type}'")
                            
                            # Test forward pass with fake input
                            fake_input = {
                                'input_ids': torch.ones(1, 10, dtype=torch.long),
                                'attention_mask': torch.ones(1, 10, dtype=torch.long)
                            }
                            
                            # Set up forward pass behavior for transformer
                            # Create a real tensor output instead of a MagicMock
                            transformer_output = torch.ones(1, 10, 768)  # [batch, seq_len, hidden_size]
                            mock_transformer.return_value = transformer_output
                            
                            # Patch the transformer_pooling method to return a real tensor
                            original_pooling = model.transformer_pooling
                            model.transformer_pooling = mock.MagicMock(return_value=torch.ones(1, 768))
                            
                            # Ensure fusion_layer is a mock that returns a real tensor
                            if hasattr(model, 'fusion_layer') and isinstance(model.fusion_layer, mock.MagicMock):
                                model.fusion_layer.return_value = torch.ones(1, 16)
                            
                            # Execute forward pass
                            try:
                                result = model(fake_input)
                                print(f"✅ Successfully ran forward pass with observation_type: '{obs_type}'")
                            except Exception as e:
                                print(f"❌ Error during forward pass: {e}")
                                return False
                            finally:
                                # Restore original pooling method
                                model.transformer_pooling = original_pooling
                        except Exception as e:
                            print(f"❌ Error with observation_type '{obs_type}': {e}")
                            return False
                    
                    print("\n✅ All fusion tests passed successfully!")
                    return True

if __name__ == "__main__":
    print("Testing observation_type case handling in fusion.py...")
    if run_fusion_test():
        print("\nFix validated: The local variable reference issue in fusion.py is resolved!")
    else:
        print("\nFix failed: Some observation types still cause errors in fusion.py.") 