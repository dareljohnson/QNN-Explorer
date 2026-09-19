import sys
import os
import torch
import unittest.mock as mock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary functions for text preprocessing
from data.preprocessing import preprocess_text
from core.fusion import HybridModel
from core.quantum_models import amplitude_embedding

# Mock the tokenizer
class MockTokenizer:
    def __init__(self):
        self.name_or_path = "mock-bert-tokenizer"
    
    def __call__(self, text_input, add_special_tokens=True, max_length=512, padding='max_length', 
                truncation=True, return_attention_mask=True, return_tensors=None):
        # Generate mock token ids based on text length
        if isinstance(text_input, str):
            length = min(len(text_input.split()), max_length-2)  # Account for special tokens
            input_ids = torch.ones(1, length+2, dtype=torch.long)  # +2 for [CLS] and [SEP]
        else:
            # Handle lists of text
            input_ids = torch.ones(len(text_input), max_length, dtype=torch.long)
        
        # Generate attention mask (1s for tokens, 0s for padding)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
    def encode(self, text):
        # Simulate tokenization by counting words and adding 2 for [CLS] and [SEP]
        if isinstance(text, str):
            return [1] * (len(text.split()) + 2)
        return [1] * (len(str(text).split()) + 2)


class MockTransformerOutput:
    """Mock for transformer output with pooler_output and last_hidden_state"""
    def __init__(self, batch_size=1, hidden_size=768):
        self.pooler_output = torch.randn(batch_size, hidden_size)
        self.last_hidden_state = torch.randn(batch_size, 10, hidden_size)


def test_text_prediction_pipeline():
    """Test the full text prediction pipeline from tokenization to quantum processing"""
    print("\nTesting text prediction pipeline...")
    
    # Step 1: Prepare mock text input
    test_text = "What is a quantum neural network?"
    
    # Step 2: Create mock tokenizer
    tokenizer = MockTokenizer()
    
    # Step 3: Tokenize text
    print("Tokenizing text...")
    text_encoding = preprocess_text(test_text, tokenizer)
    print(f"Tokenized text shape: input_ids={text_encoding['input_ids'].shape}, "
          f"attention_mask={text_encoding['attention_mask'].shape}")
    
    # Step 4: Create mock model and components for testing
    # Create mock transformer and quantum components
    mock_transformer = mock.MagicMock()
    mock_transformer.return_value = MockTransformerOutput(batch_size=1)
    
    mock_qnn = mock.MagicMock()
    mock_qnn.return_value = torch.ones(16)  # Mock quantum output
    
    mock_torch_layer = mock.MagicMock()
    mock_torch_layer.return_value = torch.ones(16)
    
    # Model config
    config = {
        'classical_backbone_type': 'transformer',
        'classical_model_name': 'bert-base-uncased',
        'num_qubits': 4,  # 2^4 = 16 quantum input size
        'num_layers': 1,
        'use_gpu': False,
        'observation_type': 'state'
    }
    
    # Create model with mock components
    with mock.patch('core.fusion.get_transformer_model', return_value=(mock_transformer, 768)), \
         mock.patch('core.fusion.create_qnn', return_value=(mock_qnn, 8)), \
         mock.patch('pennylane.qnn.TorchLayer', return_value=mock_torch_layer):
        
        model = HybridModel(config)
        
        # Step 5: Process input through model
        print("Running model forward pass...")
        try:
            output = model(text_encoding)
            print(f"✅ Model successfully processed text input, output shape: {output.shape}")
            return True
        except Exception as e:
            print(f"❌ Error processing text: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = test_text_prediction_pipeline()
    assert success, "Text prediction pipeline test failed"
    print("All text prediction pipeline tests passed!") 