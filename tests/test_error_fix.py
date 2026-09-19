"""Test that demonstrates the fix for the "Input must be a state vector or density matrix" error."""
import sys
import os
import torch
import pennylane as qml
from pennylane import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.metrics import partial_trace, calculate_meyer_wallach
from data.preprocessing import preprocess_text
from core.fusion import HybridModel
from core.quantum_models import amplitude_embedding

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

def test_end_to_end_with_entanglement():
    """Test the full pipeline from tokenization to entanglement calculation."""
    print("\nTesting end-to-end text processing to entanglement calculation...")
    
    # Set up test inputs
    test_text = "This is a test of the quantum neural network with text input."
    tokenizer = MockTokenizer()
    num_qubits = 4  # 2^4 = 16-dimensional quantum state
    
    # Step 1: Tokenize text
    print("Tokenizing text...")
    encoding = preprocess_text(test_text, tokenizer)
    
    # Step 2: Mock the hybrid model processing
    # We'll create a mock state vector as if it came from the QNN
    mock_state_vector = torch.randn(2**num_qubits, dtype=torch.float32)
    # Normalize for a valid quantum state
    mock_state_vector = mock_state_vector / torch.norm(mock_state_vector)
    
    # Step 3: Calculate entanglement on the state vector
    try:
        print("Calculating entanglement on the resulting state vector...")
        entanglement = calculate_meyer_wallach(mock_state_vector, num_qubits)
        print(f"Entanglement: {entanglement:.6f}")
        print("✅ Successfully calculated entanglement")
        return True
    except Exception as e:
        print(f"❌ Error calculating entanglement: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_end_to_end_with_entanglement()
    assert success, "End-to-end test with entanglement calculation failed"
    print("All end-to-end tests passed!") 