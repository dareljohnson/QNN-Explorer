import unittest
import torch
from transformers import AutoTokenizer

class TestTokenization(unittest.TestCase):
    """Unit tests for tokenization functionality"""
    
    def setUp(self):
        # Load a small, fast tokenizer for testing
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Sample text data of different types
        self.sample_texts = [
            "This is a simple text example",
            "Another example with numbers 123",
            "Text with special characters: @#$%!",
            ["This", "is", "a", "list"],  # Complex structure
            {"key": "Text in a dictionary"},  # Complex structure
            ""  # Empty text
        ]
        
    def test_tokenization_with_string_inputs(self):
        """Test that string inputs can be tokenized properly"""
        # Process texts to proper format
        processed_texts = []
        for text in self.sample_texts:
            if isinstance(text, (list, tuple, dict)):
                text_str = str(text).strip()
                processed_texts.append(text_str)
            else:
                processed_texts.append(str(text).strip())
        
        # Tokenize the texts
        tokens = self.tokenizer(
            processed_texts,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        # Verify the results
        self.assertIn('input_ids', tokens)
        self.assertIn('attention_mask', tokens)
        
        # Check shapes
        self.assertEqual(tokens['input_ids'].shape[0], len(processed_texts))
        self.assertEqual(tokens['input_ids'].shape[1], 128)
        
        # Check we can convert tokens to lists without errors
        for i in range(min(3, len(processed_texts))):
            token_list = tokens['input_ids'][i][:10].tolist()
            self.assertIsInstance(token_list, list)
            
    def test_tokenization_batched(self):
        """Test that batched tokenization works"""
        batch_size = 2
        all_input_ids = []
        all_attention_masks = []
        
        # Process in smaller batches
        for i in range(0, len(self.sample_texts), batch_size):
            batch_texts = self.sample_texts[i:i+batch_size]
            # Convert to strings
            batch_texts = [str(text).strip() for text in batch_texts]
            
            # Tokenize batch
            batch_tokens = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=64,
                return_tensors='pt'
            )
            
            all_input_ids.append(batch_tokens['input_ids'])
            all_attention_masks.append(batch_tokens['attention_mask'])
        
        # Combine batches
        combined_tokens = {
            'input_ids': torch.cat(all_input_ids, dim=0),
            'attention_mask': torch.cat(all_attention_masks, dim=0)
        }
        
        # Verify combined results
        self.assertEqual(combined_tokens['input_ids'].shape[0], len(self.sample_texts))
        self.assertEqual(combined_tokens['attention_mask'].shape[0], len(self.sample_texts))
        

if __name__ == '__main__':
    unittest.main() 