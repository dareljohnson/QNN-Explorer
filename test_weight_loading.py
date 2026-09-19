import os
import sys
import unittest
import torch
import streamlit as st
from unittest.mock import patch, MagicMock
import numpy as np
import shutil

# Add current directory to path for imports
sys.path.append(os.path.abspath('.'))

# Import the function we need to test
import app

# Import the necessary functions
from core.fusion import HybridModel
from core.quantum_models import ansatz1

class TestWeightLoading(unittest.TestCase):
    def setUp(self):
        # Create a mock model
        self.model = MagicMock()
        self.model.state_dict = MagicMock(return_value={})
        self.model.load_state_dict = MagicMock()
        self.model.eval = MagicMock()
        
        # Set up test directories
        self.test_weights_dir = os.path.join("test_data", "weights")
        os.makedirs(self.test_weights_dir, exist_ok=True)
        
        # Create multiple test weight files for different patterns
        self.test_weight_files = [
            os.path.join(self.test_weights_dir, "test_model.pt"),
            os.path.join(self.test_weights_dir, "weights_test_model.pt"),
            os.path.join(self.test_weights_dir, "weights_last_train")
        ]
        
        # Create each test file
        for file_path in self.test_weight_files:
            torch.save({"weight": torch.tensor([1.0])}, file_path)
        
        # Mock session state
        if 'device' not in st.session_state:
            st.session_state.device = 'cpu'
    
    def tearDown(self):
        # Clean up the test files
        for file_path in self.test_weight_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Remove the test directory if it's empty
        if os.path.exists(self.test_weights_dir) and not os.listdir(self.test_weights_dir):
            os.rmdir(self.test_weights_dir)
    
    def test_load_model_weights_standard_pattern(self):
        # Create a patcher for the SAVED_WEIGHTS_DIR
        with patch('app.SAVED_WEIGHTS_DIR', self.test_weights_dir), \
             patch('streamlit.success'), \
             patch('streamlit.error'):
            
            # Test with the standard pattern (config.pt)
            filename = "test_model.json"  # Will be converted to test_model.pt
            result = app.load_model_weights(self.model, filename)
            
            # Check that load_state_dict and model.eval were called
            self.model.load_state_dict.assert_called_once()
            self.model.eval.assert_called_once()
            
            # Check that result is True (success)
            self.assertTrue(result)
    
    def test_load_model_weights_weights_prefix_pattern(self):
        # Remove the standard pattern file to force it to try the weights_ pattern
        if os.path.exists(self.test_weight_files[0]):
            os.remove(self.test_weight_files[0])
        
        with patch('app.SAVED_WEIGHTS_DIR', self.test_weights_dir), \
             patch('streamlit.success'), \
             patch('streamlit.error'):
            
            # Test with a config that will need the weights_ prefix
            filename = "test_model.json"
            result = app.load_model_weights(self.model, filename)
            
            # Check success
            self.model.load_state_dict.assert_called_once()
            self.assertTrue(result)
    
    def test_load_model_weights_last_train_pattern(self):
        # Remove the first two patterns to force it to try weights_last_train
        if os.path.exists(self.test_weight_files[0]):
            os.remove(self.test_weight_files[0])
        if os.path.exists(self.test_weight_files[1]):
            os.remove(self.test_weight_files[1])
        
        with patch('app.SAVED_WEIGHTS_DIR', self.test_weights_dir), \
             patch('streamlit.success'), \
             patch('streamlit.error'):
            
            # Test loading with any config name - should find weights_last_train
            filename = "nonexistent_model.json"
            result = app.load_model_weights(self.model, filename)
            
            # Check success
            self.model.load_state_dict.assert_called_once()
            self.assertTrue(result)
    
    def test_load_model_weights_with_invalid_file(self):
        # Create an empty directory for this test
        empty_dir = os.path.join("test_data", "empty_weights")
        os.makedirs(empty_dir, exist_ok=True)
        
        try:
            with patch('app.SAVED_WEIGHTS_DIR', empty_dir), \
                 patch('streamlit.success'), \
                 patch('streamlit.error'):
                
                # Test with a file that doesn't exist in any pattern
                filename = "nonexistent_model.json"
                result = app.load_model_weights(self.model, filename)
                
                # Check that load_state_dict was not called
                self.model.load_state_dict.assert_not_called()
                
                # Check that result is False (failure)
                self.assertFalse(result)
        finally:
            # Clean up
            if os.path.exists(empty_dir):
                os.rmdir(empty_dir)
    
    def test_load_model_weights_path_normalization(self):
        with patch('app.SAVED_WEIGHTS_DIR', self.test_weights_dir), \
             patch('streamlit.success'), \
             patch('streamlit.error'), \
             patch('os.path.exists', return_value=True), \
             patch('torch.load', return_value={}):
            
            # Create a filename with backslashes
            filename = "test\\model.json"  # Contains backslash
            
            # Mock exists to check the path normalization
            result = app.load_model_weights(self.model, filename)
            
            # Check success
            self.model.load_state_dict.assert_called_once()
            self.assertTrue(result)

def test_model_weight_loading():
    """Test loading weights between models with different configurations."""
    print("\nTesting model weight loading with different configurations...")
    
    # Create test directory
    test_dir = "test_weights"
    os.makedirs(test_dir, exist_ok=True)
    
    # 1. Create a model with output head (classification)
    config_with_head = {
        'classical_backbone_type': 'none',
        'num_qubits': 4,
        'num_layers': 2,
        'ansatz_func': ansatz1,
        'use_gpu': False,
        'observation_type': 'state vector',
        'num_classes': 2  # This adds an output head
    }
    model_with_head = HybridModel(config_with_head)
    
    # Save weights
    weights_with_head_path = os.path.join(test_dir, 'weights_with_head.pt')
    torch.save(model_with_head.state_dict(), weights_with_head_path)
    print(f"Saved model weights with output head to {weights_with_head_path}")
    
    # 2. Create a model without output head
    config_no_head = {
        'classical_backbone_type': 'none',
        'num_qubits': 4,
        'num_layers': 2,
        'ansatz_func': ansatz1,
        'use_gpu': False,
        'observation_type': 'state vector'
        # No num_classes, so no output head
    }
    model_no_head = HybridModel(config_no_head)
    
    # Save weights
    weights_no_head_path = os.path.join(test_dir, 'weights_no_head.pt')
    torch.save(model_no_head.state_dict(), weights_no_head_path)
    print(f"Saved model weights without output head to {weights_no_head_path}")
    
    # 3. Test loading weights with head into model without head
    print("\nTest 1: Loading weights WITH head into model WITHOUT head")
    try:
        # This should fail with strict=True
        model_no_head.load_state_dict(torch.load(weights_with_head_path), strict=True)
        print("❌ ERROR: Should have failed with strict=True!")
        assert False
    except Exception as e:
        print(f"✅ Expected error with strict=True: {e}")
    
    try:
        # This should succeed with strict=False
        model_no_head.load_state_dict(torch.load(weights_with_head_path), strict=False)
        print("✅ Successfully loaded weights with strict=False")
    except Exception as e:
        print(f"❌ ERROR: Should have succeeded with strict=False: {e}")
        assert False
    
    # 4. Test loading weights without head into model with head
    print("\nTest 2: Loading weights WITHOUT head into model WITH head")
    model_with_head_2 = HybridModel(config_with_head)  # New instance
    
    try:
        # This should fail with strict=True
        model_with_head_2.load_state_dict(torch.load(weights_no_head_path), strict=True)
        print("❌ ERROR: Should have failed with strict=True!")
        assert False
    except Exception as e:
        print(f"✅ Expected error with strict=True: {e}")
    
    try:
        # This should succeed with strict=False
        model_with_head_2.load_state_dict(torch.load(weights_no_head_path), strict=False)
        print("✅ Successfully loaded weights with strict=False")
    except Exception as e:
        print(f"❌ ERROR: Should have succeeded with strict=False: {e}")
        assert False
    
    # Clean up
    shutil.rmtree(test_dir)
    
    print("\n✅ All weight loading tests passed!")
    return True

if __name__ == "__main__":
    unittest.main()
    test_model_weight_loading() 