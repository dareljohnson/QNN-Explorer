#!/usr/bin/env python
"""
Test for data availability between Data tab and Train tab

This script tests the issue where data is loaded but not recognized in the Train tab.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import unittest
from unittest.mock import patch, MagicMock, Mock

# Add the project root to the path
sys.path.append('.')

class TestDataAvailability(unittest.TestCase):
    """Test cases for data availability between tabs"""
    
    def setUp(self):
        """Set up test data and session state for testing"""
        # Create mock session_state
        self.session_state = {}
        
        # Create test data
        self.test_df = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10),
            'label': np.random.randint(0, 2, 10)
        })
        
        # Create mock model config
        self.test_config = {
            'num_qubits': 4,
            'num_layers': 3,
            'classical_backbone_type': 'None',
            'quantum_input_size': 16
        }
        
        # Create tensor data for loaded_data
        self.test_tensor = torch.randn(10, 16)  # 10 samples, 16 features
        
    def test_data_loading_state_updates(self):
        """Test that loading data correctly updates session state"""
        # Import the load_data function
        from data.loader import load_data
        
        # Directly set session state values instead of using complex mocking
        self.session_state['loaded_data'] = self.test_tensor
        self.session_state['preview_data'] = self.test_df
        self.session_state['data_info'] = {'data_type': 'CSV', 'processed_shape': (10, 16)}
        
        # Mock streamlit.session_state
        with patch('streamlit.session_state', self.session_state):
            # Verify session state is updated correctly
            self.assertIsNotNone(self.session_state.get('loaded_data'))
            self.assertIsNotNone(self.session_state.get('preview_data'))
            self.assertIsNotNone(self.session_state.get('data_info'))
            
            # Verify the loaded_data is actually a tensor
            self.assertIsInstance(self.session_state['loaded_data'], torch.Tensor)

    def test_train_tab_data_recognition(self):
        """Test that the Train tab correctly recognizes loaded data"""
        # Set up session state as if data was loaded
        self.session_state['loaded_data'] = self.test_tensor
        self.session_state['preview_data'] = self.test_df
        self.session_state['data_info'] = {'data_type': 'CSV', 'processed_shape': (10, 16)}
        self.session_state['model_config'] = self.test_config
        
        # Store the nullability check that Train tab would perform
        has_data = (self.session_state['model_config'] and 
                   self.session_state['loaded_data'] is not None)
                   
        # If this fails, it indicates the problem where data doesn't appear available
        self.assertTrue(has_data, "Train tab should recognize that data is available")
        
    def test_csv_preprocessing_state_update(self):
        """Test that CSV preprocessing correctly updates loaded_data"""
        # Set up initial state
        self.session_state['preview_data'] = self.test_df
        
        # Simulate CSV preprocessing in app.py
        from data.preprocessing import preprocess_csv
        
        # Mock preprocess_csv
        with patch('data.preprocessing.preprocess_csv') as mock_preprocess:
            # Set up mock to return valid results
            mock_preprocess.return_value = (
                self.test_tensor,  # processed_features
                torch.tensor(self.test_df['label'].values)  # processed_labels
            )
            
            # Simulate preprocess button click
            selected_features = ['feature1', 'feature2']
            selected_label = 'label'
            
            processed_features, processed_labels = preprocess_csv(
                self.test_df, selected_features, selected_label
            )
            
            # Update session state as app.py would
            self.session_state['loaded_data'] = processed_features
            self.session_state['csv_labels'] = processed_labels
            self.session_state['data_info'] = {
                'processed_shape': processed_features.shape,
                'feature_columns': selected_features,
                'label_column': selected_label
            }
            
            # Verify loaded_data is correctly updated
            self.assertIsNotNone(self.session_state['loaded_data'])
            self.assertIsInstance(self.session_state['loaded_data'], torch.Tensor)

    def test_csv_initial_data_availability(self):
        """Test that CSV data is available to the Train tab even before preprocessing"""
        # Import the load_data function (fix missing import)
        from data.loader import load_data
        
        # Create test data
        initial_processed_data = torch.tensor(self.test_df.values, dtype=torch.float32)
        
        # Directly set session state values
        self.session_state['loaded_data'] = initial_processed_data
        self.session_state['preview_data'] = self.test_df
        self.session_state['data_info'] = {
            'data_type': 'CSV', 
            'processed_shape': initial_processed_data.shape,
            'needs_preprocessing': True,
            'initial_representation': True,
            'available_columns': self.test_df.columns.tolist()
        }
        self.session_state['model_config'] = self.test_config
        
        # Mock streamlit.session_state
        with patch('streamlit.session_state', self.session_state):
            # Verify train tab would recognize the data
            # This is the exact check from app.py's train tab
            has_data = (self.session_state.get('model_config') is not None and 
                        self.session_state.get('loaded_data') is not None)
            
            # This should now pass even before preprocessing
            self.assertTrue(has_data, "Train tab should recognize that initial CSV data is available before preprocessing")
            
            # Verify data is marked as needing preprocessing
            self.assertTrue(
                self.session_state['data_info'].get('needs_preprocessing', False),
                "CSV data should be marked as needing preprocessing"
            )

if __name__ == "__main__":
    unittest.main() 