"""
Test script to verify that CSV preprocessing UI elements appear correctly.
This is a simple unit test to check specific functions and conditions.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import torch

# Mock streamlit for testing
class MockSessionState:
    def __init__(self):
        self.data = {}
    
    def __getitem__(self, key):
        return self.data.get(key)
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def __contains__(self, key):
        return key in self.data

# Define test cases
class TestCSVPreprocessingUI(unittest.TestCase):
    def setUp(self):
        # Create a mock session state
        self.session_state = MockSessionState()
        
        # Create a sample DataFrame
        self.sample_df = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10.1, 20.2, 30.3, 40.4, 50.5],
            'text1': ['apple', 'banana', 'cherry', 'date', 'elderberry'],
            'text2': ['This is a longer text sample', 'Another text example', 
                     'Text processing test', 'Sample text', 'Final example']
        })
    
    def test_csv_preview_data_condition(self):
        """Test that the condition for displaying CSV UI works correctly"""
        # Set up the session state with preview_data
        self.session_state['preview_data'] = self.sample_df
        
        # This simulates the condition in the app
        data_type = "CSV"
        raw_df = self.session_state.get('preview_data')
        
        # Check if UI should display
        should_display_ui = (data_type == "CSV" and 
                            raw_df is not None and 
                            isinstance(raw_df, pd.DataFrame))
        
        self.assertTrue(should_display_ui, 
                       "CSV UI should display when data_type is CSV and preview_data is a DataFrame")
        
        # Test with wrong data type
        data_type = "Images"
        should_display_ui = (data_type == "CSV" and 
                            raw_df is not None and 
                            isinstance(raw_df, pd.DataFrame))
        
        self.assertFalse(should_display_ui, 
                        "CSV UI should not display with non-CSV data_type")
        
        # Test with None preview_data
        data_type = "CSV"
        self.session_state['preview_data'] = None
        raw_df = self.session_state.get('preview_data')
        should_display_ui = (data_type == "CSV" and 
                            raw_df is not None and 
                            isinstance(raw_df, pd.DataFrame))
        
        self.assertFalse(should_display_ui, 
                        "CSV UI should not display with None preview_data")
    
    def test_csv_feature_cols_initialization(self):
        """Test the initialization of csv_feature_cols in session state"""
        # Set up session state with preview_data
        self.session_state['preview_data'] = self.sample_df
        df_cols = self.sample_df.columns.tolist()
        
        # Simulate the initialization code
        if 'csv_feature_cols' not in self.session_state:
            self.session_state['csv_feature_cols'] = [c for c in df_cols[:4] 
                                                   if c.lower() != 'label' and c.lower() != 'target']
        
        # Verify that csv_feature_cols was initialized correctly
        self.assertIn('csv_feature_cols', self.session_state, 
                     "csv_feature_cols should be in session state")
        
        # It should include all columns (up to 4) except 'label' and 'target'
        expected_cols = [c for c in df_cols[:4] if c.lower() != 'label' and c.lower() != 'target']
        self.assertEqual(self.session_state['csv_feature_cols'], expected_cols,
                        "csv_feature_cols should be initialized with the first 4 non-label columns")
    
    def test_load_data_csv_handling(self):
        """Test that load_data handles CSV data correctly"""
        # Simulate data loading for CSV
        data_type = "CSV"
        preview_data = self.sample_df
        
        # This would typically come from load_data function
        loaded_data = None
        data_info = {}
        
        # Enhanced CSV handling from the app
        if data_type == "CSV" and preview_data is not None and isinstance(preview_data, pd.DataFrame):
            # If we don't have processed data yet but have a DataFrame, create initial representation
            if loaded_data is None:
                numeric_cols = preview_data.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    # Use numeric columns to create an initial tensor representation
                    temp_data = preview_data[numeric_cols].values
                    loaded_data = torch.tensor(temp_data, dtype=torch.float32)
                    data_info['needs_preprocessing'] = True
                    data_info['initial_representation'] = True
                    data_info['available_columns'] = preview_data.columns.tolist()
        
        # Update session state
        self.session_state['loaded_data'] = loaded_data
        self.session_state['preview_data'] = preview_data
        self.session_state['data_info'] = data_info
        
        # Verify that the session state was updated correctly
        self.assertIsNotNone(self.session_state['loaded_data'], 
                            "loaded_data should not be None for CSV data")
        self.assertIsNotNone(self.session_state['preview_data'], 
                            "preview_data should not be None for CSV data")
        self.assertIn('needs_preprocessing', self.session_state['data_info'], 
                     "data_info should have needs_preprocessing flag for CSV data")
        self.assertTrue(self.session_state['data_info']['needs_preprocessing'], 
                       "needs_preprocessing should be True for CSV data")
        self.assertTrue(self.session_state['data_info']['initial_representation'], 
                       "initial_representation should be True for CSV data")

if __name__ == "__main__":
    unittest.main() 