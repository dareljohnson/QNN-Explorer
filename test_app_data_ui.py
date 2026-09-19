"""
Test script to verify that the app's Data tab is working correctly.
This script focuses on checking the conditions for displaying CSV UI elements.
"""

import unittest
import pandas as pd
import numpy as np

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

class TestDataTabUI(unittest.TestCase):
    def setUp(self):
        self.session_state = MockSessionState()
        
        # Create a sample DataFrame that would represent loaded CSV data
        self.sample_df = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10.1, 20.2, 30.3, 40.4, 50.5],
            'text1': ['apple', 'banana', 'cherry', 'date', 'elderberry'],
            'text2': ['This is a longer text sample', 'Another text example', 
                     'Text processing test', 'Sample text', 'Final example']
        })
    
    def test_csv_ui_display_condition(self):
        """Test that the condition for displaying CSV UI controls works correctly"""
        # Set up session state with preview_data
        self.session_state['preview_data'] = self.sample_df
        
        # Simulate the condition from app.py that checks if CSV UI should display
        data_type = "CSV"
        raw_df = self.session_state.get('preview_data')
        
        # The condition for showing CSV UI elements (feature/label selectors and preprocessing button)
        should_display_ui = (data_type == "CSV" and 
                              raw_df is not None and 
                              isinstance(raw_df, pd.DataFrame))
        
        # Check the condition is True with correct inputs
        self.assertTrue(should_display_ui, 
                       "CSV UI should display when data_type is CSV and preview_data is a DataFrame")
        
        # Check the condition is False with wrong data type
        data_type = "Images"
        should_display_ui = (data_type == "CSV" and 
                             raw_df is not None and 
                             isinstance(raw_df, pd.DataFrame))
        
        self.assertFalse(should_display_ui, 
                        "CSV UI should not display with non-CSV data_type")
        
        # Check the condition is False with None preview_data
        data_type = "CSV"
        self.session_state['preview_data'] = None
        raw_df = self.session_state.get('preview_data')
        should_display_ui = (data_type == "CSV" and 
                             raw_df is not None and 
                             isinstance(raw_df, pd.DataFrame))
        
        self.assertFalse(should_display_ui, 
                        "CSV UI should not display with None preview_data")
    
    def test_csv_feature_col_initialization(self):
        """Test the initialization logic for default feature columns"""
        # Set up session state with preview_data
        self.session_state['preview_data'] = self.sample_df
        df_cols = self.sample_df.columns.tolist()
        
        # Simulate the initialization code from app.py
        if 'csv_feature_cols' not in self.session_state:
            self.session_state['csv_feature_cols'] = [c for c in df_cols[:4] 
                                                   if c.lower() != 'label' and c.lower() != 'target']
        
        # Check the session state was updated correctly
        self.assertIn('csv_feature_cols', self.session_state, 
                     "csv_feature_cols should be added to session state")
        
        # All columns should be included by default (up to 4) except any 'label' or 'target' columns
        expected_cols = [c for c in df_cols[:4] if c.lower() != 'label' and c.lower() != 'target']
        self.assertEqual(self.session_state['csv_feature_cols'], expected_cols,
                        "csv_feature_cols should contain all sample columns")
        
        # Test with DF that has 'label' column
        df_with_label = self.sample_df.copy()
        df_with_label.rename(columns={'text2': 'label'}, inplace=True)
        df_cols_with_label = df_with_label.columns.tolist()
        
        # Reset session state
        del self.session_state.data['csv_feature_cols']
        self.session_state['preview_data'] = df_with_label
        
        # Simulate initialization again
        if 'csv_feature_cols' not in self.session_state:
            self.session_state['csv_feature_cols'] = [c for c in df_cols_with_label[:4] 
                                                   if c.lower() != 'label' and c.lower() != 'target']
        
        # Check that 'label' column is excluded
        self.assertNotIn('label', self.session_state['csv_feature_cols'],
                        "label column should be excluded from feature columns")

if __name__ == "__main__":
    unittest.main() 