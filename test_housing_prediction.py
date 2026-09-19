#!/usr/bin/env python
"""
Test for the housing price prediction UI components.

This script tests that the housing price prediction UI components
render correctly without any nested expander errors.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Create mocks for the streamlit module before importing it
sys.modules['streamlit'] = MagicMock()
import streamlit as st

# Mock Streamlit components to test UI rendering
class TestHousingPrediction(unittest.TestCase):
    """Test the housing price prediction UI components."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a mock for st.expander
        self.expander_patcher = patch('streamlit.expander')
        self.mock_expander = self.expander_patcher.start()
        self.mock_expander.side_effect = self.mock_expander_side_effect
        
        # Create a mock for st.checkbox
        self.checkbox_patcher = patch('streamlit.checkbox')
        self.mock_checkbox = self.checkbox_patcher.start()
        self.mock_checkbox.return_value = True
        
        # Create a mock for st.form and st.form_submit_button
        self.form_patcher = patch('streamlit.form')
        self.mock_form = self.form_patcher.start()
        self.mock_form_context = MagicMock()
        self.mock_form.return_value = self.mock_form_context
        
        self.form_submit_patcher = patch('streamlit.form_submit_button')
        self.mock_form_submit = self.form_submit_patcher.start()
        self.mock_form_submit.return_value = False
        
        # Track nested expanders
        self.expander_depth = 0
        self.max_expander_depth = 0

        # Add mocks for other Streamlit functions used in the components
        self.write_patcher = patch('streamlit.write')
        self.mock_write = self.write_patcher.start()
        
        self.columns_patcher = patch('streamlit.columns')
        self.mock_columns = self.columns_patcher.start()
        # Return a list of 3 MagicMock objects that can be used in 'with' statements
        self.column_contexts = [MagicMock(), MagicMock(), MagicMock()]
        for col in self.column_contexts:
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=None)
        # Make sure we return only the requested number of columns
        self.mock_columns.side_effect = lambda n: self.column_contexts[:n]
        
        # Add mocks for various input widgets
        self.number_input_patcher = patch('streamlit.number_input')
        self.mock_number_input = self.number_input_patcher.start()
        self.mock_number_input.return_value = 0
        
        self.selectbox_patcher = patch('streamlit.selectbox')
        self.mock_selectbox = self.selectbox_patcher.start()
        self.mock_selectbox.return_value = 'CA'
        
        self.slider_patcher = patch('streamlit.slider')
        self.mock_slider = self.slider_patcher.start()
        self.mock_slider.return_value = 3
        
        self.metric_patcher = patch('streamlit.metric')
        self.mock_metric = self.metric_patcher.start()
        
        self.success_patcher = patch('streamlit.success')
        self.mock_success = self.success_patcher.start()
        
        self.info_patcher = patch('streamlit.info')
        self.mock_info = self.info_patcher.start()
        
    def tearDown(self):
        """Clean up after the test."""
        self.expander_patcher.stop()
        self.checkbox_patcher.stop()
        self.form_patcher.stop()
        self.form_submit_patcher.stop()
        self.write_patcher.stop()
        self.columns_patcher.stop()
        self.number_input_patcher.stop()
        self.selectbox_patcher.stop()
        self.slider_patcher.stop()
        self.metric_patcher.stop()
        self.success_patcher.stop()
        self.info_patcher.stop()
    
    def mock_expander_side_effect(self, *args, **kwargs):
        """Mock the st.expander function and track nesting depth."""
        self.expander_depth += 1
        self.max_expander_depth = max(self.max_expander_depth, self.expander_depth)
        
        # Create a context manager that decreases depth when exiting
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cm)
        mock_cm.__exit__ = MagicMock(side_effect=lambda *args: self.decrease_depth())
        
        return mock_cm
    
    def decrease_depth(self):
        """Decrease the expander depth when exiting an expander."""
        self.expander_depth -= 1
        return False
    
    def test_housing_price_form_no_nested_expanders(self):
        """Test that the housing price form doesn't have nested expanders."""
        # Import the housing price form component from our utilities file
        sys.path.append('.')
        from housing_utils import render_housing_price_form
        
        # Call the function to render the form
        render_housing_price_form()
        
        # Check if any nested expanders were created
        self.assertLessEqual(self.max_expander_depth, 1, 
                            "Nested expanders detected in housing price form")
    
    def test_prediction_details_no_nested_expanders(self):
        """Test that the prediction details don't have nested expanders."""
        # Import the prediction details component from our utilities file
        sys.path.append('.')
        from housing_utils import render_prediction_details
        
        # Call the function with test parameters
        render_prediction_details(
            price=500000, 
            state="CA", 
            sqft=2000, 
            bedrooms=3, 
            bathrooms=2.0,
            year_built=1990, 
            condition=3, 
            multiplier=1.5,
            age_factor=0.8, 
            condition_factor=0.9, 
            waterfront=False
        )
        
        # Check if any nested expanders were created
        self.assertLessEqual(self.max_expander_depth, 1, 
                            "Nested expanders detected in prediction details")

# Create a simple test runner function
def run_tests():
    """Run the tests and return True if all tests pass."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHousingPrediction)
    result = unittest.TextTestRunner().run(suite)
    return result.wasSuccessful()

# Run the tests if executed directly
if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 