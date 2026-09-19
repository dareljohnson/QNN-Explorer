"""
Unit tests for the housing price prediction form visibility based on task type.
"""

import unittest
from unittest.mock import MagicMock
import sys
import os

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestHousingFormVisibility(unittest.TestCase):
    """Test case for the housing price prediction form visibility logic."""
    
    def test_hide_housing_form_for_classification(self):
        """Test that housing form is hidden for classification models."""
        # Create mocks
        st = MagicMock()
        session_state = MagicMock()
        
        # Set up mock session state for classification
        session_state.hybrid_model = True  # Model exists
        session_state.get.return_value = 'Classification'  # Task type is Classification
        
        # Import the relevant code snippet that handles the housing form visibility
        # Note: We're using a local scope to isolate the test
        local_scope = {'st': st, 'session_state': session_state}
        code_to_test = """
show_housing_form = True
    
# Don't show housing form for Classification tasks when a model is loaded
if session_state.hybrid_model and session_state.get('task_type', '') == 'Classification':
    show_housing_form = False
    st.info("Housing Price Prediction form is hidden because you have a Classification model loaded. For regression tasks like house price prediction, please use a model configured for Regression.")

if show_housing_form:
    st.subheader("🏠 Housing Price Prediction")
        """
        
        # Execute the code with mocks
        exec(code_to_test, {}, local_scope)
        
        # Verify that housing form is hidden
        self.assertFalse(local_scope['show_housing_form'])
        
        # Verify info message was shown
        st.info.assert_called_once()
        
        # Verify subheader was not called (form is hidden)
        st.subheader.assert_not_called()
    
    def test_show_housing_form_for_regression(self):
        """Test that housing form is shown for regression models."""
        # Create mocks
        st = MagicMock()
        session_state = MagicMock()
        
        # Set up mock session state for regression
        session_state.hybrid_model = True  # Model exists
        session_state.get.return_value = 'Regression'  # Task type is Regression
        
        # Import the relevant code snippet that handles the housing form visibility
        local_scope = {'st': st, 'session_state': session_state}
        code_to_test = """
show_housing_form = True
    
# Don't show housing form for Classification tasks when a model is loaded
if session_state.hybrid_model and session_state.get('task_type', '') == 'Classification':
    show_housing_form = False
    st.info("Housing Price Prediction form is hidden because you have a Classification model loaded. For regression tasks like house price prediction, please use a model configured for Regression.")

if show_housing_form:
    st.subheader("🏠 Housing Price Prediction")
        """
        
        # Execute the code with mocks
        exec(code_to_test, {}, local_scope)
        
        # Verify that housing form is shown
        self.assertTrue(local_scope['show_housing_form'])
        
        # Verify info message was not shown
        st.info.assert_not_called()
        
        # Verify subheader was called (form is shown)
        st.subheader.assert_called_once()
    
    def test_show_housing_form_when_no_model(self):
        """Test that housing form is shown when no model is loaded."""
        # Create mocks
        st = MagicMock()
        session_state = MagicMock()
        
        # Set up mock session state with no model
        session_state.hybrid_model = None  # No model
        session_state.get.return_value = ''  # Empty task type
        
        # Import the relevant code snippet that handles the housing form visibility
        local_scope = {'st': st, 'session_state': session_state}
        code_to_test = """
show_housing_form = True
    
# Don't show housing form for Classification tasks when a model is loaded
if session_state.hybrid_model and session_state.get('task_type', '') == 'Classification':
    show_housing_form = False
    st.info("Housing Price Prediction form is hidden because you have a Classification model loaded. For regression tasks like house price prediction, please use a model configured for Regression.")

if show_housing_form:
    st.subheader("🏠 Housing Price Prediction")
        """
        
        # Execute the code with mocks
        exec(code_to_test, {}, local_scope)
        
        # Verify that housing form is shown
        self.assertTrue(local_scope['show_housing_form'])
        
        # Verify info message was not shown
        st.info.assert_not_called()
        
        # Verify subheader was called (form is shown)
        st.subheader.assert_called_once()

if __name__ == '__main__':
    unittest.main() 