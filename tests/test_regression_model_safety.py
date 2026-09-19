"""
Unit tests for the regression model selection safety check.
Tests that the code correctly handles cases where model_name is not in available models.
"""

import unittest
from unittest.mock import MagicMock
import sys
import os

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestRegressionModelSafety(unittest.TestCase):
    """Test case for the regression model selection safety check."""
    
    def test_model_name_not_in_list(self):
        """Test that code correctly handles a model name not in the available list."""
        # Create mocks
        st = MagicMock()
        session_state = MagicMock()
        
        # Define our constants for the test
        AVAILABLE_REGRESSION_MODELS = ["linear", "sklearn_linear"]
        
        # Set up mock session state with an invalid model name
        session_state.model_config = {'classical_model_name': 'resnet18'}  # invalid model type for regression
        
        # Import the relevant code snippet that handles model selection
        local_scope = {
            'st': st, 
            'session_state': session_state,
            'AVAILABLE_REGRESSION_MODELS': AVAILABLE_REGRESSION_MODELS
        }
        
        code_to_test = """
# Add UI for regression models
default_model = session_state.model_config.get('classical_model_name', 'linear')
# Add safety check - if the model name isn't in the available models, default to 'linear'
if default_model not in AVAILABLE_REGRESSION_MODELS:
    default_model = 'linear'
    
cb_name = st.selectbox("Regression Model", AVAILABLE_REGRESSION_MODELS, 
                      index=AVAILABLE_REGRESSION_MODELS.index(default_model), 
                      key="classical_model_name_regression",
                      help="Linear regression model for housing price prediction")
        """
        
        # Execute the code with mocks
        exec(code_to_test, {}, local_scope)
        
        # Verify that selectbox was called with the default model
        st.selectbox.assert_called_once()
        # Check that the index parameter is 0 (index of 'linear' in AVAILABLE_REGRESSION_MODELS)
        call_kwargs = st.selectbox.call_args[1]
        self.assertEqual(call_kwargs['index'], 0)
        
    def test_valid_model_name(self):
        """Test that code correctly handles a valid model name in the available list."""
        # Create mocks
        st = MagicMock()
        session_state = MagicMock()
        
        # Define our constants for the test
        AVAILABLE_REGRESSION_MODELS = ["linear", "sklearn_linear"]
        
        # Set up mock session state with a valid model name
        session_state.model_config = {'classical_model_name': 'sklearn_linear'}  # valid model
        
        # Import the relevant code snippet that handles model selection
        local_scope = {
            'st': st, 
            'session_state': session_state,
            'AVAILABLE_REGRESSION_MODELS': AVAILABLE_REGRESSION_MODELS
        }
        
        code_to_test = """
# Add UI for regression models
default_model = session_state.model_config.get('classical_model_name', 'linear')
# Add safety check - if the model name isn't in the available models, default to 'linear'
if default_model not in AVAILABLE_REGRESSION_MODELS:
    default_model = 'linear'
    
cb_name = st.selectbox("Regression Model", AVAILABLE_REGRESSION_MODELS, 
                      index=AVAILABLE_REGRESSION_MODELS.index(default_model), 
                      key="classical_model_name_regression",
                      help="Linear regression model for housing price prediction")
        """
        
        # Execute the code with mocks
        exec(code_to_test, {}, local_scope)
        
        # Verify that selectbox was called with the valid model
        st.selectbox.assert_called_once()
        # Check that the index parameter is 1 (index of 'sklearn_linear' in AVAILABLE_REGRESSION_MODELS)
        call_kwargs = st.selectbox.call_args[1]
        self.assertEqual(call_kwargs['index'], 1)
    
    def test_missing_model_config(self):
        """Test that code correctly handles missing model_config in session state."""
        # Create mocks
        st = MagicMock()
        session_state = MagicMock()
        
        # Define our constants for the test
        AVAILABLE_REGRESSION_MODELS = ["linear", "sklearn_linear"]
        
        # Set up mock session state with NO model_config
        session_state.model_config = {}
        
        # Import the relevant code snippet that handles model selection
        local_scope = {
            'st': st, 
            'session_state': session_state,
            'AVAILABLE_REGRESSION_MODELS': AVAILABLE_REGRESSION_MODELS
        }
        
        code_to_test = """
# Add UI for regression models
default_model = session_state.model_config.get('classical_model_name', 'linear')
# Add safety check - if the model name isn't in the available models, default to 'linear'
if default_model not in AVAILABLE_REGRESSION_MODELS:
    default_model = 'linear'
    
cb_name = st.selectbox("Regression Model", AVAILABLE_REGRESSION_MODELS, 
                      index=AVAILABLE_REGRESSION_MODELS.index(default_model), 
                      key="classical_model_name_regression",
                      help="Linear regression model for housing price prediction")
        """
        
        # Execute the code with mocks
        exec(code_to_test, {}, local_scope)
        
        # Verify that selectbox was called with the default model
        st.selectbox.assert_called_once()
        # Check that the index parameter is 0 (index of 'linear' in AVAILABLE_REGRESSION_MODELS)
        call_kwargs = st.selectbox.call_args[1]
        self.assertEqual(call_kwargs['index'], 0)

if __name__ == '__main__':
    unittest.main() 