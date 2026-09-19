"""
Unit tests for the run history selection dropdown functionality.
"""

import os
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys

# Add the parent directory to the path to make relative imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.run_history import add_run, load_run_history, save_run_details

# Patch the STREAMLIT_AVAILABLE variable
import utils.run_history_tab
utils.run_history_tab.STREAMLIT_AVAILABLE = True

class TestRunHistorySelectDropdown(unittest.TestCase):
    """Test case for the run history selection dropdown functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for the test history
        self.test_dir = os.path.join("tests", "test_history")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Path for the test history file
        self.test_history_path = os.path.join(self.test_dir, "test_run_history.csv")
        self.test_details_path = os.path.join(self.test_dir, "run_details")
        os.makedirs(self.test_details_path, exist_ok=True)
        
        # Create some test run history data
        self.test_runs = pd.DataFrame({
            'run_id': [1, 2, 3],
            'timestamp': ['2023-05-01', '2023-05-02', '2023-05-03'],
            'config_name': ['Test Config 1', 'Test Config 2', 'Test Config 3'],
            'duration': [60, 120, 180],
            'status': ['Completed', 'Failed', 'Completed']
        })
        
        # Save the test data
        self.test_runs.to_csv(self.test_history_path, index=False)
        
        # Create sample run details
        for run_id in self.test_runs['run_id']:
            run_details = {
                'run_id': run_id,
                'config_name': f'Test Config {run_id}',
                'task_type': 'Classification',
                'model_config': {
                    'num_qubits': 4,
                    'num_layers': 2
                }
            }
            
            # Save the details to a file
            details_file = os.path.join(self.test_details_path, f"run_{run_id}.json")
            with open(details_file, 'w') as f:
                import json
                json.dump(run_details, f)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up test files
        if os.path.exists(self.test_history_path):
            os.remove(self.test_history_path)
        
        # Clean up test details files
        for run_id in self.test_runs['run_id']:
            details_file = os.path.join(self.test_details_path, f"run_{run_id}.json")
            if os.path.exists(details_file):
                os.remove(details_file)
        
        # Remove test directories
        if os.path.exists(self.test_details_path):
            os.rmdir(self.test_details_path)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
    
    @patch('utils.run_history_tab.st')
    @patch('utils.run_history_tab.load_run_history')
    @patch('utils.run_history_tab.load_run_details')
    def test_run_history_table_display(self, mock_load_details, mock_load_history, mock_st):
        """Test that the run history table is displayed correctly."""
        # Reset the mock's session state
        mock_session_state = {}
        
        def mock_session_state_get(key, default=None):
            return mock_session_state.get(key, default)
            
        mock_st.session_state = MagicMock()
        mock_st.session_state.get.side_effect = mock_session_state_get
        
        # Configure the mock to return our test data
        mock_load_history.return_value = self.test_runs
        mock_load_details.return_value = {'task_type': 'Classification'}
        
        # Call the function that renders the run history tab
        from utils.run_history_tab import render_run_history_tab
        render_run_history_tab()
        
        # Check that the table was displayed
        mock_st.dataframe.assert_called()
        
        # Check that the dropdown was created
        mock_st.selectbox.assert_called()
        
        # Verify the selectbox was called with the correct key
        selectbox_calls = [call for call in mock_st.selectbox.call_args_list 
                          if 'key' in call[1] and call[1]['key'] == 'run_selector']
        self.assertTrue(len(selectbox_calls) > 0, "Selectbox with key 'run_selector' should be created")
    
    @patch('utils.run_history_tab.st')
    @patch('utils.run_history_tab.load_run_history')
    @patch('utils.run_history_tab.load_run_details')
    def test_dropdown_selection_updates_session_state(self, mock_load_details, mock_load_history, mock_st):
        """Test that selecting from the dropdown updates the session state."""
        # Configure the mock to return our test data
        mock_load_history.return_value = self.test_runs
        mock_load_details.return_value = {'task_type': 'Classification'}
        
        # Setup mock session state
        session_state_dict = {}
        
        # Create a mock session state that supports dict-like access
        class MockSessionState(dict):
            def __getitem__(self, key):
                return session_state_dict.get(key)
            
            def __setitem__(self, key, value):
                session_state_dict[key] = value
            
            def get(self, key, default=None):
                return session_state_dict.get(key, default)
        
        mock_st.session_state = MockSessionState()
        
        # Simulate dropdown selection for run_id = 2
        test_run_id = 2
        
        # Make selectbox return our test run ID
        def mock_selectbox_side_effect(*args, **kwargs):
            if 'key' in kwargs and kwargs['key'] == 'run_selector':
                return test_run_id
            return None
        
        mock_st.selectbox.side_effect = mock_selectbox_side_effect
        
        # Call the function that renders the run history tab
        from utils.run_history_tab import render_run_history_tab
        render_run_history_tab()
        
        # Check that session_state was updated with the selected run ID
        self.assertEqual(session_state_dict.get('selected_run_id'), test_run_id,
                        "Session state should be updated with selected dropdown run ID")
        
        # Verify that we asked for a rerun after the selection change
        mock_st.rerun.assert_called()

if __name__ == '__main__':
    unittest.main() 