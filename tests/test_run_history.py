import unittest
import sys
import os
import pandas as pd
import tempfile
import shutil
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a temporary directory for test data
TEST_DIR = tempfile.mkdtemp()

# Override the run history paths for testing
import utils.run_history
utils.run_history.RUN_HISTORY_PATH = os.path.join(TEST_DIR, "run_history.csv")
utils.run_history.RUN_DETAILS_PATH = os.path.join(TEST_DIR, "details")

# Now import the functions
from utils.run_history import (
    load_run_history,
    save_run_history,
    add_run,
    get_run,
    update_run_status,
    save_run_details,
    load_run_details,
    delete_run
)

class TestRunHistory(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Create required directories
        os.makedirs(utils.run_history.RUN_DETAILS_PATH, exist_ok=True)
        
    def tearDown(self):
        """Clean up test environment"""
        # Remove test files
        if os.path.exists(utils.run_history.RUN_HISTORY_PATH):
            os.remove(utils.run_history.RUN_HISTORY_PATH)
            
        # Remove test directory
        shutil.rmtree(TEST_DIR, ignore_errors=True)
    
    def test_load_save_history(self):
        """Test loading and saving run history"""
        # Create a test DataFrame
        df = pd.DataFrame({
            'run_id': [1, 2],
            'timestamp': ['2023-01-01 12:00:00', '2023-01-01 13:00:00'],
            'config_name': ['config1', 'config2'],
            'duration': [60.0, 120.0],
            'status': ['Completed', 'Failed']
        })
        
        # Save the DataFrame
        save_run_history(df)
        
        # Load the DataFrame
        loaded_df = load_run_history()
        
        # Check that the loaded DataFrame matches the original
        pd.testing.assert_frame_equal(df, loaded_df)
        
    def test_add_run(self):
        """Test adding a new run"""
        # Add a run
        run_id = add_run('test_config', 60.0, 'Completed')
        
        # Check that the run ID is correct
        self.assertEqual(run_id, 1)
        
        # Add another run
        run_id = add_run('test_config2', 120.0, 'Failed')
        
        # Check that the run ID is incremented
        self.assertEqual(run_id, 2)
        
        # Load the history and check that both runs are there
        df = load_run_history()
        self.assertEqual(len(df), 2)
        self.assertEqual(df['config_name'].tolist(), ['test_config', 'test_config2'])
        
    def test_get_run(self):
        """Test getting a run"""
        # Add a run
        run_id = add_run('test_config', 60.0, 'Completed')
        
        # Get the run
        run = get_run(run_id)
        
        # Check that the run data is correct
        self.assertEqual(run['run_id'], run_id)
        self.assertEqual(run['config_name'], 'test_config')
        self.assertEqual(run['duration'], 60.0)
        self.assertEqual(run['status'], 'Completed')
        
        # Try to get a non-existent run
        run = get_run(999)
        self.assertIsNone(run)
        
    def test_update_run_status(self):
        """Test updating a run's status"""
        # Add a run
        run_id = add_run('test_config', 60.0, 'Completed')
        
        # Update the status
        update_run_status(run_id, 'Failed')
        
        # Get the run and check that the status was updated
        run = get_run(run_id)
        self.assertEqual(run['status'], 'Failed')
        
    def test_save_load_details(self):
        """Test saving and loading run details"""
        # Add a run
        run_id = add_run('test_config', 60.0, 'Completed')
        
        # Create some test details
        details = {
            'metrics': {
                'accuracy': 0.85,
                'precision': 0.80,
                'recall': 0.90,
                'f1': 0.85
            },
            'confusion_matrix': [[100, 20], [10, 90]],
            'class_names': ['Class 0', 'Class 1']
        }
        
        # Save the details
        save_run_details(run_id, details)
        
        # Load the details
        loaded_details = load_run_details(run_id)
        
        # Check that the loaded details match the original
        self.assertEqual(loaded_details['metrics']['accuracy'], details['metrics']['accuracy'])
        self.assertEqual(loaded_details['confusion_matrix'], details['confusion_matrix'])
        self.assertEqual(loaded_details['class_names'], details['class_names'])
        
    def test_delete_run(self):
        """Test deleting a run"""
        # Add a run
        run_id = add_run('test_config', 60.0, 'Completed')
        
        # Save some details
        details = {'metrics': {'accuracy': 0.85}}
        save_run_details(run_id, details)
        
        # Delete the run
        delete_run(run_id)
        
        # Check that the run is no longer in the history
        df = load_run_history()
        self.assertEqual(len(df), 0)
        
        # Check that the details file was deleted
        details_file = os.path.join(utils.run_history.RUN_DETAILS_PATH, f"run_{run_id}.pkl")
        self.assertFalse(os.path.exists(details_file))

if __name__ == "__main__":
    unittest.main() 