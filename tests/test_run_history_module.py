import unittest
import sys
import os
import pandas as pd
import tempfile
import shutil
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a temporary directory for test data
TEST_DIR = tempfile.mkdtemp()

# Override the run history paths for testing
import utils.run_history
original_history_dir = utils.run_history.HISTORY_DIR
original_history_file = utils.run_history.HISTORY_FILE
original_details_dir = utils.run_history.DETAILS_DIR

# Set test paths
utils.run_history.HISTORY_DIR = os.path.join(TEST_DIR, "history")
utils.run_history.HISTORY_FILE = os.path.join(utils.run_history.HISTORY_DIR, "run_history.csv")
utils.run_history.DETAILS_DIR = os.path.join(utils.run_history.HISTORY_DIR, "run_details")
utils.run_history.RUN_HISTORY_PATH = utils.run_history.HISTORY_FILE
utils.run_history.RUN_DETAILS_PATH = utils.run_history.DETAILS_DIR

# Now import the functions
from utils.run_history import (
    init_history_dirs,
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
        init_history_dirs()
        
    def tearDown(self):
        """Clean up test environment"""
        # Remove test directory
        shutil.rmtree(TEST_DIR, ignore_errors=True)
        
        # Restore original paths
        utils.run_history.HISTORY_DIR = original_history_dir
        utils.run_history.HISTORY_FILE = original_history_file
        utils.run_history.DETAILS_DIR = original_details_dir
        utils.run_history.RUN_HISTORY_PATH = original_history_file
        utils.run_history.RUN_DETAILS_PATH = original_details_dir
    
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
        # Create a separate test directory for this test
        test_delete_dir = os.path.join(TEST_DIR, "delete_test")
        os.makedirs(test_delete_dir, exist_ok=True)
        
        # Temporarily override paths for this test
        old_history_dir = utils.run_history.HISTORY_DIR
        old_history_file = utils.run_history.HISTORY_FILE
        old_details_dir = utils.run_history.DETAILS_DIR
        old_history_path = utils.run_history.RUN_HISTORY_PATH
        old_details_path = utils.run_history.RUN_DETAILS_PATH
        
        try:
            # Set paths to our test directory
            utils.run_history.HISTORY_DIR = test_delete_dir
            utils.run_history.HISTORY_FILE = os.path.join(test_delete_dir, "run_history.csv")
            utils.run_history.DETAILS_DIR = os.path.join(test_delete_dir, "run_details")
            utils.run_history.RUN_HISTORY_PATH = utils.run_history.HISTORY_FILE
            utils.run_history.RUN_DETAILS_PATH = utils.run_history.DETAILS_DIR
            
            # Create directories
            os.makedirs(utils.run_history.DETAILS_DIR, exist_ok=True)
            
            # Make sure we're starting with a clean slate
            df = pd.DataFrame(columns=['run_id', 'timestamp', 'config_name', 'duration', 'status'])
            save_run_history(df)
            
            # Add a run
            run_id = add_run('test_config', 60.0, 'Completed')
            
            # Save some details
            details = {'metrics': {'accuracy': 0.85}}
            save_run_details(run_id, details)
            
            # Verify run was added
            df = load_run_history()
            self.assertEqual(len(df), 1)
            
            # Delete the run
            delete_run(run_id)
            
            # Check that the run is no longer in the history
            df = load_run_history()
            self.assertEqual(len(df), 0)
            
            # Check that the details file was deleted
            details_file = os.path.join(utils.run_history.DETAILS_DIR, f"run_{run_id}.json")
            self.assertFalse(os.path.exists(details_file))
        
        finally:
            # Restore original paths
            utils.run_history.HISTORY_DIR = old_history_dir
            utils.run_history.HISTORY_FILE = old_history_file
            utils.run_history.DETAILS_DIR = old_details_dir
            utils.run_history.RUN_HISTORY_PATH = old_history_path
            utils.run_history.RUN_DETAILS_PATH = old_details_path
    
    def test_history_directories(self):
        """Test that the history directories are created correctly"""
        # The directories should have been created in setUp
        self.assertTrue(os.path.exists(utils.run_history.HISTORY_DIR))
        self.assertTrue(os.path.exists(utils.run_history.DETAILS_DIR))
        
    def test_run_details_format(self):
        """Test that run details are saved in the correct format"""
        # Add a run
        run_id = add_run('test_config', 60.0, 'Completed')
        
        # Create details with various data types
        details = {
            'model_config': {
                'num_qubits': 4,
                'num_layers': 2,
                'use_gpu': False
            },
            'training_params': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 10
            },
            'metrics': {
                'accuracy': 0.95,
                'loss': 0.05
            },
            'numpy_array': [1, 2, 3, 4, 5]  # This would be a numpy array normally
        }
        
        # Save the details
        save_run_details(run_id, details)
        
        # Load the details
        loaded_details = load_run_details(run_id)
        
        # Check that the details were loaded correctly
        self.assertEqual(loaded_details['model_config']['num_qubits'], 4)
        self.assertEqual(loaded_details['training_params']['learning_rate'], 0.001)
        self.assertEqual(loaded_details['metrics']['accuracy'], 0.95)
        self.assertEqual(loaded_details['numpy_array'], [1, 2, 3, 4, 5])

if __name__ == "__main__":
    unittest.main() 