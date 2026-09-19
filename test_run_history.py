import os
import sys
import shutil
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
from datetime import datetime
import importlib

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# First, get a reference to the module we'll be testing
import utils.run_history

class TestRunHistory(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Create a common temp directory for all tests."""
        cls.class_temp_dir = tempfile.mkdtemp()
        print(f"Created class temp dir: {cls.class_temp_dir}")
        
        # Store original paths
        cls.original_HISTORY_DIR = utils.run_history.HISTORY_DIR
        cls.original_HISTORY_FILE = utils.run_history.HISTORY_FILE
        cls.original_DETAILS_DIR = utils.run_history.DETAILS_DIR
        cls.original_RUN_HISTORY_PATH = utils.run_history.RUN_HISTORY_PATH
        cls.original_RUN_DETAILS_PATH = utils.run_history.RUN_DETAILS_PATH
    
    @classmethod
    def tearDownClass(cls):
        """Clean up the temp directory when done with all tests."""
        # Restore original module paths
        utils.run_history.HISTORY_DIR = cls.original_HISTORY_DIR
        utils.run_history.HISTORY_FILE = cls.original_HISTORY_FILE
        utils.run_history.DETAILS_DIR = cls.original_DETAILS_DIR
        utils.run_history.RUN_HISTORY_PATH = cls.original_RUN_HISTORY_PATH
        utils.run_history.RUN_DETAILS_PATH = cls.original_RUN_DETAILS_PATH
        
        # Clean up
        try:
            shutil.rmtree(cls.class_temp_dir)
        except Exception as e:
            print(f"Error cleaning up class temp dir: {e}")
    
    def setUp(self):
        """Set up an isolated test environment for each test case."""
        # Create a test directory
        self.test_dir = os.path.join(self.class_temp_dir, self.id().split('.')[-1])
        os.makedirs(self.test_dir, exist_ok=True)
        print(f"Test dir: {self.test_dir}")
        
        # Create test subdirectories
        self.test_history_dir = os.path.join(self.test_dir, "history")
        self.test_details_dir = os.path.join(self.test_history_dir, "run_details")
        os.makedirs(self.test_details_dir, exist_ok=True)
        
        # Set up test files
        self.test_history_file = os.path.join(self.test_history_dir, "run_history.csv")
        
        # Override module paths for this test
        utils.run_history.HISTORY_DIR = self.test_history_dir
        utils.run_history.HISTORY_FILE = self.test_history_file
        utils.run_history.DETAILS_DIR = self.test_details_dir
        utils.run_history.RUN_HISTORY_PATH = self.test_history_file
        utils.run_history.RUN_DETAILS_PATH = self.test_details_dir
        
        # Initialize with empty history file
        self._create_empty_history_file()
        
        # Print current paths for debugging
        print(f"Test HISTORY_DIR: {utils.run_history.HISTORY_DIR}")
        print(f"Test HISTORY_FILE: {utils.run_history.HISTORY_FILE}")
        print(f"Test DETAILS_DIR: {utils.run_history.DETAILS_DIR}")
    
    def tearDown(self):
        """Clean up after each test."""
        try:
            # Remove test directory
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
        except Exception as e:
            print(f"Error cleaning up test dir: {e}")
    
    def _create_empty_history_file(self):
        """Create an empty history file for testing."""
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.test_history_file), exist_ok=True)
        
        # Create an empty CSV file
        df = pd.DataFrame(columns=['run_id', 'timestamp', 'config_name', 'duration', 'status'])
        df.to_csv(self.test_history_file, index=False)
    
    def test_basic_operations(self):
        """Test basic run history operations."""
        print(f"Basic Operations test using file: {utils.run_history.HISTORY_FILE}")
        
        # Initialize directories
        utils.run_history.init_history_dirs()
        
        # Add a run
        run_id = utils.run_history.add_run("Test Config", 60, "Completed")
        
        # Test that the run was added
        df = utils.run_history.load_run_history()
        print(f"Run history has {len(df)} rows: {df}")
        self.assertFalse(df.empty, "DataFrame should not be empty")
        self.assertEqual(len(df), 1, f"Expected 1 row, got {len(df)}: {df}")
        self.assertEqual(df.iloc[0]['run_id'], run_id)
        
        # Get the run
        run = utils.run_history.get_run(run_id)
        self.assertIsNotNone(run)
        self.assertEqual(run['run_id'], run_id)
        self.assertEqual(run['config_name'], "Test Config")
        
        # Update status
        self.assertTrue(utils.run_history.update_run_status(run_id, "Failed"))
        
        # Check update
        run = utils.run_history.get_run(run_id)
        self.assertEqual(run['status'], "Failed")
        
        # Save details
        details = {
            'model_config': {
                'num_qubits': 4, 
                'num_layers': 2
            },
            'metrics': {
                'accuracy': 0.85
            }
        }
        self.assertTrue(utils.run_history.save_run_details(run_id, details))
        
        # Verify the JSON file was created
        json_file = os.path.join(self.test_details_dir, f"run_{run_id}.json")
        self.assertTrue(os.path.exists(json_file), f"JSON file {json_file} should exist but was not found")
        
        # Load details
        loaded_details = utils.run_history.load_run_details(run_id)
        self.assertIsNotNone(loaded_details)
        self.assertEqual(loaded_details['model_config']['num_qubits'], 4)
        
        # Delete run
        self.assertTrue(utils.run_history.delete_run(run_id))
        
        # Verify deletion
        df = utils.run_history.load_run_history()
        self.assertTrue(df[df['run_id'] == run_id].empty)
    
    def test_corrupted_json(self):
        """Test handling of corrupted JSON files."""
        # Initialize directories
        utils.run_history.init_history_dirs()
        
        # Add a run
        run_id = utils.run_history.add_run("Test Corrupted JSON", 120, "Completed")
        
        # Save details
        details = {
            'model_config': {
                'num_qubits': 6, 
                'num_layers': 3
            },
            'metrics': {
                'accuracy': 0.92
            }
        }
        
        # Save details
        result = utils.run_history.save_run_details(run_id, details)
        self.assertTrue(result, "save_run_details should return True")
        
        # Verify the JSON file was created
        json_file = os.path.join(self.test_details_dir, f"run_{run_id}.json")
        print(f"Looking for JSON file at: {json_file}")
        
        if not os.path.exists(json_file):
            # List directory contents for debugging
            parent_dir = os.path.dirname(json_file)
            print(f"Contents of {parent_dir}: {os.listdir(parent_dir) if os.path.exists(parent_dir) else 'Directory not found'}")
        
        self.assertTrue(os.path.exists(json_file), f"JSON file {json_file} should exist but was not found")
        
        # Corrupt the JSON file with completely invalid content
        with open(json_file, 'w') as f:
            f.write("THIS IS NOT JSON AT ALL")
        
        # Now try to load the corrupted JSON
        loaded_details = utils.run_history.load_run_details(run_id)
        
        # Even with corrupted JSON, we should get something (fallback mechanism)
        self.assertIsNotNone(loaded_details, "Should get recovery data for corrupted JSON")
        self.assertTrue('recovery_note' in loaded_details, "Recovery data should have a recovery note")
    
    def test_missing_files(self):
        """Test handling of missing files."""
        # Initialize directories
        utils.run_history.init_history_dirs()
        
        # Add a run without saving details
        run_id = utils.run_history.add_run("Test Missing Files", 180, "Completed")
        
        # Make sure there are no detail files
        json_file = os.path.join(self.test_details_dir, f"run_{run_id}.json")
        pkl_file = os.path.join(self.test_details_dir, f"run_{run_id}.pkl")
        
        if os.path.exists(json_file):
            os.remove(json_file)
        if os.path.exists(pkl_file):
            os.remove(pkl_file)
        
        # Check that the files don't exist
        self.assertFalse(os.path.exists(json_file), f"JSON file {json_file} should NOT exist")
        self.assertFalse(os.path.exists(pkl_file), f"Pickle file {pkl_file} should NOT exist")
        
        # Try to load non-existent details
        loaded_details = utils.run_history.load_run_details(run_id)
        
        # With our new implementation, we create a minimal recovery for existing runs
        # So we'll check for recovery_note instead
        self.assertIsNotNone(loaded_details, "Should get minimal recovery data for missing files")
        self.assertTrue('recovery_note' in loaded_details, "Recovery data should have a recovery note")
    
    def test_recovery_mechanism(self):
        """Test the recovery mechanism for corrupted files."""
        # Initialize directories
        utils.run_history.init_history_dirs()
        
        # Add a run
        run_id = utils.run_history.add_run("Test Recovery", 240, "In Progress")
        
        # Save details
        details = {
            'model_config': {
                'num_qubits': 8, 
                'ansatz_name': 'Test Ansatz'
            },
            'data_info': {
                'samples': 1000
            }
        }
        utils.run_history.save_run_details(run_id, details)
        
        # Verify files exist before corrupting
        json_file = os.path.join(self.test_details_dir, f"run_{run_id}.json")
        pkl_file = os.path.join(self.test_details_dir, f"run_{run_id}.pkl")
        
        self.assertTrue(os.path.exists(json_file), f"JSON file {json_file} should exist but was not found")
        
        # Completely corrupt the JSON file
        with open(json_file, 'w') as f:
            # Make completely corrupted JSON
            f.write("{ THIS IS NOT VALID JSON AT ALL - NO REPAIR POSSIBLE }")
        
        # Delete pickle file to ensure it can't be used as fallback
        if os.path.exists(pkl_file):
            os.remove(pkl_file)
        
        # Try to load the corrupted details - should create minimal recovery data
        loaded_details = utils.run_history.load_run_details(run_id)
        
        # Verify recovery data was generated
        self.assertIsNotNone(loaded_details, "Recovery data should not be None")
        self.assertTrue('recovery_note' in loaded_details, "Recovery data should have a recovery note")
        
        # The recovered data might be partial, so we can't make strong assertions about its content
        # Instead, just check that it's a valid dictionary with a recovery note
        self.assertIsInstance(loaded_details, dict)
        self.assertIn('recovery_note', loaded_details)
        
        # If run_id is present, make sure it's correct
        if 'run_id' in loaded_details:
            self.assertEqual(loaded_details['run_id'], run_id)
            
        # If config_name is present, make sure it matches what we saved
        if 'config_name' in loaded_details:
            self.assertEqual(loaded_details['config_name'], "Test Recovery")
    
    def test_numpy_conversion(self):
        """Test conversion of NumPy arrays in details."""
        # Initialize directories
        utils.run_history.init_history_dirs()
        
        # Add a run
        run_id = utils.run_history.add_run("Test NumPy", 60, "Completed")
        
        # Create details with NumPy arrays
        numpy_array = np.array([1, 2, 3, 4])
        details = {
            'array': numpy_array,
            'nested': {
                'array': numpy_array
            }
        }
        
        # Save details
        utils.run_history.save_run_details(run_id, details)
        
        # Verify the JSON file was created
        json_file = os.path.join(self.test_details_dir, f"run_{run_id}.json")
        self.assertTrue(os.path.exists(json_file), f"JSON file {json_file} should exist but was not found")
        
        # Load details
        loaded_details = utils.run_history.load_run_details(run_id)
        
        # Verify arrays were converted to lists
        self.assertIsNotNone(loaded_details)
        self.assertEqual(loaded_details['array'], [1, 2, 3, 4])
        self.assertEqual(loaded_details['nested']['array'], [1, 2, 3, 4])
    
    def test_complex_data(self):
        """Test handling of complex data structures."""
        # Initialize directories
        utils.run_history.init_history_dirs()
        
        # Add a run
        run_id = utils.run_history.add_run("Test Complex", 300, "Completed")
        
        # Create a complex nested structure
        details = {
            'training_history': {
                'loss': [0.9, 0.8, 0.7, 0.6, 0.5],
                'entanglement': [None, 0.2, 0.25, np.nan, 0.4]
            },
            'model_config': {
                'layers': [
                    {'type': 'dense', 'units': 64},
                    {'type': 'quantum', 'qubits': 4}
                ]
            },
            'metrics': {
                'confusion_matrix': np.array([[10, 2], [3, 15]])
            }
        }
        
        # Save details
        utils.run_history.save_run_details(run_id, details)
        
        # Verify the JSON file was created
        json_file = os.path.join(self.test_details_dir, f"run_{run_id}.json")
        self.assertTrue(os.path.exists(json_file), f"JSON file {json_file} should exist but was not found")
        
        # Load details
        loaded_details = utils.run_history.load_run_details(run_id)
        
        # Verify complex structure
        self.assertIsNotNone(loaded_details)
        self.assertEqual(loaded_details['training_history']['loss'], [0.9, 0.8, 0.7, 0.6, 0.5])
        self.assertEqual(len(loaded_details['model_config']['layers']), 2)
        self.assertEqual(loaded_details['metrics']['confusion_matrix'], [[10, 2], [3, 15]])

if __name__ == '__main__':
    unittest.main() 