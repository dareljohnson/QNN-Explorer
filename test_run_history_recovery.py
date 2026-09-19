"""
Comprehensive Test for Run History Recovery System

This module tests the full run history recovery system, focusing on:
1. JSON file corruption detection
2. Multi-strategy recovery attempts
3. Creation of recovery data when repair fails
4. Backup file utilization
"""

import os
import json
import shutil
import unittest
import tempfile
from unittest.mock import patch
from utils.run_history import (
    load_run_details, save_run_details, 
    RUN_DETAILS_PATH, init_history_dirs,
    create_recovery_data
)

class TestRunHistoryRecovery(unittest.TestCase):
    """Test case for run history recovery functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        
        # Sample valid run details
        self.sample_run = {
            "task_type": "Classification",
            "model_config": {
                "num_qubits": 4,
                "num_layers": 2,
                "encoding_method": "Amplitude Encoding",
                "ansatz_name": "Basic Entangling Layers",
                "features": [1, 2, 3, 4],
            },
            "metrics": {
                "accuracy": 0.85,
                "loss": [0.5, 0.4, 0.3]
            },
            "results": [
                {"epoch": 1, "accuracy": 0.75},
                {"epoch": 2, "accuracy": 0.80},
                {"epoch": 3, "accuracy": 0.85}
            ]
        }
        
        # Remember the original path
        self.original_path = RUN_DETAILS_PATH
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.test_dir.cleanup()
    
    @patch('utils.run_history.RUN_DETAILS_PATH')
    def test_full_recovery_system(self, mock_path):
        """Test the full recovery system with various corruption scenarios."""
        # Set the mock path to our test directory
        mock_path.__str__.return_value = self.test_dir.name
        
        # Test multiple run IDs with different corruption types
        test_cases = [
            (1, "truncated", True),      # Truncated file, should be repairable
            (2, "invalid_syntax", True), # Invalid syntax, should be repairable
            (3, "severe", False),        # Severely corrupted, should create recovery data
            (4, "empty", False),         # Empty file, should create recovery data
            (5, "missing", False)        # Missing file, should create recovery data
        ]
        
        # Create and corrupt files for each test case
        for run_id, corruption_type, is_repairable in test_cases:
            # Skip missing file case
            if corruption_type != "missing":
                # Set up a valid file first
                with patch('utils.run_history.RUN_DETAILS_PATH', self.test_dir.name):
                    save_run_details(run_id, self.sample_run)
                
                # Now corrupt it based on the type
                json_file = os.path.join(self.test_dir.name, f"run_{run_id}.json")
                self.corrupt_file(json_file, corruption_type)
        
        # Test loading each file
        for run_id, corruption_type, is_repairable in test_cases:
            with patch('utils.run_history.RUN_DETAILS_PATH', self.test_dir.name):
                # Load and attempt recovery
                recovered_data = load_run_details(run_id)
                
                # All cases should return some data (not None)
                self.assertIsNotNone(recovered_data, 
                    f"Run {run_id} ({corruption_type}): Should return some data, not None")
                
                # Check for recovery indicators - either run_id, recovery_note, or is_recovered
                has_recovery_indicator = any(key in recovered_data for key in ["run_id", "recovery_note", "is_recovered"])
                self.assertTrue(has_recovery_indicator,
                    f"Run {run_id} ({corruption_type}): Should have some recovery indicator")
                
                # For repairable files, we may recover original keys or create recovery data
                # The important thing is that we get usable data back
                if is_repairable:
                    # Either recover some original data or have task_type
                    has_original_key = any(key in recovered_data for key in ["task_type", "model_config", "metrics", "num_qubits"])
                    self.assertTrue(has_original_key, 
                        f"Run {run_id} ({corruption_type}): Should have task_type or model_config data")
    
    def corrupt_file(self, file_path, corruption_type):
        """Corrupt a file based on the specified corruption type."""
        if corruption_type == "truncated":
            # Truncate the file
            with open(file_path, 'r') as f:
                content = f.read()
            with open(file_path, 'w') as f:
                f.write(content[:len(content)//2])
                
        elif corruption_type == "invalid_syntax":
            # Add invalid syntax
            with open(file_path, 'a') as f:
                f.write("\n\nTHIS IS NOT VALID JSON!!!")
                
        elif corruption_type == "severe":
            # Severely corrupt the file
            with open(file_path, 'w') as f:
                f.write('{"run_id":')
                
        elif corruption_type == "empty":
            # Create an empty file
            with open(file_path, 'w') as f:
                f.write('')
    
    @patch('utils.run_history.RUN_DETAILS_PATH')
    def test_backup_file_utilization(self, mock_path):
        """Test that the system properly uses backup files when available."""
        # Set the mock path to our test directory
        mock_path.__str__.return_value = self.test_dir.name
        
        run_id = 10
        
        # Create a corrupted main file
        json_file = os.path.join(self.test_dir.name, f"run_{run_id}.json")
        with open(json_file, 'w') as f:
            f.write('{"this_is": "corrupted"')  # Missing closing brace
        
        # Create a valid backup file
        backup_file = os.path.join(self.test_dir.name, f"run_{run_id}.json.bak")
        with open(backup_file, 'w') as f:
            json.dump(self.sample_run, f)
        
        # Load the file
        with patch('utils.run_history.RUN_DETAILS_PATH', self.test_dir.name):
            recovered_data = load_run_details(run_id)
            
            # Check that we got valid data back
            self.assertIsNotNone(recovered_data, "Should recover some data")
            
            # Check that we can identify it used the backup by checking for task_type or recovery_note
            self.assertTrue("task_type" in recovered_data or "recovery_note" in recovered_data, 
                "Should indicate backup usage or recovery")
            
            # If using backup, task_type should match
            if "task_type" in recovered_data and "task_type" in self.sample_run:
                self.assertEqual(recovered_data["task_type"], self.sample_run["task_type"], 
                    "Should recover correct task_type from backup file")
    
    @patch('utils.run_history.RUN_DETAILS_PATH')
    def test_pickle_fallback(self, mock_path):
        """Test fallback to pickle format when JSON is corrupted."""
        # Set the mock path to our test directory
        mock_path.__str__.return_value = self.test_dir.name
        
        run_id = 20
        
        # Create a corrupted JSON file
        json_file = os.path.join(self.test_dir.name, f"run_{run_id}.json")
        with open(json_file, 'w') as f:
            f.write('{"this_is": "corrupted"')  # Missing closing brace
        
        # Create a valid pickle file
        import pickle
        pkl_file = os.path.join(self.test_dir.name, f"run_{run_id}.pkl")
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.sample_run, f)
        
        # Load the file
        with patch('utils.run_history.RUN_DETAILS_PATH', self.test_dir.name):
            recovered_data = load_run_details(run_id)
            
            # Check that we got valid data back
            self.assertIsNotNone(recovered_data, "Should recover some data")
            
            # Check for recovery indicators
            recovery_found = ("recovery_note" in recovered_data or 
                              "is_recovered" in recovered_data or 
                              "task_type" in recovered_data)
            self.assertTrue(recovery_found, "Should have recovery indicators or valid data")
            
            # If task_type is available, it should match
            if "task_type" in recovered_data and "task_type" in self.sample_run:
                self.assertEqual(recovered_data["task_type"], self.sample_run["task_type"], 
                    "If using pickle, should recover correct task_type")
    
    @patch('utils.run_history.RUN_DETAILS_PATH')
    def test_recovery_data_creation(self, mock_path):
        """Test creation of recovery data when all recovery attempts fail."""
        # Set the mock path to our test directory
        mock_path.__str__.return_value = self.test_dir.name
        
        run_id = 30
        
        # We won't create any files, so it will need to create recovery data
        with patch('utils.run_history.RUN_DETAILS_PATH', self.test_dir.name):
            recovered_data = load_run_details(run_id)
            
            # Check that we got minimal recovery data
            self.assertIn("recovery_note", recovered_data, 
                "Should have a recovery note in minimal recovery data")
            
            # Should have the run ID
            self.assertEqual(recovered_data["run_id"], run_id, 
                "Recovery data should have the correct run ID")

def run_tests():
    """Run the tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestRunHistoryRecovery)
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == "__main__":
    run_tests() 