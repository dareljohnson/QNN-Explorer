"""
Unit tests for the Run History module's JSON repair mechanisms.

This module tests the JSON file handling and recovery mechanisms
implemented in the utils/run_history.py module.
"""

import os
import json
import shutil
import unittest
from unittest.mock import patch, mock_open
from utils.run_history import (
    load_run_details, save_run_details, 
    RUN_DETAILS_PATH, init_history_dirs
)

class TestRunHistoryJsonRepair(unittest.TestCase):
    """Test case for run history JSON repair functionality."""
    
    def setUp(self):
        """Set up test fixtures, if any."""
        # Create a test directory for run history files
        self.test_dir = os.path.join("tests", "test_run_details")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Sample valid run details
        self.sample_run = {
            "task_type": "Classification",
            "model_config": {
                "num_qubits": 4,
                "num_layers": 2,
                "ansatz_name": "Basic Entangling Layers",
                "features": [1, 2, 3, 4],
                "complex_data": {"key1": [1, 2], "key2": "value"}
            },
            "metrics": {
                "accuracy": 0.85,
                "loss": [0.5, 0.4, 0.3]
            }
        }
        
        # Path where the RUN_DETAILS_PATH originally pointed
        self.original_path = RUN_DETAILS_PATH
    
    def tearDown(self):
        """Tear down test fixtures, if any."""
        # Delete test files
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('utils.run_history.RUN_DETAILS_PATH')
    def test_json_truncation_repair(self, mock_path):
        """Test repairing a truncated JSON file."""
        # Set the mock path to our test directory
        mock_path.__str__.return_value = self.test_dir
        
        # Create sample run
        run_id = 99
        json_file = os.path.join(self.test_dir, f"run_{run_id}.json")
        
        # Save full run details
        with patch('utils.run_history.RUN_DETAILS_PATH', self.test_dir):
            save_run_details(run_id, self.sample_run)
        
        # Verify the file was created
        self.assertTrue(os.path.exists(json_file))
        
        # Now corrupt the file by truncating it
        with open(json_file, 'r') as f:
            content = f.read()
        
        # Truncate at a random position that leaves valid JSON structure
        truncate_pos = content.find('"complex_data":')
        with open(json_file, 'w') as f:
            f.write(content[:truncate_pos])
        
        # Try to load the truncated file
        with patch('utils.run_history.RUN_DETAILS_PATH', self.test_dir):
            recovered_data = load_run_details(run_id)
        
        # Verify we got something back
        self.assertIsNotNone(recovered_data)
        self.assertIn('task_type', recovered_data)
        self.assertIn('model_config', recovered_data)
        
        # Verify recovery note is present
        self.assertIn('recovery_note', recovered_data)
    
    @patch('utils.run_history.RUN_DETAILS_PATH')
    def test_invalid_json_syntax(self, mock_path):
        """Test handling JSON with invalid syntax."""
        # Set the mock path to our test directory
        mock_path.__str__.return_value = self.test_dir
        
        # Create sample run
        run_id = 100
        json_file = os.path.join(self.test_dir, f"run_{run_id}.json")
        
        # Create a file with invalid JSON syntax
        with open(json_file, 'w') as f:
            f.write('{"key": "value", "broken": ')  # Missing closing value and brace
        
        # Try to load the invalid file
        with patch('utils.run_history.RUN_DETAILS_PATH', self.test_dir):
            recovered_data = load_run_details(run_id)
        
        # Verify we got recovery data
        self.assertIsNotNone(recovered_data)
        self.assertIn('recovery_note', recovered_data)
    
    @patch('utils.run_history.RUN_DETAILS_PATH')
    def test_backup_file_fallback(self, mock_path):
        """Test fallback to backup file when main file is corrupted."""
        # Set the mock path to our test directory
        mock_path.__str__.return_value = self.test_dir
        
        # Create sample run
        run_id = 101
        json_file = os.path.join(self.test_dir, f"run_{run_id}.json")
        backup_file = json_file + ".bak"
        
        # Create a backup file with valid data
        os.makedirs(os.path.dirname(backup_file), exist_ok=True)
        with open(backup_file, 'w') as f:
            json.dump(self.sample_run, f)
        
        # Create main file with invalid data
        with open(json_file, 'w') as f:
            f.write('{"this": "is invalid" "json"')
        
        # Try to load the invalid file
        with patch('utils.run_history.RUN_DETAILS_PATH', self.test_dir):
            recovered_data = load_run_details(run_id)
        
        # Verify we got data from backup
        self.assertIsNotNone(recovered_data)
        self.assertEqual(recovered_data['task_type'], self.sample_run['task_type'])
        self.assertIn('recovery_note', recovered_data)

def run_tests():
    """Run the tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    run_tests() 