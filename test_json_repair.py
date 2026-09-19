"""
Unit tests for the JSON repair module.

This module tests the JSON repair functionality implemented in the utils/json_repair.py module.
"""

import os
import unittest
import tempfile
import json
from utils.json_repair import (
    repair_json_file,
    repair_truncated_file,
    repair_unbalanced_braces,
    extract_complete_objects,
    repair_trailing_commas,
    repair_missing_quotes
)

class TestJsonRepair(unittest.TestCase):
    """Test case for JSON repair functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample valid JSON data
        self.sample_data = {
            "name": "Quantum Model Test",
            "config": {
                "num_qubits": 4,
                "depth": 2,
                "learning_rate": 0.01
            },
            "metrics": {
                "accuracy": 0.85,
                "loss": [0.5, 0.4, 0.3, 0.2, 0.1]
            },
            "results": [
                {"epoch": 1, "loss": 0.5, "accuracy": 0.7},
                {"epoch": 2, "loss": 0.3, "accuracy": 0.8},
                {"epoch": 3, "loss": 0.2, "accuracy": 0.85}
            ]
        }
        
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary directory
        self.test_dir.cleanup()
    
    def create_test_file(self, content):
        """Create a test file with the given content."""
        # Create a temporary file
        file_path = os.path.join(self.test_dir.name, "test.json")
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path
    
    def test_repair_truncated_file(self):
        """Test repairing a truncated JSON file."""
        # Create a valid JSON string
        json_str = json.dumps(self.sample_data)
        
        # Truncate the string at different positions
        truncations = [
            json_str[:json_str.find('"metrics"')],  # Cut in the middle of an object
            json_str[:json_str.find('[0.5, 0.4')],  # Cut in the middle of an array
            json_str[:json_str.find('"results"')],  # Cut before a key
        ]
        
        for i, truncated in enumerate(truncations):
            # Test the repair function
            success, repaired_data, info = repair_truncated_file(truncated)
            
            # Check if repair was successful
            self.assertTrue(success, f"Repair should succeed for truncation {i+1}")
            
            # Check if the repaired data is valid JSON
            self.assertIsNotNone(repaired_data, f"Repaired data should not be None for truncation {i+1}")
            
            # For truncated files, we can't always recover all data, but we should have at least some keys
            common_keys = set(repaired_data.keys()) & set(self.sample_data.keys())
            self.assertGreater(len(common_keys), 0, f"Repaired data should have at least one key for truncation {i+1}")
    
    def test_repair_unbalanced_braces(self):
        """Test repairing JSON with unbalanced braces."""
        # Create JSON strings with unbalanced braces
        unbalanced_samples = [
            '{"name": "Test", "values": [1, 2, 3',  # Missing closing bracket and brace
            '{"config": {"nested": true',  # Missing nested closing brace and outer brace
            '{"items": ["a", "b", {"key": "value"',  # Complex missing closures
        ]
        
        for i, sample in enumerate(unbalanced_samples):
            # Test the repair function
            success, repaired_data, info = repair_unbalanced_braces(sample)
            
            # Check if repair was successful
            self.assertTrue(success, f"Repair should succeed for unbalanced sample {i+1}")
            
            # Check if the repaired data is valid JSON
            self.assertIsNotNone(repaired_data, f"Repaired data should not be None for unbalanced sample {i+1}")
    
    def test_extract_complete_objects(self):
        """Test extracting complete objects from corrupted JSON."""
        # Create samples with complete objects buried in corrupted data
        samples = [
            'garbage{"valid": true}more garbage',  # Complete object with garbage
            '{"obj1": true}{"obj2": false}',  # Multiple objects
            '{"name": "test"} # Invalid comment',  # Object with trailing invalid content
        ]
        
        for i, sample in enumerate(samples):
            # Test the extraction function
            success, extracted_data, info = extract_complete_objects(sample)
            
            # Check if extraction was successful
            self.assertTrue(success, f"Extraction should succeed for sample {i+1}")
            
            # Check if the extracted data is valid JSON
            self.assertIsNotNone(extracted_data, f"Extracted data should not be None for sample {i+1}")
    
    def test_repair_trailing_commas(self):
        """Test repairing JSON with trailing commas."""
        # Create samples with trailing commas
        samples = [
            '{"name": "test", "items": [1, 2, 3,], "valid": true,}',  # Trailing commas
            '{"array": [1, 2, 3,]}',  # Trailing comma in array
            '{"obj": {"nested": true,},}',  # Multiple trailing commas
        ]
        
        for i, sample in enumerate(samples):
            # Test the repair function
            success, repaired_data, info = repair_trailing_commas(sample)
            
            # Check if repair was successful
            self.assertTrue(success, f"Repair should succeed for trailing comma sample {i+1}")
            
            # Check if the repaired data is valid JSON
            self.assertIsNotNone(repaired_data, f"Repaired data should not be None for trailing comma sample {i+1}")
    
    def test_repair_missing_quotes(self):
        """Test repairing JSON with missing quotes around keys."""
        # Create samples with missing quotes
        samples = [
            '{name: "test", items: [1, 2, 3]}',  # Missing quotes around keys
            '{config: {nested: true}}',  # Missing quotes in nested object
            '{mixed: "test", "valid": true}',  # Mixed quoted and unquoted
        ]
        
        for i, sample in enumerate(samples):
            # Test the repair function
            success, repaired_data, info = repair_missing_quotes(sample)
            
            # Check if repair was successful
            self.assertTrue(success, f"Repair should succeed for missing quotes sample {i+1}")
            
            # Check if the repaired data is valid JSON
            self.assertIsNotNone(repaired_data, f"Repaired data should not be None for missing quotes sample {i+1}")
    
    def test_repair_json_file(self):
        """Test the main repair_json_file function with a file."""
        # Create a truncated JSON file
        json_str = json.dumps(self.sample_data)
        truncated = json_str[:json_str.find('"results"')]
        file_path = self.create_test_file(truncated)
        
        # Test the repair function
        success, repaired_data, info = repair_json_file(file_path)
        
        # Check if repair was successful
        self.assertTrue(success, "File repair should succeed")
        
        # Check if the repaired data is valid JSON
        self.assertIsNotNone(repaired_data, "Repaired data should not be None")
        
        # Check that a backup was created
        backup_files = [f for f in os.listdir(self.test_dir.name) if f.endswith('.bak')]
        self.assertGreaterEqual(len(backup_files), 1, "A backup file should be created")
    
    def test_multiple_repair_strategies(self):
        """Test that the repair system tries multiple strategies."""
        # Create a complex corrupted JSON file that needs multiple repair strategies
        corrupted = '{unquoted_key: "value", "items": [1, 2, 3,], "nested": {"key": "value",}'
        file_path = self.create_test_file(corrupted)
        
        # Test the repair function
        success, repaired_data, info = repair_json_file(file_path)
        
        # Check if repair was successful
        self.assertTrue(success, "Complex repair should succeed")
        
        # Check if the repaired data is valid JSON
        self.assertIsNotNone(repaired_data, "Repaired data should not be None")
        
        # The repaired data should have a repair_info key
        if isinstance(repaired_data, dict):
            self.assertIn('repair_info', repaired_data, "Repair info should be added to the data")
    
    def test_severely_corrupted_file(self):
        """Test handling of a severely corrupted file."""
        # Create a severely corrupted file
        corrupted = 'This is not JSON at all {{ invalid ][ data'
        file_path = self.create_test_file(corrupted)
        
        # Test the repair function
        success, repaired_data, info = repair_json_file(file_path)
        
        # Check the result - it's okay if repair fails for extremely corrupted files
        if not success:
            self.assertIn("All repair strategies failed", info, "Should report failure of all strategies")
        else:
            self.assertIsNotNone(repaired_data, "If repair succeeded, data should not be None")

def run_tests():
    """Run the tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    run_tests() 