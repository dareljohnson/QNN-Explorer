"""
Test script for data loader module.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock, mock_open

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if requirements are available
try:
    import pandas as pd
    import torch
    from io import StringIO
    from data.loader import validate_file_type, get_sample_files
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Required imports not available: {e}")
    IMPORTS_AVAILABLE = False

@unittest.skipUnless(IMPORTS_AVAILABLE, "Required imports not available")
class TestDataLoader(unittest.TestCase):
    """Test case for the data loader module."""
    
    def setUp(self):
        """Set up test resources."""
        # Create a mock file object
        self.mock_file = MagicMock()
        self.mock_file.name = "test.csv"
        self.mock_file.type = "text/csv"
    
    def test_validate_file_type_csv(self):
        """Test validating a CSV file."""
        # Set up mock file as CSV
        self.mock_file.name = "test.csv"
        self.mock_file.type = "text/csv"
        
        # Test validation
        is_valid, error = validate_file_type(self.mock_file, "CSV")
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        # Test invalid CSV type
        self.mock_file.type = "text/plain"
        self.mock_file.name = "test.txt"
        is_valid, error = validate_file_type(self.mock_file, "CSV")
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
    
    def test_validate_file_type_image(self):
        """Test validating an image file."""
        # Set up mock file as image
        self.mock_file.name = "test.jpg"
        self.mock_file.type = "image/jpeg"
        
        # Test validation
        is_valid, error = validate_file_type(self.mock_file, "Images")
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        # Test invalid image type
        self.mock_file.type = "application/pdf"
        is_valid, error = validate_file_type(self.mock_file, "Images")
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
    
    def test_validate_file_type_text(self):
        """Test validating a text file."""
        # Set up mock file as text
        self.mock_file.name = "test.txt"
        self.mock_file.type = "text/plain"
        
        # Test validation
        is_valid, error = validate_file_type(self.mock_file, "Text")
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        # Test validation with octet-stream type (common for .txt files in some browsers)
        self.mock_file.type = "application/octet-stream"
        is_valid, error = validate_file_type(self.mock_file, "Text")
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_validate_file_type_video(self):
        """Test validating a video file."""
        # Set up mock file as video
        self.mock_file.name = "test.mp4"
        self.mock_file.type = "video/mp4"
        
        # Test validation
        is_valid, error = validate_file_type(self.mock_file, "Video")
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        # Test invalid video type
        self.mock_file.type = "image/jpeg"
        is_valid, error = validate_file_type(self.mock_file, "Video")
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
    
    def test_validate_file_type_none(self):
        """Test validating a None file."""
        is_valid, error = validate_file_type(None, "CSV")
        self.assertFalse(is_valid)
        self.assertEqual(error, "No file uploaded")
    
    @patch('os.path.exists')
    @patch('glob.glob')
    def test_get_sample_files(self, mock_glob, mock_exists):
        """Test getting sample files."""
        # Mock os.path.exists to return True
        mock_exists.return_value = True
        
        # Mock glob.glob to return a list of files
        mock_glob.return_value = [
            "/path/to/file1.csv",
            "/path/to/file2.csv"
        ]
        
        # Test getting CSV files
        files = get_sample_files("CSV", "/path/to")
        self.assertEqual(len(files), 2)
        self.assertEqual(files[0], "/path/to/file1.csv")
        self.assertEqual(files[1], "/path/to/file2.csv")
        
        # Test directory doesn't exist
        mock_exists.return_value = False
        files = get_sample_files("CSV", "/nonexistent/path")
        self.assertEqual(len(files), 0)
        
        # Test unknown data type
        mock_exists.return_value = True
        files = get_sample_files("Unknown", "/path/to")
        self.assertEqual(len(files), 0)

if __name__ == '__main__':
    unittest.main() 