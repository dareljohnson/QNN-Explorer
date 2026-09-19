#!/usr/bin/env python
"""
Test housing dataset path handling

This test script verifies that housing dataset paths are correctly handled and can be
accessed without permission errors.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import joblib

# Import the path utilities that were created to fix the path issue
sys.path.append('.')
from data.path_utils import ensure_path_sep, make_path

class TestHousingDatasetPaths(unittest.TestCase):
    """Test cases for housing dataset path handling"""
    
    def setUp(self):
        """Set up the test environment with correct paths"""
        # Define the correct paths using our utilities
        self.housing_dir = ensure_path_sep(os.path.join("data", "demo_data", "housing"))
        self.raw_dir = ensure_path_sep(os.path.join(self.housing_dir, "raw"))
        self.processed_dir = ensure_path_sep(os.path.join(self.housing_dir, "processed"))
        
        # Ensure the directories exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def test_directories_exist(self):
        """Test that the housing dataset directories exist"""
        self.assertTrue(os.path.exists(self.housing_dir), f"Housing directory does not exist: {self.housing_dir}")
        self.assertTrue(os.path.exists(self.raw_dir), f"Raw directory does not exist: {self.raw_dir}")
        self.assertTrue(os.path.exists(self.processed_dir), f"Processed directory does not exist: {self.processed_dir}")
    
    def test_write_permissions(self):
        """Test that we have write permissions to the housing dataset directories"""
        # Test writing to raw directory
        test_file_raw = os.path.join(self.raw_dir, "test_file.txt")
        try:
            with open(test_file_raw, 'w') as f:
                f.write("Testing write permissions")
            self.assertTrue(os.path.exists(test_file_raw), f"Failed to write to {test_file_raw}")
        finally:
            if os.path.exists(test_file_raw):
                os.remove(test_file_raw)
        
        # Test writing to processed directory
        test_file_processed = os.path.join(self.processed_dir, "test_file.txt")
        try:
            with open(test_file_processed, 'w') as f:
                f.write("Testing write permissions")
            self.assertTrue(os.path.exists(test_file_processed), f"Failed to write to {test_file_processed}")
        finally:
            if os.path.exists(test_file_processed):
                os.remove(test_file_processed)
    
    def test_dataset_files_exist(self):
        """Test that key dataset files exist or can be created"""
        # Define paths to key files
        csv_path = os.path.join(self.raw_dir, "USA-house-prices.csv")
        train_path = os.path.join(self.processed_dir, "train_housing.csv")
        test_path = os.path.join(self.processed_dir, "test_housing.csv")
        scaler_path = os.path.join(self.processed_dir, "house_price_scaler.joblib")
        encoders_path = os.path.join(self.processed_dir, "house_price_encoders.joblib")
        
        # Check if files exist or create test versions if needed
        if not os.path.exists(csv_path):
            # Create a minimal test CSV file
            print(f"Creating test CSV file: {csv_path}")
            test_data = pd.DataFrame({
                'price': np.random.normal(250000, 50000, 100),
                'bedrooms': np.random.randint(1, 6, 100),
                'bathrooms': np.random.randint(1, 5, 100),
                'sqft_living': np.random.normal(2000, 500, 100),
                'state': np.random.choice(['CA', 'NY', 'TX', 'FL', 'WA'], 100)
            })
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            test_data.to_csv(csv_path, index=False)
        
        self.assertTrue(os.path.exists(csv_path), f"Raw CSV file does not exist: {csv_path}")
        
        # Process the dataset if needed
        if not all(os.path.exists(p) for p in [train_path, test_path, scaler_path, encoders_path]):
            try:
                from data.download_house_prices import create_synthetic_dataset, process_dataset
                print("Creating and processing synthetic housing dataset...")
                create_synthetic_dataset(samples=100)
                process_dataset()
            except Exception as e:
                self.fail(f"Failed to create/process housing dataset: {e}")
        
        # Verify that all required files exist
        self.assertTrue(os.path.exists(train_path), f"Train file does not exist: {train_path}")
        self.assertTrue(os.path.exists(test_path), f"Test file does not exist: {test_path}")
        self.assertTrue(os.path.exists(scaler_path), f"Scaler file does not exist: {scaler_path}")
        self.assertTrue(os.path.exists(encoders_path), f"Encoders file does not exist: {encoders_path}")
    
    def test_load_processed_data(self):
        """Test that we can load the processed data without errors"""
        train_path = os.path.join(self.processed_dir, "train_housing.csv")
        test_path = os.path.join(self.processed_dir, "test_housing.csv")
        scaler_path = os.path.join(self.processed_dir, "house_price_scaler.joblib")
        encoders_path = os.path.join(self.processed_dir, "house_price_encoders.joblib")
        
        # Ensure files exist
        if not all(os.path.exists(p) for p in [train_path, test_path, scaler_path, encoders_path]):
            self.test_dataset_files_exist()
        
        # Try loading each file
        try:
            train_df = pd.read_csv(train_path)
            self.assertGreater(len(train_df), 0, "Train dataset is empty")
            self.assertIn('price', train_df.columns, "Train dataset missing 'price' column")
            
            test_df = pd.read_csv(test_path)
            self.assertGreater(len(test_df), 0, "Test dataset is empty")
            self.assertIn('price', test_df.columns, "Test dataset missing 'price' column")
            
            scaler = joblib.load(scaler_path)
            self.assertIsNotNone(scaler, "Failed to load scaler")
            
            encoders = joblib.load(encoders_path)
            self.assertIsNotNone(encoders, "Failed to load encoders")
            
            print("Successfully loaded all housing dataset files")
        except Exception as e:
            self.fail(f"Error loading processed data: {e}")

if __name__ == "__main__":
    unittest.main() 