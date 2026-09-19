"""
Unit tests for regression target scaling functionality.
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
import torch
import joblib
from unittest.mock import patch, MagicMock

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestRegressionScaling(unittest.TestCase):
    """Test case for the regression target scaling functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create a temporary directory for test scalers
        self.test_scaler_dir = os.path.join("data", "scalers", "test")
        os.makedirs(self.test_scaler_dir, exist_ok=True)
        
        # Create a sample dataframe with housing price data
        self.df = pd.DataFrame({
            'sqft': [1500, 2000, 2500, 3000, 3500],
            'bedrooms': [2, 3, 3, 4, 5],
            'bathrooms': [1.5, 2.0, 2.5, 3.0, 3.5],
            'year': [1990, 2000, 2010, 2015, 2020],
            'price': [250000, 350000, 450000, 550000, 650000]  # High values that would cause large MSE
        })
        
        self.feature_cols = ['sqft', 'bedrooms', 'bathrooms', 'year']
        self.label_col = 'price'
        self.scaler_name = "test_scaler.joblib"
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove test scaler files
        test_scaler_path = os.path.join("data", "scalers", self.scaler_name)
        test_target_scaler_path = os.path.join("data", "scalers", f"target_{self.scaler_name}")
        
        if os.path.exists(test_scaler_path):
            os.remove(test_scaler_path)
        if os.path.exists(test_target_scaler_path):
            os.remove(test_target_scaler_path)
    
    def test_target_scaling_for_regression(self):
        """Test that regression target values are properly scaled."""
        from data.preprocessing import preprocess_csv
        
        # Process the data
        features, labels = preprocess_csv(
            self.df, 
            self.feature_cols, 
            self.label_col,
            scaler_name=self.scaler_name,
            is_prediction=False
        )
        
        # Verify features and labels are not None
        self.assertIsNotNone(features)
        self.assertIsNotNone(labels)
        
        # Check that features tensor has the right shape
        self.assertEqual(features.shape, (5, 4))
        
        # Check that labels tensor has the right shape
        self.assertEqual(labels.shape, (5,))
        
        # Verify labels are scaled with mean close to 0 and std close to 1
        self.assertAlmostEqual(labels.mean().item(), 0.0, delta=0.1)
        self.assertAlmostEqual(labels.std().item(), 1.0, delta=0.2)
        
        # Verify target scaler was created
        target_scaler_path = os.path.join("data", "scalers", f"target_{self.scaler_name}")
        self.assertTrue(os.path.exists(target_scaler_path))
        
        # Check the scaler properties
        target_scaler = joblib.load(target_scaler_path)
        # Target scaler mean should be close to the mean of the prices
        self.assertAlmostEqual(target_scaler.mean_[0], self.df['price'].mean(), delta=1.0)
    
    def test_target_scaling_for_prediction(self):
        """Test that scaling works consistently for prediction."""
        from data.preprocessing import preprocess_csv
        
        # First process as training data to create scalers
        preprocess_csv(
            self.df, 
            self.feature_cols, 
            self.label_col,
            scaler_name=self.scaler_name,
            is_prediction=False
        )
        
        # Now create a prediction dataframe with one row
        pred_df = pd.DataFrame({
            'sqft': [2200],
            'bedrooms': [3],
            'bathrooms': [2.5],
            'year': [2005],
            'price': [400000]  # This is what we want to predict
        })
        
        # Process as prediction data
        pred_features, pred_labels = preprocess_csv(
            pred_df, 
            self.feature_cols, 
            self.label_col,
            scaler_name=self.scaler_name,
            is_prediction=True
        )
        
        # Verify prediction features and labels are not None
        self.assertIsNotNone(pred_features)
        self.assertIsNotNone(pred_labels)
        
        # Check that prediction tensor has the right shape
        self.assertEqual(pred_features.shape, (1, 4))
        
        # Load target scaler to verify prediction scaling is correct
        target_scaler_path = os.path.join("data", "scalers", f"target_{self.scaler_name}")
        target_scaler = joblib.load(target_scaler_path)
        
        # Calculate expected scaled value manually
        expected_scaled = target_scaler.transform(np.array([[400000]]))[0][0]
        
        # Check that scaled value matches expectation
        self.assertAlmostEqual(pred_labels[0].item(), expected_scaled, delta=0.001)

if __name__ == '__main__':
    unittest.main() 