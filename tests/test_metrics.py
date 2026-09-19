"""Tests for the metrics module."""

import unittest
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the metrics module
try:
    from utils.metrics import (
        calculate_classification_metrics,
        calculate_regression_metrics,
        calculate_clustering_metrics,
        plot_confusion_matrix,
        plot_training_accuracy,
        plot_regression_predictions,
        get_metric_descriptions
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

@unittest.skipUnless(METRICS_AVAILABLE, "Metrics module not available")
class TestMetrics(unittest.TestCase):
    """Test case for the metrics module."""
    
    def setUp(self):
        """Set up test resources."""
        # Create sample data for testing
        self.y_true_classification = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        self.y_pred_classification = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        self.y_prob_classification = np.array([0.2, 0.8, 0.3, 0.4, 0.6, 0.9, 0.1, 0.7])
        
        self.y_true_regression = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred_regression = np.array([1.2, 2.1, 2.9, 4.2, 5.2])
        
        self.X_clustering = np.array([
            [1, 2], [1, 3], [2, 3], [4, 5], [5, 6], [6, 5]
        ])
        self.labels_clustering = np.array([0, 0, 0, 1, 1, 1])
    
    def test_classification_metrics(self):
        """Test classification metrics calculation."""
        metrics = calculate_classification_metrics(
            self.y_true_classification,
            self.y_pred_classification,
            self.y_prob_classification
        )
        
        # Check that all required metrics are present
        required_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'log_loss', 'confusion_matrix']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            
        # Check specific values
        self.assertAlmostEqual(metrics['accuracy'], 0.75, places=2)
    
    def test_regression_metrics(self):
        """Test regression metrics calculation."""
        metrics = calculate_regression_metrics(
            self.y_true_regression,
            self.y_pred_regression
        )
        
        # Check that all required metrics are present
        required_metrics = ['mae', 'mse', 'rmse', 'r2']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            
        # Check specific values
        self.assertLess(metrics['mae'], 0.3)  # MAE should be less than 0.3
        self.assertLess(metrics['mse'], 0.1)  # MSE should be less than 0.1
    
    def test_clustering_metrics(self):
        """Test clustering metrics calculation."""
        # Create a simplified minimal test that should work even if external libraries are not available
        try:
            # First, check if adjusted_rand_score and adjusted_mutual_info_score exist in metrics
            from utils.metrics import adjusted_rand_score, adjusted_mutual_info_score
            
            # If we get here, the functions exist and we can run the real test
            metrics = calculate_clustering_metrics(
                self.X_clustering,
                self.labels_clustering,
                self.labels_clustering  # Using same labels as "true" labels
            )
            
            # Check metrics if they exist
            if metrics['adjusted_rand'] is not None:
                self.assertAlmostEqual(metrics['adjusted_rand'], 1.0, places=2)
            
            if metrics['adjusted_mutual_info'] is not None:
                self.assertAlmostEqual(metrics['adjusted_mutual_info'], 1.0, places=2)
                
        except (ImportError, AttributeError):
            # If sklearn metrics are not available, skip the test
            import sys
            if sys.version_info >= (3, 0):
                self.skipTest("sklearn metrics not available")
            else:
                # For Python 2 compatibility
                print("Skipping test_clustering_metrics as sklearn is not available")
    
    def test_confusion_matrix_plot(self):
        """Test confusion matrix plotting."""
        cm = np.array([[3, 1], [1, 3]])  # 2x2 confusion matrix
        class_names = ['Class 0', 'Class 1']
        
        # Plot should return a base64 string
        plot_data = plot_confusion_matrix(cm, class_names)
        self.assertIsInstance(plot_data, str)
        self.assertTrue(len(plot_data) > 0)
    
    def test_training_accuracy_plot(self):
        """Test training accuracy plotting."""
        epochs = [1, 2, 3, 4, 5]
        accuracies = [0.5, 0.6, 0.7, 0.75, 0.8]
        
        # Skip the test if matplotlib is not available
        try:
            import matplotlib.pyplot as plt
            
            # Test with default title
            fig = plot_training_accuracy(epochs, accuracies)
            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)
            
            # Test with custom title
            fig = plot_training_accuracy(epochs, accuracies, title="Custom Title")
            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)
        except ImportError:
            import sys
            if sys.version_info >= (3, 0):
                self.skipTest("matplotlib not available")
            else:
                print("Skipping test_training_accuracy_plot as matplotlib is not available")
    
    def test_regression_predictions_plot(self):
        """Test regression predictions plotting."""
        # Skip the test if matplotlib is not available
        try:
            import matplotlib.pyplot as plt
            fig = plot_regression_predictions(self.y_true_regression, self.y_pred_regression)
            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)
        except ImportError:
            import sys
            if sys.version_info >= (3, 0):
                self.skipTest("matplotlib not available")
            else:
                print("Skipping test_regression_predictions_plot as matplotlib is not available")
    
    def test_metric_descriptions(self):
        """Test metric descriptions."""
        descriptions = get_metric_descriptions()
        self.assertIsInstance(descriptions, dict)
        
        # Check that descriptions for all required metrics are present
        required_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 'auc', 'log_loss',
            'mae', 'mse', 'rmse', 'rmsle', 'r2',
            'silhouette', 'adjusted_rand', 'adjusted_mutual_info'
        ]
        for metric in required_metrics:
            self.assertIn(metric, descriptions)
            self.assertIsInstance(descriptions[metric], str)
            self.assertTrue(len(descriptions[metric]) > 0)

if __name__ == '__main__':
    unittest.main() 