"""Tests for the visualization module."""

import unittest
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the visualization module
try:
    from utils.visualization import (
        plot_circuit,
        plot_state_vector,
        plot_training_history,
        plot_entanglement,
        plot_training_accuracy,
        plot_regression_predictions
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

@unittest.skipUnless(VISUALIZATION_AVAILABLE, "Visualization module not available")
class TestVisualization(unittest.TestCase):
    """Test case for the visualization module."""
    
    def setUp(self):
        """Set up test resources."""
        # Create sample data for testing
        self.y_true_regression = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred_regression = np.array([1.2, 2.1, 2.9, 4.2, 5.2])
    
    def test_plot_regression_predictions(self):
        """Test regression predictions plotting."""
        # Skip the test if matplotlib is not available
        try:
            fig = plot_regression_predictions(self.y_true_regression, self.y_pred_regression)
            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)
        except ImportError:
            import sys
            if sys.version_info >= (3, 0):
                self.skipTest("matplotlib not available")
            else:
                print("Skipping test_regression_predictions_plot as matplotlib is not available")

if __name__ == '__main__':
    unittest.main() 