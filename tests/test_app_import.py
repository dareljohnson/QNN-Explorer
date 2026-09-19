"""Test if the app can import the required functions from visualization."""

import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_app_import():
    try:
        # This import pattern should match the one in app.py
        from utils.visualization import (
            plot_circuit, 
            plot_state_vector, 
            plot_training_history, 
            plot_entanglement, 
            plot_training_accuracy, 
            plot_regression_predictions
        )
        print(f"SUCCESS: All imports from utils.visualization work for the app context.")
        return True
    except ImportError as e:
        print(f"ERROR: Import failed - {e}")
        return False

if __name__ == "__main__":
    test_app_import() 