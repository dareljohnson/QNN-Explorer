"""Simple test to check if imports work correctly."""

import os
import sys
import warnings

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    # First try to import the module itself
    try:
        import utils.visualization
        print("SUCCESS: utils.visualization module can be imported.")
    except ImportError as e:
        print(f"ERROR: utils.visualization module import failed - {e}")
        return False
        
    # Now check if the specific function exists
    try:
        # Try getting attribute directly without importing
        if hasattr(utils.visualization, 'plot_regression_predictions'):
            print("SUCCESS: plot_regression_predictions function exists in the module.")
            return True
        else:
            print("ERROR: plot_regression_predictions function does not exist in the module.")
            return False
    except Exception as e:
        print(f"ERROR: Checking for function failed - {e}")
        return False

if __name__ == "__main__":
    test_imports() 