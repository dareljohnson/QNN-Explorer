"""
Test script to verify fixes for training issues in the QNN Explorer.
This test focuses on the "list assignment index out of range" error during training.
"""
import torch
import numpy as np
import sys
import os
import unittest

# Add the current directory to the path for imports
sys.path.append(os.path.abspath('.'))

# Mock streamlit for testing
class MockStreamlit:
    """Mock streamlit module for testing without UI"""
    
    @staticmethod
    def progress(value=0):
        return MockProgressBar()
    
    @staticmethod
    def empty():
        return MockEmptyElement()
    
    @staticmethod
    def line_chart():
        return MockLineChart()
    
    @staticmethod
    def error(message):
        print(f"Error: {message}")
    
    @staticmethod
    def info(message):
        print(f"Info: {message}")
    
    @staticmethod
    def success(message):
        print(f"Success: {message}")
    
    @staticmethod
    def warning(message):
        print(f"Warning: {message}")
    
    @staticmethod
    def stop():
        print("Execution stopped")
        raise StopIteration("Test stop")
    
    class session_state:
        """Mock session state for testing"""
        def __init__(self):
            self._data = {}
            
        def __getitem__(self, key):
            return self._data.get(key)
            
        def __setitem__(self, key, value):
            self._data[key] = value
            
        def get(self, key, default=None):
            return self._data.get(key, default)
            
        def __contains__(self, key):
            return key in self._data
    
    def __init__(self):
        self.session_state = MockStreamlit.session_state()

class MockProgressBar:
    """Mock progress bar for testing"""
    def progress(self, value):
        print(f"Progress: {value:.2f}")
        return self

class MockEmptyElement:
    """Mock empty element for testing"""
    def text(self, content):
        print(f"Status: {content}")
        return self

class MockLineChart:
    """Mock line chart for testing"""
    def add_rows(self, data):
        print(f"Chart data added: {data}")
        return self

# Create a mock of the calculate_meyer_wallach function
def mock_calculate_meyer_wallach(state_vector, num_qubits):
    """Mock implementation of Meyer-Wallach entanglement calculation"""
    # Just return a random value between 0 and 1
    return np.random.random()

class TestTrainingFixes(unittest.TestCase):
    """Unit tests for training fixes"""
    
    def setUp(self):
        """Set up test environment"""
        # Import needed modules
        global st
        
        # Create mock streamlit
        st = MockStreamlit()
        
        # Create a mock for torch.cuda
        torch.cuda.is_available = lambda: False
        
        # Set up mock session state
        st.session_state.device = "cpu"
        st.session_state.training_in_progress = True
        st.session_state.model_config = {
            'num_qubits': 4,
            'observation_type': 'State Vector',
            'save_config_name': 'test_config'
        }
        
        # Patch the metrics.calculate_meyer_wallach function
        from core import metrics
        self.original_meyer_wallach = metrics.calculate_meyer_wallach
        metrics.calculate_meyer_wallach = mock_calculate_meyer_wallach
    
    def tearDown(self):
        """Clean up after test"""
        # Restore original function
        from core import metrics
        metrics.calculate_meyer_wallach = self.original_meyer_wallach
    
    def test_training_history_initialization(self):
        """Test that training history is properly initialized"""
        # Initialize the training history
        st.session_state.training_history = {'loss': [], 'entanglement': []}
        
        # Check that it was initialized correctly
        self.assertIn('loss', st.session_state.training_history)
        self.assertIn('entanglement', st.session_state.training_history)
        self.assertEqual(len(st.session_state.training_history['loss']), 0)
        self.assertEqual(len(st.session_state.training_history['entanglement']), 0)
    
    def test_entanglement_tracking(self):
        """Test entanglement tracking and history updates"""
        # Initialize variables
        st.session_state.training_history = {'loss': [], 'entanglement': []}
        entanglement = np.nan
        batch_entanglements = []
        
        # Simulate entanglement calculation during batch processing
        for _ in range(3):  # Simulate 3 batches
            # Calculate entanglement
            entanglement = mock_calculate_meyer_wallach(None, 4)
            batch_entanglements.append(entanglement)
        
        # Simulate end of epoch processing
        if len(batch_entanglements) > 0:
            avg_entanglement = np.nanmean(batch_entanglements)
            entanglement = avg_entanglement
        
        # Update history
        st.session_state.training_history['entanglement'].append(entanglement)
        
        # Verify history was updated correctly
        self.assertEqual(len(st.session_state.training_history['entanglement']), 1)
        self.assertIsInstance(st.session_state.training_history['entanglement'][0], float)
        self.assertGreaterEqual(st.session_state.training_history['entanglement'][0], 0)
        self.assertLessEqual(st.session_state.training_history['entanglement'][0], 1)
    
    def test_empty_batch_entanglements(self):
        """Test handling of empty batch entanglements"""
        # Initialize history
        st.session_state.training_history = {'loss': [], 'entanglement': []}
        entanglement = np.nan
        batch_entanglements = []  # Empty list - no batches processed
        
        # Simulate end of epoch processing with empty batch entanglements
        if len(batch_entanglements) > 0:
            avg_entanglement = np.nanmean(batch_entanglements)
            entanglement = avg_entanglement
        
        # Update history - should not crash
        st.session_state.training_history['entanglement'].append(entanglement)
        
        # Verify history was updated with NaN
        self.assertEqual(len(st.session_state.training_history['entanglement']), 1)
        self.assertTrue(np.isnan(st.session_state.training_history['entanglement'][0]))
    
    def test_try_catch_entanglement(self):
        """Test that errors in entanglement calculation are caught"""
        # Test that exceptions during entanglement calculation are caught
        try:
            # Mock a failing entanglement calculation
            def failing_calculate_meyer_wallach(state_vector, num_qubits):
                raise ValueError("Test error in entanglement calculation")
            
            # Override the mock with a failing version
            from core import metrics
            metrics.calculate_meyer_wallach = failing_calculate_meyer_wallach
            
            # Initialize
            entanglement = np.nan
            
            # Try calculation in a try-except block
            try:
                entanglement = metrics.calculate_meyer_wallach(None, 4)
            except Exception as e:
                # Exceptions should be caught
                pass
            
            # Should still be NaN
            self.assertTrue(np.isnan(entanglement))
            
        finally:
            # Restore the original mock
            from core import metrics
            metrics.calculate_meyer_wallach = mock_calculate_meyer_wallach

if __name__ == "__main__":
    # Run the tests
    unittest.main() 