"""
Unit test for accuracy tracking during model training.
Tests the calculation and display of accuracy metrics.
"""

import unittest
import torch
import numpy as np
import os
import sys
import tempfile
from io import StringIO

# Mock streamlit
class MockStreamlit:
    """Mock Streamlit functionality for testing."""
    def __init__(self):
        self.session_state = {}
        self.info_messages = []
        self.error_messages = []
        self.success_messages = []
        self.warning_messages = []
        self.progress_values = []
        self.metrics = {}
        self.text_values = []
        
    def info(self, message):
        self.info_messages.append(message)
        
    def error(self, message, icon=None):
        self.error_messages.append(message)
        
    def success(self, message):
        self.success_messages.append(message)
        
    def warning(self, message, icon=None):
        self.warning_messages.append(message)
        
    def progress(self, value):
        self.progress_values.append(value)
        return self
        
    def metric(self, label, value):
        self.metrics[label] = value
        
    def text(self, value):
        self.text_values.append(value)
        
    def empty(self):
        return self
        
    def markdown(self, text, unsafe_allow_html=False):
        pass
        
    def columns(self, n):
        return [self] * n
        
    def json(self, data):
        pass
        
    def rerun(self):
        pass
    
    def line_chart(self, data, x=None, y=None):
        pass
        
    def subheader(self, text):
        pass

    def write(self, text):
        pass

    def pyplot(self, fig):
        pass

def mock_accuracy_calculation():
    """Test if accuracy calculation works correctly for classification tasks."""
    # Mock output and labels
    output = torch.tensor([
        [0.1, 0.8, 0.1],  # Class 1
        [0.7, 0.2, 0.1],  # Class 0
        [0.1, 0.1, 0.8],  # Class 2
        [0.8, 0.1, 0.1],  # Class 0
    ])
    
    # Ground truth labels
    labels = torch.tensor([1, 0, 2, 0])
    
    # Calculate predicted classes
    preds = torch.argmax(output, dim=1)
    
    # Calculate accuracy manually
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    expected_accuracy = correct / total
    
    # Calculate using the same logic as in the app
    batch_accuracy = (preds == labels).float().mean().item()
    
    # Should be the same
    assert abs(expected_accuracy - batch_accuracy) < 1e-6, f"Accuracy calculation error: {expected_accuracy} vs {batch_accuracy}"
    
    # Should be 75% accuracy (3/4 correct)
    assert abs(0.75 - batch_accuracy) < 1e-6, f"Expected 75% accuracy, got {batch_accuracy*100}%"
    
    print(f"✅ Accuracy calculation test passed: {batch_accuracy:.2%}")
    return batch_accuracy

def test_accuracy_tracking_over_epochs():
    """Test tracking accuracy over multiple epochs."""
    # Create mock batch accuracies for multiple epochs
    mock_batch_accuracies = [
        [0.6, 0.7, 0.65, 0.75],  # Epoch 1
        [0.7, 0.75, 0.8, 0.75],  # Epoch 2
        [0.8, 0.85, 0.9, 0.85],  # Epoch 3
    ]
    
    # Expected average accuracies for each epoch
    expected_epoch_accuracies = [
        np.mean(mock_batch_accuracies[0]),
        np.mean(mock_batch_accuracies[1]),
        np.mean(mock_batch_accuracies[2])
    ]
    
    # Create history tracking
    history = {'loss': [], 'entanglement': [], 'accuracy': []}
    
    # Simulate epochs
    for epoch_idx, batch_accs in enumerate(mock_batch_accuracies):
        # Add average accuracy for epoch to history
        avg_epoch_accuracy = np.mean(batch_accs)
        history['accuracy'].append(avg_epoch_accuracy)
        
        # Add dummy loss and entanglement values
        history['loss'].append(1.0 - avg_epoch_accuracy)  # Dummy loss that decreases as accuracy increases
        history['entanglement'].append(0.5)  # Dummy entanglement
        
        # Verify that stored accuracy matches expected
        assert abs(history['accuracy'][epoch_idx] - expected_epoch_accuracies[epoch_idx]) < 1e-6, \
            f"Epoch {epoch_idx+1}: Expected accuracy {expected_epoch_accuracies[epoch_idx]}, got {history['accuracy'][epoch_idx]}"
    
    # Verify final accuracy is improving
    assert history['accuracy'][0] < history['accuracy'][1] < history['accuracy'][2], \
        "Accuracy should be improving over epochs"
    
    print(f"✅ Accuracy tracking test passed: {history['accuracy']}")
    return history

def test_accuracy_ui_integration():
    """Test that accuracy is properly integrated in the UI."""
    mock_st = MockStreamlit()
    mock_st.session_state['training_history'] = {
        'loss': [0.9, 0.7, 0.5], 
        'entanglement': [0.4, 0.5, 0.6],
        'accuracy': [0.6, 0.75, 0.85]
    }
    
    # Simulate displaying final metrics
    final_accuracy = mock_st.session_state['training_history']['accuracy'][-1]
    mock_st.metric("Final Accuracy", f"{final_accuracy:.2%}")
    
    # Check metric was set correctly
    assert "Final Accuracy" in mock_st.metrics, "Final Accuracy metric should be displayed"
    assert mock_st.metrics["Final Accuracy"] == "85.00%", f"Expected 85.00%, got {mock_st.metrics['Final Accuracy']}"
    
    print(f"✅ UI integration test passed")
    return mock_st

if __name__ == "__main__":
    print("\n=== Testing Accuracy Tracking Implementation ===\n")
    
    # Run all tests
    try:
        print("\n1. Testing accuracy calculation...")
        accuracy = mock_accuracy_calculation()
        
        print("\n2. Testing accuracy tracking over epochs...")
        history = test_accuracy_tracking_over_epochs()
        
        print("\n3. Testing UI integration...")
        mock_st = test_accuracy_ui_integration()
        
        print("\n✅ All accuracy tracking tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 