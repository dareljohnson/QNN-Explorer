"""
Test script to verify the training flow executes without errors.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock, ANY

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if requirements are available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from utils.tensor_utils import ensure_real
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Required imports not available: {e}")
    IMPORTS_AVAILABLE = False

@unittest.skipUnless(IMPORTS_AVAILABLE, "Required imports not available")
class TestTrainingFlow(unittest.TestCase):
    """Test case for the training flow in app.py."""
    
    def setUp(self):
        """Set up test resources."""
        # Create a simple model for testing
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Create loss function
        self.loss_fn = nn.MSELoss()
        
        # Create sample batch
        self.batch_features = torch.rand(16, 10)
        self.batch_labels = torch.rand(16, 1)
    
    def test_training_step(self):
        """Test that a single training step executes without errors."""
        # Reset gradients
        self.optimizer.zero_grad()
        
        # --- Forward Pass ---
        try:
            # Apply dtype consistency to batch_features (for all model types)
            batch_features = ensure_real(self.batch_features, target_dtype=torch.float32)
            batch_labels = ensure_real(self.batch_labels, target_dtype=torch.float32)
            
            # Run forward pass (for all model types)
            output = self.model(batch_features)
            
            # Apply dtype consistency to output (for all model types)
            output = ensure_real(output, target_dtype=torch.float32)
            
        except Exception as e:
            self.fail(f"Forward pass failed: {e}")
        
        # --- Calculate Loss ---
        try:
            # Ensure output has consistent dtype for loss calculation
            output = ensure_real(output, target_dtype=torch.float32)
            
            # Ensure labels are on the same device as output and have consistent dtype
            batch_labels = batch_labels.to(output.device)
            batch_labels = ensure_real(batch_labels, target_dtype=torch.float32)
            loss = self.loss_fn(output, batch_labels)
        except Exception as e:
            self.fail(f"Loss calculation failed: {e}")
        
        # --- Backward Pass & Optimize ---
        try:
            # Final check for dtype consistency before backward pass
            loss = ensure_real(loss, target_dtype=torch.float32)
            
            loss.backward()
            self.optimizer.step()
        except Exception as e:
            self.fail(f"Backward pass failed: {e}")
        
        # If we get here, the test passed
        self.assertTrue(True, "Training step executed successfully")
    
    def test_mixed_dtype_handling(self):
        """Test that mixed dtype tensors are handled correctly."""
        # Create tensors with different dtypes
        double_features = self.batch_features.to(torch.float64)
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        # --- Forward Pass with mixed dtypes ---
        try:
            # Apply dtype consistency to batch_features (for all model types)
            batch_features = ensure_real(double_features, target_dtype=torch.float32)
            batch_labels = ensure_real(self.batch_labels, target_dtype=torch.float32)
            
            # Run forward pass (for all model types)
            output = self.model(batch_features)
            
            # Apply dtype consistency to output (for all model types)
            output = ensure_real(output, target_dtype=torch.float32)
            
        except Exception as e:
            self.fail(f"Forward pass with mixed dtypes failed: {e}")
        
        # --- Calculate Loss ---
        try:
            # Ensure output has consistent dtype for loss calculation
            output = ensure_real(output, target_dtype=torch.float32)
            
            # Ensure labels are on the same device as output and have consistent dtype
            batch_labels = batch_labels.to(output.device)
            batch_labels = ensure_real(batch_labels, target_dtype=torch.float32)
            loss = self.loss_fn(output, batch_labels)
        except Exception as e:
            self.fail(f"Loss calculation with mixed dtypes failed: {e}")
        
        # --- Backward Pass & Optimize ---
        try:
            # Final check for dtype consistency before backward pass
            loss = ensure_real(loss, target_dtype=torch.float32)
            
            loss.backward()
            self.optimizer.step()
        except Exception as e:
            self.fail(f"Backward pass with mixed dtypes failed: {e}")
        
        # If we get here, the test passed
        self.assertTrue(True, "Training step with mixed dtypes executed successfully")

if __name__ == '__main__':
    unittest.main() 