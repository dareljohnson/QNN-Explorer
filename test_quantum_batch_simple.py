import sys
import os
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the amplitude_embedding function
from core.quantum_models import amplitude_embedding

def test_amplitude_embedding():
    """Test amplitude_embedding with different input shapes."""
    # Test standard 1D input
    x1 = torch.ones(16)
    y1 = amplitude_embedding(x1)
    print(f"Input shape: {x1.shape} → Output shape: {y1.shape}")
    assert len(y1.shape) == 1, "Output should be 1D"
    
    # Test batch dimension handling
    x2 = torch.ones(1, 16)
    y2 = amplitude_embedding(x2)
    print(f"Input shape: {x2.shape} → Output shape: {y2.shape}")
    assert len(y2.shape) == 1, "Output should be 1D (batch dim removed)"
    
    # Test reshaping 2D tensor
    x3 = torch.ones(4, 4)
    y3 = amplitude_embedding(x3)
    print(f"Input shape: {x3.shape} → Output shape: {y3.shape}")
    assert y3.shape[0] == 16, "Output should have 16 elements"
    
    # Test all outputs are normalized
    assert torch.isclose(torch.norm(y1), torch.tensor(1.0)), "Output vector should be normalized"
    assert torch.isclose(torch.norm(y2), torch.tensor(1.0)), "Output vector should be normalized"
    assert torch.isclose(torch.norm(y3), torch.tensor(1.0)), "Output vector should be normalized"
    
    print("✅ All amplitude_embedding tests passed!")

if __name__ == "__main__":
    print("Testing improved amplitude_embedding function...")
    test_amplitude_embedding()
    print("\nAmplitude embedding function is now properly handling different input shapes")
    print("This should fix the 'Broadcasting with MottonenStatePreparation is not supported' error") 