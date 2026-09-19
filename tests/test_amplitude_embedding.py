import sys
import os
import torch
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import amplitude_embedding function
from core.quantum_models import amplitude_embedding

def test_basic_functionality():
    print("Testing basic functionality of amplitude_embedding...")
    
    # Test case 1: Standard input vector
    x1 = torch.rand(16)
    y1 = amplitude_embedding(x1)
    print(f"Input shape: {x1.shape} → Output shape: {y1.shape}")
    assert y1.shape == torch.Size([16]), "Output shape should match input"
    assert torch.isclose(torch.norm(y1), torch.tensor(1.0, dtype=y1.dtype)), "Output should be normalized"
    
    # Test case 2: Input with batch dimension = 1
    x2 = torch.rand(1, 16)
    y2 = amplitude_embedding(x2)
    print(f"Input shape: {x2.shape} → Output shape: {y2.shape}")
    assert y2.shape == torch.Size([16]), "Batch dimension should be removed"
    assert torch.isclose(torch.norm(y2), torch.tensor(1.0, dtype=y2.dtype)), "Output should be normalized"
    
    # Test case 3: Flattening multi-dimensional input
    x3 = torch.rand(4, 4)  # Should be flattened to 16
    y3 = amplitude_embedding(x3)
    print(f"Input shape: {x3.shape} → Output shape: {y3.shape}")
    assert y3.shape == torch.Size([16]), "Multi-dimensional input should be flattened to 1D"
    assert torch.isclose(torch.norm(y3), torch.tensor(1.0, dtype=y3.dtype)), "Output should be normalized"
    
    print("✅ Basic amplitude_embedding tests passed!")
    return True

def test_edge_cases():
    print("\nTesting edge cases for amplitude_embedding...")
    
    # Test case 4: Empty tensor (should handle gracefully)
    try:
        x4 = torch.zeros(0)
        y4 = amplitude_embedding(x4)
        print("⚠️ Empty tensor: function did not raise error but this might need attention")
    except Exception as e:
        print(f"✅ Empty tensor handling: {type(e).__name__}: {e}")
    
    # Test case 5: Zero vector (division by zero risk)
    x5 = torch.zeros(16)
    y5 = amplitude_embedding(x5)
    print(f"Zero vector → Output shape: {y5.shape}")
    assert not torch.isnan(y5).any(), "Output should not contain NaN values"
    
    # Test case 6: NumPy array input
    x6 = np.random.rand(16)
    y6 = amplitude_embedding(x6)
    print(f"Numpy input → Output shape: {y6.shape}")
    assert isinstance(y6, torch.Tensor), "Output should be a torch.Tensor"
    assert torch.isclose(torch.norm(y6), torch.tensor(1.0, dtype=y6.dtype)), "Output should be normalized"
    
    # Test case 7: Single element tensor
    x7 = torch.tensor([42.0])
    y7 = amplitude_embedding(x7)
    print(f"Single element tensor → Output shape: {y7.shape}")
    assert y7.shape == torch.Size([1]), "Shape should be preserved for single element"
    assert torch.isclose(torch.norm(y7), torch.tensor(1.0, dtype=y7.dtype)), "Output should be normalized"
    
    print("✅ Edge case tests passed!")
    return True

if __name__ == "__main__":
    basic_result = test_basic_functionality()
    edge_result = test_edge_cases()
    
    if basic_result and edge_result:
        print("\n✅✅ All amplitude_embedding tests passed successfully! ✅✅")
    else:
        print("\n❌ Some tests failed.")
        exit(1) 