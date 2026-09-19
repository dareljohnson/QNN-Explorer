"""
Test cases for tensor utility functions.
"""

import sys
import os
import unittest

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if requirements are available
try:
    import torch
    import numpy as np
    from utils.tensor_utils import ensure_real, is_complex_tensor_present, get_tensor_stats
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Required imports not available: {e}")
    IMPORTS_AVAILABLE = False

@unittest.skipUnless(IMPORTS_AVAILABLE, "Required imports not available")
class TestTensorUtils(unittest.TestCase):
    """Test cases for tensor utility functions."""
    
    def test_ensure_real_tensor(self):
        """Test ensure_real on a single tensor."""
        # Real tensor
        real_tensor = torch.tensor([1.0, 2.0, 3.0])
        result = ensure_real(real_tensor)
        self.assertIs(result, real_tensor, "Real tensor should be returned unchanged")
        
        # Complex tensor
        complex_tensor = torch.complex(
            torch.tensor([1.0, 2.0]), 
            torch.tensor([3.0, 4.0])
        )
        result = ensure_real(complex_tensor)
        self.assertFalse(torch.is_complex(result), "Result should be real")
        self.assertTrue(torch.allclose(
            result, torch.tensor([
                (1.0**2 + 3.0**2)**0.5,
                (2.0**2 + 4.0**2)**0.5,
            ])
        ), "Result should be the absolute value of complex tensor")
    
    def test_ensure_real_dict(self):
        """Test ensure_real on a dictionary."""
        test_dict = {
            'real': torch.tensor([1.0, 2.0]),
            'complex': torch.complex(
                torch.tensor([1.0]), 
                torch.tensor([2.0])
            ),
            'nested': {
                'real': torch.tensor([3.0]),
                'complex': torch.complex(
                    torch.tensor([4.0]), 
                    torch.tensor([5.0])
                )
            }
        }
        
        result = ensure_real(test_dict)
        
        # Check structure
        self.assertEqual(set(result.keys()), set(test_dict.keys()))
        self.assertEqual(set(result['nested'].keys()), set(test_dict['nested'].keys()))
        
        # Check types
        self.assertFalse(torch.is_complex(result['complex']))
        self.assertFalse(torch.is_complex(result['nested']['complex']))
        
        # Check values
        self.assertTrue(torch.allclose(
            result['complex'], 
            torch.tensor([(1.0**2 + 2.0**2)**0.5])
        ))
        self.assertTrue(torch.allclose(
            result['nested']['complex'], 
            torch.tensor([(4.0**2 + 5.0**2)**0.5])
        ))
    
    def test_ensure_real_list(self):
        """Test ensure_real on a list."""
        test_list = [
            torch.tensor([1.0, 2.0]),
            torch.complex(
                torch.tensor([3.0]), 
                torch.tensor([4.0])
            ),
            [
                torch.tensor([5.0]),
                torch.complex(
                    torch.tensor([6.0]), 
                    torch.tensor([7.0])
                )
            ]
        ]
        
        result = ensure_real(test_list)
        
        # Check structure
        self.assertEqual(len(result), len(test_list))
        self.assertEqual(len(result[2]), len(test_list[2]))
        
        # Check types
        self.assertFalse(torch.is_complex(result[1]))
        self.assertFalse(torch.is_complex(result[2][1]))
        
        # Check values
        self.assertTrue(torch.allclose(
            result[1], 
            torch.tensor([(3.0**2 + 4.0**2)**0.5])
        ))
        self.assertTrue(torch.allclose(
            result[2][1], 
            torch.tensor([(6.0**2 + 7.0**2)**0.5])
        ))
    
    def test_ensure_real_nan(self):
        """Test ensure_real handles NaN values."""
        tensor_with_nan = torch.tensor([1.0, float('nan'), 3.0])
        result = ensure_real(tensor_with_nan)
        self.assertFalse(torch.isnan(result).any(), "Result should not contain NaN")
        self.assertTrue(torch.allclose(
            result, torch.tensor([1.0, 0.0, 3.0]), 
            equal_nan=True
        ), "NaN should be replaced with 0.0")
    
    def test_ensure_real_numpy(self):
        """Test ensure_real on numpy arrays."""
        # Real array
        real_array = np.array([1.0, 2.0, 3.0])
        result = ensure_real(real_array)
        self.assertIs(result, real_array, "Real array should be returned unchanged")
        
        # Complex array
        complex_array = np.array([1.0 + 2.0j, 3.0 + 4.0j])
        result = ensure_real(complex_array)
        self.assertFalse(np.iscomplex(result).any(), "Result should be real")
        self.assertTrue(np.allclose(
            result, np.array([
                abs(1.0 + 2.0j),
                abs(3.0 + 4.0j)
            ])
        ), "Result should be the absolute value of complex array")
    
    def test_is_complex_tensor_present(self):
        """Test is_complex_tensor_present function."""
        # No complex tensor
        test_dict = {
            'real1': torch.tensor([1.0, 2.0]),
            'real2': torch.tensor([3.0, 4.0]),
            'nested': {
                'real3': torch.tensor([5.0, 6.0])
            }
        }
        self.assertFalse(is_complex_tensor_present(test_dict))
        
        # With complex tensor
        test_dict['nested']['complex'] = torch.complex(
            torch.tensor([1.0]), 
            torch.tensor([2.0])
        )
        self.assertTrue(is_complex_tensor_present(test_dict))
        
        # List with complex tensor
        test_list = [torch.tensor([1.0]), torch.tensor([2.0])]
        self.assertFalse(is_complex_tensor_present(test_list))
        
        test_list.append(torch.complex(
            torch.tensor([1.0]), 
            torch.tensor([2.0])
        ))
        self.assertTrue(is_complex_tensor_present(test_list))
    
    def test_get_tensor_stats(self):
        """Test get_tensor_stats function."""
        # Real tensor
        real_tensor = torch.tensor([1.0, 2.0, 3.0])
        stats = get_tensor_stats(real_tensor)
        
        self.assertEqual(stats["type"], "Tensor")
        self.assertEqual(stats["shape"], "torch.Size([3])")
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 3.0)
        self.assertEqual(stats["mean"], 2.0)
        self.assertFalse(stats["has_nan"])
        self.assertFalse(stats["has_inf"])
        
        # Complex tensor
        complex_tensor = torch.complex(
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0])
        )
        stats = get_tensor_stats(complex_tensor)
        
        self.assertEqual(stats["type"], "Tensor")
        self.assertEqual(stats["shape"], "torch.Size([2])")
        self.assertEqual(stats["real_min"], 1.0)
        self.assertEqual(stats["real_max"], 2.0)
        self.assertEqual(stats["imag_min"], 3.0)
        self.assertEqual(stats["imag_max"], 4.0)

    def test_ensure_real_dtype_conversion(self):
        """Test ensure_real converts tensors to the target dtype."""
        # Double tensor
        double_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = ensure_real(double_tensor, target_dtype=torch.float32)
        self.assertEqual(result.dtype, torch.float32, "Result should be float32")
        self.assertTrue(torch.allclose(
            result, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        ))
        
        # Float tensor to double
        float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = ensure_real(float_tensor, target_dtype=torch.float64)
        self.assertEqual(result.dtype, torch.float64, "Result should be float64")
        self.assertTrue(torch.allclose(
            result, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        ))
        
    def test_ensure_real_mixed_dtypes(self):
        """Test ensure_real handles mixed dtypes in nested structures."""
        mixed_dict = {
            'float32': torch.tensor([1.0, 2.0], dtype=torch.float32),
            'float64': torch.tensor([3.0, 4.0], dtype=torch.float64),
            'complex': torch.complex(
                torch.tensor([1.0], dtype=torch.float64), 
                torch.tensor([2.0], dtype=torch.float64)
            ),
            'nested': {
                'float32': torch.tensor([5.0], dtype=torch.float32),
                'float64': torch.tensor([6.0], dtype=torch.float64)
            }
        }
        
        result = ensure_real(mixed_dict, target_dtype=torch.float32)
        
        # Check all dtypes are float32
        self.assertEqual(result['float32'].dtype, torch.float32)
        self.assertEqual(result['float64'].dtype, torch.float32)
        self.assertEqual(result['complex'].dtype, torch.float32)
        self.assertEqual(result['nested']['float32'].dtype, torch.float32)
        self.assertEqual(result['nested']['float64'].dtype, torch.float32)
        
        # Check values are preserved
        self.assertTrue(torch.allclose(result['float32'], torch.tensor([1.0, 2.0], dtype=torch.float32)))
        self.assertTrue(torch.allclose(result['float64'], torch.tensor([3.0, 4.0], dtype=torch.float32)))
        self.assertTrue(torch.allclose(result['complex'], torch.tensor([(1.0**2 + 2.0**2)**0.5], dtype=torch.float32)))

if __name__ == '__main__':
    unittest.main() 