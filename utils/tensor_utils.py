"""
Tensor utility functions for quantum-classical neural networks.
"""

import torch
import numpy as np

# Dtypes that carry indices/masks rather than values; never cast these to float.
_INTEGRAL_DTYPES = (
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)

def ensure_real(tensor_or_dict, eps=1e-10, target_dtype=torch.float32):
    """
    Convert complex tensors to real and ensure consistent dtype.
    
    This function recursively processes nested dictionaries, lists, and tuples
    to ensure all tensors contained within are real-valued with consistent dtype.
    
    Args:
        tensor_or_dict: A tensor, dictionary, list, or tuple potentially containing complex tensors
        eps: Small epsilon value to add when replacing NaN/Inf values
        target_dtype: The target dtype to convert tensors to (default: torch.float32)
        
    Returns:
        Same structure as input, but with all complex tensors converted to real
        and all tensors having the same target_dtype
    """
    # Handle dictionaries (recursively process each value)
    if isinstance(tensor_or_dict, dict):
        return {k: ensure_real(v, eps, target_dtype) for k, v in tensor_or_dict.items()}
    
    # Handle lists (recursively process each element)
    elif isinstance(tensor_or_dict, list):
        return [ensure_real(item, eps, target_dtype) for item in tensor_or_dict]
    
    # Handle tuples (recursively process each element)
    elif isinstance(tensor_or_dict, tuple):
        return tuple(ensure_real(item, eps, target_dtype) for item in tensor_or_dict)
    
    # Handle tensors
    elif isinstance(tensor_or_dict, torch.Tensor):
        # Integer/bool tensors are indices and masks (token ids, attention
        # masks), never quantum values. Downcasting them to float breaks
        # nn.Embedding lookups inside transformer backbones:
        #   "Expected tensor for argument #1 'indices' to have one of the
        #    following scalar types: Long, Int; but got ... FloatTensor"
        if tensor_or_dict.dtype in _INTEGRAL_DTYPES:
            return tensor_or_dict

        # Convert complex to real
        if torch.is_complex(tensor_or_dict):
            print(f"Converting complex tensor with shape {tensor_or_dict.shape} to real")
            tensor = tensor_or_dict.abs()
        else:
            tensor = tensor_or_dict
        
        # Convert dtype if needed
        if tensor.dtype != target_dtype:
            current_dtype = tensor.dtype
            print(f"Converting tensor dtype from {current_dtype} to {target_dtype}")
            tensor = tensor.to(target_dtype)
        
        # Handle NaN/Inf values if present
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"Replacing NaN/Inf values in tensor with shape {tensor.shape}")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0/eps, neginf=-1.0/eps)
        
        return tensor
    
    # Handle numpy arrays
    elif isinstance(tensor_or_dict, np.ndarray):
        array = tensor_or_dict
        is_complex = np.iscomplex(array).any()
        has_bad = np.isnan(array).any() or np.isinf(array).any()

        # A real, finite array already satisfies the contract - return it
        # unchanged so callers can rely on object identity.
        if not is_complex and not has_bad:
            return array

        if is_complex:
            print(f"Converting complex numpy array with shape {array.shape} to real")
            array = np.abs(array)

        if has_bad:
            print(f"Replacing NaN/Inf values in numpy array with shape {array.shape}")
            array = np.nan_to_num(array, nan=0.0, posinf=1.0/eps, neginf=-1.0/eps)

        return array
    
    # All other types pass through unchanged
    return tensor_or_dict


def is_complex_tensor_present(tensor_or_dict):
    """
    Check if a complex tensor exists anywhere in the input structure.
    
    Args:
        tensor_or_dict: A tensor, dictionary, list, or tuple potentially containing complex tensors
        
    Returns:
        bool: True if any complex tensor is found, False otherwise
    """
    # Handle dictionaries
    if isinstance(tensor_or_dict, dict):
        return any(is_complex_tensor_present(v) for v in tensor_or_dict.values())
    
    # Handle lists and tuples
    elif isinstance(tensor_or_dict, (list, tuple)):
        return any(is_complex_tensor_present(item) for item in tensor_or_dict)
    
    # Handle tensors
    elif isinstance(tensor_or_dict, torch.Tensor):
        return torch.is_complex(tensor_or_dict)
    
    # Handle numpy arrays
    elif isinstance(tensor_or_dict, np.ndarray):
        return np.iscomplex(tensor_or_dict).any()
    
    # Default for other types
    return False


def get_tensor_stats(tensor):
    """
    Get basic statistics about a tensor for debugging.
    
    Args:
        tensor: A PyTorch tensor
        
    Returns:
        dict: Statistics about the tensor
    """
    if not isinstance(tensor, torch.Tensor):
        return {"type": str(type(tensor))}
    
    stats = {
        "type": "Tensor",
        "dtype": str(tensor.dtype),
        "shape": str(tensor.shape),
        "device": str(tensor.device),
        "requires_grad": tensor.requires_grad
    }
    
    # Add more stats for non-complex tensors
    if not torch.is_complex(tensor):
        try:
            stats.update({
                "min": tensor.min().item(),
                "max": tensor.max().item(),
                "mean": tensor.mean().item(),
                "std": tensor.std().item(),
                "has_nan": torch.isnan(tensor).any().item(),
                "has_inf": torch.isinf(tensor).any().item()
            })
        except (RuntimeError, ValueError):
            # Some operations might fail on empty tensors
            pass
    else:
        # For complex tensors
        real_part = tensor.real
        imag_part = tensor.imag
        stats.update({
            "real_min": real_part.min().item(),
            "real_max": real_part.max().item(),
            "real_mean": real_part.mean().item(),
            "imag_min": imag_part.min().item(),
            "imag_max": imag_part.max().item(),
            "imag_mean": imag_part.mean().item(),
        })
    
    return stats 