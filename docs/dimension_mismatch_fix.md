# Hybrid Model Fixes: Dimension Mismatch & Data Type Consistency

This document describes three important fixes to the hybrid model:

1. **Adaptive Fusion Layer**: Handles dimension mismatches between input features and quantum input size
2. **Data Type Consistency**: Ensures consistent data types throughout forward propagation
3. **Transformer Pooling**: Adds missing method for handling transformer outputs

## Problem 1: Dimension Mismatch

The error "mat1 and mat2 shapes cannot be multiplied (64x2 and 32x2)" occurs during model training when the input feature dimension doesn't match the quantum input size required by the amplitude embedding.

### Root Cause

When using the Quantum Neural Network without a classical backbone (or with a backbone that produces outputs of different dimensions than expected), there's a mismatch between:

1. **Input data dimension**: Features with shape (batch_size, feature_dim) where feature_dim is 2 in this case
2. **Quantum input dimension**: The quantum circuit expects amplitude inputs of size 2^num_qubits (32 for 5 qubits)

In the original implementation, when no classical backbone is specified, the fusion layer is set to an `nn.Identity()`, which assumes the input features will already match the quantum input size.

### Solution: Adaptive Fusion Layer

We've implemented an adaptive fusion layer that:

1. **Detects dimension mismatches** during the first forward pass
2. **Dynamically creates** a linear projection layer to map from input feature dimension to quantum input dimension
3. **Maintains compatibility** with existing models by only adapting when necessary

The fix adds the following code to the `forward` method in `core/fusion.py`:

```python
# If no backbone, check if input dimensions match quantum dimensions
# and initialize fusion layer to match if needed
if (isinstance(x, torch.Tensor) and 
    len(x.shape) > 1 and 
    not isinstance(self.fusion_layer, nn.Linear) and 
    x.shape[1] != self.quantum_input_size):
    
    input_feature_size = x.shape[1]
    print(f"Input feature size ({input_feature_size}) doesn't match quantum input size ({self.quantum_input_size}). Creating adaptive fusion layer.")
    
    # Create a new fusion layer on first use
    self.fusion_layer = nn.Linear(input_feature_size, self.quantum_input_size)
    # Initialize with small values to avoid large gradients
    nn.init.xavier_uniform_(self.fusion_layer.weight, gain=0.1)
    nn.init.zeros_(self.fusion_layer.bias)
    
    # Move to same device as input
    self.fusion_layer = self.fusion_layer.to(x.device)
    
# Apply fusion layer if it's been created
if isinstance(self.fusion_layer, nn.Linear):
    x = self.fusion_layer(x)
```

## Problem 2: Data Type Mismatch

The error "mat1 and mat2 must have the same dtype, but got Double and Float" occurs when tensors with different data types (typically Float32 vs Float64/Double) are multiplied.

### Root Cause

The quantum circuit operations return complex values which are then converted to real values using `abs()`. However, these values might have a different data type (often `torch.float64`/double precision) than the model parameters (typically `torch.float32`/single precision).

### Solution: Data Type Consistency

We've added explicit type conversion to ensure consistent data types throughout the forward propagation:

```python
# After converting complex state vector to real values
if quantum_output.dtype != torch.float32:
    quantum_output = quantum_output.to(torch.float32)

# Before passing to the output head
if quantum_output.dtype != self.output_head.weight.dtype:
    quantum_output = quantum_output.to(self.output_head.weight.dtype)
```

This ensures that quantum outputs are always converted to the same data type as the neural network parameters before multiplication.

## Problem 3: Missing Transformer Pooling Method

The error "'HybridModel' object has no attribute 'transformer_pooling'" occurs when trying to use a transformer model backbone, as the method is referenced but was not implemented.

### Root Cause

In the `forward` method of `HybridModel`, there's a call to `self.transformer_pooling(x)` when processing transformer backbone outputs, but this method wasn't defined in the class.

### Solution: Implement Transformer Pooling

We've implemented the missing method to handle transformer outputs properly:

```python
def transformer_pooling(self, x):
    """
    Apply pooling operation to transformer output to get a fixed-size representation.
    
    Args:
        x: Transformer output (typically shape [batch_size, seq_len, hidden_size])
        
    Returns:
        Pooled representation (shape [batch_size, hidden_size])
    """
    # Use the [CLS] token (first token) representation - most common for BERT
    cls_token = x[:, 0]  # Shape: [batch_size, hidden_size]
    
    return cls_token
```

This implementation extracts the first token representation (CLS token), which is a common approach for transformer models like BERT.

## Benefits

These fixes offer several advantages:

1. **Automatic adaptation**: The model automatically adapts to input dimensions at runtime
2. **Type consistency**: Tensors are properly converted to compatible types during computation
3. **Transformer support**: Proper handling of transformer outputs with pooling
4. **Backward compatibility**: Existing models continue to work as before
5. **More flexibility**: Users can provide data with various feature dimensions and types
6. **Better error handling**: Avoids confusing dimension and type mismatch errors

## Testing

We've created comprehensive tests to verify the fixes:

1. `test_dimension_and_dtype_fixes()`: Verifies both dimension and type fixes work together
2. `test_training_with_fixes()`: Ensures training works with the fixes in place
3. `test_transformer_pooling()`: Validates that transformer outputs are correctly pooled

All tests pass, confirming that:
- The fusion layer is created with correct dimensions
- Data types are properly converted
- Transformer outputs are correctly pooled to fixed-size representations
- Forward and backward passes work without errors
- The model trains successfully with inputs of different dimensions

## Usage

No change is required in your workflow. The model will now automatically:
- Handle inputs with mismatched dimensions by creating an appropriate fusion layer
- Ensure consistent data types during computation
- Properly pool transformer outputs to fixed-size representations

These changes make the QNN Explorer more robust and user-friendly, especially when working with transformers and preprocessed data that might have different dimensions or precision than expected. 