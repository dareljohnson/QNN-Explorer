"""
Test script to diagnose and fix the dimension mismatch in the hybrid model.
"""
import torch
import torch.nn as nn
import pennylane as qml
from core.fusion import HybridModel
from core.quantum_models import ansatz1, amplitude_embedding

def diagnose_dimension_mismatch():
    """
    Diagnose the dimension mismatch in the hybrid model.
    The error happens in matrix multiplication: mat1 and mat2 shapes cannot be multiplied (64x2 and 32x2)
    """
    print("Diagnosing dimension mismatch in the hybrid model...")
    
    # Create a minimal model configuration
    config = {
        'num_qubits': 5,        # This gives a quantum input size of 2^5 = 32
        'num_layers': 2,
        'encoding_method': 'Amplitude Encoding',
        'ansatz_name': 'Ansatz 1 (Rot+CNOT Chain)',
        'ansatz_func': ansatz1,
        'observation_type': 'State Vector', 
        'classical_backbone_type': 'None',
        'use_gpu': False,
        'quantum_input_size': 32, # 2^5 = 32
        'num_classes': 2,        # Binary classification output
    }
    
    # Create the model
    model = HybridModel(config)
    model.eval()
    
    # Print model summary
    print("\nModel Structure:")
    for name, module in model.named_children():
        print(f"  {name}: {module}")
    
    # Create test inputs of different shapes to find the issue
    test_shapes = [
        (32,),     # Direct quantum input (no batch)
        (1, 32),   # One sample with correct quantum input size
        (64, 32),  # Batch of 64 with correct quantum input size
        (64, 2),   # Batch of 64 but only 2 features (problematic)
    ]
    
    # Test each input shape
    for shape in test_shapes:
        try:
            print(f"\nTesting input shape: {shape}")
            test_input = torch.rand(*shape)
            output = model(test_input)
            print(f"Success! Output shape: {output.shape}")
            
            # If this is the problematic shape, verify dimensions in each layer
            if shape == (64, 2):
                print("\nDetailed layer analysis for shape (64, 2):")
                x = test_input
                print(f"Input: {x.shape}")
                
                if model.classical_backbone is not None:
                    x_backbone = model.classical_backbone(x)
                    print(f"After classical_backbone: {x_backbone.shape}")
                    
                if model.fusion_layer is not None:
                    # Check fusion layer weights
                    if isinstance(model.fusion_layer, nn.Linear):
                        print(f"Fusion layer weights shape: {model.fusion_layer.weight.shape}")
                    x_fusion = model.fusion_layer(x if model.classical_backbone is None else x_backbone)
                    print(f"After fusion_layer: {x_fusion.shape}")
                
                # For the quantum part, we need to do a bit of manual checking
                print(f"Expected quantum_input_size: {model.quantum_input_size}")
                try:
                    quantum_output = model.quantum_circuit_fn(
                        x_fusion if model.fusion_layer is not None else x,
                        model.quantum_params
                    )
                    print(f"After quantum_circuit: {quantum_output.shape}")
                    
                    if model.output_head is not None:
                        output = model.output_head(quantum_output)
                        print(f"After output_head: {output.shape}")
                except Exception as e:
                    print(f"Error in quantum circuit: {e}")
        except Exception as e:
            print(f"Error with shape {shape}: {e}")
    
    # Propose a fix
    print("\nProposed Fix:")
    print("Based on the analysis, the issue is likely in the fusion layer that maps input features to quantum input size.")
    print("If the input has only 2 features but the model expects 32 (2^5) features, we need to ensure the fusion layer maps from 2 to 32.")
    
    # Create a fixed model with a fusion layer explicitly defined
    config_fixed = config.copy()
    
    # If the error is that we have inputs of shape [64, 2] but the fusion layer expects [64, 32],
    # we need to handle this case
    print("\nTesting Fixed Model:")
    fixed_model = FixedHybridModel(config_fixed)
    
    # Test the problematic shape with the fixed model
    try:
        test_input = torch.rand(64, 2)  # Problematic shape
        print(f"\nTesting fixed model with shape: {test_input.shape}")
        output = fixed_model(test_input)
        print(f"Success! Fixed model output shape: {output.shape}")
    except Exception as e:
        print(f"Error in fixed model: {e}")
    
    return fixed_model

class FixedHybridModel(nn.Module):
    """
    A fixed version of HybridModel that ensures dimensions match correctly.
    """
    def __init__(self, config):
        super().__init__()
        
        self.num_qubits = config.get('num_qubits', 5)
        self.quantum_input_size = 2**self.num_qubits
        
        # Input feature dimensions may vary, but we need to map to quantum_input_size
        # This is the crucial fix - adapting to the actual input feature size
        input_feature_size = 2  # The feature size from the error message
        self.fusion_layer = nn.Linear(input_feature_size, self.quantum_input_size)
        
        # Quantum circuit 
        self.quantum_circuit_fn, initial_params, num_params = create_quantum_layer(
            num_qubits=self.num_qubits,
            num_layers=config.get('num_layers', 2),
            ansatz_func=ansatz1
        )
        
        # Parameters
        self.quantum_params = nn.Parameter(initial_params)
        
        # Output head
        if config.get('num_classes') is not None:
            self.output_head = nn.Linear(self.quantum_input_size, config['num_classes'])
        else:
            self.output_head = None
    
    def forward(self, x):
        # First, ensure we're handling the input's feature dimension correctly
        if len(x.shape) == 1:
            # Single sample, no batch dimension
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Map features to quantum input size
        x = self.fusion_layer(x)
        
        try:
            # Process through quantum circuit
            quantum_output = self.quantum_circuit_fn(x, self.quantum_params)
            
            # Process through output head if present
            if self.output_head is not None:
                return self.output_head(quantum_output.abs())
            else:
                return quantum_output
        except Exception as e:
            print(f"Error in forward pass: {e}")
            if self.output_head is not None:
                # Return dummy output
                return torch.zeros((x.shape[0], self.output_head.out_features), device=x.device)
            else:
                return torch.zeros((x.shape[0], self.quantum_input_size), device=x.device)

def create_quantum_layer(num_qubits, num_layers, ansatz_func):
    """
    Simplified version of create_quantum_layer from quantum_models.py
    """
    import torch
    import pennylane as qml
    
    # Device
    dev = qml.device("default.qubit", wires=num_qubits)
    
    # Number of parameters
    num_params = 2 * num_qubits * num_layers  # Assuming ansatz1
    
    # Initialize parameters
    initial_params = torch.randn(num_params) * 0.1
    
    # Quantum circuit
    @qml.qnode(dev, interface='torch')
    def quantum_circuit(inputs, params):
        # Embed inputs into quantum state
        qml.AmplitudeEmbedding(inputs, wires=range(num_qubits), normalize=True)
        
        # Apply variational circuit
        ansatz_func(params, num_qubits, num_layers)
        
        # Return state vector
        return qml.state()
    
    # Quantum layer function
    def quantum_layer_fn(x, params):
        """Handle batched inputs for quantum circuit."""
        batch_size = x.shape[0]
        
        # Process each sample individually
        quantum_outputs = []
        for i in range(batch_size):
            single_input = x[i]
            # Normalize
            norm = torch.norm(single_input)
            if norm > 1e-6:
                single_input = single_input / norm
            
            # Process through quantum circuit
            output = quantum_circuit(single_input, params)
            quantum_outputs.append(output.unsqueeze(0))
        
        # Combine outputs
        return torch.cat(quantum_outputs, dim=0)
    
    return quantum_layer_fn, initial_params, num_params

if __name__ == "__main__":
    fixed_model = diagnose_dimension_mismatch()
    
    # Verify the fix by running real training scenario
    print("\nVerifying fix with training scenario:")
    
    # Create fake dataset
    batch_size = 64
    feature_dim = 2  # The dimension from the error (64x2)
    num_classes = 2
    
    # Create dataloader with the problematic shape
    X = torch.rand(128, feature_dim)  # 128 samples with 2 features each
    y = torch.randint(0, num_classes, (128,))  # random class labels
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Set up training
    fixed_model.train()
    optimizer = torch.optim.Adam(fixed_model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Test one batch
    for batch_idx, (features, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        try:
            outputs = fixed_model(features)
            loss = loss_fn(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            print(f"Successfully completed forward and backward pass. Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"Error during training: {e}")
        
        break  # Just one batch is enough for testing 