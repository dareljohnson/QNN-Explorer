"""
Test file to verify that our fix for quantum state gradient issues works.
This test creates a simple quantum model in both state vector and expectation value modes
and verifies that it can perform forward and backward passes successfully.
"""
import torch
import pennylane as qml

# Try both observation types
OBSERVATION_TYPES = ['state', 'expval']

def test_quantum_gradient(observation_type='state', num_qubits=4, num_layers=2):
    """Test that a quantum circuit can compute gradients with the given observation type."""
    print(f"\nTesting quantum gradients with observation_type='{observation_type}'")
    
    # Import necessary functions from our codebase
    from core.quantum_models import create_qnn, ansatz1, amplitude_embedding
    
    # Create a quantum circuit
    qnn, num_params = create_qnn(
        num_qubits=num_qubits,
        num_layers=num_layers,
        ansatz_func=ansatz1,
        use_gpu=torch.cuda.is_available(),
        observation_type=observation_type
    )
    
    # Create a TorchLayer for the quantum circuit
    quantum_layer = qml.qnn.TorchLayer(qnn, weight_shapes={"params": num_params})
    
    # Create a small network with the quantum layer
    class SimpleQuantumModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.quantum_layer = quantum_layer
            
            # For output processing
            output_size = 2**num_qubits if observation_type.lower() == 'state' else num_qubits
            self.output_head = torch.nn.Linear(output_size, 2)  # 2 classes (e.g., cats vs dogs)
        
        def forward(self, x):
            # Process each sample individually to avoid broadcasting issues
            batch_size = x.shape[0]
            results = []
            
            for i in range(batch_size):
                # Get single sample without batch dimension
                sample = x[i]
                
                # Normalize for amplitude embedding
                normalized_sample = amplitude_embedding(sample)
                
                # Process through quantum layer - one sample at a time
                quantum_output = self.quantum_layer(normalized_sample)
                
                # Handle complex outputs from state vector
                if torch.is_complex(quantum_output):
                    quantum_output = quantum_output.abs()
                    
                # Add batch dimension back
                results.append(quantum_output.unsqueeze(0))
            
            # Combine results
            combined_results = torch.cat(results, dim=0)
                
            # Process with classical output head
            return self.output_head(combined_results)
    
    # Create model and prepare for training
    model = SimpleQuantumModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Create dummy data - IMPORTANT CORRECTION: Each sample needs to be size 2^num_qubits
    # for amplitude embedding
    input_size = 2**num_qubits
    print(f"Creating input data of shape [2, {input_size}] for {num_qubits} qubits")
    inputs = torch.rand(2, input_size)
    
    # Normalize inputs (optional as amplitude_embedding will normalize again)
    inputs = inputs / torch.norm(inputs, dim=1, keepdim=True)
    labels = torch.tensor([0, 1])  # Binary classification
    
    # Try forward and backward pass
    try:
        # Forward pass
        print("Performing forward pass...")
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        print(f"Forward pass successful, loss: {loss.item():.4f}")
        
        # Backward pass
        print("Performing backward pass...")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Backward pass successful!")
        
        return True
    except Exception as e:
        print(f"Error during test: {e}")
        return False

if __name__ == "__main__":
    print("Testing quantum gradient computation with different observation types")
    
    results = {}
    for obs_type in OBSERVATION_TYPES:
        results[obs_type] = test_quantum_gradient(observation_type=obs_type)
    
    # Print summary
    print("\n--- SUMMARY ---")
    for obs_type, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"Observation type '{obs_type}': {status}")
    
    # Overall result
    if all(results.values()):
        print("\n✅ ALL TESTS PASSED - The fix is working! Both state vector and expectation value modes work.")
    else:
        print("\n❌ SOME TESTS FAILED - Check the errors above for details.") 