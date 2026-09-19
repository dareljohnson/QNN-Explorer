"""
Hardware-compatible quantum model configuration for cats vs dogs dataset.
This configuration uses expectation values instead of full state output,
making it compatible with hardware backends and parameter-shift differentiation.
"""

def get_hardware_compatible_config(num_qubits=10, num_layers=8):
    """
    Returns a model configuration that is compatible with quantum hardware.
    
    This configuration uses expectation values instead of full state output,
    making it compatible with parameter-shift differentiation which is the
    method of choice for real quantum hardware.
    
    Args:
        num_qubits: Number of qubits to use (default: 10)
        num_layers: Number of ansatz layers (default: 8)
        
    Returns:
        dict: Configuration dictionary for a hardware-compatible model
    """
    from core.quantum_models import ansatz1
    
    config = {
        'classical_backbone_type': 'cnn',
        'classical_model_name': 'resnet18',
        'num_qubits': num_qubits,
        'num_layers': num_layers,
        'ansatz_func': ansatz1,
        'use_gpu': True,
        # Use expectation values instead of state vector for hardware compatibility
        'observation_type': 'expval',
        'num_classes': 2  # For cats vs dogs
    }
    
    print("Created hardware-compatible model configuration with:")
    print(f"- {num_qubits} qubits")
    print(f"- {num_layers} layers")
    print(f"- Using expectation values (hardware-compatible)")
    print(f"- Classification with 2 classes (cats/dogs)")
    
    return config

if __name__ == "__main__":
    # Example of how to use this configuration
    config = get_hardware_compatible_config()
    
    print("\nTo use this configuration:")
    print("```python")
    print("from hardware_compatible_config import get_hardware_compatible_config")
    print("from core.fusion import HybridModel")
    print("")
    print("# Get hardware-compatible configuration")
    print("config = get_hardware_compatible_config()")
    print("")
    print("# Create model with hardware-compatible settings")
    print("model = HybridModel(config)")
    print("```") 