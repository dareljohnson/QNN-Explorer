import pennylane as qml
from pennylane import numpy as np # Use PennyLane's wrapped NumPy
import torch
import sys
import os

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import tensor utilities
try:
    from utils.tensor_utils import ensure_real
except ImportError:
    # Fallback if utils isn't installed
    def ensure_real(tensor_or_dict, eps=1e-10, target_dtype=torch.float32):
        if isinstance(tensor_or_dict, torch.Tensor):
            tensor = tensor_or_dict
            # Keep indices/masks integral; casting token ids to float breaks
            # nn.Embedding lookups inside transformer backbones.
            if tensor.dtype in (torch.bool, torch.uint8, torch.int8, torch.int16,
                                torch.int32, torch.int64):
                return tensor
            if torch.is_complex(tensor):
                print(f"Converting complex tensor with shape {tensor.shape} to real")
                tensor = tensor.abs()
            if tensor.dtype != target_dtype:
                print(f"Converting tensor dtype from {tensor.dtype} to {target_dtype}")
                tensor = tensor.to(target_dtype)
            return tensor
        return tensor_or_dict

def get_device(num_qubits, use_gpu=False):
    """Selects the appropriate PennyLane device."""
    # Consider lightning.gpu if performance becomes critical and it's installed/compatible
    # For now, leverage PyTorch's CUDA backend via default.qubit.torch
    if use_gpu and torch.cuda.is_available():
        try:
            # Try to use GPU-accelerated device
            print("Attempting to use GPU-accelerated quantum simulation...")
            return qml.device('default.qubit.torch', wires=num_qubits)
        except Exception as e:
            print(f"Error creating GPU device: {e}")
            print("To enable GPU acceleration for PennyLane, you need:")
            print("  1. PyTorch with CUDA support:")
            print("     pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html")
            print("  2. For even faster simulation (optional):")
            print("     pip install pennylane-lightning-gpu")
            print("Falling back to CPU device.")
            return qml.device('default.qubit', wires=num_qubits)
    else:
        if use_gpu and not torch.cuda.is_available():
            print("GPU requested but CUDA not available. Using CPU instead.")
        else:
            print("Using default.qubit for CPU simulation.")
        return qml.device('default.qubit', wires=num_qubits)

def amplitude_embedding(features, target_dtype=torch.float32):
    """Normalizes and prepares features for AmplitudeEmbedding."""
    # Ensure input is a torch tensor
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features)

    # Convert complex tensor to real if needed and ensure consistent dtype
    features = ensure_real(features, target_dtype=target_dtype)

    # If this is a batch with a single sample, remove batch dimension for quantum processing
    if len(features.shape) > 1 and features.shape[0] == 1:
        features = features.squeeze(0)
    
    # Ensure the input is flattened (for quantum processing)
    if len(features.shape) > 1:
        features = features.reshape(-1)
        
    # Normalize the feature vector
    norm = torch.norm(features, p=2, keepdim=True)
    # Add small epsilon to prevent division by zero
    normalized_features = features / (norm + 1e-10)
    
    # Final safety check for NaN values
    if torch.isnan(normalized_features).any():
        print("Warning: NaN values detected in normalized features, replacing with zeros")
        normalized_features = torch.nan_to_num(normalized_features, nan=0.0)
        
    # Final safety check for consistent dtype
    normalized_features = ensure_real(normalized_features, target_dtype=target_dtype)
    
    return normalized_features


def ansatz1(params, num_qubits, num_layers):
    """Ansatz 1: Layers of RY, RZ rotations and CNOT entanglers."""
    param_idx = 0
    
    # Start with Hadamard gates on all qubits
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
        
    for layer in range(num_layers):
        # Rotation layer
        for i in range(num_qubits):
            qml.RY(params[param_idx], wires=i)
            param_idx += 1
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1

        # Entanglement layer (linear chain)
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        # Add a final CNOT for circular entanglement
        if layer == num_layers - 1 and num_qubits > 2:
            qml.CNOT(wires=[num_qubits - 1, 0])


def hardware_efficient_ansatz(params, num_qubits, num_layers):
    """
    Hardware-efficient ansatz: Designed to maximize entanglement while using
    gates that are typically efficient to implement on quantum hardware.
    
    Args:
        params: Parameters for the circuit
        num_qubits: Number of qubits
        num_layers: Number of circuit layers
        
    Requires: 3 * num_qubits * num_layers parameters
    """
    param_idx = 0
    
    # Create special superposition state with phase differences (higher entanglement potential)
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
        # Add phase that varies by position to break symmetry
        qml.PhaseShift(np.pi/3 * (i+1), wires=i)
    
    # First create a GHZ-like entangled state for high initial entanglement
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    # Complete the cycle
    if num_qubits > 2:
        qml.CNOT(wires=[num_qubits - 1, 0])
        
    # Add an additional layer of parity-preserving gates for higher entanglement
    for i in range(0, num_qubits, 2):
        next_idx = (i + 1) % num_qubits
        qml.CZ(wires=[i, next_idx])
    
    # Add parameterized layers
    for layer in range(num_layers):
        # Rotation layer with full X, Y, Z rotations on each qubit
        for i in range(num_qubits):
            qml.RX(params[param_idx], wires=i)
            param_idx += 1
            qml.RY(params[param_idx], wires=i)
            param_idx += 1
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1
        
        # Entangling layer
        # Alternating direction CNOTs for enhanced entanglement
        if layer % 2 == 0:
            # Forward direction
            for i in range(num_qubits):
                target = (i + 1) % num_qubits
                qml.CNOT(wires=[i, target])
        else:
            # Reverse direction
            for i in range(num_qubits):
                target = (i - 1) % num_qubits
                qml.CNOT(wires=[i, target])
        
        # Add all-to-all connectivity with CZ gates for even more entanglement
        if layer == num_layers - 1:  # Last layer
            for i in range(num_qubits):
                for j in range(i+2, num_qubits):  # Skip adjacent qubits (already entangled)
                    qml.CZ(wires=[i, j])
    
    # Final special rotations to maximize entanglement measure
    # These specific angles were found to increase the Meyer-Wallach measure
    for i in range(num_qubits):
        qml.RY(np.pi/2, wires=i)  # Creates maximal superpositions
    
    # Final entangling layer for maximum connectivity
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[i, i + 1])


def ghz_type_ansatz(params, num_qubits, num_layers):
    """
    GHZ-type ansatz: Creates entanglement similar to GHZ states (|000...0〉 + |111...1〉).
    
    Args:
        params: Parameters for the circuit
        num_qubits: Number of qubits
        num_layers: Number of circuit layers
        
    Requires: 3 * num_qubits * num_layers parameters
    """
    param_idx = 0
    
    # Create GHZ state
    qml.Hadamard(wires=0)
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    
    # Parameterized layers
    for layer in range(num_layers):
        # Rotation layer
        for i in range(num_qubits):
            qml.RX(params[param_idx], wires=i)
            param_idx += 1
            qml.RY(params[param_idx], wires=i)
            param_idx += 1
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1
            
        # Star entanglement pattern (all qubits entangled with qubit 0)
        if num_qubits > 1:
            for i in range(1, num_qubits):
                qml.CNOT(wires=[0, i])


def brickwall_ansatz(params, num_qubits, num_layers):
    """
    Brickwall ansatz: Alternating layers of entangling gates in a "brick wall" pattern.
    
    This alternates between even and odd qubit pairs for entanglement, creating a pattern 
    that efficiently entangles all qubits and is known for generating highly entangled states.
    
    Args:
        params: Parameters for the circuit
        num_qubits: Number of qubits
        num_layers: Number of circuit layers
        
    Requires: 2 * num_qubits * num_layers parameters
    """
    param_idx = 0
    
    # Start with a superposition on all qubits
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
    
    for layer in range(num_layers):
        # Rotation layer first
        for i in range(num_qubits):
            qml.RY(params[param_idx], wires=i)
            param_idx += 1
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1
        
        # Even layer: entangle qubits (0,1), (2,3), ...
        if layer % 2 == 0:
            for i in range(0, num_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
        
        # Odd layer: entangle qubits (1,2), (3,4), ...
        else:
            for i in range(1, num_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])
        
        # Connect the ends to ensure all qubits can get entangled
        if num_qubits > 2:
            qml.CNOT(wires=[num_qubits - 1, 0])


def quantum_volume_ansatz(params, num_qubits, num_layers):
    """
    Quantum Volume ansatz: Based on the quantum volume circuit design.
    
    This implements a pattern similar to IBM's Quantum Volume benchmark circuits,
    with random two-qubit gates applied in a pattern that maximizes entanglement
    between all pairs of qubits, often used to benchmark quantum computers.
    
    Args:
        params: Parameters for the circuit
        num_qubits: Number of qubits
        num_layers: Number of circuit layers
        
    Requires: 3 * num_qubits * num_layers parameters
    """
    param_idx = 0
    
    # Start with superposition on all qubits
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
    
    for layer in range(num_layers):
        # First apply single-qubit SU(2) rotations (using three angles)
        for i in range(num_qubits):
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1
            qml.RY(params[param_idx], wires=i)
            param_idx += 1
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1
        
        # Apply two-qubit gates in a pattern similar to the Quantum Volume circuit model
        # Pair qubits based on a permutation that depends on the layer
        for pair_idx in range(num_qubits // 2):
            # Determine the pair of qubits to entangle
            # This creates a different pairing pattern for each layer
            q1 = (pair_idx * 2 + layer) % num_qubits
            q2 = (pair_idx * 2 + 1 + layer) % num_qubits
            
            # Apply the CNOT (or CZ) gate to the pair
            qml.CNOT(wires=[q1, q2])
        
        # Add additional entangling gates for better connectivity
        if layer % 2 == 0 and num_qubits > 3:
            # Add some extra entanglement between distant qubits
            for i in range(num_qubits):
                # Connect to a qubit that's halfway around the register
                target = (i + num_qubits//2) % num_qubits
                qml.CZ(wires=[i, target])


def create_qnn(num_qubits, num_layers, ansatz_func, use_gpu=False, observation_type='state'):
    """Creates a QNode representing the Quantum Neural Network."""
    dev = get_device(num_qubits, use_gpu)

    # Calculate the number of parameters needed by the ansatz
    # The parameter count depends on the specific ansatz function
    if ansatz_func == ansatz1 or ansatz_func == brickwall_ansatz:
        # These ansatzes use 2 params (RY, RZ) per qubit per layer
        num_params = 2 * num_qubits * num_layers
    elif ansatz_func == hardware_efficient_ansatz or ansatz_func == ghz_type_ansatz or ansatz_func == quantum_volume_ansatz:
        # These ansatzes use 3 params (RX, RY, RZ) or (RZ, RY, RZ) per qubit per layer
        num_params = 3 * num_qubits * num_layers
    else:
        # Default to 2 params per qubit per layer if unknown
        num_params = 2 * num_qubits * num_layers
        print(f"Warning: Unknown ansatz function. Assuming 2 parameters per qubit per layer.")
    
    # Convert observation_type to lowercase outside the inner function
    observation_type_lower = observation_type.lower()
    is_state_output = observation_type_lower in ('state', 'state vector')
    is_expval_output = observation_type_lower in (
        'expval', 'expectation value', 'expectation value (pauliz)')
    
    # Select appropriate differentiation method based on observation type
    # parameter-shift doesn't work with quantum state output but backprop does
    if is_state_output:
        print("Using 'backprop' differentiation method for state vector output")
        diff_method = 'backprop'
    else:
        print("Using 'parameter-shift' differentiation method for expectation values")
        diff_method = 'parameter-shift'

    @qml.qnode(dev, interface='torch', diff_method=diff_method)
    def _qnode(inputs, params):
        # Amplitude Encoding
        # The input 'inputs' MUST be pre-normalized and have size 2**num_qubits
        qml.AmplitudeEmbedding(inputs, wires=range(num_qubits), normalize=False, pad_with=0.)

        # Variational Ansatz
        ansatz_func(params, num_qubits, num_layers)

        # Measurement / Output
        if is_state_output:
            # Returns the full state vector |psi> = Sum(c_i |i>)
            # Output shape: (2**num_qubits,)
            return qml.state()
        elif is_expval_output:
            # Expectation value of Pauli Z on each qubit
            # Output shape: (num_qubits,)
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
        else:
            raise ValueError(f"Unsupported observation_type: {observation_type}")

    def quantum_circuit(inputs, params):
        """Run the QNode and return a single tensor with a consistent dtype.

        PennyLane returns a Python list for multiple expectation values and uses
        float64/complex128 by default.  Downstream code (the output head,
        autograd) expects a single float32/complex64 tensor, so normalise here.
        """
        result = _qnode(inputs, params)
        if is_state_output:
            tensor = result if isinstance(result, torch.Tensor) else torch.as_tensor(result)
            # Default qubit simulation is complex128; use complex64 for consistency.
            return tensor.to(torch.complex64)

        # Expectation values: PennyLane gives a list/tuple of scalars -> stack.
        if isinstance(result, (list, tuple)):
            tensors = [r if isinstance(r, torch.Tensor) else torch.as_tensor(r) for r in result]
            result = torch.stack(tensors)
        else:
            result = result if isinstance(result, torch.Tensor) else torch.as_tensor(result)
        return result.to(torch.float32)

    print(f"Created QNode with {num_qubits} qubits, {num_layers} layers, {num_params} parameters.")
    return quantum_circuit, num_params

def create_quantum_layer(num_qubits, num_layers, ansatz_func, use_gpu=False, observation_type='state'):
    """Creates a quantum layer function that properly handles batched inputs."""
    quantum_circuit, num_params = create_qnn(num_qubits, num_layers, ansatz_func, use_gpu, observation_type)
    
    # Initialize random parameters for the circuit
    # NOTE: Simply creating a Parameter here doesn't associate it with a module
    # so the optimizer won't see it - we'll register it properly in HybridModel
    initial_params = torch.randn(num_params) * 0.1
    
    def quantum_layer_function(x, params):
        """
        Process inputs through the quantum circuit, handling batches appropriately.
        
        Args:
            x: Input tensor with shape [batch_size, features]
            params: Tensor containing the quantum circuit parameters
            
        Returns:
            Tensor with shape [batch_size, output_features]
            where output_features is 2^num_qubits for state vector or num_qubits for expectation values
        """
        # Ensure consistent dtype and handle complex tensors
        target_dtype = torch.float32  # Default dtype for better compatibility
        x = ensure_real(x, target_dtype=target_dtype)
            
        # Ensure parameters require gradients for proper backprop and have consistent dtype
        if not params.requires_grad:
            print("WARNING: Parameters don't require gradients! Creating a differentiable copy.")
            # This should generally not happen since we register params as nn.Parameter
            params = params.clone().detach().requires_grad_(True)
        
        # Ensure parameters have consistent dtype
        params = ensure_real(params, target_dtype=target_dtype)
            
        # Process each sample in the batch individually for quantum operations
        # to avoid broadcasting issues with MottonenStatePreparation
        batch_size = x.shape[0]
        
        # Determine output size based on observation type
        observation_type_lower = observation_type.lower()
        if observation_type_lower == 'state' or observation_type_lower == 'state vector':
            output_size = 2**num_qubits
        else:  # expectation values
            output_size = num_qubits
            
        # Handle batch processing
        if batch_size > 1:
            # Process each sample individually
            quantum_outputs = []
            for i in range(batch_size):
                try:
                    # Get single sample (as a vector, not keeping batch dimension)
                    single_sample = x[i]
                    
                    # Ensure the sample has consistent dtype
                    single_sample = ensure_real(single_sample, target_dtype=target_dtype)
                        
                    # Normalize for amplitude embedding
                    quantum_input = amplitude_embedding(single_sample, target_dtype=target_dtype)
                    
                    # Process through quantum circuit
                    single_output = quantum_circuit(quantum_input, params)
                    
                    # Ensure the output has consistent dtype
                    single_output = ensure_real(single_output, target_dtype=target_dtype)
                        
                    quantum_outputs.append(single_output.unsqueeze(0))  # Add batch dim back
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    # Create fallback tensor that requires gradients
                    if observation_type_lower == 'state' or observation_type_lower == 'state vector':
                        # Create a dummy tensor with parameters as an input to ensure gradient flow
                        fallback = torch.zeros(1, 2**num_qubits, device=x.device, dtype=target_dtype)
                        # Make a small contribution from parameters to ensure gradient flow
                        fallback = fallback + 0.0 * params.sum() * 0.0  # Doesn't change values but connects graph
                    else:
                        fallback = torch.zeros(1, num_qubits, device=x.device, dtype=target_dtype)
                        fallback = fallback + 0.0 * params.sum() * 0.0  # Connect to params for gradient flow
                    quantum_outputs.append(fallback)
            
            # Combine results
            quantum_output = torch.cat(quantum_outputs, dim=0)
            
            # Final safety check for dtype consistency
            quantum_output = ensure_real(quantum_output, target_dtype=target_dtype)
        else:
            # Single sample case
            try:
                # Get the sample without batch dimension
                single_sample = x.squeeze(0)
                
                # Ensure the sample has consistent dtype
                single_sample = ensure_real(single_sample, target_dtype=target_dtype)
                
                # Normalize for amplitude embedding
                quantum_input = amplitude_embedding(single_sample, target_dtype=target_dtype)
                
                # Process through quantum circuit
                quantum_output = quantum_circuit(quantum_input, params)
                
                # Ensure the output has consistent dtype
                quantum_output = ensure_real(quantum_output, target_dtype=target_dtype)
                
                # Ensure output has batch dimension for consistency
                if len(quantum_output.shape) == 1:
                    quantum_output = quantum_output.unsqueeze(0)
            except Exception as e:
                print(f"Error processing single sample: {e}")
                # Create fallback that requires gradients
                if observation_type_lower == 'state' or observation_type_lower == 'state vector':
                    # Create a dummy tensor with parameters as input to ensure gradient flow
                    quantum_output = torch.zeros(1, 2**num_qubits, device=x.device, dtype=target_dtype)
                    # Connect to parameters for gradient flow
                    quantum_output = quantum_output + 0.0 * params.sum() * 0.0
                else:
                    quantum_output = torch.zeros(1, num_qubits, device=x.device, dtype=target_dtype)
                    # Connect to parameters for gradient flow
                    quantum_output = quantum_output + 0.0 * params.sum() * 0.0
                
        # Final safety check for dtype consistency
        quantum_output = ensure_real(quantum_output, target_dtype=target_dtype)
        
        return quantum_output
    
    return quantum_layer_function, initial_params, num_params 