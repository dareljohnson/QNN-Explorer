import torch
import torch.nn as nn
import pennylane as qml
from .classical_models import get_cnn_model, get_transformer_model, get_gnn_model, get_regression_model, GNNWrapper
from .quantum_models import amplitude_embedding, ansatz1, create_qnn, create_quantum_layer, hardware_efficient_ansatz, ghz_type_ansatz, brickwall_ansatz, quantum_volume_ansatz
import numpy as np
import logging

# Define available ansatzes that can be used in the quantum circuit
AVAILABLE_ANSATZES = {
    "Ansatz 1 (Rot+CNOT Chain)": ansatz1,
    "Hardware Efficient (All-to-All)": hardware_efficient_ansatz,
    "GHZ-type (Star Topology)": ghz_type_ansatz,
    "Brickwall (Alternating Pairs)": brickwall_ansatz,
    "Quantum Volume (Maximally Entangling)": quantum_volume_ansatz,
    # Add more ansatzes as they become available
    # "Ansatz 2 (Rot+CZ AllPairs)": None,  # To be implemented later
}

class ComplexModuleWrapper(nn.Module):
    """
    Wrapper for nn.Module instances that handles complex-to-real conversion automatically.
    This ensures any module that expects real inputs will work even if given complex inputs.
    """
    def __init__(self, module):
        super().__init__()
        self.module = module
        
    def forward(self, x):
        # Convert complex inputs to real before passing to the wrapped module
        if torch.is_complex(x):
            print(f"ComplexModuleWrapper: Converting complex input (dtype={x.dtype}) to real")
            x = x.abs()
        return self.module(x)


class HybridModel(nn.Module):
    def __init__(self, config):
        """
        Initializes the Hybrid Quantum-Classical Model.

        Args:
            config (dict): Configuration dictionary containing keys like:
                'classical_backbone_type': 'cnn', 'gnn', 'transformer', 'regression', or 'none'
                'classical_model_name': e.g., 'resnet18', 'bert-base-uncased', 'gcn', 'linear'
                'gnn_input_features': (Required for GNN) Number of node features.
                'num_qubits' (N): Number of qubits.
                'num_layers' (L): Number of layers in the QNN ansatz.
                'ansatz_func': The ansatz function to use (e.g., ansatz1).
                'use_gpu': Boolean flag for GPU usage.
                'observation_type': 'state' or 'expval'.
                'encoding_method': 'amplitude' or other encoding method.
                'regression_output_size': Number of regression outputs (for housing price prediction).
        """
        super().__init__()
        self.config = config
        self.classical_backbone_type = config.get('classical_backbone_type', 'none')
        self.num_qubits = config.get('num_qubits', 4)
        self.encoding_method = config.get('encoding_method', 'Amplitude Encoding')
        self.regression_output_size = config.get('regression_output_size')
        
        # GPU config with more robust handling
        requested_gpu = config.get('use_gpu', False)
        has_cuda = torch.cuda.is_available()
        self.use_gpu = requested_gpu and has_cuda
        
        if requested_gpu and not has_cuda:
            print("Warning: GPU was requested but CUDA is not available. Using CPU instead.")
            
        # Check if GPU is being used overall
        device_info = f"Using device: {'CUDA' if self.use_gpu else 'CPU'}"
        if self.use_gpu:
            device_info += f" ({torch.cuda.get_device_name()})"
        print(device_info)
        
        self.observation_type = config.get('observation_type', 'state vector')
        self.param_update_counter = 0
        
        # Calculate quantum input size early - needed by multiple parts of the code
        self.quantum_input_size = 2**self.num_qubits
        
        # Storage for the quantum output before it passes through the output head
        self.last_quantum_output = None

        # --- Classical Backbone ---
        self.classical_backbone = None
        classical_feature_size = None
        backbone_type = self.classical_backbone_type.lower()  # Convert to lowercase for consistent matching
        
        if backbone_type == 'cnn':
            self.classical_backbone, classical_feature_size = get_cnn_model(
                model_name=config['classical_model_name']
            )
        elif backbone_type == 'transformer':
            self.classical_backbone, classical_feature_size = get_transformer_model(
                model_name=config['classical_model_name']
            )
        elif backbone_type == 'gnn':
            gnn_base, gnn_feature_size = get_gnn_model(
                model_name=config['classical_model_name'],
                input_features=config.get('gnn_input_features') # Need this from data
            )
            # Wrap the GNN to handle graph-level feature extraction (pooling)
            self.classical_backbone = GNNWrapper(gnn_base) # Uses mean pooling by default
            classical_feature_size = gnn_feature_size
        elif backbone_type == 'regression':
            # Add support for regression models (for house price prediction)
            self.classical_backbone, classical_feature_size = get_regression_model(
                model_name=config['classical_model_name'],
                output_features=config.get('regression_output_size', 1)
            )
            # Note: classical_feature_size might be None initially, will be set during forward pass
        elif backbone_type == 'none':
            # If no classical backbone, the input features must match quantum_input_size
            print("No classical backbone selected. Input data must have size {}".format(self.quantum_input_size))
            classical_feature_size = self.quantum_input_size
        else:
            raise ValueError(f"Unsupported classical_backbone_type: {self.classical_backbone_type}")

        # --- Fusion Layer ---
        # This layer maps the classical features to the size required for amplitude embedding (2^N)
        if self.classical_backbone is not None and classical_feature_size is not None:
            print(f"Classical feature size: {classical_feature_size}, Quantum input size (2^N): {self.quantum_input_size}")
            fusion = nn.Linear(classical_feature_size, self.quantum_input_size)
            # Wrap the fusion layer to handle complex inputs automatically
            self.fusion_layer = ComplexModuleWrapper(fusion)
        else:
            # If no backbone or dynamic feature size, we'll create the fusion layer later
            # or use identity if input size matches quantum input size
            self.fusion_layer = nn.Identity() # Default to identity for now
            # We'll check the actual input size during the first forward pass

        # --- Quantum Layer ---
        # Create the quantum circuit
        print(f"Creating quantum circuit with {self.num_qubits} qubits, {config.get('num_layers', 2)} layers")
        # Quantum layer returns a tuple of (circuit_fn, initial_params, num_params)
        self.quantum_circuit_fn, initial_params, num_params = create_quantum_layer(
            num_qubits=self.num_qubits,
            num_layers=config.get('num_layers', 2),
            ansatz_func=config.get('ansatz_func', ansatz1), # Default to ansatz1 if not specified
            use_gpu=self.use_gpu,
            observation_type=self.observation_type
        )
        
        # Register the quantum parameters as a module parameter
        self.quantum_params = nn.Parameter(initial_params)
        self.num_quantum_params = num_params
        print(f"Registered {num_params} quantum circuit parameters")
        
        # Optional: Add a final classical head if doing classification/regression
        # For now, we output the quantum result (state vector or expvals)
        # self.output_head = nn.Linear(...) # Example

        # --- Optional Classical Output Head ---
        self.output_head = None
        quantum_output_size = 0
        # Convert observation_type to lowercase for case-insensitive comparison
        observation_type_lower = self.observation_type.lower()
        if observation_type_lower == 'state' or observation_type_lower == 'state vector':
            # State vector output size is complex (2**N)
            # Using it directly for classification/regression might be tough
            # Usually, you'd measure expectation values instead.
            # For now, let's assume we use its magnitude for simplicity if needed.
            quantum_output_size = self.quantum_input_size # 2**N
            print("Warning: Using state vector directly for output head might be inefficient.")
        elif observation_type_lower == 'expval' or observation_type_lower == 'expectation value' or observation_type_lower == 'expectation value (pauliz)':
            # Expectation values output size is num_qubits
            quantum_output_size = self.num_qubits
        else:
            raise ValueError(f"Unsupported observation_type: {self.observation_type}")

        # --- Residual / skip path ---
        # Give the classifier direct access to the classical features (in
        # addition to the quantum features). This keeps hybrid models accurate
        # even when the quantum layer is a narrow bottleneck (only 2**N or N
        # features). Can be disabled via config['use_classical_skip'] = False.
        self.use_classical_skip = config.get('use_classical_skip', True)
        backbone_type_lower = (self.classical_backbone_type or 'none').lower()
        # With no classical backbone the input feature count is only known at
        # run time (e.g. TF-IDF vectors), so the head is built lazily.
        self._dynamic_input = (backbone_type_lower == 'none')
        if self.use_classical_skip:
            if self._dynamic_input:
                self.skip_size = None  # determined per batch
            elif classical_feature_size is not None:
                self.skip_size = classical_feature_size
            else:
                self.skip_size = 0
        else:
            self.skip_size = 0

        self.quantum_output_size = quantum_output_size
        self.head_hidden = config.get('head_hidden_size', 256)
        self.head_dropout = config.get('dropout', 0.1)
        self._head_out_size = None
        if config.get('num_classes') is not None and config['num_classes'] > 0:
            self._head_out_size = config['num_classes']
        elif config.get('regression_output_size') is not None and config['regression_output_size'] > 0:
            self._head_out_size = config['regression_output_size']

        self.output_head = None
        if self._head_out_size is not None and not self._dynamic_input:
            self._build_head(quantum_output_size + (self.skip_size or 0))

    def _build_head(self, head_input_size):
        """Create the (MLP) output head for the given input feature size."""
        out_size = self._head_out_size
        if self.head_hidden and self.head_hidden > 0:
            module = nn.Sequential(
                nn.Linear(head_input_size, self.head_hidden),
                nn.GELU(),
                nn.Dropout(self.head_dropout),
                nn.Linear(self.head_hidden, out_size),
            )
        else:
            module = nn.Linear(head_input_size, out_size)
        self.output_head = ComplexModuleWrapper(module)
        try:
            self.output_head = self.output_head.to(self.quantum_params.device)
        except Exception:
            pass
        print(f"Built output head: in={head_input_size} -> out={out_size}")
        return self.output_head

    def forward(self, x):
        """
        Forward pass through the model. This will:
        1. Pass inputs through the classical backbone (if present)
        2. Process quantum features through QNN
        3. Apply output transformation (if needed)
        
        Args:
            x: Input data which can be:
               - Tensor of shape [batch_size, input_features] for direct quantum input
               - Dictionary of tensors for transformer inputs
               - Image tensor [batch_size, channels, height, width] for CNN backbones
        
        Returns:
            Tensor of processed outputs
        """
        # Check for complex input at the very beginning
        if isinstance(x, torch.Tensor) and torch.is_complex(x):
            print(f"Converting complex input tensor (dtype={x.dtype}) to real values at model entry point")
            x = x.abs()

        # Adapt 1-D/2-D tabular input to the shape expected by an image backbone.
        x = self._prepare_backbone_input(x)

        batch_size = self.get_batch_size(x)
        print(f"Processing batch of size {batch_size}")
        
        # --- Classical Feature Extraction ---
        if self.classical_backbone is not None:
            try:
                classical_features = self.classical_backbone(x)
                print(f"Classical backbone output: {classical_features.shape}")
                
                # Check if classical features are complex and convert to real if necessary
                if torch.is_complex(classical_features):
                    print(f"Converting complex classical features to real values")
                    classical_features = classical_features.abs()
            except Exception as e:
                message = str(e)
                if (self.classical_backbone_type or '').lower() == 'cnn' and (
                        'conv2d' in message or 'Expected 3D' in message or 'Expected 4D' in message):
                    features = x.shape[1] if isinstance(x, torch.Tensor) and x.dim() == 2 else 'unknown'
                    raise ValueError(
                        f"Input with {features} features cannot be reshaped into an RGB image "
                        f"for the CNN backbone. Please provide image-shaped input or use a "
                        f"'None' backbone.") from e
                print(f"Error in classical backbone: {e}")
                raise
        else:
            # Direct input to quantum - x should already be a tensor
            classical_features = x
            
            # Check if input is complex and convert to real if necessary
            if torch.is_complex(classical_features):
                print(f"Converting complex input tensor to real values")
                classical_features = classical_features.abs()
        
        # Keep a copy of the classical features for the residual/skip path
        # (before the fusion layer transforms them into the quantum input).
        skip_features = classical_features
        if isinstance(skip_features, torch.Tensor) and skip_features.dim() == 1:
            skip_features = skip_features.unsqueeze(0)

        # Apply fusion layer if present and not Identity
        if hasattr(self, 'fusion_layer') and self.fusion_layer is not None and not isinstance(self.fusion_layer, nn.Identity):
            try:
                # Ensure no complex values enter the fusion layer
                if torch.is_complex(classical_features):
                    print(f"Converting complex features to real before fusion layer")
                    classical_features = classical_features.abs()
                    
                classical_features = self.fusion_layer(classical_features)
                
                # Check if fusion output is complex and convert to real if necessary
                if torch.is_complex(classical_features):
                    print(f"Converting complex fusion output to real values")
                    classical_features = classical_features.abs()
            except Exception as e:
                error_msg = str(e)
                if "mat1 and mat2 must have the same dtype" in error_msg:
                    print(f"Dtype mismatch in fusion layer. Converting to real values before fusion.")
                    # Try to convert to real and reapply fusion
                    if torch.is_complex(classical_features):
                        classical_features = classical_features.abs()
                        classical_features = self.fusion_layer(classical_features)
                else:
                    print(f"Error in fusion layer: {e}")
                    raise
                    
        # Universal safety check for complex values after fusion - backup in case previous checks missed something
        if torch.is_complex(classical_features):
            print(f"Converting complex features to real values after fusion (final safety check)")
            classical_features = classical_features.abs()

        # No classical backbone: the input may not match 2**N (e.g. TF-IDF
        # vectors). Project it to the quantum input size for the quantum branch
        # while the skip path keeps the original features.
        if self.classical_backbone is None and isinstance(classical_features, torch.Tensor):
            feat_dim = classical_features.shape[-1]
            if feat_dim != self.quantum_input_size:
                if getattr(self, '_input_proj', None) is None or self._input_proj.in_features != feat_dim:
                    self._input_proj = nn.Linear(feat_dim, self.quantum_input_size).to(classical_features.device)
                    print(f"Created input projection {feat_dim} -> {self.quantum_input_size} for the quantum branch")
                classical_features = self._input_proj(classical_features)
        
        # --- Quantum Processing ---
        
        # Prepare for per-sample processing if needed
        needs_individual_processing = False
        single_outputs = []
        
        # Check type and shape of classical features
        fusion_output_type = type(classical_features).__name__
        fusion_output_shape = getattr(classical_features, 'shape', 'unknown')
        fusion_output_device = getattr(classical_features, 'device', 'unknown')
        print(f"Fusion output: type={fusion_output_type}, shape={fusion_output_shape}, device={fusion_output_device}")
        
        # Handle different output types from classical backbone
        if isinstance(classical_features, torch.Tensor):
            # Make sure it's at least 2D [batch, features]
            if len(classical_features.shape) == 1:
                classical_features = classical_features.unsqueeze(0)
                
            # Check for NaN values which might cause issues
            if torch.isnan(classical_features).any():
                print("WARNING: NaN values detected in classical features")
                classical_features = torch.nan_to_num(classical_features, nan=0.0)
                
            # For batch processing, we need to process each sample individually if:
            # 1. Batch size > 1 because quantum ops often don't natively support batching
            # 2. And we're in amplitude encoding which broadcasts poorly with quantum ops
            if batch_size > 1 and self.encoding_method.lower() == 'amplitude encoding':
                needs_individual_processing = True
        else:
            # Unexpected output type - try to convert or fail
            try:
                classical_features = torch.tensor(classical_features)
                print(f"Converted non-tensor output to tensor: {classical_features.shape}")
            except:
                raise ValueError(f"Classical backbone output is not a tensor and cannot be converted: {type(classical_features)}")
        
        # Process quantum features
        if needs_individual_processing:
            # Process each sample individually
            for i in range(batch_size):
                sample_features = classical_features[i:i+1]  # Keep batch dimension
                try:
                    # Ensure no complex values before normalization
                    if torch.is_complex(sample_features):
                        sample_features = sample_features.abs()
                    
                    # Normalize for amplitude encoding (L2 norm = 1)
                    if self.encoding_method.lower() == 'amplitude encoding':
                        sample_features = sample_features / torch.norm(sample_features)
                    
                    # Process through quantum circuit
                    sample_output = self.quantum_circuit_fn(sample_features, self.quantum_params)
                    
                    # Convert complex output to real if needed
                    if torch.is_complex(sample_output):
                        # For state vector output, take absolute values (amplitudes)
                        sample_output = sample_output.abs()
                    
                    single_outputs.append(sample_output)
                except Exception as e:
                    print(f"Error in quantum processing for sample {i}: {e}")
                    # Provide a fallback output to avoid stopping the entire batch
                    if i > 0:
                        # Use the previous valid output if we have one
                        print(f"Using previous valid output for sample {i}")
                        single_outputs.append(single_outputs[-1])
                    else:
                        # For the first sample, create a zero tensor of correct expected shape
                        expected_size = 2**self.num_qubits if self.observation_type.lower() == 'state vector' else self.num_qubits
                        fallback = torch.zeros(1, expected_size, device=sample_features.device)
                        print(f"Using zero tensor for failed sample: {fallback.shape}")
                        single_outputs.append(fallback)
            
            # Combine all outputs
            quantum_features = torch.cat(single_outputs, dim=0)
            # Store for entanglement calculation
            self.last_quantum_output = quantum_features
        else:
            # Process the entire batch at once
            try:
                # Ensure no complex values before normalization
                if torch.is_complex(classical_features):
                    classical_features = classical_features.abs()
                
                # Normalize for amplitude encoding if needed
                if self.encoding_method.lower() == 'amplitude encoding':
                    # Calculate norm for each sample, reshape to broadcast properly
                    norms = torch.norm(classical_features, dim=1, keepdim=True)
                    norms = torch.clamp(norms, min=1e-8)  # Avoid division by zero
                    classical_features = classical_features / norms
                
                # Process through quantum circuit
                quantum_features = self.quantum_circuit_fn(classical_features, self.quantum_params)
                
                # Convert complex output to real if needed
                if torch.is_complex(quantum_features):
                    # For state vector output, take absolute values (amplitudes)
                    print(f"Converting complex quantum output (shape={quantum_features.shape}, dtype={quantum_features.dtype}) to real values")
                    quantum_features = quantum_features.abs()
                
                # Store for entanglement calculation
                self.last_quantum_output = quantum_features
            except Exception as e:
                print(f"Error in quantum batch processing: {e}")
                raise
        
        # Ensure quantum features are real, not complex
        if torch.is_complex(quantum_features):
            print(f"Converting complex quantum features to real before output layer (final safety check)")
            quantum_features = quantum_features.abs()
        
        # --- Output Transformation ---
        if self.output_head is not None or self._head_out_size is not None:
            # For classification/regression tasks
            # Ensure no complex values enter the output head
            if torch.is_complex(quantum_features):
                quantum_features = quantum_features.abs()

            head_input = quantum_features
            if isinstance(skip_features, torch.Tensor):
                skip = skip_features
                if torch.is_complex(skip):
                    skip = skip.abs()
                if skip.dim() == 1:
                    skip = skip.unsqueeze(0)
                skip = skip.to(device=quantum_features.device, dtype=quantum_features.dtype)
                if skip.shape[0] == quantum_features.shape[0]:
                    # Lazily build the head for dynamic input sizes ('none' backbone).
                    if self.output_head is None:
                        self._build_head(quantum_features.shape[-1] + skip.shape[-1])
                    # Align fixed skip dims (slice/pad).
                    if self.skip_size is not None and skip.shape[-1] != self.skip_size:
                        if skip.shape[-1] > self.skip_size:
                            skip = skip[..., :self.skip_size]
                        else:
                            skip = torch.nn.functional.pad(skip, (0, self.skip_size - skip.shape[-1]))
                    head_input = torch.cat([quantum_features, skip], dim=-1)

            if self.output_head is not None:
                output = self.output_head(head_input)
            else:
                output = head_input

            # Final safety check for complex output
            if torch.is_complex(output):
                output = output.abs()
        else:
            # For direct quantum output
            output = quantum_features

            # Final safety check for complex output
            if torch.is_complex(output):
                print(f"Converting complex output to real values (final check)")
                output = output.abs()
            
        # Add metadata to output tensor for regression scaling
        if hasattr(self, 'regression_output_size') and self.regression_output_size is not None:
            # Set a flag on the tensor to indicate it's a regression output that needs scaling
            # This is used in the loss function to apply appropriate scaling
            output.is_regression = True
            
        return output

    def transformer_pooling(self, x):
        """
        Apply pooling operation to transformer output to get a fixed-size representation.
        
        Args:
            x: Transformer output (typically shape [batch_size, seq_len, hidden_size])
            
        Returns:
            Pooled representation (shape [batch_size, hidden_size])
        """
        # There are several common approaches to pooling transformer outputs:
        
        # 1. Use the [CLS] token (first token) representation - most common for BERT
        # This assumes the model has been trained with a [CLS] token at the beginning
        cls_token = x[:, 0]  # Shape: [batch_size, hidden_size]
        
        # If needed, we could implement other pooling strategies:
        # 2. Mean pooling over sequence length
        # mean_pooled = torch.mean(x, dim=1)  # Shape: [batch_size, hidden_size]
        
        # 3. Max pooling over sequence length
        # max_pooled, _ = torch.max(x, dim=1)  # Shape: [batch_size, hidden_size]
        
        # Return the chosen pooling method (using CLS token by default)
        return cls_token 

    def _prepare_backbone_input(self, x):
        """Reshape tabular (2-D) input into an image tensor for a CNN backbone.

        Convolutional backbones require 4-D input ``(batch, channels, height,
        width)``.  When the incoming data is tabular we try to reinterpret it as
        an RGB image.  If that is impossible we raise a clear ``ValueError``
        instead of letting ``conv2d`` fail with a cryptic message.
        """
        if not isinstance(x, torch.Tensor):
            return x
        if (self.classical_backbone_type or '').lower() != 'cnn':
            return x

        if x.dim() == 4:
            return x
        if x.dim() == 3:
            # (batch, H, W) grayscale images -> add a channel dimension
            return x.unsqueeze(1)
        if x.dim() != 2:
            return x

        batch, features = x.shape
        pixels = features // 3
        if features % 3 != 0 or pixels == 0:
            # Cannot be interpreted as an RGB image. Leave the tensor as-is so
            # that a custom/mocked backbone can still consume it; the real CNN
            # call will surface a helpful error instead (see forward()).
            return x

        height = int(pixels ** 0.5)
        while height > 1 and pixels % height != 0:
            height -= 1
        width = pixels // height if height > 0 else 0
        if height * width != pixels:
            return x

        print(f"Reshaping tabular input {tuple(x.shape)} -> ({batch}, 3, {height}, {width})")
        return x.reshape(batch, 3, height, width)

    def get_batch_size(self, x):
        """
        Determine the batch size of the input data, handling different input types.
        
        Args:
            x: Input data - can be tensor, dictionary (for transformer inputs), etc.
            
        Returns:
            int: The batch size
        """
        logger = logging.getLogger(__name__)
        
        if isinstance(x, torch.Tensor):
            return x.shape[0]
        elif isinstance(x, dict) and 'input_ids' in x:
            return x['input_ids'].shape[0]
        else:
            # Try to infer batch size or default to 1
            try:
                return len(x)
            except:
                logger.warning("Could not determine batch size, defaulting to 1")
                return 1 