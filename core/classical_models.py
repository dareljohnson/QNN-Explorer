import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel
from sklearn.linear_model import LinearRegression
import numpy as np
# torch_geometric requires separate installation, assuming it's available
try:
    import torch_geometric.nn as pyg_nn
except ImportError:
    pyg_nn = None # Handle missing optional dependency

def get_cnn_model(model_name='resnet18', pretrained=True, output_features=None):
    """Loads a pretrained CNN model and modifies its final layer."""
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Identity() # Remove the final classification layer
    # Add other models like EfficientNet here if needed
    # elif model_name == 'efficientnet_b0':
    #     model = models.efficientnet_b0(pretrained=pretrained)
    #     num_ftrs = model.classifier[1].in_features
    #     model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported CNN model: {model_name}")

    if output_features is not None:
        # Add a new linear layer to project features if needed
        # This might be better handled in the fusion layer
        print(f"CNN base features: {num_ftrs}. Output layer added in Fusion Layer if needed.")
        # Example: model.fc = nn.Linear(num_ftrs, output_features)
        pass # Keep original features for now, projection in Fusion Layer

    return model, num_ftrs

def get_transformer_model(model_name='bert-base-uncased', output_features=None):
    """Loads a pretrained Transformer model as a fixed-size feature extractor.

    Returns ``(TransformerFeatureExtractor, hidden_size)``. The wrapper is what
    makes the model usable as a backbone: HuggingFace ``AutoModel`` takes
    keyword arguments and returns a ``BaseModelOutput``, while ``HybridModel``
    expects a single argument and a ``[batch, features]`` tensor.
    """
    model = AutoModel.from_pretrained(model_name)
    # The hidden size of the [CLS] token embedding is used as the feature vector
    num_ftrs = model.config.hidden_size

    if output_features is not None:
        # Projection to output_features will typically happen in the Fusion Layer
        print(f"Transformer base features: {num_ftrs}. Output layer added in Fusion Layer if needed.")
        pass

    return TransformerFeatureExtractor(model), num_ftrs


class TransformerFeatureExtractor(nn.Module):
    """Adapts a HuggingFace transformer for use as a classical backbone.

    Two incompatibilities are bridged here:

    1. ``HybridModel`` calls its backbone positionally, but HuggingFace models
       are keyword-only -- ``input_ids=``, ``attention_mask=``. A tokenizer
       output dict is therefore dispatched as ``**kwargs`` rather than being
       passed as the first positional argument (which used to fail with
       ``TypeError: unhashable type: 'slice'``).
    2. ``AutoModel`` returns a ``BaseModelOutput``, not a tensor, so callers
       cannot use ``.shape``. The ``last_hidden_state`` is pooled to a
       ``[batch, hidden_size]`` vector here.

    Pooling uses the ``[CLS]`` token (position 0), the standard BERT-family
    sentence representation; this matches ``HybridModel.transformer_pooling``.

    The wrapper forwards whatever the model supports, so it works for both
    BERT (which emits ``token_type_ids``) and DistilBERT (which does not).
    """

    def __init__(self, hf_model, pool='cls'):
        super().__init__()
        self.hf_model = hf_model
        self.pool = pool
        self.hidden_size = hf_model.config.hidden_size

    def _pool(self, hidden_state):
        if self.pool == 'cls':
            return hidden_state[:, 0]
        if self.pool == 'mean':
            return hidden_state.mean(dim=1)
        if self.pool == 'max':
            return hidden_state.max(dim=1).values
        raise ValueError(f"Unsupported pooling method: {self.pool}")

    def forward(self, x):
        if isinstance(x, dict):
            outputs = self.hf_model(**x)
        elif isinstance(x, torch.Tensor):
            outputs = self.hf_model(input_ids=x)
        else:
            raise TypeError(
                f"Transformer backbone expects a tensor or a tokenizer dict, got {type(x).__name__}"
            )

        hidden_state = getattr(outputs, 'last_hidden_state', None)
        if hidden_state is None:
            # Older/other model classes return a plain tuple with hidden states first
            hidden_state = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        pooled = self._pool(hidden_state)
        # Some checkpoints emit complex activations; the quantum layer needs real input.
        if torch.is_complex(pooled):
            pooled = pooled.abs()
        return pooled

def get_regression_model(model_name='linear', output_features=None):
    """
    Creates a regression model for tasks like house price prediction.
    
    Args:
        model_name: Type of regression model ('linear' is currently supported)
        output_features: Number of output features (usually 1 for regression)
        
    Returns:
        model: A PyTorch module implementing the regression model
        input_features: Number of input features (set after first data pass)
    """
    class LinearRegressionModel(nn.Module):
        """PyTorch implementation of Linear Regression."""
        def __init__(self):
            super(LinearRegressionModel, self).__init__()
            self.input_features = None
            self.output_features = output_features or 1
            self.linear = None  # Will be initialized on first forward pass
            
        def forward(self, x):
            # Initialize on first forward pass when we know input size
            if self.linear is None:
                input_size = x.shape[1]
                self.input_features = input_size
                self.linear = nn.Linear(input_size, self.output_features)
                print(f"Initialized LinearRegression with input size: {input_size}, output size: {self.output_features}")
                
                # Initialize with reasonable values for housing price prediction
                # Scale is important to prevent exploding gradients
                nn.init.xavier_uniform_(self.linear.weight, gain=0.01)
                nn.init.zeros_(self.linear.bias)
                
                # Move to same device as input
                self.linear = self.linear.to(x.device)
                
            return self.linear(x)
    
    # For sklearn model (alternative implementation)
    class SklearnLinearRegression(nn.Module):
        """Wrapper for scikit-learn's LinearRegression."""
        def __init__(self):
            super(SklearnLinearRegression, self).__init__()
            self.model = LinearRegression()
            self.input_features = None
            self.is_fitted = False
            self.weights = None
            self.bias = None
            
        def fit(self, X, y):
            """Fit the model using scikit-learn."""
            self.model.fit(X, y)
            self.is_fitted = True
            self.weights = torch.tensor(self.model.coef_, dtype=torch.float32)
            self.bias = torch.tensor(self.model.intercept_, dtype=torch.float32)
            self.input_features = X.shape[1]
            
        def forward(self, x):
            """Forward pass using fitted weights or dynamic initialization."""
            if not self.is_fitted:
                if self.input_features is None:
                    self.input_features = x.shape[1]
                
                # Random initialization if not fitted
                if self.weights is None or self.bias is None:
                    self.weights = torch.randn(1, self.input_features, dtype=torch.float32, device=x.device) * 0.01
                    self.bias = torch.zeros(1, dtype=torch.float32, device=x.device)
                    
            # Ensure weights and inputs are on same device
            if self.weights.device != x.device:
                self.weights = self.weights.to(x.device)
                self.bias = self.bias.to(x.device)
                
            # Manual matrix multiplication equivalent to W*x + b
            return torch.matmul(x, self.weights.T) + self.bias
    
    if model_name.lower() == 'linear':
        model = LinearRegressionModel()
        # Return None for num_ftrs since it will be set dynamically
        return model, None
    elif model_name.lower() == 'sklearn_linear':
        model = SklearnLinearRegression()
        return model, None
    else:
        raise ValueError(f"Unsupported regression model: {model_name}")

def get_gnn_model(model_name='gcn', input_features=None, hidden_features=64, output_features=None):
    """Creates a GNN model using torch_geometric."""
    if pyg_nn is None:
        raise ImportError("torch_geometric is required for GNN models but not installed.")
    if input_features is None:
        raise ValueError("input_features must be specified for GNN models.")

    if model_name == 'gcn':
        # Example: 2-layer GCN
        model = pyg_nn.GCN(input_features, hidden_features, num_layers=2, out_channels=hidden_features) # Intermediate output
        num_output_ftrs = hidden_features
    elif model_name == 'graphsage':
        model = pyg_nn.GraphSAGE(input_features, hidden_features, num_layers=2, out_channels=hidden_features)
        num_output_ftrs = hidden_features
    else:
        raise ValueError(f"Unsupported GNN model: {model_name}")

    if output_features is not None:
        # Add projection layer if needed, likely handled in Fusion Layer
        print(f"GNN base features: {num_output_ftrs}. Output layer added in Fusion Layer if needed.")
        pass

    # The GNN itself needs to be wrapped in a module handling graph pooling if needed
    # For simplicity, returning the convolutional part; pooling might be added in the fusion model
    return model, num_output_ftrs

# Example of a wrapper for GNN to handle pooling (needed for graph-level features)
class GNNWrapper(nn.Module):
    def __init__(self, gnn_model, pool_type='mean'):
        super().__init__()
        self.gnn = gnn_model
        if pool_type == 'mean':
            self.pool = pyg_nn.global_mean_pool
        elif pool_type == 'add':
            self.pool = pyg_nn.global_add_pool
        else:
            raise ValueError(f"Unsupported pooling type: {pool_type}")

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gnn(x, edge_index)
        # Apply pooling to get graph-level embedding
        graph_embedding = self.pool(x, batch)
        return graph_embedding 