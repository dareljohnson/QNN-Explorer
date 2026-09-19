"""
Model builder module for constructing models from configuration.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from .fusion import HybridModel, AVAILABLE_ANSATZES
from .classical_models import get_cnn_model, get_transformer_model, get_gnn_model, get_regression_model

def build_model(config: Dict[str, Any], device: Optional[torch.device] = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Build a model based on the provided configuration.
    
    Args:
        config: Dictionary containing model configuration parameters
        device: The device to place the model on (None for auto-detection)
        
    Returns:
        tuple: (model, updated_config)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Building model on device: {device}")
    
    # Extract model type from config
    model_type = config.get('model_type', 'hybrid').lower()
    print(f"Model type: {model_type}")
    
    # Update the config with the device
    config['device'] = device
    
    if model_type == 'hybrid':
        # Build a hybrid quantum-classical model
        model = HybridModel(config)
    elif model_type == 'classical':
        # Get the specific classical model type
        classical_type = config.get('classical_backbone_type', '').lower()
        input_size = config.get('input_size', 4)
        output_size = config.get('output_size', 1)
        
        if classical_type == 'cnn':
            # Build a CNN model
            model = get_cnn_model(
                input_shape=config.get('input_shape', (3, 32, 32)),
                num_classes=output_size,
                pretrained=config.get('pretrained', False)
            )
        elif classical_type == 'transformer':
            # Build a transformer model
            model = get_transformer_model(
                model_name=config.get('transformer_model', 'bert-base-uncased'),
                num_classes=output_size,
                pretrained=config.get('pretrained', True)
            )
        elif classical_type == 'gnn':
            # Build a graph neural network
            model = get_gnn_model(
                node_features=config.get('node_features', 64),
                num_classes=output_size,
                gnn_type=config.get('gnn_type', 'GCN')
            )
        elif classical_type == 'mlp' or classical_type.startswith('regression'):
            # Build a simple MLP for regression/classification
            model = get_regression_model(
                input_size=input_size,
                output_size=output_size,
                hidden_layers=config.get('hidden_layers', [64, 32]),
                dropout=config.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"Unsupported classical model type: {classical_type}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Move the model to the specified device
    model = model.to(device)
    print(f"Model built and moved to {device}")
    
    # Return both the model and the potentially updated config
    return model, config 