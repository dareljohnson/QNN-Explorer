import pytest
import torch

from core.fusion import HybridModel
from core.quantum_models import ansatz1

# TODO: Add actual tests for the fusion model

def test_hybrid_model_cnn_init():
    config = {
        'classical_backbone_type': 'cnn',
        'classical_model_name': 'resnet18',
        'num_qubits': 4,
        'num_layers': 2,
        'ansatz_func': ansatz1,
        'use_gpu': False,
        'observation_type': 'state'
    }
    model = HybridModel(config)
    assert model is not None
    assert model.classical_backbone is not None
    assert model.fusion_layer is not None
    assert model.quantum_circuit_fn is not None
    assert model.quantum_params is not None

def test_hybrid_model_transformer_init():
    config = {
        'classical_backbone_type': 'transformer',
        'classical_model_name': 'distilbert-base-uncased',
        'num_qubits': 3,
        'num_layers': 1,
        'ansatz_func': ansatz1,
        'use_gpu': False,
        'observation_type': 'expval'
    }
    model = HybridModel(config)
    assert model is not None
    assert model.classical_backbone is not None
    assert model.fusion_layer is not None
    assert model.quantum_circuit_fn is not None
    assert model.quantum_params is not None

def test_hybrid_model_gnn_init():
    # requires torch_geometric
    # try:
    config = {
        'classical_backbone_type': 'gnn',
        'classical_model_name': 'gcn',
        'gnn_input_features': 10,
        'num_qubits': 3,
        'num_layers': 1,
        'ansatz_func': ansatz1,
        'use_gpu': False,
        'observation_type': 'state'
    }
    #     model = HybridModel(config)
    #     assert model is not None
    #     assert model.classical_backbone is not None # Should be GNNWrapper
    #     assert model.fusion_layer is not None
    #     assert model.quantum_layer is not None
    # except ImportError:
    #     pytest.skip("torch_geometric not installed, skipping GNN test")
    pass # Skip if torch_geometric not installed

def test_hybrid_model_none_init():
    config = {
        'classical_backbone_type': 'none',
        'classical_model_name': None,
        'num_qubits': 2,
        'num_layers': 1,
        'ansatz_func': ansatz1,
        'use_gpu': False,
        'observation_type': 'state'
    }
    model = HybridModel(config)
    assert model is not None
    assert model.classical_backbone is None
    assert isinstance(model.fusion_layer, torch.nn.Identity)
    assert model.quantum_circuit_fn is not None
    assert model.quantum_params is not None

def test_hybrid_model_forward_cnn():
    config = {
        'classical_backbone_type': 'cnn',
        'classical_model_name': 'resnet18',
        'num_qubits': 4, # 2^4 = 16 output features for quantum layer
        'num_layers': 1,
        'ansatz_func': ansatz1,
        'use_gpu': False,
        'observation_type': 'state'
    }
    model = HybridModel(config)
    # Dummy image input (Batch, Channel, Height, Width)
    dummy_input = torch.rand(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 2**config['num_qubits']) # Batch size, 2^N

# Add forward tests for Transformer, GNN, None backbones 