import pytest
import torch

# from core.classical_models import get_cnn_model, get_transformer_model, get_gnn_model, GNNWrapper
# from transformers import AutoTokenizer
# from torch_geometric.data import Data, Batch

# TODO: Add actual tests for classical models

def test_cnn_loading():
    # model, features = get_cnn_model('resnet18')
    # assert model is not None
    # assert features > 0
    pass

def test_transformer_loading():
    # model, features = get_transformer_model('distilbert-base-uncased')
    # assert model is not None
    # assert features > 0
    pass

def test_gnn_loading():
    # requires torch_geometric
    # try:
    #     model, features = get_gnn_model('gcn', input_features=10)
    #     assert model is not None
    #     assert features > 0
    # except ImportError:
    #     pytest.skip("torch_geometric not installed, skipping GNN test")
    pass

def test_gnn_wrapper():
    # requires torch_geometric
    # try:
    #     gnn_model, features = get_gnn_model('gcn', input_features=10, hidden_features=16)
    #     wrapper = GNNWrapper(gnn_model)
    #     # Create dummy graph data
    #     edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    #     x = torch.randn(3, 10) # 3 nodes, 10 features
    #     data = Data(x=x, edge_index=edge_index)
    #     batch = Batch.from_data_list([data]) # Create a batch
    #     output = wrapper(batch)
    #     assert output.shape == (1, 16) # Batch size 1, output features 16
    # except ImportError:
    #      pytest.skip("torch_geometric not installed, skipping GNN test")
    pass 