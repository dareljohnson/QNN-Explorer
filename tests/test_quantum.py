import pytest
import pennylane as qml
from pennylane import numpy as pnp
import torch

from core.quantum_models import get_device, amplitude_embedding, ansatz1, create_qnn

# TODO: Add actual tests for quantum models

def test_device_selection():
    cpu_dev = get_device(num_qubits=2, use_gpu=False)
    assert cpu_dev.name == 'default.qubit'
    # Test GPU if available (tricky in CI)
    # if torch.cuda.is_available():
    #     gpu_dev = get_device(num_qubits=2, use_gpu=True)
    #     assert gpu_dev.name == 'default.qubit.torch' # Or lightning.gpu if implemented
    pass

def test_amplitude_embedding_prep():
    features = torch.tensor([1., 2., 3., 4.])
    normalized = amplitude_embedding(features)
    assert torch.allclose(torch.norm(normalized), torch.tensor(1.0))
    assert normalized.shape == features.shape

def test_ansatz1_runs():
    num_qubits = 3
    num_layers = 2
    num_params = 2 * num_qubits * num_layers
    params = torch.rand(num_params)
    dev = qml.device('default.qubit', wires=num_qubits)

    @qml.qnode(dev)
    def circuit(params):
        ansatz1(params, num_qubits, num_layers)
        return qml.state()

    state = circuit(params)
    assert state.shape == (2**num_qubits,)

def test_create_qnn_state_vector():
    num_qubits = 2
    num_layers = 1
    qnn, num_params = create_qnn(num_qubits, num_layers, ansatz1, observation_type='state')
    assert callable(qnn)
    assert num_params == 2 * num_qubits * num_layers

    # Test execution
    input_size = 2**num_qubits
    dummy_input = torch.rand(input_size)
    dummy_input = dummy_input / torch.norm(dummy_input)
    dummy_params = torch.rand(num_params)

    output_state = qnn(dummy_input, dummy_params)
    assert isinstance(output_state, torch.Tensor)
    assert output_state.shape == (input_size,)
    assert torch.allclose(torch.norm(output_state), torch.tensor(1.0))

def test_create_qnn_expval():
    num_qubits = 3
    num_layers = 1
    qnn, num_params = create_qnn(num_qubits, num_layers, ansatz1, observation_type='expval')
    assert callable(qnn)
    assert num_params == 2 * num_qubits * num_layers

    input_size = 2**num_qubits
    dummy_input = torch.rand(input_size)
    dummy_input = dummy_input / torch.norm(dummy_input)
    dummy_params = torch.rand(num_params)

    output_expval = qnn(dummy_input, dummy_params)
    assert isinstance(output_expval, torch.Tensor)
    # Expectation value for each qubit
    assert output_expval.shape == (num_qubits,)
    # Check values are within [-1, 1]
    assert torch.all(output_expval >= -1.0) and torch.all(output_expval <= 1.0) 