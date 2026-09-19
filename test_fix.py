import sys
import os

# Add core directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the fixed function and mock dependencies
from core.quantum_models import create_qnn

# Mock the dependencies
import unittest.mock as mock

def run_test():
    # Create mocks
    mock_device = mock.MagicMock()
    mock_get_device = mock.MagicMock(return_value=mock_device)
    mock_qnode = mock.MagicMock(return_value=lambda x: x)
    
    # Patch the dependencies
    with mock.patch('core.quantum_models.get_device', mock_get_device):
        with mock.patch('pennylane.qnode', mock_qnode):
            # Test different case variations
            observation_types = [
                'state',
                'State',
                'STATE',
                'state vector',
                'State Vector',
                'STATE VECTOR',
                'expval',
                'Expval',
                'EXPVAL',
                'expectation value',
                'Expectation Value',
                'expectation value (pauliz)'
            ]
            
            # Try each variation
            for obs_type in observation_types:
                try:
                    print(f"Testing observation type: '{obs_type}'")
                    # Mock ansatz function
                    mock_ansatz = mock.MagicMock()
                    # Create QNN with the observation type
                    qnn, _ = create_qnn(
                        num_qubits=2,
                        num_layers=1,
                        ansatz_func=mock_ansatz,
                        observation_type=obs_type
                    )
                    print(f"✅ Successfully handled observation type: '{obs_type}'")
                except Exception as e:
                    print(f"❌ Error with observation type '{obs_type}': {e}")
                    return False
            
            print("\n✅ All observation types were processed successfully!")
            return True

if __name__ == "__main__":
    print("Testing observation_type case handling in quantum_models.py...")
    if run_test():
        print("\nFix validated: The local variable reference issue is resolved!")
    else:
        print("\nFix failed: Some observation types still cause errors.") 