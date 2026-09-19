import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path so we can import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quantum_models import create_qnn

class TestObservationTypes(unittest.TestCase):
    """Test that observation types are handled properly regardless of case."""
    
    @patch('pennylane.qnode')
    @patch('core.quantum_models.get_device')
    def test_state_vector_case_insensitivity(self, mock_get_device, mock_qnode):
        # Mock device and qnode for testing
        mock_device = MagicMock()
        mock_get_device.return_value = mock_device
        mock_qnode.return_value = lambda x: x  # Just return a simple function
        
        # List of case variations for 'state' observation type
        state_variations = [
            'state',
            'STATE',
            'State',
            'state vector',
            'State Vector',
            'STATE VECTOR'
        ]
        
        # Test each variation
        for obs_type in state_variations:
            try:
                # This should not raise ValueError
                qnn, _ = create_qnn(
                    num_qubits=2,
                    num_layers=1,
                    ansatz_func=lambda x, y, z: None,
                    observation_type=obs_type
                )
                # If we get here, no exception was raised
                self.assertTrue(True, f"Observation type '{obs_type}' was accepted")
            except ValueError as e:
                self.fail(f"Observation type '{obs_type}' raised ValueError: {e}")
    
    @patch('pennylane.qnode')
    @patch('core.quantum_models.get_device')
    def test_expval_case_insensitivity(self, mock_get_device, mock_qnode):
        # Mock device and qnode for testing
        mock_device = MagicMock()
        mock_get_device.return_value = mock_device
        mock_qnode.return_value = lambda x: x  # Just return a simple function
        
        # List of case variations for 'expval' observation type
        expval_variations = [
            'expval',
            'EXPVAL',
            'Expval',
            'expectation value',
            'Expectation Value',
            'EXPECTATION VALUE',
            'expectation value (pauliz)',
            'Expectation Value (PauliZ)'
        ]
        
        # Test each variation
        for obs_type in expval_variations:
            try:
                # This should not raise ValueError
                qnn, _ = create_qnn(
                    num_qubits=2,
                    num_layers=1,
                    ansatz_func=lambda x, y, z: None,
                    observation_type=obs_type
                )
                # If we get here, no exception was raised
                self.assertTrue(True, f"Observation type '{obs_type}' was accepted")
            except ValueError as e:
                self.fail(f"Observation type '{obs_type}' raised ValueError: {e}")

if __name__ == '__main__':
    unittest.main() 