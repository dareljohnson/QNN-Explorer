"""Tests for circuit visualization functionality."""

import unittest
import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if PennyLane is available
try:
    import pennylane as qml
    import torch
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    warnings.warn("PennyLane or torch not available. Skipping circuit visualization tests.")

# Import the visualization module
from utils.visualization import plot_circuit

@unittest.skipUnless(PENNYLANE_AVAILABLE, "PennyLane not available")
class TestCircuitVisualization(unittest.TestCase):
    """Test case for circuit visualization."""
    
    def setUp(self):
        """Set up test resources."""
        # Define a simple quantum circuit
        self.dev = qml.device("default.qubit", wires=2)
        
        @qml.qnode(self.dev)
        def simple_circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
        
        self.simple_circuit = simple_circuit
        self.input_shape = [0.1, 0.2]  # Representative input
        
        # Define a more complex circuit similar to what's used in the app
        dev_complex = qml.device("default.qubit", wires=4)
        
        @qml.qnode(dev_complex)
        def complex_circuit(amplitudes, params):
            # Amplitude embedding
            qml.AmplitudeEmbedding(amplitudes, wires=range(4), normalize=True)
            
            # Parameterized rotation gates
            for i in range(4):
                qml.RX(params[i][0], wires=i)
                qml.RY(params[i][1], wires=i)
                qml.RZ(params[i][2], wires=i)
            
            # Entangling gates
            for i in range(3):
                qml.CNOT(wires=[i, i+1])
            
            # Return state vector
            return qml.state()
        
        self.complex_circuit = complex_circuit
        
        # Create sample inputs for the complex circuit
        self.amp_input = np.random.random(16)  # 2^4 = 16 amplitudes
        self.amp_input = self.amp_input / np.linalg.norm(self.amp_input)
        self.param_input = np.random.random((4, 3))  # 4 qubits, 3 params each
        
    def test_plot_circuit_with_input(self):
        """Test circuit visualization with input."""
        # First execute the circuit with input to ensure qtape is created
        self.simple_circuit(self.input_shape)
        
        # Now try to visualize the circuit
        result = plot_circuit(self.simple_circuit)
        self.assertIsNotNone(result, "Circuit visualization should return a result")
        
    def test_plot_circuit_without_execution(self):
        """Test circuit visualization without prior execution."""
        # Try to visualize without executing first - should show message about needing execution
        result = plot_circuit(self.simple_circuit)
        self.assertIsNotNone(result, "Circuit visualization should handle non-executed circuit")
    
    def test_plot_circuit_with_input_args(self):
        """Test circuit visualization with input_args parameter."""
        # New test using the input_args parameter - should execute circuit automatically
        result = plot_circuit(self.simple_circuit, input_args=self.input_shape)
        self.assertIsNotNone(result, "Circuit visualization with input_args should work")
    
    def test_plot_complex_circuit(self):
        """Test visualization of a more complex circuit similar to the app's usage."""
        # Test with the complex circuit using amplitude embedding (like in the app)
        result = plot_circuit(self.complex_circuit, input_args=[self.amp_input, self.param_input])
        self.assertIsNotNone(result, "Complex circuit visualization should work")
        
    def test_invalid_input_args(self):
        """Test handling of invalid input arguments."""
        # Should handle the error gracefully and return an explanatory message
        result = plot_circuit(self.simple_circuit, input_args=["invalid"])
        self.assertIsNotNone(result, "Circuit visualization should handle invalid input_args")

if __name__ == '__main__':
    unittest.main() 