import unittest
import sys
import os

# Add the project root to the Python path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestAnsatzesImport(unittest.TestCase):
    """Test class to verify AVAILABLE_ANSATZES can be imported correctly."""
    
    def test_import_available_ansatzes(self):
        """Test that AVAILABLE_ANSATZES can be imported from core.fusion."""
        try:
            from core.fusion import AVAILABLE_ANSATZES
            # Verify it's a dictionary
            self.assertIsInstance(AVAILABLE_ANSATZES, dict)
            # Verify it contains at least one ansatz
            self.assertGreater(len(AVAILABLE_ANSATZES), 0)
            # Verify the expected ansatz is present
            self.assertIn("Ansatz 1 (Rot+CNOT Chain)", AVAILABLE_ANSATZES)
            print("✅ Successfully imported AVAILABLE_ANSATZES from core.fusion")
        except ImportError as e:
            self.fail(f"Failed to import AVAILABLE_ANSATZES: {e}")
            
    def test_ansatz_function_works(self):
        """Test that the ansatz function in AVAILABLE_ANSATZES works correctly."""
        try:
            from core.fusion import AVAILABLE_ANSATZES
            import torch
            import pennylane as qml
            
            # Get the first ansatz function
            ansatz_name = "Ansatz 1 (Rot+CNOT Chain)"
            ansatz_func = AVAILABLE_ANSATZES[ansatz_name]
            
            # Test parameters for the ansatz
            num_qubits = 2
            num_layers = 1
            # 2 parameters per qubit per layer (RY and RZ)
            num_params = 2 * num_qubits * num_layers
            params = torch.rand(num_params)
            
            # Create a simple test circuit
            dev = qml.device("default.qubit", wires=num_qubits)
            
            @qml.qnode(dev)
            def test_circuit(params):
                ansatz_func(params, num_qubits, num_layers)
                return qml.state()
            
            # Execute the circuit
            result = test_circuit(params)
            
            # Verify the result has the expected shape (2^num_qubits)
            self.assertEqual(len(result), 2**num_qubits)
            print(f"✅ Successfully executed the {ansatz_name} ansatz function")
        except Exception as e:
            self.fail(f"Error testing ansatz function: {e}")

if __name__ == "__main__":
    unittest.main() 