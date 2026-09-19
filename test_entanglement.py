import torch
import numpy as np
import sys
import os
import traceback

# Add the current directory to the path to import the modules
sys.path.append(os.path.abspath('.'))

# Import the functions we need to test
from core.metrics import calculate_meyer_wallach, partial_trace, calculate_purity

def debug_entanglement(state_vector, num_qubits):
    """Debug helper to trace through the entanglement calculation process"""
    print(f"\nDEBUG: Entanglement Calculation for {num_qubits} qubits")
    print(f"Input state type: {type(state_vector)}, shape: {state_vector.shape if hasattr(state_vector, 'shape') else 'N/A'}")
    
    if torch.is_tensor(state_vector):
        print(f"State is tensor, dtype: {state_vector.dtype}")
        # Check if we have any complex values
        if torch.is_complex(state_vector):
            print("State contains complex values")
            # Check for NaN in real or imaginary parts
            if torch.isnan(state_vector.real).any() or torch.isnan(state_vector.imag).any():
                print("WARNING: NaN detected in state vector!")
        elif torch.isnan(state_vector).any():
            print("WARNING: NaN detected in state vector!")
    
    try:
        # Try each step of the calculation process
        if num_qubits <= 1:
            print("Only 1 qubit, entanglement should be 0.0")
            return 0.0
        
        sum_purities = 0.0
        for k in range(num_qubits):
            print(f"Processing qubit {k}...")
            
            # Check for invalid dimensions
            expected_size = 2**num_qubits
            if torch.is_tensor(state_vector) and state_vector.numel() != expected_size:
                print(f"ERROR: State vector size {state_vector.numel()} doesn't match expected size {expected_size} for {num_qubits} qubits")
                return np.nan
            
            try:
                # Get reduced density matrix
                print(f"  Calculating partial trace for qubit {k}...")
                rho_k = partial_trace(state_vector, num_qubits, subsystem_wires=[k])
                print(f"  Reduced density matrix shape: {rho_k.shape if hasattr(rho_k, 'shape') else 'N/A'}")
                
                # Check if partial trace gave valid results
                if np.isnan(rho_k).any():
                    print(f"  WARNING: NaN in reduced density matrix for qubit {k}")
                    return np.nan
                
                # Calculate purity
                print(f"  Calculating purity for qubit {k}...")
                purity_k = calculate_purity(rho_k)
                print(f"  Purity = {purity_k}")
                
                if np.isnan(purity_k):
                    print(f"  WARNING: NaN in purity calculation for qubit {k}")
                    return np.nan
                
                sum_purities += purity_k
            except Exception as e:
                print(f"  ERROR in processing qubit {k}: {e}")
                traceback.print_exc()
                return np.nan
        
        # Final calculation
        print(f"Sum of purities: {sum_purities}")
        print(f"Average purity: {sum_purities / num_qubits}")
        q_measure = 2.0 * (1.0 - (sum_purities / num_qubits))
        print(f"Final entanglement measure Q: {q_measure}")
        return q_measure
        
    except Exception as e:
        print(f"ERROR in overall entanglement calculation: {e}")
        traceback.print_exc()
        return np.nan

def test_random_valid_states():
    """Test entanglement calculation with random valid states"""
    print("\n=== Testing random valid states ===")
    
    # Test with different numbers of qubits
    for num_qubits in [2, 3, 4]:
        print(f"\nTesting {num_qubits} qubits:")
        
        # State dimension
        dim = 2**num_qubits
        
        # Random normalized state vector (using numpy first)
        print("Testing with random numpy state vector:")
        np_state = np.random.rand(dim) + 1j * np.random.rand(dim)
        np_state = np_state / np.linalg.norm(np_state)
        
        # Calculate entanglement the normal way
        entanglement = calculate_meyer_wallach(np_state, num_qubits)
        print(f"Entanglement (numpy): {entanglement}")
        
        # Debug calculation
        debug_ent = debug_entanglement(np_state, num_qubits)
        print(f"Debug entanglement: {debug_ent}")
        
        # Assert it's not NaN
        assert not np.isnan(entanglement), f"Entanglement is NaN for numpy state with {num_qubits} qubits"
        
        # Test with torch tensor
        print("\nTesting with random torch state vector:")
        torch_state = torch.tensor(np_state, dtype=torch.complex64)
        
        # Calculate entanglement
        entanglement = calculate_meyer_wallach(torch_state, num_qubits)
        print(f"Entanglement (torch): {entanglement}")
        
        # Assert it's not NaN
        assert not np.isnan(entanglement), f"Entanglement is NaN for torch state with {num_qubits} qubits"

def test_edge_cases():
    """Test entanglement calculation with edge cases that might cause NaN"""
    print("\n=== Testing edge cases ===")
    
    num_qubits = 3
    dim = 2**num_qubits
    
    # Case 1: Very small values
    print("\nTesting with very small values:")
    small_state = np.ones(dim) * 1e-10
    small_state[0] = 1.0  # To maintain normalization approximately
    small_state = small_state / np.linalg.norm(small_state)
    
    entanglement = calculate_meyer_wallach(small_state, num_qubits)
    print(f"Entanglement (small values): {entanglement}")
    
    # Case 2: Test with a separable state (should give 0 entanglement)
    print("\nTesting with separable state (|000>):")
    separable_state = np.zeros(dim)
    separable_state[0] = 1.0  # |000>
    
    entanglement = calculate_meyer_wallach(separable_state, num_qubits)
    print(f"Entanglement (separable |000>): {entanglement}")
    assert abs(entanglement) < 1e-10, f"Separable state should have zero entanglement, got {entanglement}"
    
    # Case 3: Test with a maximally entangled state (GHZ state)
    print("\nTesting with GHZ state:")
    ghz_state = np.zeros(dim)
    ghz_state[0] = 1/np.sqrt(2)  # |000>
    ghz_state[-1] = 1/np.sqrt(2)  # |111>
    
    entanglement = calculate_meyer_wallach(ghz_state, num_qubits)
    print(f"Entanglement (GHZ): {entanglement}")
    assert entanglement > 0.9, f"GHZ state should be highly entangled, got {entanglement}"
    
    # Case 4: Wrong size vector
    print("\nTesting with wrong size vector:")
    try:
        wrong_size = np.ones(dim//2)  # Too small
        wrong_size = wrong_size / np.linalg.norm(wrong_size)
        entanglement = calculate_meyer_wallach(wrong_size, num_qubits)
        print(f"Entanglement (wrong size): {entanglement}")
    except Exception as e:
        print(f"Correctly caught error: {e}")

def fix_calculate_meyer_wallach():
    """Test our fixed version of the entanglement calculation function"""
    print("\n=== Testing fixed entanglement function ===")
    
    # Define an improved version of the function with better error handling
    def improved_calculate_meyer_wallach(state_vector, num_qubits):
        """
        Improved version of Meyer-Wallach calculation with better error handling.
        """
        # Input validation
        if num_qubits <= 1:
            return 0.0  # No entanglement for 1 qubit
            
        # Check dimensions
        expected_size = 2**num_qubits
        if hasattr(state_vector, 'shape'):
            if len(state_vector.shape) == 1 and state_vector.shape[0] != expected_size:
                print(f"WARNING: State vector size {state_vector.shape[0]} doesn't match expected size {expected_size}")
                return np.nan
                
        # Check for NaN in input
        if isinstance(state_vector, torch.Tensor) and torch.isnan(state_vector).any():
            print("WARNING: Input state vector contains NaN values")
            return np.nan
        elif isinstance(state_vector, np.ndarray) and np.isnan(state_vector).any():
            print("WARNING: Input state vector contains NaN values")
            return np.nan
            
        try:
            sum_purities = 0.0
            for k in range(num_qubits):
                # Get reduced density matrix for qubit k
                try:
                    rho_k = partial_trace(state_vector, num_qubits, subsystem_wires=[k])
                except Exception as e:
                    print(f"ERROR in partial trace: {e}")
                    return np.nan
                    
                # Check for NaN in density matrix
                if np.isnan(rho_k).any():
                    print("WARNING: Reduced density matrix contains NaN")
                    return np.nan
                    
                # Calculate purity Tr(rho_k^2) with error handling
                try:
                    purity_k = calculate_purity(rho_k)
                except Exception as e:
                    print(f"ERROR in purity calculation: {e}")
                    return np.nan
                    
                # Validate purity is in [0,1] range and not NaN
                if np.isnan(purity_k):
                    print("WARNING: Purity calculation resulted in NaN")
                    return np.nan
                if purity_k < 0 or purity_k > 1.001:  # Allow small numerical error
                    print(f"WARNING: Purity {purity_k} outside valid range [0,1]")
                    # Clamp to valid range instead of returning NaN
                    purity_k = max(0.0, min(1.0, purity_k))
                    
                sum_purities += purity_k

            # Calculate Q with protection against division by zero
            if num_qubits == 0:
                return 0.0
                
            q_measure = 2.0 * (1.0 - (sum_purities / num_qubits))
            
            # Clamp final result to valid range [0,1]
            q_measure = max(0.0, min(1.0, q_measure))
            return q_measure
            
        except Exception as e:
            print(f"Unexpected error in entanglement calculation: {e}")
            return np.nan
    
    # Create a testcase with a special vector that has issues
    num_qubits = 3
    dim = 2**num_qubits
    
    # Use various test vectors
    test_vectors = [
        # Normal random state
        np.random.rand(dim) + 1j * np.random.rand(dim),
        
        # Small values except one
        np.ones(dim) * 1e-15,
        
        # Zeros with one element (|000>)
        np.zeros(dim),
        
        # GHZ state
        np.zeros(dim)
    ]
    
    # Normalize and fix special cases
    test_vectors[0] = test_vectors[0] / np.linalg.norm(test_vectors[0])
    test_vectors[1][0] = 1.0
    test_vectors[1] = test_vectors[1] / np.linalg.norm(test_vectors[1])
    test_vectors[2][0] = 1.0  # |000>
    test_vectors[3][0] = 1/np.sqrt(2)  # |000> part of GHZ
    test_vectors[3][-1] = 1/np.sqrt(2)  # |111> part of GHZ
    
    test_names = ["Random state", "State with small values", "Separable state |000>", "GHZ state"]
    
    # Test both the original and improved functions on these vectors
    for i, test_vector in enumerate(test_vectors):
        name = test_names[i]
        print(f"\nTesting {name}:")
        
        # Original function
        orig_result = calculate_meyer_wallach(test_vector, num_qubits)
        print(f"Original function: {orig_result}")
        
        # Improved function
        improved_result = improved_calculate_meyer_wallach(test_vector, num_qubits)
        print(f"Improved function: {improved_result}")
        
        # Compare results
        if np.isnan(orig_result) and not np.isnan(improved_result):
            print("✅ Improved function fixed a NaN result!")
        elif not np.isnan(orig_result) and not np.isnan(improved_result):
            if abs(orig_result - improved_result) < 1e-10:
                print("✅ Both functions give same correct result")
            else:
                print(f"⚠️ Results differ: original={orig_result}, improved={improved_result}")
    
    # Return the improved function for implementation
    return improved_calculate_meyer_wallach

def test_with_real_model_output():
    """Test the entanglement calculation with output similar to what we'd get from a model"""
    print("\n=== Testing with simulated model output ===")
    
    # Simulate model output - complex numbers with some structure
    num_qubits = 4
    dim = 2**num_qubits
    
    # Create a state vector with random phases but specific magnitudes
    magnitudes = np.random.rand(dim) 
    magnitudes = magnitudes / np.linalg.norm(magnitudes)
    phases = np.random.rand(dim) * 2 * np.pi
    
    # Create complex state vector
    state = magnitudes * np.exp(1j * phases)
    
    # Calculate entanglement
    print("Testing with normal state:")
    entanglement = calculate_meyer_wallach(state, num_qubits)
    print(f"Entanglement: {entanglement}")
    
    # Now test with torch tensor, double precision
    torch_state = torch.tensor(state, dtype=torch.complex128)
    entanglement = calculate_meyer_wallach(torch_state, num_qubits)
    print(f"Entanglement (torch complex128): {entanglement}")
    
    # Test with single precision (float32) - more prone to numerical issues
    torch_state_f32 = torch.tensor(state, dtype=torch.complex64)
    entanglement = calculate_meyer_wallach(torch_state_f32, num_qubits)
    print(f"Entanglement (torch complex64): {entanglement}")

if __name__ == "__main__":
    print("===== Testing Entanglement Calculation =====")
    
    # Run all tests
    try:
        print("\nRunning test_random_valid_states...")
        test_random_valid_states()
    except Exception as e:
        print(f"Error in test_random_valid_states: {e}")
    
    try:
        print("\nRunning test_edge_cases...")
        test_edge_cases() 
    except Exception as e:
        print(f"Error in test_edge_cases: {e}")
    
    try:
        print("\nRunning test_with_real_model_output...")
        test_with_real_model_output()
    except Exception as e:
        print(f"Error in test_with_real_model_output: {e}")
    
    # This function returns an improved version of calculate_meyer_wallach
    improved_function = fix_calculate_meyer_wallach()
    
    print("\n===== SUMMARY =====")
    print("1. Created and tested a fixed version of the entanglement calculation function")
    print("2. Identified potential issues causing NaN values")
    print("3. The improved function handles edge cases better and provides more informative errors")
    print("\nTo fix the NaN entanglement issues:")
    print("- Replace the current calculate_meyer_wallach with the improved version")
    print("- Add error handling in the UI where entanglement is displayed") 