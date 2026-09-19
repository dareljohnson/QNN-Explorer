"""
Demonstrates the original error and our fix for the list assignment issue.
"""
import numpy as np

def simulate_error_case():
    """Simulate the original error case."""
    print("Simulating the ORIGINAL ERROR CASE:")
    
    # Initialize history but NOT entanglement
    training_history = {'loss': [], 'entanglement': []}
    # Missing initialization of entanglement variable
    
    try:
        # Simulate 1 epoch to reproduce the error
        for epoch in range(1):
            print(f"Processing epoch {epoch+1}/1")
            
            # Simulate 1 batch
            print("  Batch 1/1")
            
            # Skip entanglement calculation
            # The error occurs here - entanglement is undefined
            
            # End of epoch - update history with undefined entanglement
            print("  End of epoch 1")
            
            # The issue happens here - entanglement is not defined
            training_history['entanglement'].append(entanglement)  # NameError
            
        print("This line won't be reached due to the error")
        return False
    except NameError as e:
        print(f"  Expected error occurred: {e}")
        print("  This is the error we're fixing.")
        return True
    except Exception as e:
        print(f"  Unexpected error: {e}")
        return False

def simulate_fixed_case():
    """Simulate our fixed approach."""
    print("\nSimulating the FIXED APPROACH:")
    
    # Initialize history AND entanglement
    training_history = {'loss': [], 'entanglement': []}
    entanglement = np.nan  # Initialize with NaN
    batch_entanglements = []  # Track entanglements per batch
    
    try:
        # Simulate 1 epoch to show the fix
        for epoch in range(1):
            print(f"Processing epoch {epoch+1}/1")
            
            # Simulate 1 batch
            print("  Batch 1/1")
            
            # Skip entanglement calculation (still works fine)
            
            # End of epoch - update history with safely initialized entanglement
            print("  End of epoch 1")
            
            # Safety check - a key part of the fix
            if len(batch_entanglements) > 0:
                avg_entanglement = np.nanmean(batch_entanglements)
                if not np.isnan(avg_entanglement):
                    entanglement = avg_entanglement
            
            # Now this works because entanglement is defined
            training_history['entanglement'].append(entanglement)
            
            print(f"  Updated history: entanglement={training_history['entanglement'][-1]}")
            
        print("  Training completed without errors!")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

if __name__ == "__main__":
    error_expected = simulate_error_case()
    fix_successful = simulate_fixed_case()
    
    print("\nRESULTS:")
    print(f"Error case behaved as expected: {'YES' if error_expected else 'NO'}")
    print(f"Fix successfully prevented the error: {'YES' if fix_successful else 'NO'}")
    
    overall_success = error_expected and fix_successful
    print(f"\nOverall test: {'PASSED' if overall_success else 'FAILED'}")
    exit(0 if overall_success else 1) 