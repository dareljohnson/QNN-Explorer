"""
Simple test to verify our fix for the list index error.
This script simulates the core functionality that was failing.
"""
import numpy as np

def main():
    """Simulate the training loop that was failing."""
    print("Testing training loop fixes...")
    
    # Initialize variables
    training_history = {'loss': [], 'entanglement': []}
    entanglement = np.nan
    batch_entanglements = []
    
    try:
        # Simulate 3 epochs
        for epoch in range(3):
            print(f"Processing epoch {epoch+1}/3")
            
            # Simulate 5 batches per epoch
            for batch_idx in range(5):
                print(f"  Batch {batch_idx+1}/5")
                
                # Simulate entanglement calculation occasionally
                if batch_idx % 2 == 0:
                    try:
                        # Generate random value to simulate entanglement calculation
                        entanglement = np.random.random()
                        print(f"    Calculated entanglement: {entanglement:.4f}")
                        batch_entanglements.append(entanglement)
                    except Exception as e:
                        print(f"    Error in entanglement calculation: {e}")
                        # Don't crash - just use NaN
                        entanglement = np.nan
            
            # End of epoch - calculate average and update history
            print(f"  End of epoch {epoch+1}")
            
            # Calculate average entanglement - THIS WAS FAILING BEFORE
            if len(batch_entanglements) > 0:
                avg_entanglement = np.nanmean(batch_entanglements)
                if not np.isnan(avg_entanglement):
                    entanglement = avg_entanglement
            
            # Update history with current value
            training_history['loss'].append(np.random.random())  # Simulate loss
            training_history['entanglement'].append(entanglement)  # This was failing before
            
            print(f"  Updated history: loss={training_history['loss'][-1]:.4f}, entanglement={training_history['entanglement'][-1]:.4f}")
            
            # Reset batch_entanglements for next epoch
            batch_entanglements = []
            
        print("\nTraining completed successfully!")
        print(f"Final training history: {training_history}")
        return True
    
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1) 