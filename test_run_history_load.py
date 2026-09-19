#!/usr/bin/env python
# Simple test script to check run history functionality

from utils.run_history import load_run_history

if __name__ == "__main__":
    print("Testing run_history module...")
    try:
        history = load_run_history()
        print(f"Successfully loaded run history with {len(history)} entries")
        print(history.head() if not history.empty else "No history entries found")
        print("Test passed!")
    except Exception as e:
        print(f"Error loading run history: {e}")
        print("Test failed!") 