#!/usr/bin/env python
"""
Test script to diagnose permission issues with the housing dataset directories.
"""

import os
import sys
import shutil
import tempfile

def test_dir_permissions():
    """Test if we can create, write to, and read from the necessary directories."""
    print("Testing directory permissions for housing dataset...")
    
    # Define paths (same as in data/download_house_prices.py)
    DATA_DIR = os.path.join("data", "demo_data", "housing")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    
    # Print the paths for verification
    print(f"DATA_DIR: {os.path.abspath(DATA_DIR)}")
    print(f"RAW_DATA_DIR: {os.path.abspath(RAW_DATA_DIR)}")
    print(f"PROCESSED_DATA_DIR: {os.path.abspath(PROCESSED_DATA_DIR)}")
    
    # Test 1: Can we create the directories?
    print("\nTest 1: Creating directories...")
    try:
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        print("✅ Successfully created directories")
    except Exception as e:
        print(f"❌ Failed to create directories: {e}")
        return False
    
    # Test 2: Can we write to the directories?
    print("\nTest 2: Writing test files...")
    try:
        # Write to raw directory
        raw_test_file = os.path.join(RAW_DATA_DIR, "test_file.txt")
        with open(raw_test_file, 'w') as f:
            f.write("Test content")
        print(f"✅ Successfully wrote to {raw_test_file}")
        
        # Write to processed directory
        processed_test_file = os.path.join(PROCESSED_DATA_DIR, "test_file.txt")
        with open(processed_test_file, 'w') as f:
            f.write("Test content")
        print(f"✅ Successfully wrote to {processed_test_file}")
    except Exception as e:
        print(f"❌ Failed to write test files: {e}")
        return False
    
    # Test 3: Can we read from the directories?
    print("\nTest 3: Reading test files...")
    try:
        # Read from raw directory
        with open(raw_test_file, 'r') as f:
            content = f.read()
        print(f"✅ Successfully read from {raw_test_file}: {content}")
        
        # Read from processed directory
        with open(processed_test_file, 'r') as f:
            content = f.read()
        print(f"✅ Successfully read from {processed_test_file}: {content}")
    except Exception as e:
        print(f"❌ Failed to read test files: {e}")
        return False
    
    # Test 4: Can we delete the test files?
    print("\nTest 4: Cleaning up test files...")
    try:
        os.remove(raw_test_file)
        os.remove(processed_test_file)
        print("✅ Successfully removed test files")
    except Exception as e:
        print(f"❌ Failed to remove test files: {e}")
        return False
    
    # Test 5: Try creating a temporary directory and file
    print("\nTest 5: Creating and writing to a temporary file...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "temp_test.txt")
            with open(temp_file, 'w') as f:
                f.write("Temporary test content")
            print(f"✅ Successfully created and wrote to temporary file {temp_file}")
    except Exception as e:
        print(f"❌ Failed to work with temporary file: {e}")
        return False
    
    print("\nAll permission tests passed successfully!")
    return True

if __name__ == "__main__":
    success = test_dir_permissions()
    sys.exit(0 if success else 1) 