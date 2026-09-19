#!/usr/bin/env python
"""
Verify Housing Dataset Path Fix

This script verifies that the housing dataset path handling fixes have been applied correctly
and that the demo dataset can be loaded without permission errors.
"""

import os
import sys
import pandas as pd
import streamlit as st
import traceback

# Add the current directory to the path to find modules
sys.path.append('.')

# Import the path utilities
from data.path_utils import ensure_path_sep, make_path

def test_path_utils():
    """Test the path utility functions"""
    print("\n1. Testing path utility functions...")
    
    # Test ensure_path_sep
    mixed_path = "data/demo_data\\housing\\processed"
    fixed_path = ensure_path_sep(mixed_path)
    correct_path = os.path.join("data", "demo_data", "housing", "processed")
    
    print(f"Mixed path: {mixed_path}")
    print(f"Fixed path: {fixed_path}")
    print(f"Correct path: {correct_path}")
    
    assert os.path.normpath(fixed_path) == os.path.normpath(correct_path), "Path fix failed"
    print("✅ Path utility functions working correctly")

def test_demo_datasets_dict():
    """Test that the DEMO_DATASETS dictionary is correctly configured"""
    print("\n2. Testing DEMO_DATASETS dictionary...")
    
    # Import the loader module
    from data.loader import DEMO_DATASETS
    
    # Check that the housing dataset path is correct
    housing_path = DEMO_DATASETS["USA Housing Prices (Regression)"]["path"]
    print(f"Housing dataset path: {housing_path}")
    
    # Verify it's a valid path
    assert os.path.exists(housing_path), f"Housing path does not exist: {housing_path}"
    print("✅ Housing dataset path exists")
    
    # Check that it doesn't have mixed separators
    assert not ('/' in housing_path and '\\' in housing_path), "Path still has mixed separators"
    print("✅ No mixed separators in housing dataset path")

def test_directory_permissions():
    """Test that the directories have correct permissions"""
    print("\n3. Testing directory permissions...")
    
    # Define paths using the utility functions
    housing_dir = ensure_path_sep(os.path.join("data", "demo_data", "housing"))
    raw_dir = ensure_path_sep(os.path.join(housing_dir, "raw"))
    processed_dir = ensure_path_sep(os.path.join(housing_dir, "processed"))
    
    # Ensure directories exist
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Test permissions by writing temp files
    test_file_raw = os.path.join(raw_dir, "permission_test.txt")
    test_file_processed = os.path.join(processed_dir, "permission_test.txt")
    
    try:
        with open(test_file_raw, 'w') as f:
            f.write("Testing permissions")
        with open(test_file_processed, 'w') as f:
            f.write("Testing permissions")
        
        print(f"✅ Successfully wrote to {test_file_raw}")
        print(f"✅ Successfully wrote to {test_file_processed}")
        
        # Clean up
        os.remove(test_file_raw)
        os.remove(test_file_processed)
    except Exception as e:
        print(f"❌ Permission error: {e}")
        raise

def test_load_demo_housing_dataset():
    """Test loading the housing demo dataset through the loader function"""
    print("\n4. Testing the loader function with housing dataset...")
    
    # Import the loader function
    from data.loader import load_data
    
    # Mock the streamlit error and success functions
    original_error = st.error
    original_success = st.success
    original_info = st.info
    original_write = st.write
    
    try:
        # Replace st.error with print to avoid errors if running outside Streamlit
        st.error = lambda msg: print(f"[ERROR]: {msg}")
        st.success = lambda msg: print(f"[SUCCESS]: {msg}")
        st.info = lambda msg: print(f"[INFO]: {msg}")
        st.write = lambda msg: print(f"[WRITE]: {msg}")
        
        # Try loading the housing dataset
        processed_data, preview_data, data_info = load_data(
            source="Demo",
            data_type="CSV",
            demo_dataset_name="USA Housing Prices (Regression)"
        )
        
        # Check the returned values
        print(f"data_info: {data_info}")
        assert data_info is not None, "data_info should not be None"
        assert 'demo_name' in data_info, "data_info should contain demo_name"
        assert 'path' in data_info, "data_info should contain path"
        assert os.path.exists(data_info['path']), f"Path should exist: {data_info['path']}"
        
        # Check preview_data
        assert preview_data is not None, "preview_data should not be None"
        assert isinstance(preview_data, pd.DataFrame), "preview_data should be a DataFrame"
        
        print("✅ Housing dataset loaded successfully through loader function")
    except Exception as e:
        print(f"❌ Error loading housing dataset: {e}")
        traceback.print_exc()
        raise
    finally:
        # Restore original functions
        st.error = original_error
        st.success = original_success
        st.info = original_info
        st.write = original_write

def test_full_path_scenario():
    """Test a full path scenario simulating the error case"""
    print("\n5. Testing full path scenario that previously caused errors...")
    
    # The path that was causing issues
    problem_path = "data/demo_data\\housing\\processed"
    
    # Fix it
    fixed_path = ensure_path_sep(problem_path)
    
    # Try to access it
    try:
        # Make sure the directories exist
        os.makedirs(fixed_path, exist_ok=True)
        
        # Try to write to it
        test_file = os.path.join(fixed_path, "scenario_test.txt")
        with open(test_file, 'w') as f:
            f.write("Testing the full error scenario")
        
        print(f"✅ Successfully wrote to {test_file}")
        
        # Try to read from it
        with open(test_file, 'r') as f:
            content = f.read()
        
        print(f"✅ Successfully read from {test_file}: {content}")
        
        # Clean up
        os.remove(test_file)
    except Exception as e:
        print(f"❌ Error in full path scenario: {e}")
        raise

def main():
    """Main function to run all tests"""
    print("=" * 60)
    print("Housing Dataset Path Fix Verification")
    print("=" * 60)
    
    try:
        # Run all tests
        test_path_utils()
        test_demo_datasets_dict()
        test_directory_permissions()
        test_load_demo_housing_dataset()
        test_full_path_scenario()
        
        print("\n✅ All tests passed! The housing dataset path fix has been verified.")
        return True
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 