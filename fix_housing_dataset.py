#!/usr/bin/env python
"""
Fix Housing Dataset Path and Permission Issues

This script resolves path inconsistencies and permission problems with the housing dataset
by ensuring directories exist with proper permissions and running the dataset creation process.
"""

import os
import sys
import shutil
import traceback

def ensure_directories():
    """Ensure that all required directories exist with proper permissions."""
    print("Creating necessary directories with proper permissions...")
    
    # Define paths consistently using os.path.join
    data_dir = os.path.join("data", "demo_data", "housing")
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    
    # Create parent directories first
    os.makedirs(os.path.join("data"), exist_ok=True)
    os.makedirs(os.path.join("data", "demo_data"), exist_ok=True)
    
    # Create housing directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Verify directories exist and are writable
    for path in [data_dir, raw_dir, processed_dir]:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            print(f"ERROR: Failed to create directory: {abs_path}")
            return False
        
        # Test write permissions by creating a test file
        test_file = os.path.join(abs_path, "test_permissions.txt")
        try:
            with open(test_file, 'w') as f:
                f.write("Testing write permissions")
            os.remove(test_file)
            print(f"✅ Directory created and writable: {abs_path}")
        except Exception as e:
            print(f"❌ Directory not writable: {abs_path}")
            print(f"   Error: {e}")
            return False
    
    return True

def generate_dataset():
    """Generate the housing dataset using the existing download_house_prices.py script."""
    try:
        print("\nGenerating housing dataset...")
        
        # Import the dataset creation functions
        sys.path.append('.')
        from data.download_house_prices import create_synthetic_dataset, process_dataset
        
        # Create a synthetic dataset (if needed)
        create_synthetic_dataset(samples=1000)
        
        # Process the dataset
        if process_dataset():
            print("✅ Housing dataset successfully created and processed!")
            return True
        else:
            print("❌ Failed to process the housing dataset.")
            return False
    
    except Exception as e:
        print(f"❌ Error generating housing dataset: {e}")
        traceback.print_exc()
        return False

def main():
    """Main function to fix housing dataset issues."""
    print("=" * 60)
    print("Housing Dataset Path and Permission Fix")
    print("=" * 60)
    
    # Step 1: Ensure directories exist with proper permissions
    if not ensure_directories():
        print("❌ Failed to create or verify directories. Please check permissions.")
        sys.exit(1)
    
    # Step 2: Generate the dataset
    if not generate_dataset():
        print("❌ Failed to generate housing dataset.")
        sys.exit(1)
    
    print("\n✅ Housing dataset fix completed successfully!")
    print("You should now be able to use the housing dataset without permission errors.")
    sys.exit(0)

if __name__ == "__main__":
    main() 