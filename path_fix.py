#!/usr/bin/env python
"""
Fix path handling for the housing dataset

This script ensures proper path handling across the application by consistently using
os.path.join() for path construction instead of mixing forward slashes and backslashes.
"""

import os
import sys
import re
import time

def fix_path_handling():
    """Fix the path inconsistencies in the loader.py file and elsewhere."""
    print("Fixing path inconsistencies in the application...")
    
    # Update DEMO_DATA_DIR in loader.py to use os.path.join()
    loader_path = os.path.join("data", "loader.py")
    if os.path.exists(loader_path):
        with open(loader_path, 'r') as f:
            content = f.read()
        
        # Replace string paths with os.path.join
        updated_content = content.replace(
            'DEMO_DATA_DIR = "data/demo_data"',
            'DEMO_DATA_DIR = os.path.join("data", "demo_data")'
        )
        
        with open(loader_path, 'w') as f:
            f.write(updated_content)
        
        print(f"✅ Updated {loader_path} to use os.path.join() for DEMO_DATA_DIR")
    else:
        print(f"❌ Could not find {loader_path}")
    
    # Create wrapper function to fix path handling in app.py
    wrapper_path = os.path.join("data", "path_utils.py")
    with open(wrapper_path, 'w') as f:
        f.write("""import os

def ensure_path_sep(path):
    \"\"\"
    Ensure path uses consistent separators regardless of how it was created.
    
    This function normalizes paths that might have been created with mixed
    separators (forward slashes and backslashes on Windows).
    
    Args:
        path (str): The path to normalize
        
    Returns:
        str: The normalized path with consistent separators
    \"\"\"
    # First, normalize by splitting on both types of separators
    parts = re.split(r'[\\\\/]', path)
    # Then rejoin using the OS-specific separator
    return os.path.join(*parts)

def make_path(*parts):
    \"\"\"
    Create a path from parts ensuring consistent separators.
    
    This is a wrapper around os.path.join that ensures paths are created
    consistently even if some parts already include separators.
    
    Args:
        *parts: Path parts to join
        
    Returns:
        str: The resulting path with consistent separators
    \"\"\"
    # First, flatten parts by splitting each on separators
    flat_parts = []
    for part in parts:
        flat_parts.extend(re.split(r'[\\\\/]', part))
    # Remove empty parts
    flat_parts = [p for p in flat_parts if p]
    # Join with the OS-specific separator
    return os.path.join(*flat_parts)
""")
    
    print(f"✅ Created {wrapper_path} with path handling utilities")
    
    # Create symbolic paths for housing dataset
    housing_dir = os.path.join("data", "demo_data", "housing")
    raw_dir = os.path.join(housing_dir, "raw")
    processed_dir = os.path.join(housing_dir, "processed")
    
    # Ensure directories exist with proper permissions
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Test permissions
    test_file_raw = os.path.join(raw_dir, "test_file.txt")
    test_file_processed = os.path.join(processed_dir, "test_file.txt")
    
    try:
        with open(test_file_raw, 'w') as f:
            f.write("Testing write permissions")
        with open(test_file_processed, 'w') as f:
            f.write("Testing write permissions")
        
        print(f"✅ Successfully created and wrote to {test_file_raw}")
        print(f"✅ Successfully created and wrote to {test_file_processed}")
        
        # Clean up test files
        os.remove(test_file_raw)
        os.remove(test_file_processed)
    except Exception as e:
        print(f"❌ Permission error during testing: {e}")
        return False
    
    # Create a fix guidance file for manual review
    with open("PATH_HANDLING_GUIDE.md", 'w') as f:
        f.write("""# Path Handling Guide for Quantum QNN

## Problem: Mixed Path Separators

The application was experiencing errors due to inconsistent path separators:
- Forward slashes (/) in string literals
- Backslashes (\\) when using os.path.join() on Windows
- Mixing both in combined paths like: `data/demo_data\\housing\\processed`

## Fix: Consistent Path Handling

1. **Use `os.path.join()` consistently**
   - Instead of: `"data/demo_data/housing"`
   - Use: `os.path.join("data", "demo_data", "housing")`

2. **For existing paths with mixed separators:**
   - Import and use the new utilities in `data/path_utils.py`:
   ```python
   from data.path_utils import ensure_path_sep, make_path
   
   # Fix a potentially mixed path:
   safe_path = ensure_path_sep(ensure_path_sep(os.path.join("data", "demo_data", "housing", "processed")))
   
   # Create a new path from parts:
   new_path = make_path("data/demo_data", "housing/processed")
   ```

3. **When loading files from paths:**
   ```python
   # Before loading or saving files, normalize the path:
   normalized_path = ensure_path_sep(path)
   with open(normalized_path, 'r') as f:
       # ...
   ```

This approach ensures cross-platform compatibility while fixing existing mixed paths.
""")
    
    print(f"✅ Created PATH_HANDLING_GUIDE.md with guidance for developers")
    
    return True

if __name__ == "__main__":
    if fix_path_handling():
        print("\n✅ Path handling fix completed successfully!")
        print("To complete the fix, follow the guidance in PATH_HANDLING_GUIDE.md")
        sys.exit(0)
    else:
        print("\n❌ Failed to fix path handling")
        sys.exit(1) 