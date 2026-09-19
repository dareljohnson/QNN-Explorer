#!/usr/bin/env python
"""
Fix specific path issues in app.py and data loading modules

This script addresses the specific error where mixed path separators are causing permission denied errors.
"""

import os
import sys
import re
import glob

# Import our path utilities
sys.path.append('.')
from data.path_utils import ensure_path_sep, make_path

def fix_paths_in_app():
    """Find and fix problematic paths in app.py"""
    print("Searching for problematic path strings in app.py...")
    
    app_path = "app.py"
    if not os.path.exists(app_path):
        print(f"❌ Could not find {app_path}")
        return False
    
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern for string paths with mixed separators
    pattern = r'["\']data\/demo_data\\\\?housing\\\\?processed["\']'
    
    # Check if the pattern exists
    if re.search(pattern, content):
        print(f"✅ Found problematic path pattern in {app_path}")
        
        # Fix: Add path_utils import
        if "from data.path_utils import " not in content:
            # Add import after other imports
            content = re.sub(
                r'(import\s+[^\n]+\n+)',
                r'\1from data.path_utils import ensure_path_sep, make_path\n\n',
                content,
                count=1
            )
        
        # Fix the problematic paths
        content = re.sub(
            r'(["\'])data\/demo_data\\\\?housing\\\\?processed(["\'])',
            r'ensure_path_sep(os.path.join("data", "demo_data", "housing", "processed"))',
            content
        )
        
        # Also fix other similar paths
        content = re.sub(
            r'(["\'])data\/demo_data\\\\?housing\\\\?raw(["\'])',
            r'ensure_path_sep(os.path.join("data", "demo_data", "housing", "raw"))',
            content
        )
        
        with open(app_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Fixed problematic paths in {app_path}")
        return True
    else:
        print(f"ℹ️ Did not find problematic path pattern in {app_path}")
        return False

def apply_path_fix_to_data_loader():
    """Fix path issues in data/loader.py"""
    loader_path = os.path.join("data", "loader.py")
    if not os.path.exists(loader_path):
        print(f"❌ Could not find {loader_path}")
        return False
    
    with open(loader_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add import
    if "from .path_utils import " not in content:
        import_line = "from .path_utils import ensure_path_sep, make_path\n"
        # Add after other imports
        content = re.sub(
            r'(from .preprocessing import.*?\n)',
            r'\1' + import_line,
            content
        )
    
    # Fix any paths like "data/demo_data/housing/processed"
    content = re.sub(
        r'(["\'])data\/demo_data\/housing\/processed(["\'])',
        r'ensure_path_sep(os.path.join("data", "demo_data", "housing", "processed"))',
        content
    )
    
    with open(loader_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Updated {loader_path} with path utilities")
    return True

def test_housing_paths():
    """Test housing dataset paths to ensure they work correctly"""
    print("\nTesting housing dataset paths...")
    
    # Use our path utility to construct paths
    housing_path = ensure_path_sep(os.path.join("data", "demo_data", "housing", "processed"))
    raw_path = ensure_path_sep(os.path.join("data", "demo_data", "housing", "raw"))
    
    # Make sure directories exist
    os.makedirs(housing_path, exist_ok=True)
    os.makedirs(raw_path, exist_ok=True)
    
    # Test permissions by writing test files
    try:
        test_file_processed = os.path.join(housing_path, "path_test.txt")
        with open(test_file_processed, 'w') as f:
            f.write("Testing path fix")
        
        test_file_raw = os.path.join(raw_path, "path_test.txt")
        with open(test_file_raw, 'w') as f:
            f.write("Testing path fix")
        
        print(f"✅ Successfully wrote to {test_file_processed}")
        print(f"✅ Successfully wrote to {test_file_raw}")
        
        # Clean up test files
        os.remove(test_file_processed)
        os.remove(test_file_raw)
        
        return True
    except Exception as e:
        print(f"❌ Error testing housing paths: {e}")
        return False

def find_and_fix_mixed_paths():
    """Scan the entire project for mixed path separators and fix them"""
    print("\nScanning for mixed path separators in Python files...")
    
    # Pattern to find problematic mixed path separators
    pattern = r'["\']data\/demo_data\\\\?housing'
    
    # Find all Python files in the project
    python_files = glob.glob("**/*.py", recursive=True)
    
    fixed_count = 0
    for file_path in python_files:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                content = f.read()
            except UnicodeDecodeError:
                print(f"⚠️ Could not read {file_path} due to encoding issues. Skipping.")
                continue
        
        if re.search(pattern, content):
            print(f"Found mixed paths in {file_path}")
            
            # Add import if needed
            if "path_utils" not in content and file_path != "data/path_utils.py":
                if file_path.startswith("data/"):
                    import_line = "from .path_utils import ensure_path_sep, make_path\n"
                else:
                    import_line = "from data.path_utils import ensure_path_sep, make_path\n"
                
                # Add after imports
                if re.search(r'^import\s+', content, re.MULTILINE):
                    content = re.sub(
                        r'(import\s+[^\n]+\n+)',
                        r'\1' + import_line,
                        content,
                        count=1
                    )
                else:
                    # Add at the beginning
                    content = import_line + content
            
            # Fix mixed paths
            content = re.sub(
                r'(["\'])data\/demo_data\\\\?housing\\\\?processed(["\'])',
                r'ensure_path_sep(os.path.join("data", "demo_data", "housing", "processed"))',
                content
            )
            
            content = re.sub(
                r'(["\'])data\/demo_data\\\\?housing\\\\?raw(["\'])',
                r'ensure_path_sep(os.path.join("data", "demo_data", "housing", "raw"))',
                content
            )
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            fixed_count += 1
    
    print(f"✅ Fixed mixed paths in {fixed_count} files")
    return fixed_count > 0

def main():
    """Main function to fix path issues"""
    print("=" * 60)
    print("Path Issue Fix Utility")
    print("=" * 60)
    
    # Step 1: Update app.py
    fix_paths_in_app()
    
    # Step 2: Update data/loader.py
    apply_path_fix_to_data_loader()
    
    # Step 3: Find and fix any other mixed paths in the project
    find_and_fix_mixed_paths()
    
    # Step 4: Test housing dataset paths
    if test_housing_paths():
        print("\n✅ Path fixes applied and tested successfully!")
        print("You should now be able to use the housing dataset without permission errors.")
        return True
    else:
        print("\n❌ Path fix testing failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 