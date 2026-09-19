import sys
import os
import importlib.util

def test_module_import(module_name):
    """Test importing a module."""
    try:
        module = __import__(module_name)
        print(f"✅ Successfully imported {module_name}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return False

def test_app_imports():
    """Test importing all necessary modules for the app."""
    modules = [
        "utils.run_history",
        "utils.run_history_tab",
        "utils.metrics"
    ]
    
    success = True
    for module in modules:
        if not test_module_import(module):
            success = False
    
    if success:
        print("\n✅ All modules imported successfully!")
    else:
        print("\n❌ Some modules failed to import.")
    
    return success

if __name__ == "__main__":
    print("Testing app startup...")
    test_app_imports() 