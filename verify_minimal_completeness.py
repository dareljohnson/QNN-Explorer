#!/usr/bin/env python
"""
Verify that app_minimal.py has all the necessary components
to resolve the original data availability issue.
"""

import sys

def test_imports_and_fixes():
    """Test that app_minimal.py has the right imports and fixes."""
    print("Checking app_minimal.py for required components...")
    
    required_components = [
        # Important imports
        "import streamlit as st",
        "import torch",
        "import pandas as pd",
        "from data.loader import load_data",
        "from housing_utils import render_housing_price_form, render_prediction_details",
        
        # Fixed ending structure
        "with tab_run_history:",
        "render_run_history_tab()",
        
        # Fixed UI elements
        'st.tabs(',
        '["Model Configuration", "Data", "Train", "Predict", "Visualize", "Help", "Run History"]',
        
        # Check that session state is properly initialized
        "if 'loaded_data' not in st.session_state:",
        "st.session_state.loaded_data = None",
        
        # Check for other key components
        "from core.fusion import HybridModel",
        "AVAILABLE_CLASSICAL_BACKBONES",
    ]
    
    try:
        # Read the file
        with open("app_minimal.py", "r", encoding="utf-8") as f:
            source = f.read()
        
        # Check for each required component
        missing = []
        for component in required_components:
            if component not in source:
                missing.append(component)
        
        if missing:
            print("❌ Missing required components:")
            for item in missing:
                print(f"  - {item}")
            return False
        
        # Check that the CSV data availability issue is potentially resolved
        if "from data.loader import load_data" in source and "if 'loaded_data' not in st.session_state:" in source:
            print("✅ Basic structure for handling loaded_data is present")
        else:
            print("❌ Missing basic structure for handling loaded_data")
            return False
        
        # Check for global references to data_type at the end
        lines = source.split("\n")
        last_20_lines = lines[-20:]
        if any("data_type" in line and not line.strip().startswith("#") and "=" in line for line in last_20_lines):
            print("❌ Found potential problematic data_type references at the end of the file")
            return False
        
        # Check that the ending is correct (housing_utils import should be the last line)
        expected_ending = "from housing_utils import render_housing_price_form, render_prediction_details"
        if source.strip().endswith(expected_ending):
            print("✅ File ends correctly with housing_utils import")
        else:
            print("❌ File doesn't end with expected housing_utils import")
            return False
        
        print("✅ All required components found in app_minimal.py")
        return True
    except Exception as e:
        print(f"❌ Error checking app_minimal.py: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("App Completeness Verification")
    print("=" * 60)
    
    success = test_imports_and_fixes()
    
    if success:
        print("\n✅ Success! app_minimal.py has all required components.")
        print("You can use this file as the base for your final app.py version.")
        sys.exit(0)
    else:
        print("\n❌ Failed! app_minimal.py is missing required components.")
        sys.exit(1) 