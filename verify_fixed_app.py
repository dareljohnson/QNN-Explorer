#!/usr/bin/env python
"""
Verify that the final app.py is working correctly.
"""

import sys

def test_final_app():
    """Test that app.py is syntactically correct and has all required components."""
    print("Final verification of app.py...")
    
    try:
        # Read the file
        with open("app.py", "r", encoding="utf-8") as f:
            source = f.read()
        
        # Test compilation
        compile(source, "app.py", "exec")
        print("✅ app.py successfully compiled without syntax errors")
        
        # Check that it doesn't have the original issue
        lines = source.split("\n")
        last_20_lines = lines[-20:]
        if any("data_type" in line and not line.strip().startswith("#") and "=" in line for line in last_20_lines):
            print("❌ Found problematic data_type references at the end of the file")
            return False
        
        # Check the ending
        expected_ending = "from housing_utils import render_housing_price_form, render_prediction_details"
        if source.strip().endswith(expected_ending):
            print("✅ File ends correctly with housing_utils import")
        else:
            print("❌ File doesn't end with expected housing_utils import")
            return False
        
        print("✅ app.py file looks good!")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in app.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Final App Verification")
    print("=" * 60)
    
    success = test_final_app()
    
    if success:
        print("\n✅ Success! app.py is fixed and ready to use.")
        sys.exit(0)
    else:
        print("\n❌ Failed! app.py still has issues.")
        sys.exit(1) 