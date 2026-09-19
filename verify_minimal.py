#!/usr/bin/env python
"""
Verify that app_minimal.py can be compiled without syntax errors.
"""

import sys

def test_compile():
    """Test that app_minimal.py compiles without syntax errors."""
    print("Testing app_minimal.py compilation...")
    
    try:
        # Read the file
        with open("app_minimal.py", "r", encoding="utf-8") as f:
            source = f.read()
        
        # Compile to check for syntax errors
        compile(source, "app_minimal.py", "exec")
        
        print("✅ app_minimal.py successfully compiled without syntax errors.")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in app_minimal.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Minimal App Verification")
    print("=" * 60)
    
    success = test_compile()
    
    if success:
        print("\n✅ Success! app_minimal.py is syntactically correct.")
        sys.exit(0)
    else:
        print("\n❌ Failed! app_minimal.py has syntax errors.")
        sys.exit(1) 