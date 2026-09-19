"""
Simple script to check if app.py can run without syntax errors.
This can detect basic issues before trying to run the full Streamlit app.
"""

import importlib.util
import sys

def test_import_app():
    """Try to import app.py to check for syntax errors"""
    try:
        # Load the module from the file path
        spec = importlib.util.spec_from_file_location("app", "app.py")
        app_module = importlib.util.module_from_spec(spec)
        sys.modules["app"] = app_module
        
        # Execute the module
        spec.loader.exec_module(app_module)
        
        print("✅ app.py imported successfully without syntax errors")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in app.py: {e}")
        return False
    except ImportError as e:
        print(f"❌ Import error in app.py: {e}")
        print("This usually happens when a module is missing. Make sure all requirements are installed.")
        return False
    except Exception as e:
        print(f"❌ Error importing app.py: {e}")
        return False

if __name__ == "__main__":
    print("Checking app.py for syntax errors...")
    success = test_import_app()
    
    if success:
        print("app.py appears to be valid Python code.")
        sys.exit(0)
    else:
        print("app.py contains errors that need to be fixed.")
        sys.exit(1) 