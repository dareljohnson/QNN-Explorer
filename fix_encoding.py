#!/usr/bin/env python
"""
Simple script to remove non-ASCII characters from app_final.py
and ensure the file has the right ending.
"""

import re
import sys

def fix_file(input_file, output_file):
    """Fix encoding issues by replacing all non-ASCII chars with spaces."""
    print(f"Processing {input_file}...")
    
    try:
        # Read the file, ignoring encoding errors
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Replace all non-ASCII characters with spaces
        content = re.sub(r'[^\x00-\x7F]+', ' ', content)
        
        # Fix the ending of the file
        if "render_run_history_tab()" in content:
            ending = "\n\n# --- Import housing utilities for prediction form ---\nfrom housing_utils import render_housing_price_form, render_prediction_details\n"
            
            # Split content at render_run_history_tab() and fix the ending
            parts = content.split("render_run_history_tab()")
            if len(parts) == 2:
                content = parts[0] + "render_run_history_tab()" + ending
                print("✅ Fixed file ending")
            
        # Write the fixed content
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Created {output_file}")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if fix_file("app_final.py", "app_clean.py"):
        print("✅ Success!")
    else:
        print("❌ Failed!")
        sys.exit(1) 