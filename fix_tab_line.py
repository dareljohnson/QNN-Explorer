#!/usr/bin/env python
"""
Fix the tab names line in app_clean.py (line 185).
"""

import sys

def fix_tab_line(input_file, output_file):
    """Fix the tab names line that has an unterminated string."""
    print(f"Reading {input_file}...")
    
    try:
        # Read all lines from the file
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Look for the tabs line
        found = False
        for i, line in enumerate(lines):
            if "st.tabs(" in line and i < len(lines) - 1 and "[" in lines[i+1]:
                # Found the tabs declaration line
                print(f"Found tabs line at line {i+1}")
                
                # Replace the tab names line with a fixed version
                lines[i+1] = '    ["Model Configuration", "Data", "Train", "Predict", "Visualize", "Help", "Run History"]\n'
                
                found = True
                print(f"Fixed tabs line (line {i+2})")
                break
        
        if not found:
            print("Could not find the tabs line")
            return False
        
        # Write the modified content back to a new file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"Created {output_file} with fixed tabs line")
        return True
    
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if fix_tab_line("app_clean.py", "app_fixed_tabs.py"):
        print("✅ Success!")
    else:
        print("❌ Failed!")
        sys.exit(1) 