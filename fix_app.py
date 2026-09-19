#!/usr/bin/env python
"""
Script to fix encoding issues in app.py and remove the erroneous code at the end of the file.
"""

import re
import os
import sys

def fix_app_file():
    """Fix the app.py file by removing problematic code at the end."""
    print("Starting app.py fix process...")
    
    # Create a backup of the original file
    backup_filename = "app.py.bak"
    if os.path.exists(backup_filename):
        print(f"Backup file {backup_filename} already exists, skipping backup creation")
    else:
        try:
            with open("app.py", "r", encoding="utf-8", errors="ignore") as f_in:
                content = f_in.read()
            
            with open(backup_filename, "w", encoding="utf-8") as f_out:
                f_out.write(content)
            print(f"Created backup at {backup_filename}")
        except Exception as e:
            print(f"Error creating backup: {e}")
            return False
    
    try:
        # Read the file content, ignoring encoding errors
        with open("app.py", "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        # Find the problematic section at the end of the file
        end_marker = "with tab_run_history:"
        render_marker = "render_run_history_tab()"
        housing_import = 'from housing_utils import render_housing_price_form, render_prediction_details'
        
        # Split by the end marker to isolate the last part
        if end_marker in content:
            parts = content.split(end_marker)
            if len(parts) == 2:
                # Check if there's problematic code after the render_run_history_tab() call
                last_part = parts[1]
                
                if render_marker in last_part:
                    # Find where to cut off the code
                    lines = last_part.split('\n')
                    
                    # Find the index of the render_marker line
                    render_line_idx = -1
                    for i, line in enumerate(lines):
                        if render_marker in line:
                            render_line_idx = i
                            break
                    
                    if render_line_idx >= 0:
                        # Keep only lines up to and including render marker, and add housing import
                        fixed_last_part = '\n'.join(lines[:render_line_idx+1]) + "\n\n# --- Import housing utilities for prediction form ---\n" + housing_import + "\n"
                        
                        # Combine with the first part
                        fixed_content = parts[0] + end_marker + fixed_last_part
                        
                        # Write the fixed content back to a new file
                        with open("app_fixed_again.py", "w", encoding="utf-8") as f_out:
                            f_out.write(fixed_content)
                        
                        print("✅ Successfully created fixed version at app_fixed_again.py")
                        print("Problematic code at the end of the file has been removed.")
                        return True
                    else:
                        print("❌ Could not find render_run_history_tab() line in the last part")
                else:
                    print(f"❌ Could not find '{render_marker}' in the last part of the file")
            else:
                print(f"❌ Unexpected splitting result: got {len(parts)} parts")
        else:
            print(f"❌ Could not find '{end_marker}' marker in file")
                
    except Exception as e:
        print(f"❌ Error fixing app.py: {e}")
    
    return False

if __name__ == "__main__":
    success = fix_app_file()
    if success:
        print("🎉 Fix completed. Please check app_fixed_again.py")
        sys.exit(0)
    else:
        print("❌ Fix failed. Please check error messages above.")
        sys.exit(1) 