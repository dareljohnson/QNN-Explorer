#!/usr/bin/env python
"""
Script to remove BOM character from file.
"""

def remove_bom(input_file, output_file):
    # Read the file as binary
    with open(input_file, 'rb') as f:
        content = f.read()
    
    # Check if it starts with a BOM
    if content.startswith(b'\xef\xbb\xbf'):  # UTF-8 BOM
        print(f"Removing UTF-8 BOM from {input_file}")
        content = content[3:]
    elif content.startswith(b'\xff\xfe'):  # UTF-16 LE BOM
        print(f"Removing UTF-16 LE BOM from {input_file}")
        content = content[2:]
    elif content.startswith(b'\xfe\xff'):  # UTF-16 BE BOM
        print(f"Removing UTF-16 BE BOM from {input_file}")
        content = content[2:]
    elif content.startswith(b'\xff\xfe\x00\x00'):  # UTF-32 LE BOM
        print(f"Removing UTF-32 LE BOM from {input_file}")
        content = content[4:]
    elif content.startswith(b'\x00\x00\xfe\xff'):  # UTF-32 BE BOM
        print(f"Removing UTF-32 BE BOM from {input_file}")
        content = content[4:]
    else:
        print(f"No BOM detected in {input_file}")
    
    # Write to output file
    with open(output_file, 'wb') as f:
        f.write(content)
    
    print(f"Created {output_file} without BOM")
    
    # Double-check the result
    with open(output_file, 'rb') as f:
        new_content = f.read(4)  # Just check the first few bytes
    
    # Check if any BOMs are present
    bom_detected = False
    if new_content.startswith(b'\xef\xbb\xbf'):
        print(f"WARNING: UTF-8 BOM still detected in {output_file}")
        bom_detected = True
    elif new_content.startswith(b'\xff\xfe'):
        print(f"WARNING: UTF-16 LE BOM still detected in {output_file}")
        bom_detected = True
    elif new_content.startswith(b'\xfe\xff'):
        print(f"WARNING: UTF-16 BE BOM still detected in {output_file}")
        bom_detected = True
    
    if not bom_detected:
        print(f"SUCCESS: No BOM detected in {output_file}")
    
    return not bom_detected

if __name__ == "__main__":
    success = remove_bom("app_fixed_again.py", "app_final.py")
    if success:
        print("BOM removal successful!")
    else:
        print("Failed to completely remove BOM.") 