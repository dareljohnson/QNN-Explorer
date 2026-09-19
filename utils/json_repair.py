"""
JSON Repair Utilities

This module provides tools for repairing corrupted JSON files.
It implements various strategies to recover data from damaged JSON.
"""

import os
import json
import re
import shutil
from datetime import datetime

def repair_json_file(file_path, backup=True):
    """
    Attempts to repair a corrupted JSON file using multiple strategies.
    
    Args:
        file_path (str): Path to the corrupted JSON file
        backup (bool): Whether to create a backup of the original file
    
    Returns:
        tuple: (success, repaired_data, error_info)
    """
    if not os.path.exists(file_path):
        return False, None, "File does not exist"
    
    # Create a backup if requested
    if backup:
        backup_path = f"{file_path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
        try:
            shutil.copy(file_path, backup_path)
        except Exception as e:
            print(f"Warning: Failed to create backup: {e}")
    
    # Read the file content
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        # For unreadable files, create minimal recovery data
        return create_minimal_recovery_data(file_path)
    
    # If file is empty or nearly empty, create minimal recovery data
    if not content or len(content.strip()) < 5:
        return create_minimal_recovery_data(file_path)
    
    # Try to parse the file directly first
    try:
        data = json.loads(content)
        # If successful, no repair needed
        return True, data, "No repair needed"
    except json.JSONDecodeError as e:
        error_info = f"Original error: {e}"
        
        # Try repair strategies in order of likely success
        repair_strategies = [
            repair_truncated_file,
            repair_unbalanced_braces,
            extract_complete_objects,
            repair_trailing_commas,
            repair_missing_quotes,
            repair_corrupted_strings
        ]
        
        for strategy in repair_strategies:
            try:
                success, repaired_data, strategy_info = strategy(content)
                if success:
                    # Add repair info to data if it's a dictionary
                    if isinstance(repaired_data, dict):
                        repaired_data['repair_info'] = {
                            'repaired': True,
                            'strategy': strategy.__name__,
                            'original_error': str(e),
                            'strategy_info': strategy_info
                        }
                    return True, repaired_data, f"Repaired with {strategy.__name__}: {strategy_info}"
            except Exception as strategy_e:
                pass
        
        # If all strategy-based repairs fail, try creating minimal recovery data
        print(f"All repair strategies failed. Attempting to create minimal recovery data.")
        return create_minimal_recovery_data(file_path)

def repair_truncated_file(content):
    """
    Attempts to repair a JSON file that was truncated during writing.
    This typically happens when a write operation is interrupted.
    
    Args:
        content (str): Content of the corrupted JSON file
        
    Returns:
        tuple: (success, repaired_data, info)
    """
    # Make sure we have some meaningful content to work with
    if not content or not content.strip():
        return False, None, "Empty content"
    
    # Count opening and closing braces/brackets
    open_braces = content.count('{')
    close_braces = content.count('}')
    open_brackets = content.count('[')
    close_brackets = content.count(']')
    
    # If braces/brackets are unbalanced, try to balance them
    repaired = content.strip()
    
    # Check if we need to add closing braces and brackets
    if open_braces > close_braces:
        # Add missing closing braces
        repaired += '}' * (open_braces - close_braces)
    
    if open_brackets > close_brackets:
        # Add missing closing brackets
        repaired += ']' * (open_brackets - close_brackets)
    
    # Check for unterminated strings and fix them
    if repaired.count('"') % 2 != 0:
        # Add a closing quote
        repaired += '"'
    
    # If the content is truncated in the middle of a property, try to complete it
    if repaired.rstrip().endswith(':') or repaired.rstrip().endswith(':', 1):
        # If it ends with a colon, add a null value
        repaired += ' null'
    
    # If it ends with a comma, remove it
    if repaired.rstrip().endswith(','):
        repaired = repaired.rstrip().rstrip(',')
    
    # Try to parse the repaired content
    try:
        data = json.loads(repaired)
        return True, data, f"Added {open_braces - close_braces} closing braces and {open_brackets - close_brackets} closing brackets"
    except json.JSONDecodeError as e:
        # If still failing, try more aggressive repairs
        
        # Try 1: Check if we have at least one top-level opening brace
        if '{' in repaired and not repaired.strip().startswith('{'):
            # Try to extract everything from the first { and re-close it
            first_brace = repaired.find('{')
            extracted = repaired[first_brace:]
            
            # Count how many braces we need to close
            open_count = extracted.count('{')
            close_count = extracted.count('}')
            if open_count > close_count:
                extracted += '}' * (open_count - close_count)
            
            try:
                data = json.loads(extracted)
                return True, data, "Extracted from first opening brace and repaired"
            except:
                pass
        
        # Try 2: Even more aggressive - create a minimal valid object with whatever we can extract
        try:
            # Look for key-value pairs
            minimal_json = '{'
            
            # Extract potential keys and values using regex
            import re
            key_value_pattern = r'"([^"]+)"\s*:\s*(?:"([^"]*)"|\d+|true|false|null|\{[^}]*\}|\[[^\]]*\])'
            matches = re.findall(key_value_pattern, repaired)
            
            if matches:
                # Construct a valid JSON with found key-value pairs
                for i, (key, value) in enumerate(matches):
                    # If value is empty, set it to null
                    if not value:
                        value = 'null'
                    # Add commas between items except for the last one
                    separator = ', ' if i < len(matches) - 1 else ''
                    minimal_json += f'"{key}": "{value}"{separator}'
                
                minimal_json += '}'
                
                data = json.loads(minimal_json)
                return True, data, "Created minimal valid JSON from extracted key-value pairs"
        except:
            pass
        
        return False, None, f"Basic balancing was insufficient: {e}"

def repair_unbalanced_braces(content):
    """
    Attempts to repair unbalanced braces by analyzing structure.
    More sophisticated than simple counting.
    
    Args:
        content (str): Content of the corrupted JSON file
        
    Returns:
        tuple: (success, repaired_data, info)
    """
    # Find the innermost unclosed object
    stack = []
    for i, char in enumerate(content):
        if char == '{' or char == '[':
            stack.append((char, i))
        elif char == '}' and stack and stack[-1][0] == '{':
            stack.pop()
        elif char == ']' and stack and stack[-1][0] == '[':
            stack.pop()
    
    if not stack:
        # No unbalanced braces found, but there might be other issues
        return False, None, "No unbalanced braces detected"
    
    # Try to balance each unclosed object
    repaired = content
    for bracket_type, _ in reversed(stack):
        if bracket_type == '{':
            repaired += '}'
        else:
            repaired += ']'
    
    # Try to parse
    try:
        data = json.loads(repaired)
        return True, data, f"Balanced {len(stack)} unclosed objects"
    except json.JSONDecodeError:
        return False, None, "Advanced balancing was insufficient"

def extract_complete_objects(content):
    """
    Attempts to extract a complete JSON object from the content.
    Finds the largest valid JSON object.
    
    Args:
        content (str): Content of the corrupted JSON file
        
    Returns:
        tuple: (success, repaired_data, info)
    """
    # Try to find the outermost object
    if content.lstrip().startswith('{'):
        # Find the last complete JSON object
        brace_level = 0
        start_index = content.find('{')
        
        for i in range(start_index, len(content)):
            if content[i] == '{':
                brace_level += 1
            elif content[i] == '}':
                brace_level -= 1
                if brace_level == 0:
                    # Found a complete object
                    try:
                        obj = content[start_index:i+1]
                        data = json.loads(obj)
                        return True, data, f"Extracted complete object of length {len(obj)}"
                    except:
                        pass
    
    # If no complete object found, try finding any valid JSON
    for i in range(len(content)):
        for j in range(len(content), i, -1):
            try:
                sub_content = content[i:j]
                if (sub_content.startswith('{') and sub_content.endswith('}')) or \
                   (sub_content.startswith('[') and sub_content.endswith(']')):
                    data = json.loads(sub_content)
                    return True, data, f"Extracted valid subset from position {i} to {j}"
            except:
                pass
    
    return False, None, "No valid JSON object could be extracted"

def repair_trailing_commas(content):
    """
    Removes trailing commas in objects and arrays which are invalid in JSON.
    
    Args:
        content (str): Content of the corrupted JSON file
        
    Returns:
        tuple: (success, repaired_data, info)
    """
    # Replace trailing commas in objects
    fixed_content = re.sub(r',\s*}', '}', content)
    # Replace trailing commas in arrays
    fixed_content = re.sub(r',\s*]', ']', fixed_content)
    
    # Try to parse
    try:
        data = json.loads(fixed_content)
        return True, data, "Removed trailing commas"
    except json.JSONDecodeError:
        return False, None, "Removing trailing commas was insufficient"

def repair_missing_quotes(content):
    """
    Attempts to repair missing quotes around keys or string values.
    
    Args:
        content (str): Content of the corrupted JSON file
        
    Returns:
        tuple: (success, repaired_data, info)
    """
    # Look for unquoted keys - a common error
    # This pattern finds keys that are not quoted, supporting alphanumeric characters and underscores
    # 1. After opening brace or comma
    # 2. Optional whitespace
    # 3. One or more alphanumeric/underscore characters
    # 4. Optional whitespace
    # 5. Colon
    fixed_content = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', content)
    
    # Handle more complex cases with nested unquoted keys
    # Look for cases where there might be nested unquoted keys after colons
    fixed_content2 = re.sub(r':\s*{([^{}]*)}\s*([,}])', lambda m: 
                         re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', 
                                ":{" + m.group(1) + "}" + m.group(2)), 
                         fixed_content)
    
    # If the content doesn't start with proper JSON delimiters, wrap it
    if not fixed_content2.strip().startswith('{') and not fixed_content2.strip().startswith('['):
        wrapped = '{' + fixed_content2 + '}'
    else:
        wrapped = fixed_content2
    
    # Try to parse with quotes added
    try:
        data = json.loads(fixed_content2)
        return True, data, "Added missing quotes around keys"
    except json.JSONDecodeError:
        # Try with wrapped content
        try:
            if wrapped != fixed_content2:
                data = json.loads(wrapped)
                return True, data, "Added missing quotes and wrapped in object"
        except:
            pass
        
        # Try more aggressive repairs - handle unquoted values
        try:
            # Replace common unquoted string values with quoted ones
            # This regex finds patterns that look like keys with unquoted values
            value_pattern = r'"([^"]+)":\s*([\w.-]+)([,}])'
            aggressive_fix = re.sub(value_pattern, r'"\1": "\2"\3', fixed_content2)
            
            data = json.loads(aggressive_fix)
            return True, data, "Added quotes to unquoted values"
        except:
            # One last attempt - try to extract any valid JSON object
            # Find substrings that might be valid JSON objects
            for i in range(len(fixed_content2)):
                if fixed_content2[i] == '{':
                    # Try to find the matching closing brace
                    brace_level = 1
                    for j in range(i+1, len(fixed_content2)):
                        if fixed_content2[j] == '{':
                            brace_level += 1
                        elif fixed_content2[j] == '}':
                            brace_level -= 1
                            if brace_level == 0:
                                # Extract this object and try to parse it
                                obj = fixed_content2[i:j+1]
                                try:
                                    data = json.loads(obj)
                                    return True, data, "Extracted valid object from content with quotes added"
                                except:
                                    pass
            
            return False, None, "Adding missing quotes was insufficient"

def repair_corrupted_strings(content):
    """
    Attempts to repair corrupted string values by finding and fixing unescaped quotes.
    
    Args:
        content (str): Content of the corrupted JSON file
        
    Returns:
        tuple: (success, repaired_data, info)
    """
    # This is a simplified approach - a real implementation would be more complex
    # Look for unescaped quotes in strings
    in_string = False
    escaped = False
    fixed_chars = []
    
    for char in content:
        if not in_string:
            if char == '"':
                in_string = True
            fixed_chars.append(char)
        else:
            if char == '\\':
                escaped = not escaped
                fixed_chars.append(char)
            elif char == '"' and not escaped:
                in_string = False
                fixed_chars.append(char)
            elif char == '"' and escaped:
                # Already escaped
                fixed_chars.append(char)
                escaped = False
            else:
                if escaped:
                    escaped = False
                fixed_chars.append(char)
    
    fixed_content = ''.join(fixed_chars)
    
    # Try to parse
    try:
        data = json.loads(fixed_content)
        return True, data, "Fixed corrupted strings"
    except json.JSONDecodeError:
        return False, None, "String repair was insufficient"

def create_minimal_recovery_data(file_path):
    """
    Creates minimal recovery data for severely corrupted files.
    Extracts what information it can from filename and content.
    
    Args:
        file_path (str): Path to the corrupted file
        
    Returns:
        tuple: (success, repaired_data, info)
    """
    # Extract run ID from filename if possible
    filename = os.path.basename(file_path)
    run_id = None
    
    if filename.startswith("run_") and filename.endswith(".json"):
        try:
            run_id = int(filename.replace("run_", "").replace(".json", ""))
        except:
            pass
    
    # Try to extract some content from the file
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Look for any key-value pairs we can extract
        import re
        key_values = {}
        
        # Look for patterns like "key": value
        key_value_pattern = r'"([^"]+)"\s*:\s*("[^"]*"|[0-9]+|true|false|null)'
        matches = re.findall(key_value_pattern, content)
        
        for key, value in matches:
            # Process the value
            if value.startswith('"') and value.endswith('"'):
                # String value
                value = value[1:-1]
            elif value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.lower() == 'null':
                value = None
            else:
                # Try to convert to number
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except:
                    pass
            
            key_values[key] = value
        
        # Create minimal recovery data
        recovery_data = {
            "recovered": True,
            "recovery_note": f"This is minimal recovery data created from a severely corrupted file: {filename}",
            "repair_info": {
                "repaired": True,
                "strategy": "create_minimal_recovery_data",
                "severity": "severe_corruption",
                "original_file": filename
            }
        }
        
        # Add run ID if available
        if run_id is not None:
            recovery_data["run_id"] = run_id
        
        # Add any extracted key-values
        for key, value in key_values.items():
            recovery_data[key] = value
        
        return True, recovery_data, "Created minimal recovery data"
    except Exception as e:
        # If all else fails, create absolute minimal data
        recovery_data = {
            "recovered": True,
            "recovery_note": f"This is minimal recovery data. Original file was unreadable: {filename}",
            "repair_info": {
                "repaired": True,
                "strategy": "create_minimal_recovery_data",
                "severity": "unreadable_file",
                "error": str(e)
            }
        }
        
        # Add run ID if available
        if run_id is not None:
            recovery_data["run_id"] = run_id
            
        return True, recovery_data, "Created absolute minimal recovery data"

def fix_run_json_file(run_id, details_path):
    """
    Attempt to repair a corrupted run JSON file.
    
    Args:
        run_id (int): Run ID number
        details_path (str): Path to run details directory
        
    Returns:
        tuple: (success, message)
    """
    json_file = os.path.join(details_path, f"run_{run_id}.json")
    
    if not os.path.exists(json_file):
        return False, f"File not found: {json_file}"
    
    # Create timestamped backup
    backup_file = f"{json_file}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        shutil.copy(json_file, backup_file)
    except Exception as e:
        return False, f"Failed to create backup: {e}"
    
    # Attempt repair
    success, repaired_data, info = repair_json_file(json_file, backup=False)
    
    if success:
        # Save the repaired data back to the file
        try:
            with open(json_file, 'w') as f:
                json.dump(repaired_data, f, indent=2)
            return True, f"Successfully repaired file: {info}"
        except Exception as e:
            return False, f"Repair succeeded but failed to save: {e}"
    else:
        return False, f"Repair failed: {info}"

if __name__ == "__main__":
    # This allows the script to be run directly to repair a file
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python json_repair.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    print(f"Attempting to repair {file_path}...")
    
    success, data, info = repair_json_file(file_path)
    
    if success:
        print(f"Repair successful: {info}")
        
        # Save the repaired data
        output_path = f"{file_path}.repaired"
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Repaired data saved to {output_path}")
    else:
        print(f"Repair failed: {info}") 