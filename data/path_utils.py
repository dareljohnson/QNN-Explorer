import os
import re

def ensure_path_sep(path):
    """
    Ensure path uses consistent separators regardless of how it was created.
    
    This function normalizes paths that might have been created with mixed
    separators (forward slashes and backslashes on Windows).
    
    Args:
        path (str): The path to normalize
        
    Returns:
        str: The normalized path with consistent separators
    """
    # First, normalize by splitting on both types of separators
    parts = re.split(r'[\\/]', path)
    # Then rejoin using the OS-specific separator
    return os.path.join(*parts)

def make_path(*parts):
    """
    Create a path from parts ensuring consistent separators.
    
    This is a wrapper around os.path.join that ensures paths are created
    consistently even if some parts already include separators.
    
    Args:
        *parts: Path parts to join
        
    Returns:
        str: The resulting path with consistent separators
    """
    # First, flatten parts by splitting each on separators
    flat_parts = []
    for part in parts:
        flat_parts.extend(re.split(r'[\\/]', part))
    # Remove empty parts
    flat_parts = [p for p in flat_parts if p]
    # Join with the OS-specific separator
    return os.path.join(*flat_parts)
