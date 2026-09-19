# Path Handling Guide for Quantum QNN

## Problem: Mixed Path Separators

The application was experiencing errors due to inconsistent path separators:
- Forward slashes (/) in string literals
- Backslashes (\) when using os.path.join() on Windows
- Mixing both in combined paths like: `data/demo_data\housing\processed`

## Fix: Consistent Path Handling

1. **Use `os.path.join()` consistently**
   - Instead of: `"data/demo_data/housing"`
   - Use: `os.path.join("data", "demo_data", "housing")`

2. **For existing paths with mixed separators:**
   - Import and use the new utilities in `data/path_utils.py`:
   ```python
   from data.path_utils import ensure_path_sep, make_path
   
   # Fix a potentially mixed path:
   safe_path = ensure_path_sep("data/demo_data\housing\processed")
   
   # Create a new path from parts:
   new_path = make_path("data/demo_data", "housing/processed")
   ```

3. **When loading files from paths:**
   ```python
   # Before loading or saving files, normalize the path:
   normalized_path = ensure_path_sep(path)
   with open(normalized_path, 'r') as f:
       # ...
   ```

This approach ensures cross-platform compatibility while fixing existing mixed paths.
