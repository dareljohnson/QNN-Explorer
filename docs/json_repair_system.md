# JSON Repair System Documentation

## Overview

The JSON Repair System is a robust solution for handling corrupted JSON files in the application's run history database. It implements multiple repair strategies to recover data from damaged files, ensuring that valuable experiment results are not lost due to file corruption.

## Key Features

- **Multi-Strategy Repair:** Applies multiple repair techniques in sequence to maximize recovery chances
- **Automatic Backup:** Creates timestamped backups before attempting repairs
- **Minimal Recovery:** When full recovery is impossible, creates minimal valid data with critical identifiers
- **Command Line Tool:** Easy-to-use CLI for fixing individual or all corrupted run files
- **Comprehensive Testing:** Verified with unit tests for all repair strategies

## Repair Strategies

The system implements the following repair strategies, applied in order of likely success:

1. **Truncated File Repair:** Detects and fixes files truncated during writing by balancing braces and handling incomplete properties
2. **Unbalanced Braces Analysis:** Performs structural analysis to repair unbalanced braces in a sophisticated way
3. **Complete Object Extraction:** Extracts valid JSON objects from surrounding corrupted content
4. **Trailing Commas Removal:** Fixes invalid trailing commas in arrays and objects
5. **Missing Quotes Addition:** Repairs unquoted keys and values, a common source of JSON errors
6. **Corrupted Strings Repair:** Fixes string values with unescaped quotes or other string-related issues
7. **Minimal Recovery Data:** When all else fails, creates minimal valid data with run identification

## Usage

### Command Line Interface

The system provides a command-line interface for repairing corrupted JSON files:

```bash
# Repair a specific run
python fix_json_runs.py 25

# Scan all runs and repair corrupted files
python fix_json_runs.py --all

# Preview what would be repaired without making changes
python fix_json_runs.py --all --dry-run
```

### Programmatic Usage

The system can also be used programmatically within Python code:

```python
from utils.json_repair import repair_json_file

# Repair a file
success, repaired_data, info = repair_json_file('path/to/file.json')

if success:
    print(f"Repair successful: {info}")
else:
    print(f"Repair failed: {info}")
```

## Recovery Mechanism

When a JSON file is detected as corrupted, the system:

1. Creates a backup of the original file
2. Tries multiple repair strategies in sequence
3. Adds recovery metadata to the repaired data
4. If all strategies fail, creates minimal recovery data

## Minimal Recovery Data

When full repair is not possible, the system creates minimal recovery data containing:

- Run ID (when available)
- Recovery timestamp
- Error information
- Recovery note explaining the situation
- Any keys/values that could be extracted from the corrupted file

## Testing

The JSON repair system is thoroughly tested with unit tests that verify:

- Truncated file repair capabilities
- Handling of unbalanced braces
- Extraction of complete objects from corrupted files
- Removal of trailing commas
- Fixing missing quotes
- Handling severely corrupted files
- End-to-end file repair workflow

## Error Handling

The system provides detailed error reporting, including:

- Original error location and message
- Applied repair strategy
- Reason for failure when repair is unsuccessful
- Recovery notes in repaired data

## Implementation Details

The implementation uses a careful sequence of repair strategies, with each subsequent strategy more aggressive than the previous. This ensures that the least invasive repair strategy that can fix the file is applied first.

The system also ensures that repaired data includes metadata about the repair process, allowing tracking of which files have been repaired and how. 