# Run History System Documentation

## Overview

The Run History system provides comprehensive tracking and management of your quantum model training sessions. It's designed to be robust, resilient to errors, and provide useful insights into your model development process.

## Features

- **Training Run Tracking**: Automatically logs all training runs with metadata
- **Detailed Metrics Storage**: Stores model configurations, training parameters, and performance metrics
- **Visualization**: Interactive charts for loss curves, entanglement metrics, and model performance
- **Error Resilience**: Multi-layered recovery mechanisms for corrupted data files
- **Data Management**: Tools to update, repair, and delete run records

## Architecture

The Run History system consists of two main components:

1. **Data Management Layer** (`utils/run_history.py`):
   - Handles data storage and retrieval
   - Implements error detection and recovery logic
   - Converts complex data types for stable storage
   - Creates backups to prevent data loss

2. **UI Layer** (`utils/run_history_tab.py`):
   - Provides the Streamlit interface for run history
   - Visualizes metrics and model performance
   - Offers tools for run management
   - Includes UI components for data recovery

## Data Storage Format

Run history data is stored in two formats:

1. **Summary CSV** (`data/history/run_history.csv`):
   - Contains basic information about all runs
   - Includes run ID, timestamp, configuration name, duration, and status
   - Serves as the primary index for run history

2. **Run Detail Files** (`data/history/run_details/run_X.json`):
   - Detailed JSON file for each run (X is the run ID)
   - Stores model configuration, training parameters, metrics, and results
   - Also maintains pickle (.pkl) versions for backward compatibility

## Error Handling and Recovery

The Run History system implements a multi-layered approach to error handling and recovery:

### 1. Corrupted JSON Handling

When a JSON file is corrupted, the system attempts to recover in this sequence:

```python
# Attempt to load JSON file
try:
    with open(json_file, 'r') as f:
        return json.load(f)
except json.JSONDecodeError as e:
    # Record error information
    print(f"Error decoding JSON file for run {run_id}: {e}")
    
    # Try partial recovery up to the error position
    try:
        with open(json_file, 'r') as f:
            content = f.read()
            
        valid_content = content[:e.pos]
        if valid_content.strip():
            # Try to close incomplete JSON objects
            if valid_content.count('{') > valid_content.count('}'):
                valid_content += '}'
                
            partial_data = json.loads(valid_content)
            partial_data['recovery_note'] = "This is partially recovered data"
            return partial_data
    except:
        pass
```

### 2. Backup File Fallback

If partial recovery fails, the system checks for backup files:

```python
# Try backup JSON file
if os.path.exists(json_backup):
    try:
        with open(json_backup, 'r') as f:
            data = json.load(f)
            data['recovery_note'] = "Recovered from backup file"
            return data
    except:
        pass
```

### 3. Pickle Format Alternative

As a third fallback, the system attempts to load the pickle version:

```python
# Try pickle format as fallback
if os.path.exists(pkl_file):
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            data['recovery_note'] = "Recovered from pickle format"
            return data
    except:
        pass
```

### 4. Minimal Recovery Data

If all previous methods fail, the system creates minimal recovery data:

```python
# Create minimal recovery data if all else fails
print(f"No valid detail files found for run {run_id}. Creating minimal details.")
return create_recovery_data(run_id, error_info)
```

## Error Prevention Strategies

1. **Multiple Storage Formats**:
   - Primary JSON format with human-readable indentation
   - Backup JSON format without indentation (more resilient to partial writes)
   - Pickle format as binary fallback option

2. **Safe Write Operations**:
   - Writes to temporary files first
   - Validates data before committing
   - Handles type conversion to ensure JSON compatibility

3. **Proactive Backups**:
   - Creates backup files automatically
   - Maintains backup integrity separate from primary files
   - Uses different serialization formats for redundancy

## User Interface Features

### Viewing Run History

The Run History tab provides a comprehensive view of all training runs:

1. **Run List Table**:
   - Shows all runs with ID, timestamp, configuration, duration, and status
   - Sortable columns for easy organization
   - Quick overview of all training sessions

2. **Run Details View**:
   - Detailed information about the selected run
   - Model configuration parameters
   - Training parameters and dataset information
   - Performance metrics specific to the task type

3. **Metric Visualizations**:
   - Loss curves showing training progress
   - Entanglement metrics for quantum circuits
   - Confusion matrices for classification tasks
   - Error metrics for regression tasks

### Error Recovery UI

When corrupted data is detected, the UI provides user-friendly recovery options:

1. **Warning Messages**:
   - Clear indication that data has been recovered
   - Information about the specific error encountered
   - Details about which recovery method was used

2. **Repair Options**:
   - "Repair Run Data" button to create clean records
   - Option to update run status or delete corrupted runs
   - Tools to create new minimal run details when needed

## Practical Examples

### Example 1: Viewing Run History

To view your run history after training models:

1. Navigate to the "Run History" tab
2. Browse the table of all runs
3. Click on a run to view its details
4. Examine metrics, configurations, and other information

### Example 2: Recovering from Corrupted Data

If you encounter corrupted data files:

1. The system automatically attempts recovery
2. A warning message indicates the file was recovered
3. Use the "Repair Run Data" button to create a clean record
4. Optionally delete runs that cannot be recovered

## Best Practices

1. **Regular Backups**:
   - Periodically export important runs
   - Maintain copies of critical configurations
   - Use version control for your entire project

2. **Naming Conventions**:
   - Use descriptive names for configurations
   - Include key parameters in configuration names
   - Maintain consistent naming patterns

3. **Data Cleanup**:
   - Delete obsolete or failed runs
   - Organize runs by project or experiment type
   - Keep the run history focused on relevant experiments

## Troubleshooting

### Common Issues

1. **"JSONDecodeError: Expecting value"**:
   - Indicates a corrupted JSON file
   - The automatic recovery should handle this
   - If problems persist, try the "Repair Run Data" button

2. **"Run data files not found"**:
   - Files may have been moved or deleted
   - Check the data/history/run_details directory
   - Use the "Create Empty Run Details" button to recreate minimal data

3. **Incorrect or missing metrics**:
   - May indicate partial data recovery
   - Check for recovery_note in the run details
   - Retrain the model if critical data is missing

### Manual Recovery

In extreme cases where automatic recovery fails:

1. Check the `data/history/run_details` directory
2. Look for backup files (`run_X.json.bak` or `run_X.pkl`)
3. Manually restore these files if needed
4. Use the UI's "Update Status" to mark corrupted runs

## Conclusion

The Run History system is designed to be robust, user-friendly, and resilient to errors. By implementing multiple layers of error recovery and providing clear user interfaces for data management, it ensures that your valuable training results are preserved even in the face of file system errors or application crashes. 