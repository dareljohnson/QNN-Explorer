# JSON Repair Tool - Quick Reference Guide

## Overview

The JSON Repair Tool helps you recover corrupted run history files in the quantum application. If you encounter errors like "Failed to repair JSON" or "Error decoding JSON file," this tool can help restore your data.

## Quick Usage

### Repair a Specific Run

If you know which run is causing problems (e.g., run 25):

```bash
python fix_json_runs.py 25
```

### Check All Files for Corruption

To scan all run files without making changes (safe to run anytime):

```bash
python fix_json_runs.py --all --dry-run
```

### Repair All Corrupted Files

To automatically repair all corrupted run files:

```bash
python fix_json_runs.py --all
```

## Common Error Messages and Solutions

| Error Message | What It Means | Solution |
|---------------|--------------|----------|
| `Expecting value: line X column Y` | JSON file was truncated during saving | Run `fix_json_runs.py <run_id>` to repair |
| `Expecting ',' delimiter` | Missing comma in JSON structure | Run `fix_json_runs.py <run_id>` to repair |
| `Error reading backup JSON` | Both main and backup files are corrupted | Run with `--all` flag to use advanced recovery |
| `No valid detail files found` | Run file is missing or severely damaged | Recovery will create minimal valid data |

## What Happens During Repair

When you run the repair tool:

1. It creates a backup of the original file before making changes
2. It applies multiple repair strategies to recover as much data as possible
3. If successful, it saves the repaired data back to the original file
4. If repair fails, it creates minimal recovery data with the run ID and metadata

## Tips for Preventing Corruption

1. Avoid interrupting the application during model training or evaluation
2. Let runs complete normally when possible
3. Use the application's save/export features to back up important results
4. Run periodic checks using `fix_json_runs.py --all --dry-run`

## Getting Help

If you encounter persistent issues with run history files, you can:

1. Check the application logs for detailed error information
2. Look for backup files with timestamps in the data/history/run_details directory
3. Contact support with the specific error messages and run IDs 