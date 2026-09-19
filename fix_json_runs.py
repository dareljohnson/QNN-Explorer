#!/usr/bin/env python
"""
JSON Run Repair Tool

This script repairs corrupted JSON run files in the run history database.
It can repair a specific run or scan all runs for corruption and repair them.

Usage:
    python fix_json_runs.py [-h] [--all] [--dry-run] [run_id]

Arguments:
    run_id      The ID of the run to repair (optional if --all is used)

Options:
    -h, --help  Show this help message and exit
    --all       Repair all corrupted run files
    --dry-run   Show what would be repaired without making changes
"""

import os
import json
import argparse
import glob
import sys
from datetime import datetime
from utils.json_repair import repair_json_file
from utils.run_history import RUN_DETAILS_PATH, init_history_dirs

def check_run_json(file_path):
    """
    Check if a JSON file is valid.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        tuple: (is_valid, error_info)
    """
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True, None
    except json.JSONDecodeError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error reading file: {e}"

def repair_run(run_id, dry_run=False):
    """
    Repair a specific run JSON file.
    
    Args:
        run_id (int): The ID of the run to repair
        dry_run (bool): If True, only check without repairing
        
    Returns:
        bool: Success status
    """
    # Define the path to the run JSON file
    run_file = os.path.join(RUN_DETAILS_PATH, f"run_{run_id}.json")
    
    if not os.path.exists(run_file):
        print(f"Error: Run {run_id} not found.")
        return False
    
    # Check if the file is valid
    is_valid, error_info = check_run_json(run_file)
    
    if is_valid:
        print(f"Run {run_id}: JSON file is valid. No repair needed.")
        return True
    
    print(f"Run {run_id}: Corrupted JSON detected. Error: {error_info}")
    
    if dry_run:
        print(f"Would repair run {run_id} (dry run)")
        return True
    
    # Attempt repair
    success, data, info = repair_json_file(run_file)
    
    if success:
        print(f"Run {run_id}: Successfully repaired! {info}")
        
        # Save the repaired data
        with open(run_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Run {run_id}: Repaired data saved.")
        return True
    else:
        print(f"Run {run_id}: Repair failed. {info}")
        return False

def scan_and_repair_all(dry_run=False):
    """
    Scan all run JSON files and repair corrupted ones.
    
    Args:
        dry_run (bool): If True, only scan without repairing
        
    Returns:
        tuple: (total_runs, corrupted_runs, repaired_runs)
    """
    init_history_dirs()
    
    # Find all JSON run files
    run_files = glob.glob(os.path.join(RUN_DETAILS_PATH, "run_*.json"))
    
    total_runs = len(run_files)
    corrupted_runs = 0
    repaired_runs = 0
    
    print(f"Scanning {total_runs} run files for corruption...")
    
    for run_file in run_files:
        # Extract run ID from filename
        run_id = os.path.basename(run_file).replace("run_", "").replace(".json", "")
        
        # Check if the file is valid
        is_valid, error_info = check_run_json(run_file)
        
        if not is_valid:
            corrupted_runs += 1
            print(f"Run {run_id}: Corrupted JSON detected. Error: {error_info}")
            
            if not dry_run:
                # Attempt repair
                success, data, info = repair_json_file(run_file)
                
                if success:
                    print(f"Run {run_id}: Successfully repaired! {info}")
                    
                    # Save the repaired data
                    with open(run_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"Run {run_id}: Repaired data saved.")
                    repaired_runs += 1
                else:
                    print(f"Run {run_id}: Repair failed. {info}")
    
    return total_runs, corrupted_runs, repaired_runs

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Repair corrupted JSON run files")
    parser.add_argument("run_id", nargs="?", type=int, help="The ID of the run to repair")
    parser.add_argument("--all", action="store_true", help="Repair all corrupted run files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be repaired without making changes")
    
    args = parser.parse_args()
    
    # Initialize directories
    init_history_dirs()
    
    # Check that at least one option is provided
    if not args.run_id and not args.all:
        parser.print_help()
        print("\nError: Either provide a run_id or use --all option.")
        sys.exit(1)
    
    # Repair mode
    if args.all:
        print("=== Scanning all run files for corruption ===")
        if args.dry_run:
            print("DRY RUN: No changes will be made")
        
        total, corrupted, repaired = scan_and_repair_all(args.dry_run)
        
        print("\n=== Summary ===")
        print(f"Total run files: {total}")
        print(f"Corrupted files: {corrupted}")
        if not args.dry_run:
            print(f"Successfully repaired: {repaired}")
            print(f"Failed to repair: {corrupted - repaired}")
        
        if corrupted == 0:
            print("\nNo corrupted files found!")
        elif args.dry_run:
            print(f"\nFound {corrupted} corrupted files. Run without --dry-run to repair.")
    else:
        print(f"=== Repairing run {args.run_id} ===")
        if args.dry_run:
            print("DRY RUN: No changes will be made")
        
        success = repair_run(args.run_id, args.dry_run)
        
        if success and not args.dry_run:
            print(f"\nRun {args.run_id} successfully repaired!")
        elif success and args.dry_run:
            print(f"\nRun {args.run_id} needs repair. Run without --dry-run to repair.")
        else:
            print(f"\nFailed to repair run {args.run_id}.")

if __name__ == "__main__":
    main() 