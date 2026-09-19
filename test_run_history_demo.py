#!/usr/bin/env python
"""
Run History Error Recovery Demonstration

This script demonstrates the error recovery capabilities of the run history system.
It intentionally creates and corrupts run data, then shows how the system recovers.

Usage:
    python test_run_history_demo.py

The demonstration will:
1. Create sample run data with metrics and configuration
2. Save the run data normally
3. Corrupt the JSON file in different ways
4. Show how the system recovers the data through various fallback mechanisms
"""

import os
import json
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import shutil
import time

# Add the parent directory to the path if needed
if "." not in sys.path:
    sys.path.append(".")

# Import the run history module
from utils.run_history import (
    init_history_dirs, load_run_history, add_run, 
    save_run_details, load_run_details, get_run,
    RUN_DETAILS_PATH
)

class ColorPrint:
    """Helper class for colored terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @staticmethod
    def print(color, text, end='\n'):
        """Print text in color."""
        print(f"{color}{text}{ColorPrint.END}", end=end)
    
    @staticmethod
    def header(text):
        """Print header text."""
        print(f"\n{ColorPrint.HEADER}{ColorPrint.BOLD}{text}{ColorPrint.END}\n")
    
    @staticmethod
    def step(text):
        """Print step text."""
        print(f"{ColorPrint.BLUE}{ColorPrint.BOLD}==> {text}{ColorPrint.END}")
    
    @staticmethod
    def success(text):
        """Print success text."""
        print(f"{ColorPrint.GREEN}✓ {text}{ColorPrint.END}")
    
    @staticmethod
    def warning(text):
        """Print warning text."""
        print(f"{ColorPrint.YELLOW}⚠ {text}{ColorPrint.END}")
    
    @staticmethod
    def error(text):
        """Print error text."""
        print(f"{ColorPrint.RED}✗ {text}{ColorPrint.END}")

def create_sample_run():
    """Create a sample run with training data and metrics."""
    ColorPrint.step("Creating a sample training run...")
    
    # Add a run
    run_id = add_run("Demo Config", 120, "Completed")
    
    # Create detailed run data
    details = {
        'model_config': {
            'num_qubits': 6,
            'num_layers': 4,
            'ansatz_name': 'Ansatz 1 (Rot+CNOT Chain)',
            'encoding_method': 'amplitude',
            'observation_type': 'state',
            'classical_backbone_type': 'CNN',
            'classical_model_name': 'resnet18',
            'quantum_input_size': 64,
            'use_gpu': False
        },
        'training_params': {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10
        },
        'task_type': 'Classification',
        'metrics': {
            'accuracy': 0.92,
            'precision': 0.91,
            'recall': 0.92,
            'f1': 0.915,
            'auc': 0.95,
            'log_loss': 0.24,
            'confusion_matrix': np.array([[45, 5], [3, 47]]).tolist()
        },
        'weights_files': [
            'saved_models/weights_last_train.pt',
            'saved_models/weights_cnn_N6_L4.pt'
        ],
        'training_history': {
            'loss': [0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.28, 0.25],
            'entanglement': [0.5, 0.7, 0.9, 1.1, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
        },
        'data_info': {
            'data_type': 'Images',
            'samples': 100,
            'feature_columns': None,
            'processed_shape': [100, 64],
            'classes': ['class_0', 'class_1']
        }
    }
    
    # Save the details
    save_run_details(run_id, details)
    
    ColorPrint.success(f"Created run ID: {run_id} with detailed metrics and configuration")
    return run_id

def verify_run_data(run_id):
    """Verify that run data can be loaded correctly."""
    ColorPrint.step(f"Verifying run data for ID: {run_id}...")
    
    # Load the run details
    details = load_run_details(run_id)
    
    # Print some info
    if details:
        ColorPrint.success("Successfully loaded run details")
        print(f"  Model: {details['model_config']['classical_backbone_type']} + {details['model_config']['num_qubits']} qubits")
        print(f"  Accuracy: {details['metrics']['accuracy']:.4f}")
        print(f"  Final Entanglement: {details['training_history']['entanglement'][-1]:.4f}")
        
        if 'recovery_note' in details:
            ColorPrint.warning(f"Recovery note: {details['recovery_note']}")
    else:
        ColorPrint.error("Failed to load run details")
    
    return details

def corrupt_json_file(run_id, corruption_type="truncate"):
    """Corrupt the JSON file in different ways to test recovery mechanisms.
    
    Args:
        run_id: The run ID to corrupt
        corruption_type: The type of corruption to apply
            - "truncate": Truncate the file in the middle
            - "invalid": Add invalid JSON content
            - "delete": Delete the file
    """
    json_file = os.path.join(RUN_DETAILS_PATH, f"run_{run_id}.json")
    
    if not os.path.exists(json_file):
        ColorPrint.error(f"JSON file not found: {json_file}")
        return False
    
    # Create a backup before corrupting
    backup_file = json_file + ".original"
    shutil.copy(json_file, backup_file)
    
    ColorPrint.step(f"Corrupting JSON file with '{corruption_type}' corruption...")
    
    if corruption_type == "truncate":
        # Read the file content
        with open(json_file, 'r') as f:
            content = f.read()
        
        # Truncate the file to half its length
        with open(json_file, 'w') as f:
            f.write(content[:len(content)//2])
        
        ColorPrint.warning("File truncated to 50% of its original size")
    
    elif corruption_type == "invalid":
        # Add invalid JSON content
        with open(json_file, 'a') as f:
            f.write("\n\n THIS IS NOT VALID JSON CONTENT !!@@## \n\n")
        
        ColorPrint.warning("Added invalid content to the end of the file")
    
    elif corruption_type == "delete":
        # Delete the file
        os.remove(json_file)
        ColorPrint.warning("File deleted")
    
    return True

def restore_backup(run_id):
    """Restore the original backup file."""
    json_file = os.path.join(RUN_DETAILS_PATH, f"run_{run_id}.json")
    backup_file = json_file + ".original"
    
    if os.path.exists(backup_file):
        if os.path.exists(json_file):
            os.remove(json_file)
        shutil.copy(backup_file, json_file)
        ColorPrint.success("Restored original file from backup")
        return True
    
    ColorPrint.error("Backup file not found")
    return False

def delete_backup_files(run_id):
    """Delete the backup files."""
    json_file = os.path.join(RUN_DETAILS_PATH, f"run_{run_id}.json")
    backup_file = json_file + ".original"
    
    if os.path.exists(backup_file):
        os.remove(backup_file)
        ColorPrint.success("Deleted backup file")
    
    pkl_file = os.path.join(RUN_DETAILS_PATH, f"run_{run_id}.pkl")
    if os.path.exists(pkl_file):
        os.remove(pkl_file)
        ColorPrint.success("Deleted pickle file")

def delete_pickle_file(run_id):
    """Delete the pickle file to test recovery from JSON only."""
    pkl_file = os.path.join(RUN_DETAILS_PATH, f"run_{run_id}.pkl")
    
    if os.path.exists(pkl_file):
        os.rename(pkl_file, pkl_file + ".hidden")
        ColorPrint.warning("Pickle file temporarily renamed")
        return True
    
    return False

def restore_pickle_file(run_id):
    """Restore the pickle file."""
    pkl_file = os.path.join(RUN_DETAILS_PATH, f"run_{run_id}.pkl")
    
    if os.path.exists(pkl_file + ".hidden"):
        os.rename(pkl_file + ".hidden", pkl_file)
        ColorPrint.success("Pickle file restored")
        return True
    
    return False

def create_detailed_corruption(run_id):
    """Create a more detailed corruption that breaks at a specific key value."""
    json_file = os.path.join(RUN_DETAILS_PATH, f"run_{run_id}.json")
    
    # Create a backup before corrupting
    backup_file = json_file + ".original"
    if not os.path.exists(backup_file):
        shutil.copy(json_file, backup_file)
    
    # Read the file content
    with open(json_file, 'r') as f:
        content = f.read()
    
    # Find a specific key value pair and corrupt it
    segments = content.split('"accuracy"')
    if len(segments) > 1:
        # Corrupt the file at the accuracy value
        corrupted = segments[0] + '"accuracy"' + segments[1][:10] + "THIS_IS_NOT_VALID" + segments[1][20:]
        
        with open(json_file, 'w') as f:
            f.write(corrupted)
        
        ColorPrint.warning("Created detailed corruption around 'accuracy' field")
        return True
    
    ColorPrint.error("Could not find 'accuracy' field to corrupt")
    return False

def demonstrate_recovery():
    """Run a full demonstration of the error recovery capabilities."""
    ColorPrint.header("RUN HISTORY ERROR RECOVERY DEMONSTRATION")
    
    # Initialize directories
    init_history_dirs()
    
    # Create sample run data
    run_id = create_sample_run()
    
    # Verify the run data
    original_details = verify_run_data(run_id)
    
    # Wait a moment
    print("\nWaiting 2 seconds...\n")
    time.sleep(2)
    
    # Test Case 1: Truncated JSON file
    ColorPrint.header("TEST CASE 1: TRUNCATED JSON FILE")
    corrupt_json_file(run_id, "truncate")
    recovered_details = verify_run_data(run_id)
    restore_backup(run_id)
    
    # Wait a moment
    print("\nWaiting 2 seconds...\n")
    time.sleep(2)
    
    # Test Case 2: Test pickle fallback
    ColorPrint.header("TEST CASE 2: PICKLE FALLBACK WHEN JSON IS DELETED")
    has_pickle = delete_pickle_file(run_id)
    corrupt_json_file(run_id, "delete")
    recovered_details = verify_run_data(run_id)
    restore_backup(run_id)
    if has_pickle:
        restore_pickle_file(run_id)
    
    # Wait a moment
    print("\nWaiting 2 seconds...\n")
    time.sleep(2)
    
    # Test Case 3: Invalid JSON content
    ColorPrint.header("TEST CASE 3: INVALID JSON CONTENT")
    corrupt_json_file(run_id, "invalid")
    recovered_details = verify_run_data(run_id)
    restore_backup(run_id)
    
    # Wait a moment
    print("\nWaiting 2 seconds...\n")
    time.sleep(2)
    
    # Test Case 4: Detailed corruption
    ColorPrint.header("TEST CASE 4: DETAILED CORRUPTION")
    create_detailed_corruption(run_id)
    recovered_details = verify_run_data(run_id)
    restore_backup(run_id)
    
    # Clean up backup files
    delete_backup_files(run_id)
    
    ColorPrint.header("DEMONSTRATION COMPLETE")
    ColorPrint.success("The run history system successfully demonstrated its error recovery capabilities")
    print("For more information, see the documentation in docs/run_history.md")

if __name__ == "__main__":
    demonstrate_recovery() 