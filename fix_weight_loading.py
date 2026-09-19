"""
Fix for the weight loading issue with weights_last_train

This script:
1. Examines the structure of weights_last_train
2. Creates a summary document of what was fixed
"""
import torch
import os
import sys

# Add the current directory to the path for imports
sys.path.append(os.path.abspath('.'))

# Define paths
WEIGHTS_DIR = os.path.join("saved_models", "weights")
WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, "weights_last_train")

def examine_weights_file(filepath):
    """Examine the structure of a weights file"""
    if not os.path.exists(filepath):
        print(f"Weights file not found at {filepath}")
        return None
        
    try:
        # Load the weights file
        weights = torch.load(filepath, map_location='cpu')
        
        # Get information about the weights
        info = {
            "num_parameters": len(weights),
            "keys": list(weights.keys()),
            "has_output_head": any('output_head' in k for k in weights.keys()),
        }
        
        # Get output head information if it exists
        if info["has_output_head"] and 'output_head.weight' in weights:
            output_shape = weights['output_head.weight'].shape
            info["num_classes"] = output_shape[0]
            info["input_features"] = output_shape[1]
            
        return info
    except Exception as e:
        print(f"Error examining weights file: {e}")
        return None

def create_fix_summary(weights_info):
    """Create a summary of the weight loading fix"""
    if not weights_info:
        return "Could not examine weights file."
        
    summary = """# Weight Loading Fix

## Issue
The application was encountering errors when loading model weights:
```
Error loading model weights from saved_models\\weights\\weights_last_train: Error(s) in loading state_dict for HybridModel: Unexpected key(s) in state_dict: "output_head.weight", "output_head.bias".
```

## Root Cause
The model being instantiated didn't have the same architecture as the model that was saved:
- The saved weights included an output head (classification layer)
- The model being loaded didn't have an output head

This is a common issue in machine learning when:
1. Model architectures evolve over time
2. Different configurations are used for training vs. inference
3. Model definition code changes but saved weights remain the same

## Analysis of Saved Weights
"""
    
    # Add information about the examined weights
    summary += f"- Number of parameters: {weights_info['num_parameters']}\n"
    summary += f"- Has output head: {weights_info['has_output_head']}\n"
    
    if weights_info["has_output_head"]:
        summary += f"- Output head parameters: {[k for k in weights_info['keys'] if 'output_head' in k]}\n"
        if "num_classes" in weights_info:
            summary += f"- Number of classes: {weights_info['num_classes']}\n"
        if "input_features" in weights_info:
            summary += f"- Input features: {weights_info['input_features']}\n"
    
    summary += """
## Solution Implemented
1. **Flexible Weight Loading**: Modified `load_model_weights()` function to use `strict=False` parameter, allowing partial loading of weights when architectures don't match exactly.

2. **Smart Model Instantiation**: Updated the model instantiation code to:
   - Examine the saved weights file structure
   - Detect whether it has an output head and how many classes
   - Configure the model to match the saved architecture
   - Provide clear feedback about architecture mismatches

3. **Better Error Handling**: Added detailed error messages that show:
   - Which parameters from the weights file were not used
   - Which parameters in the model were initialized randomly
   - Suggestions for resolving model architecture mismatches

## Benefits
- **Backward Compatibility**: Models can now load weights from different configurations
- **Graceful Degradation**: Even with mismatched architectures, usable parts of the model are loaded
- **Better Diagnostics**: Clear feedback about which parts were loaded vs. initialized
- **Future-Proofing**: Architecture changes won't break weight loading

## Testing
Created two test cases:
1. `test_weight_loading_fix.py`: Tests loading weights between models with and without output heads
2. `test_last_train_weights.py`: Specifically tests loading the actual `weights_last_train` file

## How to Use
The fix is transparent to users. The application now:
1. Detects the architecture in saved weights
2. Adjusts the model configuration accordingly
3. Loads the weights with appropriate handling of mismatches
4. Provides clear feedback about what was loaded
"""
    
    return summary

if __name__ == "__main__":
    print("Examining weights_last_train file...")
    weights_info = examine_weights_file(WEIGHTS_FILE)
    
    if weights_info:
        print(f"Weights file contains {weights_info['num_parameters']} parameters")
        print(f"Has output head: {weights_info['has_output_head']}")
        
        if weights_info["has_output_head"]:
            output_head_keys = [k for k in weights_info['keys'] if 'output_head' in k]
            print(f"Output head parameters: {output_head_keys}")
            
            if "num_classes" in weights_info:
                print(f"Number of classes: {weights_info['num_classes']}")
        
        # Create the fix summary
        summary = create_fix_summary(weights_info)
        
        # Save the summary
        summary_file = "weight_loading_fix_summary.md"
        with open(summary_file, 'w') as f:
            f.write(summary)
            
        print(f"\nSummary of fix saved to {summary_file}")
        print("\nFix is now implemented in app.py with the following changes:")
        print("1. Modified load_model_weights() to use strict=False")
        print("2. Updated model instantiation to match saved weights architecture")
        print("3. Added detailed feedback about parameter mismatches")
    else:
        print("Could not analyze weights file.") 