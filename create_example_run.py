import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import random
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import io
import base64

# Import utility functions
from utils.run_history import (
    add_run,
    save_run_details,
    plot_to_base64
)

def create_confusion_matrix_image():
    """Create a sample confusion matrix and convert to base64 image."""
    # Create a sample confusion matrix
    cm = np.array([
        [42, 8, 2],
        [7, 38, 5],
        [3, 6, 39]
    ])
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=["Class 0", "Class 1", "Class 2"],
        yticklabels=["Class 0", "Class 1", "Class 2"]
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    # Convert to base64
    img_str = plot_to_base64(plt.gcf())
    return cm, img_str

def create_example_classification_run():
    """Create an example classification run."""
    # Add the run to history
    run_id = add_run(
        config_name="Example Classification",
        duration=356.7,  # About 6 minutes
        status="Completed"
    )
    
    # Create details for the run
    # Create confusion matrix and image
    cm, cm_img = create_confusion_matrix_image()
    
    # Create metrics
    metrics = {
        "accuracy": 0.884,
        "precision": 0.892,
        "recall": 0.875,
        "f1": 0.883,
        "auc": 0.945,
        "log_loss": 0.352,
        "confusion_matrix": cm.tolist()
    }
    
    # Create run details
    details = {
        "task_type": "Classification",
        "metrics": metrics,
        "confusion_matrix_img": cm_img,
        "model_config": {
            "num_qubits": 4,
            "num_layers": 6,
            "encoding_method": "Amplitude Encoding",
            "ansatz_name": "Ansatz 1 (Rot+CNOT Chain)",
            "observation_type": "State Vector",
            "classical_backbone_type": "None",
            "use_gpu": True
        },
        "training_params": {
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 5
        },
        "data_info": {
            "data_type": "CSV",
            "samples": 1000,
            "features": 16,
            "classes": 3
        }
    }
    
    # Save run details
    save_run_details(run_id, details)
    print(f"Created example classification run with ID {run_id}")
    return run_id

def create_example_regression_run():
    """Create an example regression run."""
    # Add the run to history
    run_id = add_run(
        config_name="Example Regression",
        duration=238.2,  # About 4 minutes
        status="Completed"
    )
    
    # Create metrics
    metrics = {
        "mae": 0.324,
        "mse": 0.156,
        "rmse": 0.395,
        "rmsle": 0.187,
        "r2": 0.876
    }
    
    # Create run details
    details = {
        "task_type": "Regression",
        "metrics": metrics,
        "model_config": {
            "num_qubits": 6,
            "num_layers": 4,
            "encoding_method": "Amplitude Encoding",
            "ansatz_name": "Hardware Efficient (All-to-All)",
            "observation_type": "State Vector",
            "classical_backbone_type": "CNN",
            "classical_model_name": "resnet18",
            "use_gpu": True
        },
        "training_params": {
            "learning_rate": 0.0005,
            "batch_size": 8,
            "epochs": 10
        },
        "data_info": {
            "data_type": "Images",
            "samples": 500,
            "features": 64
        }
    }
    
    # Save run details
    save_run_details(run_id, details)
    print(f"Created example regression run with ID {run_id}")
    return run_id

def create_example_clustering_run():
    """Create an example clustering run."""
    # Add the run to history
    run_id = add_run(
        config_name="Example Clustering",
        duration=156.5,  # About 2.5 minutes
        status="Completed"
    )
    
    # Create metrics
    metrics = {
        "silhouette": 0.713,
        "adjusted_rand": 0.822,
        "adjusted_mutual_info": 0.785
    }
    
    # Create run details
    details = {
        "task_type": "Clustering",
        "metrics": metrics,
        "model_config": {
            "num_qubits": 5,
            "num_layers": 10,
            "encoding_method": "Amplitude Encoding",
            "ansatz_name": "GHZ-type (Star Topology)",
            "observation_type": "Expectation Value (PauliZ)",
            "classical_backbone_type": "None",
            "use_gpu": False
        },
        "training_params": {
            "learning_rate": 0.002,
            "batch_size": 32,
            "epochs": 15
        },
        "data_info": {
            "data_type": "CSV",
            "samples": 300,
            "features": 32
        }
    }
    
    # Save run details
    save_run_details(run_id, details)
    print(f"Created example clustering run with ID {run_id}")
    return run_id

def create_example_failed_run():
    """Create an example failed run."""
    # Add the run to history
    run_id = add_run(
        config_name="Failed Run Example",
        duration=42.3,  # Less than a minute
        status="Failed"
    )
    
    # Create run details
    details = {
        "task_type": "Classification",
        "error_message": "CUDA out of memory error during batch processing",
        "model_config": {
            "num_qubits": 8,
            "num_layers": 12,
            "encoding_method": "Amplitude Encoding",
            "ansatz_name": "Quantum Volume (Maximally Entangling)",
            "observation_type": "State Vector",
            "classical_backbone_type": "Transformer",
            "classical_model_name": "bert-base-uncased",
            "use_gpu": True
        },
        "training_params": {
            "learning_rate": 0.001,
            "batch_size": 4,
            "epochs": 5
        },
        "data_info": {
            "data_type": "Text",
            "samples": 200,
            "features": 768
        }
    }
    
    # Save run details
    save_run_details(run_id, details)
    print(f"Created example failed run with ID {run_id}")
    return run_id

def create_example_in_progress_run():
    """Create an example in-progress run."""
    # Add the run to history
    run_id = add_run(
        config_name="In Progress Example",
        duration=83.5,  # About 1.5 minutes so far
        status="In Progress"
    )
    
    # Create run details - more minimal since it's in progress
    details = {
        "task_type": "Classification",
        "current_epoch": 2,
        "total_epochs": 10,
        "current_loss": 0.563,
        "model_config": {
            "num_qubits": 6,
            "num_layers": 8,
            "encoding_method": "Amplitude Encoding",
            "ansatz_name": "Brickwall (Alternating Pairs)",
            "observation_type": "State Vector",
            "classical_backbone_type": "None",
            "use_gpu": True
        },
        "training_params": {
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 10
        },
        "data_info": {
            "data_type": "CSV",
            "samples": 800,
            "features": 64,
            "classes": 4
        }
    }
    
    # Save run details
    save_run_details(run_id, details)
    print(f"Created example in-progress run with ID {run_id}")
    return run_id

if __name__ == "__main__":
    print("Creating example runs for testing...")
    
    # Create all example runs
    create_example_classification_run()
    create_example_regression_run()
    create_example_clustering_run()
    create_example_failed_run()
    create_example_in_progress_run()
    
    print("Done creating example runs. Check the Run History tab in the app!") 