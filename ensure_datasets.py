#!/usr/bin/env python
"""
Ensure required datasets are available for the QNN Explorer application.

This script checks for the presence of required datasets and creates them
if they don't exist, including:
- Housing prices dataset
- MNIST digits
- Other demo datasets
"""

import os
import sys
import time

def ensure_housing_dataset():
    """
    Check if the housing dataset exists, and if not, create it.
    """
    housing_path = os.path.join("data", "demo_data", "housing", "processed")
    raw_path = os.path.join("data", "demo_data", "housing", "raw")
    
    # Create directories if they don't exist
    os.makedirs(housing_path, exist_ok=True)
    os.makedirs(raw_path, exist_ok=True)
    
    # Check if processed dataset exists
    train_file = os.path.join(housing_path, "train_housing.csv")
    test_file = os.path.join(housing_path, "test_housing.csv")
    scaler_file = os.path.join(housing_path, "house_price_scaler.joblib")
    encoders_file = os.path.join(housing_path, "house_price_encoders.joblib")
    
    if (not os.path.exists(train_file) or 
        not os.path.exists(test_file) or 
        not os.path.exists(scaler_file) or 
        not os.path.exists(encoders_file)):
        
        print("Housing dataset not found. Creating synthetic dataset...")
        
        try:
            from data.download_house_prices import create_synthetic_dataset, process_dataset
            
            # Create and process the dataset
            create_synthetic_dataset(samples=5000)
            process_dataset()
            
            if os.path.exists(train_file) and os.path.exists(test_file):
                print("✅ Housing dataset created successfully.")
            else:
                print("❌ Failed to create housing dataset.")
        except Exception as e:
            print(f"Error creating housing dataset: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("✅ Housing dataset already exists.")
    
    return os.path.exists(train_file) and os.path.exists(test_file)

def ensure_mnist_dataset():
    """
    Check if the MNIST dataset exists, and if not, create it.
    """
    mnist_path = os.path.join("data", "demo_data", "mnist")
    
    # Check if the dataset exists
    if not os.path.exists(os.path.join(mnist_path, "train")) or not os.path.exists(os.path.join(mnist_path, "test")):
        print("MNIST dataset not found. Downloading...")
        
        try:
            import subprocess
            result = subprocess.run(["python", "download_mnist.py"], 
                                   capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ MNIST dataset downloaded successfully.")
            else:
                print(f"❌ Failed to download MNIST dataset: {result.stderr}")
        except Exception as e:
            print(f"Error downloading MNIST dataset: {e}")
    else:
        print("✅ MNIST dataset already exists.")
    
    return os.path.exists(os.path.join(mnist_path, "train"))

def ensure_all_datasets():
    """
    Ensure all required datasets exist.
    """
    print("=" * 50)
    print("Checking for required datasets...")
    print("=" * 50)
    
    # Dictionary to track dataset status
    datasets = {
        "Housing": ensure_housing_dataset(),
        "MNIST": ensure_mnist_dataset(),
        # Add more datasets as needed
    }
    
    # Print summary
    print("\nDataset Summary:")
    for name, status in datasets.items():
        print(f"- {name}: {'✅ Available' if status else '❌ Not Available'}")
    
    # Return overall status
    return all(datasets.values())

if __name__ == "__main__":
    start_time = time.time()
    
    # Ensure all datasets exist
    if ensure_all_datasets():
        print("\n✅ All required datasets are available.")
    else:
        print("\n⚠️ Some datasets are missing. The application may not function correctly.")
    
    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time:.2f} seconds") 