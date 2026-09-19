"""
Download and prepare MNIST dataset for QNN Explorer.

This script downloads the MNIST handwritten digits dataset and prepares it 
for use with the QNN Explorer application.
"""

import os
import sys

def ensure_module_path():
    # Add the current directory to the path so imports work properly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

if __name__ == "__main__":
    ensure_module_path()
    
    print("=== MNIST Dataset Downloader for QNN Explorer ===")
    from data.mnist_loader import download_mnist_dataset
    
    print("Starting MNIST dataset download and preparation...")
    
    # Default path is data/demo_data/mnist
    data_dir = os.path.join('data', 'demo_data', 'mnist')
    
    # Allow custom path as first argument
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        print(f"Using custom data directory: {data_dir}")
    
    # Download and prepare the dataset
    path = download_mnist_dataset(data_dir)
    
    print("\nMNIST dataset has been downloaded and prepared!")
    print(f"Dataset location: {path}")
    print("\nYou can now use this dataset in QNN Explorer by selecting")
    print("'MNIST Digits (Image Classification)' from the demo datasets.") 