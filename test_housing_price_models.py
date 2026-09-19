#!/usr/bin/env python
"""
Comprehensive test comparing classical and quantum-enhanced housing price prediction models.

This script:
1. Tests data loading and preprocessing for the housing dataset
2. Implements and evaluates a classical neural network model
3. Implements and evaluates a hybrid quantum-classical model
4. Compares the performance of both approaches
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Add the parent directory to path
sys.path.append('.')

# Import necessary functions and modules
from data.download_house_prices import create_synthetic_dataset, process_dataset
from core.fusion import HybridModel

# Constants
DATA_DIR = os.path.join("data", "demo_data", "housing")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
INFO_FILE = os.path.join(DATA_DIR, "dataset_info.json")
RESULTS_DIR = "test_results"

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Class for classical housing price model
class ClassicalHousingModel(torch.nn.Module):
    """Classical neural network for housing price prediction."""
    def __init__(self, input_size, hidden_size=128):
        super(ClassicalHousingModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Class for hybrid quantum-classical model
class HybridHousingModel(torch.nn.Module):
    """Hybrid quantum-classical model for housing price prediction."""
    def __init__(self, input_size, qubits=4, layers=3):
        super(HybridHousingModel, self).__init__()
        
        # Feature reduction layers to reduce to quantum input size
        self.feature_reducer = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2**qubits),
            torch.nn.Tanh()  # Normalize to [-1,1] for better quantum input
        )
        
        # Create the quantum model configuration
        from core.quantum_models import ansatz1
        self.model_config = {
            'num_qubits': qubits,
            'num_layers': layers,
            'ansatz_name': 'Ansatz 1 (Rot+CNOT Chain)',
            'ansatz_func': ansatz1,
            'observation_type': 'State Vector',
            'classical_backbone_type': 'None',  # Direct quantum processing
            'use_gpu': False,
            'quantum_input_size': 2**qubits
        }
        
        # Create the quantum model component
        self.quantum_model = HybridModel(self.model_config)
        
        # Regression head to convert quantum output to price
        self.regressor = torch.nn.Linear(2**qubits, 1)
    
    def forward(self, x):
        # Reduce input features to quantum size
        reduced_features = self.feature_reducer(x)
        
        # Process with quantum circuit
        quantum_output = self.quantum_model(reduced_features)
        
        # Convert to price prediction
        return self.regressor(quantum_output)

def ensure_dataset_exists():
    """Ensure the housing dataset exists and is processed."""
    print("Checking for housing dataset...")
    
    # Create directories if they don't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Check if required files exist
    train_file = os.path.join(PROCESSED_DATA_DIR, "train_housing.csv")
    test_file = os.path.join(PROCESSED_DATA_DIR, "test_housing.csv")
    scaler_file = os.path.join(PROCESSED_DATA_DIR, "house_price_scaler.joblib")
    encoders_file = os.path.join(PROCESSED_DATA_DIR, "house_price_encoders.joblib")
    
    if (not os.path.exists(train_file) or not os.path.exists(test_file) or
        not os.path.exists(scaler_file) or not os.path.exists(encoders_file)):
        
        print("Housing dataset missing or incomplete. Creating now...")
        
        # Create synthetic dataset
        create_synthetic_dataset(samples=2000)
        
        # Process the dataset
        if not process_dataset():
            print("Error: Failed to process housing dataset.")
            return False
    
    print("✅ Housing dataset is available.")
    return True

def load_and_prepare_data():
    """Load and prepare the housing dataset for training and evaluation."""
    print("Loading housing dataset...")
    
    # Load the train and test datasets
    train_file = os.path.join(PROCESSED_DATA_DIR, "train_housing.csv")
    test_file = os.path.join(PROCESSED_DATA_DIR, "test_housing.csv")
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # Separate features and target
    X_train = train_df.drop('price', axis=1)
    y_train = train_df['price']
    X_test = test_df.drop('price', axis=1)
    y_test = test_df['price']
    
    # Scale the target for better training
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).reshape(-1, 1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).reshape(-1, 1)
    
    print(f"Data loaded: {len(X_train)} training samples, {len(X_test)} test samples")
    print(f"Input size: {X_train.shape[1]} features")
    
    # Return prepared data
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            y_scaler, X_train.shape[1])

def train_model(model, X_train, y_train, X_test, y_test, epochs=30, lr=0.001, name="Model"):
    """Train a PyTorch model and track metrics."""
    print(f"Training {name}...")
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Train mode
        model.train()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
        
        # Record history
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    return model, history

def evaluate_model(model, X_test, y_test, y_scaler, name="Model"):
    """Evaluate model performance and return metrics."""
    print(f"Evaluating {name}...")
    
    # Set to evaluation mode
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        y_pred_scaled = model(X_test)
        
        # Convert back to original scale
        y_pred = y_scaler.inverse_transform(y_pred_scaled.numpy()).flatten()
        y_true = y_scaler.inverse_transform(y_test.numpy()).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Print metrics
        print(f"MAE: ${mae:.2f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"R²: {r2:.4f}")
        
        # Return metrics and predictions
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred,
            'truth': y_true
        }
    
    return metrics

def plot_training_history(classical_history, quantum_history):
    """Plot training and validation loss for both models."""
    plt.figure(figsize=(12, 6))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(classical_history['train_loss'], label='Classical Train')
    plt.plot(quantum_history['train_loss'], label='Quantum Train')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation loss
    plt.subplot(1, 2, 2)
    plt.plot(classical_history['val_loss'], label='Classical Val')
    plt.plot(quantum_history['val_loss'], label='Quantum Val')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_history_comparison.png'))
    print(f"Training history plot saved to {os.path.join(RESULTS_DIR, 'training_history_comparison.png')}")

def plot_predictions(classical_metrics, quantum_metrics):
    """Plot prediction results comparison."""
    plt.figure(figsize=(15, 10))
    
    # Plot classical predictions vs actual
    plt.subplot(2, 2, 1)
    plt.scatter(classical_metrics['truth'], classical_metrics['predictions'], alpha=0.5)
    plt.plot([min(classical_metrics['truth']), max(classical_metrics['truth'])], 
             [min(classical_metrics['truth']), max(classical_metrics['truth'])], 'r--')
    plt.title('Classical Model: Predicted vs Actual')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    
    # Plot quantum predictions vs actual
    plt.subplot(2, 2, 2)
    plt.scatter(quantum_metrics['truth'], quantum_metrics['predictions'], alpha=0.5)
    plt.plot([min(quantum_metrics['truth']), max(quantum_metrics['truth'])], 
             [min(quantum_metrics['truth']), max(quantum_metrics['truth'])], 'r--')
    plt.title('Quantum Model: Predicted vs Actual')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    
    # Plot error distribution for classical model
    plt.subplot(2, 2, 3)
    classical_errors = classical_metrics['predictions'] - classical_metrics['truth']
    plt.hist(classical_errors, bins=50, alpha=0.7)
    plt.title('Classical Model: Error Distribution')
    plt.xlabel('Prediction Error ($)')
    plt.ylabel('Frequency')
    
    # Plot error distribution for quantum model
    plt.subplot(2, 2, 4)
    quantum_errors = quantum_metrics['predictions'] - quantum_metrics['truth']
    plt.hist(quantum_errors, bins=50, alpha=0.7)
    plt.title('Quantum Model: Error Distribution')
    plt.xlabel('Prediction Error ($)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'prediction_comparison.png'))
    print(f"Prediction comparison plot saved to {os.path.join(RESULTS_DIR, 'prediction_comparison.png')}")

def main():
    """Main function to run the housing price model comparison."""
    print("=" * 50)
    print("Housing Price Prediction Model Comparison")
    print("Classical vs Quantum-Enhanced Models")
    print("=" * 50)
    
    # Ensure dataset exists
    if not ensure_dataset_exists():
        print("Failed to ensure dataset exists. Exiting.")
        return False
    
    # Load and prepare data
    data = load_and_prepare_data()
    if data is None:
        print("Failed to load data. Exiting.")
        return False
    
    X_train, y_train, X_test, y_test, y_scaler, input_size = data
    
    # Create models
    classical_model = ClassicalHousingModel(input_size)
    quantum_model = HybridHousingModel(input_size, qubits=4, layers=3)
    
    # Train models
    classical_model, classical_history = train_model(
        classical_model, X_train, y_train, X_test, y_test, 
        epochs=30, lr=0.001, name="Classical Model"
    )
    
    quantum_model, quantum_history = train_model(
        quantum_model, X_train, y_train, X_test, y_test, 
        epochs=30, lr=0.001, name="Quantum-Enhanced Model"
    )
    
    # Evaluate models
    classical_metrics = evaluate_model(
        classical_model, X_test, y_test, y_scaler, name="Classical Model"
    )
    
    quantum_metrics = evaluate_model(
        quantum_model, X_test, y_test, y_scaler, name="Quantum-Enhanced Model"
    )
    
    # Plot results
    plot_training_history(classical_history, quantum_history)
    plot_predictions(classical_metrics, quantum_metrics)
    
    # Print comparison
    print("\nModel Comparison:")
    print("-" * 50)
    print(f"Classical MAE: ${classical_metrics['mae']:.2f}, RMSE: ${classical_metrics['rmse']:.2f}, R²: {classical_metrics['r2']:.4f}")
    print(f"Quantum MAE: ${quantum_metrics['mae']:.2f}, RMSE: ${quantum_metrics['rmse']:.2f}, R²: {quantum_metrics['r2']:.4f}")
    print("-" * 50)
    
    # Determine the winner
    if quantum_metrics['r2'] > classical_metrics['r2']:
        advantage = (quantum_metrics['r2'] - classical_metrics['r2']) / classical_metrics['r2'] * 100
        print(f"The quantum-enhanced model performed better by {advantage:.2f}% in R² score!")
    else:
        advantage = (classical_metrics['r2'] - quantum_metrics['r2']) / quantum_metrics['r2'] * 100
        print(f"The classical model performed better by {advantage:.2f}% in R² score.")
    
    # Save models
    torch.save(classical_model.state_dict(), os.path.join(RESULTS_DIR, 'classical_housing_model.pt'))
    torch.save(quantum_model.state_dict(), os.path.join(RESULTS_DIR, 'quantum_housing_model.pt'))
    print(f"Models saved to {RESULTS_DIR}")
    
    return True

if __name__ == "__main__":
    try:
        result = main()
        if result:
            print("\n🎉 Housing price model comparison completed successfully!")
        else:
            print("\n❌ Housing price model comparison failed.")
    except Exception as e:
        print(f"\n❌ Error during model comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 