import streamlit as st
import torch
import pennylane as qml
import pandas as pd
import numpy as np
import os
import json
import time
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import joblib
from datetime import datetime
from collections import defaultdict
from torchvision import models, transforms
from utils.helpers import check_gpu
from torch.utils.data import DataLoader

# Local imports
from core.classical_models import get_cnn_model, get_transformer_model, get_gnn_model
from core.quantum_models import ansatz1 # Add more ansatz options later
from core.fusion import HybridModel, AVAILABLE_ANSATZES
from core.metrics import calculate_meyer_wallach
from data.loader import load_data, DEMO_DATASETS
from data.datasets import TensorDatasetWrapper, ImageDatasetWrapper, TextDatasetWrapper
from utils.visualization import plot_circuit, plot_state_vector, plot_training_history, plot_entanglement
from utils.helpers import check_gpu
from torch.utils.data import DataLoader
from utils.run_history_tab import render_run_history_tab

# Import preprocessing functions
from data.preprocessing import preprocess_image, preprocess_text, preprocess_csv, extract_text_from_csv

# Constants
SAVED_CONFIG_DIR = "saved_models/configs"
SAVED_WEIGHTS_DIR = "saved_models/weights"
DEMO_DATA_DIR = "data/demo_data"

# --- Constants & Configuration ---
# Ensure directories exist
os.makedirs(SAVED_CONFIG_DIR, exist_ok=True)
os.makedirs(SAVED_WEIGHTS_DIR, exist_ok=True)

# --- Available Model Options ---
AVAILABLE_ENCODINGS = ["Amplitude Encoding"] # Add Angle Encoding later
AVAILABLE_OBSERVATIONS = ["State Vector", "Expectation Value (PauliZ)"]
AVAILABLE_DATA_TYPES = ["Images", "Text", "CSV", "Video", "Graph"]
AVAILABLE_DATA_SOURCES = ["Upload", "Demo", "URL"]
AVAILABLE_CNN_MODELS = ["resnet18"] # Add more later (e.g., efficientnet_b0)
AVAILABLE_TRANSFORMER_MODELS = ["bert-base-uncased", "distilbert-base-uncased"] # Add more later
AVAILABLE_GNN_MODELS = ["gcn", "graphsage"] # Add more later
AVAILABLE_REGRESSION_MODELS = ["linear", "sklearn_linear"] # Models for housing price prediction
AVAILABLE_CLASSICAL_BACKBONES = ["None", "CNN", "Transformer", "GNN", "Regression"]

# --- Helper Functions for UI / State ---
def get_model_config_files():
    try:
        return [f for f in os.listdir(SAVED_CONFIG_DIR) if f.endswith('.json')]
    except FileNotFoundError:
        return []

def save_model_config(config, filename):
    filepath = os.path.join(SAVED_CONFIG_DIR, filename)
    try:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        st.success(f"Configuration saved to {filepath}")
    except Exception as e:
        st.error(f"Error saving configuration: {e}")

def load_model_config(filename):
    filepath = os.path.join(SAVED_CONFIG_DIR, filename)
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        st.error(f"Configuration file not found: {filepath}")
        return None
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return None

def save_model_weights(model, filename):
    # If filename is already a full path, use it directly
    if os.path.dirname(filename):
        filepath = filename
    else:
        # Otherwise, construct the path in the weights directory
        filepath = os.path.join(SAVED_WEIGHTS_DIR, filename)
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model weights
        torch.save(model.state_dict(), filepath)
        st.success(f"Model weights saved to {filepath}")
    except Exception as e:
        st.error(f"Error saving model weights: {e}")

def load_model_weights(model, filename):
    # Try multiple possible file paths with different naming patterns
    filepath_patterns = [
        # Standard pattern: config_name.pt
        os.path.normpath(os.path.join(SAVED_WEIGHTS_DIR, filename.replace('.json', '.pt'))),
        # Alternative pattern: weights_config_name.pt
        os.path.normpath(os.path.join(SAVED_WEIGHTS_DIR, f"weights_{filename.replace('.json', '.pt')}")),
        # Last training weights
        os.path.normpath(os.path.join(SAVED_WEIGHTS_DIR, "weights_last_train"))
    ]
    
    # Try each filepath pattern
    for filepath in filepath_patterns:
        try:
            if os.path.exists(filepath):
                # First try with strict=False to handle model architecture differences
                try:
                    model.load_state_dict(torch.load(filepath, map_location=st.session_state.device), strict=False)
                    model.eval()  # Set model to evaluation mode
                    st.success(f"Model weights loaded from {filepath} (with strict=False to handle architecture differences)")
                    return True
                except Exception as e:
                    st.error(f"Error loading model weights even with strict=False: {e}")
                    return False
            
        except Exception as e:
            st.error(f"Error loading model weights from {filepath}: {e}")
    
    # If we get here, none of the file patterns worked
    st.error("No matching weights file found.")
    return False

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="QNN Explorer")
st.title("QNN Explorer: Hybrid Quantum-Classical Models")

# --- Initialize Session State ---
if 'model_config' not in st.session_state:
    st.session_state.model_config = {}
if 'loaded_data' not in st.session_state:
    st.session_state.loaded_data = None
if 'preview_data' not in st.session_state:
    st.session_state.preview_data = None
if 'data_info' not in st.session_state:
    st.session_state.data_info = {}
if 'hybrid_model' not in st.session_state:
    st.session_state.hybrid_model = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = {'loss': [], 'entanglement': []}
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None # Will be loaded when Transformer is selected
if 'device' not in st.session_state:
    st.session_state.device = check_gpu() # Check GPU availability once
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False

# --- UI Tabs ---
tab_config, tab_data, tab_train, tab_predict, tab_visualize, tab_help, tab_run_history = st.tabs(
    ["Model Configuration", "Data", "Train", "Predict", "Visualize", "Help", "Run History"]
)

# Tab implementations omitted for brevity 
# These would include the full tab code from the original app.py

# Implement the run history tab
with tab_run_history:
    render_run_history_tab()

# --- Import housing utilities for prediction form ---
from housing_utils import render_housing_price_form, render_prediction_details 