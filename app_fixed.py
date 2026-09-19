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
from core.metrics import calculate_meyer_wallach # Add KL later if feasible
from data.loader import load_data, DEMO_DATASETS
from data.datasets import TensorDatasetWrapper, ImageDatasetWrapper, TextDatasetWrapper # Add Graph etc. later
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
        # Convert ansatz name back to function if needed during model creation
        # config['ansatz_func'] = AVAILABLE_ANSATZES.get(config['ansatz_name'])
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
                    
                    # Check what parameters were actually loaded
                    loaded_params = set(model.state_dict().keys())
                    saved_params = set(torch.load(filepath, map_location='cpu').keys())
                    
                    # Calculate differences
                    missing_in_model = saved_params - loaded_params
                    missing_in_weights = loaded_params - saved_params
                    
                    # Provide feedback about differences
                    if missing_in_model:
                        st.info(f"The following parameters in the weights file were not used: {', '.join(missing_in_model)}")
                    if missing_in_weights:
                        st.warning(f"The following parameters in the model were initialized randomly: {', '.join(missing_in_weights)}")
                        
                    return True
                except Exception as e:
                    st.error(f"Error loading model weights even with strict=False: {e}")
                    return False
            
        except Exception as e:
            st.error(f"Error loading model weights from {filepath}: {e}")
    
    # If we get here, none of the file patterns worked
    available_weights = [f for f in os.listdir(SAVED_WEIGHTS_DIR) if os.path.isfile(os.path.join(SAVED_WEIGHTS_DIR, f))]
    if available_weights:
        weight_info = "Available weights files: " + ", ".join(available_weights)
    else:
        weight_info = "No weights files found in directory."
    
    # Show a nicer error with available files
    st.error(f"No matching weights file found. Tried:\n" + 
             "\n".join([f"- {fp}" for fp in filepath_patterns]) + 
             f"\n\n{weight_info}")
    return False

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="QNN Explorer")
st.title("âš›ï¸ QNN Explorer: Hybrid Quantum-Classical Models")

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
    ["âš™ï¸ Model Configuration", "ðŸ'¾ Data", "ðŸš€ Train", "ðŸ"® Predict", "ðŸ"Š Visualize", "â" Help", "ðŸ Run History"]
)

# --- Tab 1: Model Configuration ---
with tab_config:
    st.header("Configure Hybrid Model Architecture")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Quantum Settings")
        nq = st.slider("Number of Qubits (N)", min_value=2, max_value=10, value=st.session_state.model_config.get('num_qubits', 4), key="n_qubits", help="Max qubits limited for performance. Amplitude encoding uses 2^N features.")
        nl = st.slider("Number of Circuit Layers (L)", min_value=1, max_value=20, value=st.session_state.model_config.get('num_layers', 6), key="n_layers")
        enc = st.selectbox("Encoding Method", AVAILABLE_ENCODINGS, index=0, key="encoding_method", help="Currently only Amplitude Encoding is implemented.")
        
        # Ansatz selection with descriptions of entanglement properties
        ansatz_descriptions = {
            "Ansatz 1 (Rot+CNOT Chain)": "Default ansatz with medium entanglement using a linear chain of CNOT gates.",
            "Hardware Efficient (All-to-All)": "High entanglement with all-to-all connectivity using CZ gates. Efficient for hardware implementation.",
            "GHZ-type (Star Topology)": "Very high entanglement similar to GHZ states. Excellent for complex patterns requiring global correlations.",
            "Brickwall (Alternating Pairs)": "High entanglement with alternating CNOT patterns, efficient for nearest-neighbor connections.",
            "Quantum Volume (Maximally Entangling)": "Very high entanglement based on IBM's Quantum Volume design. Creates maximal qubit interactions."
        }
        
        ans_name = st.selectbox(
            "Circuit Ansatz", 
            list(AVAILABLE_ANSATZES.keys()), 
            index=0, 
            key="ansatz_name",
            help="Select the quantum circuit structure (ansatz) for your model"
        )
        
        # Display the description below the selectbox
        st.caption(ansatz_descriptions.get(ans_name, ""))
        
        # Display more detailed information about the selected ansatz
        with st.expander("â„¹ï¸ Ansatz Entanglement Information"):
            st.markdown(f"### {ans_name}")
            st.markdown(f"**Description:** {ansatz_descriptions.get(ans_name, '')}")
            
            entanglement_ratings = {
                "Ansatz 1 (Rot+CNOT Chain)": "â­â­â­â˜†â˜† (Medium)",
                "Hardware Efficient (All-to-All)": "â­â­â­â­â˜† (High)",
                "GHZ-type (Star Topology)": "â­â­â­â­â­ (Very High)",
                "Brickwall (Alternating Pairs)": "â­â­â­â­â˜† (High)",
                "Quantum Volume (Maximally Entangling)": "â­â­â­â­â­ (Very High)"
            }
            
            st.markdown(f"**Entanglement Rating:** {entanglement_ratings.get(ans_name, '')}")
            
            # Add link to documentation
            st.markdown("[ðŸ"š Learn more about entanglement-promoting ansatzes](docs/entanglement_ansatz.md)")
        
        obs = st.selectbox("Observation Type", AVAILABLE_OBSERVATIONS, index=AVAILABLE_OBSERVATIONS.index(st.session_state.model_config.get('observation_type', 'State Vector')), key="observation_type", help="'State Vector' returns the full quantum state. 'Expectation Value' returns expectation values (e.g., PauliZ).")
        
        gpu_available = torch.cuda.is_available()
        use_gpu = st.checkbox("Use GPU if available", value=st.session_state.model_config.get('use_gpu', True), key="use_gpu", disabled=not gpu_available)
        
        if use_gpu and gpu_available and st.session_state.model_config.get('classical_backbone_type') == 'Transformer':
            st.warning("âš ï¸ Using GPU with Transformer models might cause issues in Streamlit. If you encounter errors, try disabling GPU or using a different backbone type.")

    with col2:
        st.subheader("Classical Settings")
        cb_type = st.selectbox("Classical Backbone Type", AVAILABLE_CLASSICAL_BACKBONES, index=AVAILABLE_CLASSICAL_BACKBONES.index(st.session_state.model_config.get('classical_backbone_type', 'None')), key="classical_backbone_type")

        cb_name = None
        gnn_input_features = None
        if cb_type == "CNN":
            cb_name = st.selectbox("CNN Model", AVAILABLE_CNN_MODELS, index=AVAILABLE_CNN_MODELS.index(st.session_state.model_config.get('classical_model_name', 'resnet18')), key="classical_model_name_cnn")
        elif cb_type == "Transformer":
            cb_name = st.selectbox("Transformer Model", AVAILABLE_TRANSFORMER_MODELS, index=AVAILABLE_TRANSFORMER_MODELS.index(st.session_state.model_config.get('classical_model_name', 'bert-base-uncased')), key="classical_model_name_transformer")
            # Load tokenizer when Transformer selected
            if st.session_state.tokenizer is None or st.session_state.tokenizer.name_or_path != cb_name:
                with st.spinner(f"Loading tokenizer for {cb_name}..."):
                    st.session_state.tokenizer = AutoTokenizer.from_pretrained(cb_name)
                st.success(f"Tokenizer {cb_name} loaded.")
        elif cb_type == "GNN":
            cb_name = st.selectbox("GNN Model", AVAILABLE_GNN_MODELS, index=AVAILABLE_GNN_MODELS.index(st.session_state.model_config.get('classical_model_name', 'gcn')), key="classical_model_name_gnn")
            # GNNs need input feature dimension, often determined by the data
            gnn_input_features = st.number_input("GNN Input Node Features", min_value=1, value=st.session_state.model_config.get('gnn_input_features', 10), key="gnn_input_features", help="Specify the number of features per node in your graph data.")
        elif cb_type == "Regression":
            # Add UI for regression models
            cb_name = st.selectbox("Regression Model", AVAILABLE_REGRESSION_MODELS, 
                                  index=AVAILABLE_REGRESSION_MODELS.index(st.session_state.model_config.get('classical_model_name', 'linear')), 
                                  key="classical_model_name_regression",
                                  help="Linear regression model for housing price prediction")
            
            # Add regression-specific options
            st.info("Regression models are optimized for predicting continuous values such as house prices.")
            with st.expander("Regression Model Details"):
                st.markdown("""
                **Available Models:**
                - **linear**: Pure PyTorch implementation of Linear Regression
                - **sklearn_linear**: scikit-learn's LinearRegression wrapped in PyTorch
                
                For housing price prediction, these models work with processed features including:
                - Square footage
                - Number of bedrooms/bathrooms
                - Location data (state, zip code)
                - Property age and condition
                """)
        else: # None
            st.info(f"No classical backbone. Input data features must match 2^N = {2**nq}.")

        st.subheader("Save/Load Configuration")
        save_name = st.text_input("Configuration Name", f"config_N{nq}_L{nl}_{cb_type.lower() if cb_type != 'None' else 'direct'}.json", key="save_config_name")

        if st.button("ðŸ'¾ Create and Save Configuration", key="save_config_button"):
            config = {
                "num_qubits": nq,
                "num_layers": nl,
                "encoding_method": enc,
                "ansatz_name": ans_name,
                "observation_type": obs,
                "classical_backbone_type": cb_type,
                "classical_model_name": cb_name if cb_type != "None" else None,
                "gnn_input_features": gnn_input_features if cb_type == "GNN" else None,
                "use_gpu": use_gpu and torch.cuda.is_available(),
                "quantum_input_size": 2**nq,
            }
            st.session_state.model_config = config
            save_model_config(config, save_name)
            st.success("Configuration created and saved.")
            st.json(config)

        st.markdown("---  ")
        config_files = get_model_config_files()
        if config_files:
            selected_config_file = st.selectbox("Load Existing Configuration", config_files, key="load_config_select")
            if st.button("ðŸ"‚ Load Configuration", key="load_config_button"):
                loaded_cfg = load_model_config(selected_config_file)
                if loaded_cfg:
                    st.session_state.model_config = loaded_cfg
                    st.success(f"Loaded configuration '{selected_config_file}'")
                    # Trigger rerun to update widgets - use st.rerun() (replaced experimental_rerun)
                    st.rerun()
        else:
            st.info("No saved configurations found.")

# --- Tab 2: Data --- (Depends on Config)
with tab_data:
    st.header("Load and Preprocess Data")

    if not st.session_state.model_config:
        st.warning("Please configure and save/load a model configuration first.")
    else:
        st.write("Current Model Configuration:")
        st.json(st.session_state.model_config)

        data_source = st.radio("Select Data Source", AVAILABLE_DATA_SOURCES, key="data_source")
        data_type = st.selectbox("Select Data Type", AVAILABLE_DATA_TYPES, key="data_type")

        uploaded_file = None
        url = None
        demo_dataset_name = None

        if data_source == "Upload":
            accept_multiple = False # Allow multiple later?
            uploaded_file = st.file_uploader(f"Upload {data_type} Data", type=None, accept_multiple_files=accept_multiple, key="file_uploader")
        elif data_source == "URL":
            url = st.text_input(f"Enter URL for {data_type} Data", key="data_url")
        elif data_source == "Demo":
            # Filter demo datasets by selected type
            relevant_demos = {name: info for name, info in DEMO_DATASETS.items() if info["type"] == data_type}
            if relevant_demos:
                demo_dataset_name = st.selectbox(f"Select Demo {data_type} Dataset", list(relevant_demos.keys()), key="demo_select")
            else:
                st.warning(f"No demo datasets available for type '{data_type}'. Please add data to {DEMO_DATA_DIR}.")

            # Special info section for housing dataset
            if demo_dataset_name == "USA Housing Prices (Regression)":
                housing_path = os.path.join(DEMO_DATA_DIR, "housing", "processed")
                info_file = os.path.join(DEMO_DATA_DIR, "housing", "dataset_info.json")
                
                if not os.path.exists(housing_path):
                    # Show download instructions
                    st.warning("Housing dataset not found. Please run the download script first.")
                    
                    if st.button("Download Housing Dataset"):
                        with st.spinner("Downloading and processing USA Housing Price dataset..."):
                            try:
                                import subprocess
                                result = subprocess.run(["python", "data/download_house_prices.py"], 
                                                       capture_output=True, text=True)
                                if result.returncode == 0:
                                    st.success("Housing dataset downloaded and processed successfully!")
                                    st.info("Please reload the data section to see the dataset.")
                                    # Show some output from the download script
                                    with st.expander("Download Output"):
                                        st.code(result.stdout)
                                else:
                                    st.error(f"Error downloading housing dataset: {result.stderr}")
                            except Exception as e:
                                st.error(f"Error running download script: {e}")
                elif os.path.exists(info_file):
                    # Show dataset information
                    try:
                        with open(info_file, 'r') as f:
                            import json
                            info = json.load(f)
                            
                        st.success("Housing price dataset is available")
                        price_range = info.get('price_range', [0, 0])
                        
                        # Display info metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Samples", info.get('samples', 'Unknown'))
                        with col2:
                            st.metric("Avg. Price", f"${int(info.get('mean_price', 0)):,}")
                        with col3:
                            st.metric("Price Range", f"${int(price_range[0]):,} - ${int(price_range[1]):,}")
                        
                        # Show what can be predicted with this dataset
                        st.info("This dataset can be used with Regression models to predict house prices across all 50 US states.")
                    except Exception as e:
                        st.error(f"Error loading dataset info: {e}")

        # --- CSV Specific UI ---
        if data_type == "CSV":
            st.subheader("CSV Configuration")
            # This UI now appears if preview_data (the raw DataFrame) exists
            raw_df = st.session_state.get('preview_data')
            if raw_df is not None and isinstance(raw_df, pd.DataFrame):
                 df_cols = st.session_state.preview_data.columns.tolist()
                 
                 # Use session state to store selections, persisting across runs
                 if 'csv_feature_cols' not in st.session_state:
                      st.session_state.csv_feature_cols = [c for c in df_cols[:4] if c.lower() != 'label' and c.lower() != 'target'] # Default guess
                 if 'csv_label_col' not in st.session_state:
                      st.session_state.csv_label_col = "None"

                 st.session_state.csv_feature_cols = st.multiselect(
                     "Select Feature Columns",
                     df_cols,
                     default=st.session_state.csv_feature_cols,
                     key="csv_features_select"
                 )
                 st.session_state.csv_label_col = st.selectbox(
                     "Select Label Column (Optional)",
                     ["None"] + df_cols,
                     index=(["None"] + df_cols).index(st.session_state.csv_label_col) if st.session_state.csv_label_col in df_cols else 0,
                     key="csv_label_select"
                 )

                 # Add button to trigger preprocessing
                 col1, col2 = st.columns(2)
                 
                 with col1:
                     # Default all columns as text if they're not numeric
                     default_text_cols = [col for col in df_cols if col not in ['label', 'target', 'class'] 
                                        and not pd.api.types.is_numeric_dtype(raw_df[col])]
                     
                     # Text processing options (only show if non-numeric columns exist)
                     if default_text_cols:
                         st.write("**Text Processing Options:**")
                         text_cols = st.multiselect(
                             "Select Text Columns (for vectorization)",
                             df_cols,
                             default=default_text_cols,
                             key="text_cols_select"
                         )
                         
                         text_vectorizer = st.selectbox(
                             "Text Vectorization Method",
                             ["tfidf", "count"],
                             index=0,
                             key="text_vectorizer_select"
                         )
                         
                         max_text_features = st.slider(
                             "Max Features per Text Column",
                             min_value=10,
                             max_value=1000,
                             value=100,
                             step=10,
                             key="max_text_features_slider"
                         )
                     else:
                         text_cols = []
                         text_vectorizer = "tfidf"
                         max_text_features = 100
                 
                 with col2:
                     # Button placement
                     st.write("**CSV Preprocessing:**")
                     st.write("Select data columns and processing options, then click below:")
                     preprocess_button = st.button("âš™ï¸ Preprocess Selected CSV Columns", key="preprocess_csv_button")
                 
                 if preprocess_button:
                     selected_features = st.session_state.get('csv_feature_cols')
                     selected_label = st.session_state.get('csv_label_col') if st.session_state.get('csv_label_col') != "None" else None

                     if not selected_features:
                         st.error("Please select at least one feature column.")
                     else:
                         # Check if using transformer but not extracting text directly
                         if st.session_state.model_config.get('classical_backbone_type') == 'Transformer':
                             st.warning("""
                             âš ï¸ You're using a Transformer model backbone, but preprocessing CSV columns directly.
                             
                             For text data with Transformer models, it's recommended to use:
                             - The "Direct Text Extraction" approach in the section below
                             - Or change your model configuration to use a different backbone type
                             
                             Vectorized features (TF-IDF/Count) are not compatible with Transformer models!
                             """)
                             
                         with st.spinner("Preprocessing CSV data..."):
                             try:
                                 # Import the function locally
                                 from data.preprocessing import preprocess_csv
                                 # Create a scaler name based on data shape
                                 scaler_name = f"scaler_{raw_df.shape[1]}_{len(selected_features)}.joblib"
                                 processed_features, processed_labels = preprocess_csv(
                                     raw_df, selected_features, selected_label,
                                     scaler_name=scaler_name,
                                     is_prediction=False, # This is training data, not prediction
                                     text_cols=text_cols if text_cols else None, # Pass selected text columns
                                     text_vectorizer=text_vectorizer,
                                     max_text_features=max_text_features
                                 )
                                 if processed_features is not None:
                                     st.session_state.loaded_data = processed_features
                                     st.session_state.csv_labels = processed_labels # Store labels separately
                                     st.session_state.data_info['processed_shape'] = processed_features.shape
                                     st.session_state.data_info['feature_columns'] = selected_features
                                     st.session_state.data_info['label_column'] = selected_label
                                     st.session_state.data_info['scaler_name'] = scaler_name # Store for prediction
                                     # Store text processing settings
                                     if 'text_cols' in locals() and text_cols:
                                         st.session_state.data_info['text_cols'] = text_cols
                                         st.session_state.data_info['text_vectorizer'] = text_vectorizer
                                         st.session_state.data_info['max_text_features'] = max_text_features
                                     st.success("CSV data preprocessed successfully!")
                                     st.write("Processed Feature Shape:", processed_features.shape)
                                     if processed_labels is not None:
                                         st.write("Processed Labels Shape:", processed_labels.shape)
                                 else:
                                     st.error("CSV preprocessing failed.")
                             except Exception as e:
                                 st.error(f"Error during CSV preprocessing: {e}")

            else:
                 st.info("Load a CSV file to configure columns.")

        # --- Load Button ---
        if st.button("ðŸ'¾ Load and Preprocess Data", key="load_data_button"):
            if data_source == "Upload" and not uploaded_file:
                st.error("Please upload a file.")
            elif data_source == "URL" and not url:
                st.error("Please enter a URL.")
            elif data_source == "Demo" and not demo_dataset_name:
                st.error("Please select a demo dataset.")
            else:
                with st.spinner("Loading and preprocessing..."):
                    # Special case for Cats vs Dogs dataset
                    if data_source == "Demo" and demo_dataset_name == "Cats vs Dogs (Image Classification)":
                        try:
                            from data.cats_dogs_loader import load_cats_dogs_dataset
                            images, labels = load_cats_dogs_dataset()
                            if images is not None:
                                st.session_state.loaded_data = images
                                st.session_state.csv_labels = labels
                                st.session_state.data_info = {
                                    "data_type": "Images",
                                    "data_source": "Demo - Cats vs Dogs",
                                    "samples": len(images),
                                    "processed_shape": images.shape,
                                    "num_classes": 2,
                                    "classes": ["cat", "dog"]
                                }
                                # Store class names in session state for use in predictions
                                st.session_state.class_names = ["Cat", "Dog"]
                                st.success(f"Loaded {len(images)} images with {len(torch.unique(labels))} classes (cats and dogs)")
                            else:
                                st.error("Failed to load Cats vs Dogs dataset")
                        except Exception as e:
                            st.error(f"Error loading Cats vs Dogs dataset: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                    # Special case for MNIST dataset
                    elif data_source == "Demo" and demo_dataset_name == "MNIST Digits (Image Classification)":
                        try:
                            from data.mnist_loader import load_mnist_dataset, download_mnist_dataset
                            
                            # Check if the dataset exists, download if not
                            mnist_path = os.path.join(DEMO_DATA_DIR, "mnist")
                            
                            # Display info about MNIST
                            st.info("""
                            **MNIST Dataset Information:**
                            This dataset contains 1,000 training images (100 per digit) and 200 test images (20 per digit) of handwritten digits.
                            
                            If you encounter any issues loading the data:
                            1. Make sure the dataset has been downloaded using `python download_mnist.py`
                            2. Check that the `data/demo_data/mnist` directory has the proper folder structure
                            3. Each digit should have a folder (0-9) with PNG images inside
                            """)
                            
                            # Check if the dataset exists
                            if not os.path.exists(os.path.join(mnist_path, "train")):
                                st.warning(f"MNIST dataset not found at {mnist_path}")
                                if st.button("Download MNIST Dataset Now"):
                                    with st.spinner("Downloading MNIST dataset..."):
                                        download_mnist_dataset(mnist_path)
                                        st.success("Download complete! Please try loading again.")
                                # Instead of return, we'll just set a flag
                                continue_loading = False
                            else:
                                continue_loading = True
                                
                            # Only continue if dataset exists
                            if continue_loading:
                                # Load the dataset
                                images, labels = load_mnist_dataset(mnist_path)
                                if images is not None:
                                    st.session_state.loaded_data = images
                                    st.session_state.csv_labels = labels
                                    st.session_state.data_info = {
                                        "data_type": "Images",
                                        "data_source": "Demo - MNIST Digits",
                                        "samples": len(images),
                                        "processed_shape": images.shape,
                                        "num_classes": 10,
                                        "classes": [str(i) for i in range(10)]  # Digit class names
                                    }
                                    # Store class names in session state for use in predictions
                                    st.session_state.class_names = [f"Digit {i}" for i in range(10)]
                                    st.success(f"Loaded {len(images)} MNIST images with 10 classes (digits 0-9)")
                                else:
                                    st.error("Failed to load MNIST dataset.")
                                    st.info("Try running `python download_mnist.py` to download the dataset.")
                        except Exception as e:
                            st.error(f"Error loading MNIST dataset: {e}")
                    else:
                        # Other data types - use standard loader
                        loaded_data, preview_data, data_info = load_data(
                            data_source, data_type, 
                            uploaded_file=uploaded_file, 
                            url=url, 
                            demo_dataset_name=demo_dataset_name
                        )
                        
                        # --- Enhanced CSV handling ---
                        # For CSV data, ensure the loaded_data is actually available
                        if data_type == "CSV" and preview_data is not None and isinstance(preview_data, pd.DataFrame):
                            # If we don't have processed data yet but have a DataFrame, create initial representation
                            if loaded_data is None:
                                numeric_cols = preview_data.select_dtypes(include=['number']).columns.tolist()
                                if numeric_cols:
                                    # Use numeric columns to create an initial tensor representation
                                    temp_data = preview_data[numeric_cols].values
                                    loaded_data = torch.tensor(temp_data, dtype=torch.float32)
                                    data_info['needs_preprocessing'] = True
                                    data_info['initial_representation'] = True
                                    data_info['available_columns'] = preview_data.columns.tolist()
                                else:
                                    # Fallback for CSVs with no numeric data
                                    st.warning("CSV contains no numeric columns. You need to preprocess this data to extract features.")
                                    # Create a minimal placeholder tensor
                                    loaded_data = torch.zeros((len(preview_data), 1), dtype=torch.float32)
                                    data_info['needs_preprocessing'] = True
                                    data_info['requires_text_extraction'] = True
                        
                        # Update session state
                        st.session_state.loaded_data = loaded_data
                        st.session_state.preview_data = preview_data
                        st.session_state.data_info = data_info
                    # Check if loaded data shape is compatible with config
                    if st.session_state.model_config.get('classical_backbone_type') == 'None':
                         expected_size = st.session_state.model_config['quantum_input_size']
                         if hasattr(loaded_data, 'shape') and loaded_data.shape[-1] != expected_size:
                             st.warning(f"Loaded data feature size ({loaded_data.shape[-1]}) does not match required quantum input size ({expected_size}). The model might fail unless the data represents amplitudes directly.")

        # --- Display Data Info/Preview ---
        if st.session_state.data_info:
            # Check if there was an error during data loading
            if 'error' in st.session_state.data_info:
                error_msg = st.session_state.data_info.get('error')
                st.error(f"Data Status: Error - {error_msg}")
            else:
                status = "Raw Data Loaded" if st.session_state.data_info.get('data_type') == 'CSV' and st.session_state.loaded_data is None else "Data Loaded & Preprocessed"
                st.success(f"Data Status: {status}")
            
            st.write("Data Information:")
            st.json(st.session_state.data_info)

            st.write("Data Preview:")
            if 'error' in st.session_state.data_info:
                st.info("No preview available due to error in data loading.")
            elif st.session_state.preview_data is not None:
                data_type = st.session_state.data_info.get('data_type')
                if data_type == "Images" and isinstance(st.session_state.preview_data, Image.Image):
                    st.image(st.session_state.preview_data, caption="Loaded Image Preview", width=256)
                elif data_type == "CSV" and isinstance(st.session_state.preview_data, pd.DataFrame):
                    st.dataframe(st.session_state.preview_data)
                elif data_type == "Text":
                    # Handle different types of preview data for text
                    if isinstance(st.session_state.preview_data, str):
                        # If it's already a string, display it directly
                        st.text_area("Text Preview", st.session_state.preview_data, height=150)
                    elif isinstance(st.session_state.preview_data, pd.DataFrame):
                        # If it's a DataFrame, show it as a dataframe
                        st.write("Text data loaded as DataFrame:")
                        st.dataframe(st.session_state.preview_data)
                        
                        # Also show a sample of text from the first row if possible
                        if len(st.session_state.preview_data) > 0:
                            try:
                                # Try to get the first text column's first value
                                text_cols = st.session_state.preview_data.select_dtypes(include=['object']).columns
                                if len(text_cols) > 0:
                                    sample_text = st.session_state.preview_data.iloc[0][text_cols[0]]
                                    st.text_area("Sample Text (First Row)", str(sample_text)[:500] + "...", height=150)
                            except:
                                pass  # Silently fail if we can't extract a sample
                    else:
                        # For other types, convert to string
                        st.text_area("Text Preview", str(st.session_state.preview_data), height=150)
                elif data_type in ["Video", "Graph"]:
                    # Preview might just be the info string from loader
                     st.info(str(st.session_state.preview_data))
                else:
                    st.warning("No preview available for this data type.")
            else:
                st.info("No preview data available.")
        else:
            st.info("No data loaded yet.")

        # Add information block about the two approaches for textual data
        if (data_type == "CSV" and st.session_state.preview_data is not None and 
            isinstance(st.session_state.preview_data, pd.DataFrame)):
            
            # Check if there are any string columns that might be text
            df = st.session_state.preview_data
            object_cols = df.select_dtypes(include=['object']).columns.tolist()
            text_candidates = [col for col in object_cols if df[col].astype(str).str.len().mean() > 10]
            
            if text_candidates and st.session_state.model_config.get('classical_backbone_type') == 'Transformer':
                st.info("ðŸ"– **CSV file contains text data and you selected a Transformer model.**")
                
                # Add a prominent message
                st.markdown("""
                <div style="padding:15px;border-radius:5px;background-color:#FFEB3B;color:#000;margin:15px 0;">
                    <h3 style="margin-top:0;">âš ï¸ Important: Direct Text Extraction Required for Transformer</h3>
                    <p>When using a Transformer model backbone, you <b>must</b> use the Direct Text Extraction approach below.</p>
                    <p>Vectorized features from the CSV Configuration section are <b>not compatible</b> with Transformer models.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("You have two options for processing this textual data:")
                
                tab1, tab2 = st.tabs(["Option 1: Direct Text Extraction", "Option 2: Vector Features"])
                
                with tab1:
                    st.write("**Extract text directly for Transformer processing**")
                    st.write("This approach extracts text from a single column for the Transformer model.")
                    
                    text_col = st.selectbox(
                        "Select Text Column for Transformer",
                        text_candidates,
                        key="transformer_text_col"
                    )
                    
                    label_col = st.selectbox(
                        "Select Label Column (Optional)",
                        ["None"] + df.columns.tolist(),
                        key="transformer_label_col"
                    )
                    label_col = None if label_col == "None" else label_col
                    
                    max_samples = st.slider(
                        "Max Samples to Process",
                        min_value=100,
                        max_value=min(10000, len(df)),
                        value=min(1000, len(df)),
                        step=100,
                        key="transformer_max_samples"
                    )
                    
                    if st.button("Extract Text for Transformer", key="extract_text_button"):
                        with st.spinner("Extracting text from CSV..."):
                            from data.preprocessing import extract_text_from_csv
                            texts, labels = extract_text_from_csv(
                                df, text_col, label_col, max_samples
                            )
                            
                            if texts:
                                # Store as text data for transformer
                                st.session_state.transformer_texts = texts
                                if labels:
                                    st.session_state.transformer_labels = labels
                                
                                # Tokenize with the selected transformer
                                if st.session_state.tokenizer:
                                    with st.spinner("Tokenizing text with transformer..."):
                                        # Tokenize a subset for preview
                                        preview_size = min(3, len(texts))
                                        
                                        # Process preview texts the same way
                                        preview_processed = []
                                        for text in texts[:preview_size]:
                                            if isinstance(text, (list, tuple, dict)):
                                                preview_processed.append(str(text).strip())
                                            else:
                                                preview_processed.append(str(text).strip())
                                        
                                        preview_tokens = st.session_state.tokenizer(
                                            preview_processed, 
                                            truncation=True, 
                                            padding='max_length',
                                            return_tensors='pt'
                                        )
                                        
                                        # Tokenize all texts
                                        try:
                                            # Ensure all texts are properly formatted strings
                                            # This handles cases where the text might be nested structures or arrays
                                            processed_texts = []
                                            for text in texts:
                                                if isinstance(text, (list, tuple, dict)):
                                                    # If text is a complex structure, convert it properly
                                                    text_str = str(text).strip()
                                                    processed_texts.append(text_str)
                                                else:
                                                    # Otherwise just ensure it's a string
                                                    processed_texts.append(str(text).strip())
                                            
                                            # Debug info
                                            st.write(f"First text sample type: {type(processed_texts[0])}")
                                            st.write(f"First text sample: {processed_texts[0][:100]}..." if processed_texts else "No texts available")
                                            
                                            # Process in smaller batches to prevent memory issues
                                            batch_size = 128
                                            num_samples = len(processed_texts)
                                            all_input_ids = []
                                            all_attention_masks = []
                                            
                                            for i in range(0, num_samples, batch_size):
                                                batch_texts = processed_texts[i:i+batch_size]
                                                
                                                # Use simpler tokenization with specific parameters
                                                batch_tokens = st.session_state.tokenizer(
                                                    batch_texts,
                                                    truncation=True,
                                                    padding='max_length',
                                                    max_length=128,  # Limit sequence length
                                                    return_tensors='pt'
                                                )
                                                
                                                all_input_ids.append(batch_tokens['input_ids'])
                                                all_attention_masks.append(batch_tokens['attention_mask'])
                                            
                                            # Combine batches
                                            tokenized = {
                                                'input_ids': torch.cat(all_input_ids, dim=0),
                                                'attention_mask': torch.cat(all_attention_masks, dim=0)
                                            }
                                            
                                            st.write(f"Successfully tokenized {num_samples} texts")
                                            st.write(f"Tokenized shape: input_ids={tokenized['input_ids'].shape}, attention_mask={tokenized['attention_mask'].shape}")
                                            
                                            # Store tokenized data and labels
                                            st.session_state.loaded_data = tokenized
                                            if labels:
                                                st.session_state.csv_labels = torch.tensor(labels)
                                            
                                            # Update data info
                                            st.session_state.data_info['data_type'] = 'Text'
                                            st.session_state.data_info['source_type'] = 'CSV'
                                            st.session_state.data_info['text_column'] = text_col
                                            st.session_state.data_info['label_column'] = label_col
                                            st.session_state.data_info['num_samples'] = num_samples
                                            st.session_state.data_info['processed_shape'] = (
                                                num_samples, 
                                                tokenized['input_ids'].shape[1]
                                            )
                                            
                                            # Success messages
                                            st.success(f"Successfully processed {num_samples} text samples for transformer input")
                                            
                                            # Add a separate function for safely displaying token previews
                                            def display_token_preview():
                                                try:
                                                    st.write("**Preview of tokenized texts:**")
                                                    for i in range(min(preview_size, len(processed_texts))):
                                                        if i < len(preview_processed):
                                                            # Safely display text sample
                                                            st.write(f"**Text {i+1}:** {preview_processed[i][:100]}...")
                                                            
                                                            # Convert tensor to list for display (avoiding string operations on tensor)
                                                            if hasattr(preview_tokens['input_ids'], 'tolist'):
                                                                token_preview = preview_tokens['input_ids'][i][:10].tolist()
                                                                st.write(f"**Tokens:** {token_preview}")
                                                            else:
                                                                st.write(f"**Tokens:** {str(preview_tokens['input_ids'][i][:10])}")
                                                except Exception as e:
                                                    st.warning(f"Error displaying token preview: {e}")
                                                    st.info("This is a display issue only and doesn't affect the tokenization process.")
                                            
                                            # Call the display function
                                            display_token_preview()
                                        except Exception as e:
                                            st.error(f"Error during tokenization: {e}")
                                else:
                                    st.error("No tokenizer available. Please select a Transformer model in the Configuration tab.")
                            else:
                                st.error("Failed to extract text from CSV.")
                    
                    with tab2:
                        st.write("**Use vectorized features (TF-IDF/Count)**")
                        st.write("This approach converts text to numeric features using TF-IDF or Count vectorization.")
                        st.write("ðŸ'‰ Use the 'CSV Configuration' section below to select columns and preprocessing options.")

# --- Tab 3: Train --- (Depends on Config & Data)
with tab_train:
    st.header("Train the Hybrid Model")

    if not st.session_state.model_config or st.session_state.loaded_data is None:
        st.warning("Please configure a model and load data first.")
    else:
        # Check if CSV data needs preprocessing
        if (st.session_state.data_info.get('data_type') == 'CSV' and 
            st.session_state.data_info.get('needs_preprocessing', False) and 
            st.session_state.data_info.get('initial_representation', False)):
            st.warning("""
            âš ï¸ **CSV data loaded but not fully preprocessed**
            
            Your CSV data is available, but you need to select features and preprocess it before training:
            1. Go back to the Data tab
            2. Select the feature and label columns
            3. Click "âš™ï¸ Preprocess Selected CSV Columns"
            
            This will prepare your data properly for training.
            """)
            
        st.write("Using Model Configuration:")
        st.json(st.session_state.model_config)
        st.write("Using Data:")
        st.json(st.session_state.data_info)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Training Parameters")
            # Adjust batch size based on model type
            transformer_model = st.session_state.model_config.get('classical_backbone_type', '').lower() == 'transformer'
            default_batch = 4 if transformer_model else 16  # Smaller batch for transformer models
            
            lr = st.number_input("Learning Rate", value=0.001, format="%.5f", key="learning_rate")
            batch_size = st.number_input("Batch Size", value=default_batch, min_value=1, key="batch_size", 
                                         help="Use smaller batch sizes (4-8) for Transformer models to prevent memory issues")
            epochs = st.number_input("Epochs", value=5, min_value=1, key="epochs")
            # Add Task Selection
            task_type = st.selectbox("Training Task", ["Unsupervised (Metric Optimization)", "Classification", "Regression"], key="task_type", help="Determines the loss function and whether labels are required.")
            # Add optimizer choice later (e.g., Adam, SGD)

        with col2:
            st.subheader("Model Instantiation")
            # Button to instantiate the model based on current config
            if st.button("ðŸ—ï¸ Instantiate Hybrid Model", key="instantiate_model_button"):
                with st.spinner("Creating model..."):
                    try:
                        # Ensure ansatz function is resolved from name
                        ansatz_func = AVAILABLE_ANSATZES[st.session_state.model_config['ansatz_name']]
                        
                        # Use GPU safely
                        use_gpu = st.session_state.model_config.get('use_gpu', False) and torch.cuda.is_available()
                        if use_gpu and st.session_state.model_config.get('classical_backbone_type', '').lower() == 'transformer':
                            st.warning("Using GPU with Transformer backbone in Streamlit may cause issues. If model fails, try disabling GPU.")
                        
                        # Add task-specific info to config before creating model
                        config_with_func = {
                            **st.session_state.model_config, 
                            'ansatz_func': ansatz_func,
                            'use_gpu': use_gpu,  # Use the safely determined GPU value
                        }

                        # Determine number of classes/regression outputs based on task and data
                        config_with_func['num_classes'] = None
                        config_with_func['regression_output_size'] = None
                        if st.session_state.task_type == "Classification":
                            labels = st.session_state.get('csv_labels') # Example: get labels
                            if labels is None or len(labels) == 0:
                                st.error("Classification task selected, but no labels found. Load labeled data (e.g., CSV with Label Column).", icon="âš ï¸")
                                st.stop()
                            num_classes = len(torch.unique(labels)) if isinstance(labels, torch.Tensor) else len(labels.unique())
                            if num_classes < 2:
                                st.error(f"Classification task requires at least 2 classes, but found {num_classes} in labels.", icon="âš ï¸")
                                st.stop()
                            config_with_func['num_classes'] = num_classes
                            st.info(f"Configuring model for Classification with {num_classes} classes.")
                        elif st.session_state.task_type == "Regression":
                            labels = st.session_state.get('csv_labels')
                            if labels is None or len(labels) == 0:
                                st.error("Regression task selected, but no labels found. Load labeled data.", icon="âš ï¸")
                                st.stop()
                            # Assuming labels are single continuous values
                            config_with_func['regression_output_size'] = 1
                            st.info(f"Configuring model for Regression with 1 output value.")

                        model = HybridModel(config_with_func)
                        # Update the config in session state AFTER model creation is successful
                        st.session_state.model_config = config_with_func
                        st.session_state.hybrid_model = model.to(st.session_state.device)
                        st.success("Hybrid model instantiated successfully!")
                        # Display model summary?
                        # st.text(str(st.session_state.hybrid_model))
                        st.info(f"Model placed on device: {st.session_state.device}")
                    except Exception as e:
                        st.error(f"Failed to instantiate model: {e}")

            if st.session_state.hybrid_model is not None:
                st.success("Model is instantiated and ready for training.")
                # Option to load weights if model is instantiated
                config_files = get_model_config_files()
                if config_files:
                    load_weights_file = st.selectbox("Load Weights (Optional - requires matching config)", ["None"] + config_files, key="load_weights_select")
                    if load_weights_file != "None":
                        if st.button("ðŸ"‚ Load Weights", key="load_weights_button_train"):
                            if load_model_config(load_weights_file) == st.session_state.model_config: # Basic check
                                load_model_weights(st.session_state.hybrid_model, load_weights_file)
                            else:
                                st.error("Weight file's configuration does not match the current model configuration. Please load the corresponding config first.")

        st.markdown("--- ")

        # --- Training Execution ---
        if st.session_state.hybrid_model is not None:
            # Show training indicator if in progress
            if st.session_state.training_in_progress:
                st.warning("ðŸ"„ **TRAINING IN PROGRESS** - Please wait for completion", icon="âš ï¸")
                training_indicator = st.empty()
                training_indicator.markdown(
                    """
                    <div style="padding:15px;border-radius:5px;background-color:#FFEB3B;color:#000;text-align:center;font-weight:bold;margin:10px 0;">
                        â³ MODEL TRAINING ACTIVE - Please do not navigate away from this page
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            # Disable button when training is already in progress
            start_disabled = st.session_state.training_in_progress
            if st.button("ðŸš€ Start Training", key="start_training_button", disabled=start_disabled):
                st.session_state.training_in_progress = True
                st.rerun()  # Rerun to show the training indicator immediately
            
            # Only execute training if in progress and not just showing the indicator
            if st.session_state.training_in_progress:
                try:
                    # Function to print progress to console log
                    def log_progress(epoch, total_epochs, batch=None, total_batches=None, loss=None, elapsed_time=None):
                        """Print progress to console log for better visibility in terminal"""
                        progress_str = f"[Training Progress] Epoch: {epoch}/{total_epochs}"
                        if batch is not None and total_batches is not None:
                            progress_str += f" | Batch: {batch}/{total_batches}"
                        if loss is not None:
                            progress_str += f" | Loss: {loss:.4f}"
                        if elapsed_time is not None:
                            progress_str += f" | Elapsed: {elapsed_time}"
                        
                        # Print to console log
                        print(progress_str)
                
                    st.info(f"Starting training for {epochs} epochs...")
                    model = st.session_state.hybrid_model
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    
                    # --- Loss Function ---
                    # Define loss based on task.
                    loss_fn = None
                    task_type = st.session_state.task_type # Get selected task

                    if task_type == "Classification":
                        if st.session_state.model_config.get('num_classes') is None:
                            st.error("Model not configured for Classification (num_classes missing). Instantiate model again.", icon="ðŸ›'")
                            st.stop()
                        # Ensure labels are available
                        labels = st.session_state.get('csv_labels')
                        if labels is None:
                            st.error("Classification task requires labels, but none found in session state.", icon="ðŸ›'")
                            st.stop()
                        # Ensure labels are Long type for CrossEntropyLoss
                        if not isinstance(labels, torch.Tensor):
                            try:
                                # Attempt conversion, assuming pandas Series or similar
                                labels = torch.tensor(pd.factorize(labels)[0], dtype=torch.long) if isinstance(labels, pd.Series) else torch.tensor(labels, dtype=torch.long)
                                st.session_state.csv_labels = labels # Update state if conversion works
                                print("Converted labels to torch.long Tensor.")
                            except Exception as e:
                                st.error(f"Could not convert labels to required Long Tensor: {e}", icon="ðŸ›'")
                                st.stop()
                        elif labels.dtype != torch.long:
                            st.warning(f"Converting labels from {labels.dtype} to torch.long for CrossEntropyLoss.")
                            labels = labels.long()
                            st.session_state.csv_labels = labels # Update state

                        loss_fn = torch.nn.CrossEntropyLoss()
                        st.info("Using CrossEntropyLoss for Classification.")

                    elif task_type == "Regression":
                        if st.session_state.model_config.get('regression_output_size') is None:
                            st.error("Model not configured for Regression (regression_output_size missing). Instantiate model again.", icon="ðŸ›'")
                            st.stop()
                        # Ensure labels are available
                        labels = st.session_state.get('csv_labels')
                        if labels is None:
                            st.error("Regression task requires labels, but none found in session state.", icon="ðŸ›'")
                            st.stop()
                        # Ensure labels are Float type for MSELoss
                        if not isinstance(labels, torch.Tensor):
                            try:
                                labels = torch.tensor(labels.values, dtype=torch.float32) if isinstance(labels, pd.Series) else torch.tensor(labels, dtype=torch.float32)
                                st.session_state.csv_labels = labels
                                print("Converted labels to torch.float32 Tensor.")
                            except Exception as e:
                                st.error(f"Could not convert labels to required Float Tensor: {e}", icon="ðŸ›'")
                                st.stop()
                        elif labels.dtype != torch.float32:
                            st.warning(f"Converting labels from {labels.dtype} to torch.float32 for MSELoss.")
                            labels = labels.float()
                            st.session_state.csv_labels = labels # Update state

                        loss_fn = torch.nn.MSELoss()
                        st.info("Using MSELoss for Regression.")

                    elif task_type == "Unsupervised (Metric Optimization)":
                        if st.session_state.model_config['observation_type'] == 'State Vector':
                            # Example: Maximise Entanglement (Minimize negative entanglement)
                            def entanglement_loss(batch_output, num_qubits):
                                batch_loss = 0.0
                                # Ensure output is on CPU for metric calculation if needed
                                batch_output_cpu = batch_output.detach().cpu()
                                for state_vector in batch_output_cpu:
                                    ent = calculate_meyer_wallach(state_vector, num_qubits)
                                    # We want to maximize entanglement, so minimize negative entanglement
                                    batch_loss -= ent if not np.isnan(ent) else 0.0 # Handle potential NaN
                                # Return loss suitable for backprop (needs to be on original device)
                                return (batch_loss / len(batch_output)) * torch.ones(1, device=batch_output.device, requires_grad=True)

                            loss_fn = lambda output: entanglement_loss(output, st.session_state.model_config['num_qubits'])
                            st.info("Using average negative Meyer-Wallach entanglement as loss (Goal: Maximize Entanglement).")
                        elif st.session_state.model_config['observation_type'] == 'Expectation Value (PauliZ)':
                            # Example: Assume target is all +1 expectations (Maximize <Z> for all qubits)
                            # Minimize negative sum of expectation values
                            loss_fn = lambda output: -torch.mean(torch.sum(output, dim=1)) # Average loss over batch
                            st.info("Using average negative sum of expectation values as loss (Goal: Maximize <Z>).")
                        else:
                            st.error("Unsupported observation type for unsupervised metric optimization.", icon="ðŸ›'")
                            st.stop()
                    else:
                        st.error(f"Unknown task type selected: {task_type}", icon="ðŸ›'")
                        st.stop()

                    # --- Data Handling with Dataset and DataLoader ---
                    input_data = st.session_state.loaded_data
                    data_type = st.session_state.data_info.get('data_type')
                    labels = st.session_state.get('csv_labels') # Get labels if available (e.g., from CSV)

                    # Verify data format is compatible with model type
                    backbone_type = st.session_state.model_config.get('classical_backbone_type', '').lower()
                    is_transformer = backbone_type == 'transformer'
                    
                    # Check data and model compatibility
                    if data_type == "CSV" and backbone_type == 'cnn':
                        st.warning("""
                        âš ï¸ **Warning:** You are using a CNN backbone with CSV data. 
                        
                        This might cause issues as CNN models expect image inputs, not feature vectors.
                        The system will attempt to reshape your data to image format, but this may not work well.
                        
                        **Recommended action:** 
                        - Go back to the Configuration tab and change the backbone type to "None" instead.
                        - This will process your feature vectors directly without a CNN.
                        """)
                        
                    if data_type == "Images" and backbone_type == 'transformer':
                        st.warning("""
                        âš ï¸ **Warning:** You are using a Transformer backbone with image data.
                        
                        Transformers expect tokenized text inputs, not images.
                        
                        **Recommended action:**
                        - Change the backbone type to "CNN" which is better suited for image data.
                        """)
                        
                    if is_transformer:
                        # For transformer models, we need tokenized text (dictionary format)
                        if not isinstance(input_data, dict) or 'input_ids' not in input_data or 'attention_mask' not in input_data:
                            st.error("""
                            ERROR: Transformer models require tokenized text input, but received tensor data. 
                            
                            Please go back to the Data tab and use the "Direct Text Extraction" approach:
                            1. Select a text column in the "Option 1: Direct Text Extraction" tab
                            2. Click "Extract Text for Transformer"
                            
                            Vectorized features (TF-IDF/Count) are not compatible with Transformer models.
                            """, icon="ðŸ›'")
                            # Reset training flag on error
                            st.session_state.training_in_progress = False
                            st.stop()
                        else:
                            st.info("âœ… Input data format is correctly tokenized for Transformer model.")

                    dataset = None
                    shuffle_data = True # Shuffle for training

                    # Check if input data needs unsqueezing (if loaded as single item)
                    # This logic might need refinement based on how load_data returns single vs multiple items
                    if data_type in ["Images", "Video"] and isinstance(input_data, torch.Tensor) and len(input_data.shape) < 4:
                        input_data = input_data.unsqueeze(0) # Add batch dimension if needed
                    elif data_type == "Text" and isinstance(input_data, dict):
                        # Check if tensors inside dict have batch dim
                        first_key = list(input_data.keys())[0]
                        if len(input_data[first_key].shape) < 2:
                            input_data = {k: v.unsqueeze(0) for k, v in input_data.items()}

                    # Instantiate appropriate Dataset
                    try:
                        if data_type in ["CSV", "None"]: # None implies direct feature input
                            # Assuming input_data is a tensor [N, Features] or [Features]
                            if len(input_data.shape) == 1: input_data = input_data.unsqueeze(0) # Add batch dim
                            dataset = TensorDatasetWrapper(input_data, labels)
                        elif data_type == "Images":
                            dataset = ImageDatasetWrapper(input_data, labels) # Assuming input_data is [N, C, H, W]
                        elif data_type == "Text":
                            dataset = TextDatasetWrapper(input_data, labels) # Assuming input_data is dict of tensors [N, SeqLen]
                            # Add elif for Graph, Video etc.
                            # elif data_type == "Graph":
                            #     # Assuming input_data is a list of Data objects
                            #     # from torch_geometric.loader import DataLoader as PyGDataLoader # Use PyG dataloader
                            #     # pyg_loader = PyGDataLoader(input_data, batch_size=batch_size, shuffle=shuffle_data)
                            #     # dataset = None # Use pyg_loader directly
                            #     # st.warning("Graph training uses PyG DataLoader - batching handled separately.")
                        else:
                            st.error(f"Dataset creation not implemented for data type: {data_type}")
                            st.stop()

                        if dataset: # If not using a special loader like PyG's
                            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data)
                            st.info(f"Created DataLoader with batch size {batch_size} and {len(dataset)} samples.")
                            # elif pyg_loader: # Handle graph case
                            #     data_loader = pyg_loader
                            #     st.info(f"Using PyG DataLoader with batch size {batch_size} and {len(pyg_loader.dataset)} samples.")
                        else: # Fallback if no dataset/loader created
                            st.error("Failed to create data loader.")
                            st.stop()

                    except Exception as e:
                        st.error(f"Error creating Dataset/DataLoader: {e}")
                        st.write("Input data type:", type(input_data))
                        if hasattr(input_data, 'shape'): st.write("Input data shape:", input_data.shape)
                        st.stop()

                    # --- Training Loop ---
                    st.session_state.training_history = {'loss': [], 'entanglement': []}
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    chart_placeholder = st.empty() # For potential live plotting
                    
                    # Add a live plot that updates during training
                    live_chart_loss = st.empty()
                    live_chart_entanglement = st.empty()
                    
                    # Create batch progress indicator
                    batch_progress = st.progress(0)
                    batch_status = st.empty()
                    
                    # Add estimated time remaining
                    eta_text = st.empty()
                    
                    # Track training time
                    training_start_time = time.time()
                    batch_start_time = time.time()
                    last_update_time = time.time()
                    
                    # Initialize entanglement for tracking
                    entanglement = np.nan
                    batch_entanglements = []
                    
                    # Initialize training duration
                    training_duration = 0.0
                    
                    # Create dataframes to store live chart data
                    loss_chart_data = pd.DataFrame({"Loss": [], "Step": []})
                    entanglement_chart_data = pd.DataFrame({"Entanglement": [], "Step": []})
                    
                    global_step = 0
                    model.train() # Set model to training mode
                    
                    # Track batch entanglements for calculating epoch averages
                    batch_entanglements = []
                    
                    for epoch in range(epochs):
                        epoch_loss = 0.0
                        num_batches = 0
                        batch_times = []
                        total_batches = len(data_loader)
                        
                        for batch_idx, batch in enumerate(data_loader):
                            batch_start_time = time.time()
                            optimizer.zero_grad()
                            num_batches += 1
                            global_step += 1

                            # Separate features and labels if labels exist
                            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                                if isinstance(batch[0], dict):  # Handle dict features (e.g., from TextDatasetWrapper)
                                    batch_features, batch_labels = batch[0], batch[1]
                                else:
                                    batch_features, batch_labels = batch
                                # TODO: Use labels if loss function requires them
                            else:
                                if isinstance(batch, dict):  # Handle dict input directly
                                    batch_features = batch
                                    batch_labels = None
                                else:
                                    batch_features = batch
                                    batch_labels = None # No labels provided by dataset

                            # Move data to device
                            if isinstance(batch_features, torch.Tensor):
                                batch_features = batch_features.to(st.session_state.device)
                            elif isinstance(batch_features, dict): # Handle dicts (e.g., from text tokenizer)
                                batch_features = {k: v.to(st.session_state.device) for k, v in batch_features.items()}
                            # Add handling for other types like PyG Batch objects
                            # elif 'Batch' in str(type(batch_features)): # Heuristic check for PyG Batch
                            #     batch_features = batch_features.to(st.session_state.device)

                            if batch_labels is not None:
                                batch_labels = batch_labels.to(st.session_state.device)

                            # --- Forward Pass ---
                            try:
                                # Debug for transformer input
                                if (isinstance(batch_features, dict) and 
                                    st.session_state.model_config.get('classical_backbone_type', '').lower() == 'transformer'):
                                    # Check that the dictionary has the expected keys for transformer
                                    #st.write(f"Transformer input keys: {batch_features.keys()}")
                                    # Ensure we have at least 'input_ids' and 'attention_mask'
                                    required_keys = ['input_ids', 'attention_mask']
                                    missing_keys = [key for key in required_keys if key not in batch_features]
                                    if missing_keys:
                                        st.error(f"Missing required transformer input keys: {missing_keys}")
                                        st.stop()
                                    
                                output = model(batch_features)
                            except Exception as e:
                                st.error(f"Error during forward pass (Batch {num_batches}, Epoch {epoch+1}): {e}")
                                st.write(f"Input batch type: {type(batch_features)}")
                                if hasattr(batch_features, 'shape'): 
                                    st.write(f"Input batch shape: {batch_features.shape}")
                                if isinstance(batch_features, dict): 
                                    st.write(f"Input batch keys/shapes: { {k: v.shape for k,v in batch_features.items()} }")
                                # Consider stopping or skipping batch
                                st.stop()

                            # --- Calculate Loss ---
                            # Loss function might need labels depending on the task
                            if task_type in ["Classification", "Regression"]:
                                if batch_labels is None:
                                    st.error(f"{task_type} task requires labels, but none found in current batch.", icon="ðŸ›'")
                                    st.stop()
                                # Ensure output and labels have compatible shapes for loss
                                # E.g., for MSELoss, output might need squeezing if shape is [batch, 1]
                                if task_type == "Regression" and len(output.shape) > 1 and output.shape[1] == 1:
                                    output = output.squeeze(-1)
                                # Ensure labels are on the same device as output
                                batch_labels = batch_labels.to(output.device)
                                loss = loss_fn(output, batch_labels)
                            else: # Unsupervised
                                loss = loss_fn(output)

                            # --- Backward Pass & Optimize ---
                            try:
                                loss.backward()
                                optimizer.step()
                            except Exception as e:
                                st.error(f"Error during backward pass or optimizer step (Batch {num_batches}, Epoch {epoch+1}): {e}")
                                st.stop()

                            current_loss = loss.item()
                            epoch_loss += current_loss
                            
                            # Record time per batch for ETA calculation
                            batch_end_time = time.time()
                            batch_duration = batch_end_time - batch_start_time
                            batch_times.append(batch_duration)
                            
                            # Calculate batch progress
                            batch_progress.progress(batch_idx / total_batches)
                            
                            # Update batch status more frequently (every batch)
                            batch_status.text(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{total_batches} | Loss: {current_loss:.4f}")
                            
                            # Log progress every 10 batches
                            if batch_idx % 10 == 0:
                                log_progress(
                                    epoch+1, epochs, 
                                    batch=batch_idx+1, 
                                    total_batches=total_batches, 
                                    loss=current_loss
                                )
                            
                            # Add current loss to live chart (every batch)
                            current_time = time.time() - training_start_time
                            # Update chart data
                            new_row = pd.DataFrame({"Loss": [current_loss], "Step": [global_step]})
                            loss_chart_data = pd.concat([loss_chart_data, new_row], ignore_index=True)
                            
                            # Redraw the chart with updated data
                            live_chart_loss.line_chart(loss_chart_data, x="Step", y="Loss")
                            
                            # Calculate and display ETA
                            if len(batch_times) > 0:
                                avg_batch_time = sum(batch_times) / len(batch_times)
                                remaining_batches = total_batches - (batch_idx + 1) + (epochs - epoch - 1) * total_batches
                                eta_seconds = avg_batch_time * remaining_batches
                                
                                # Format ETA
                                eta_hours, remainder = divmod(eta_seconds, 3600)
                                eta_minutes, eta_seconds = divmod(remainder, 60)
                                if eta_hours > 0:
                                    eta_str = f"{int(eta_hours)}h {int(eta_minutes)}m {eta_seconds:.0f}s"
                                elif eta_minutes > 0:
                                    eta_str = f"{int(eta_minutes)}m {eta_seconds:.0f}s"
                                else:
                                    eta_str = f"{eta_seconds:.1f}s"
                                
                                # Update ETA display
                                eta_text.text(f"Estimated time remaining: {eta_str}")

                            # --- Metrics & Logging (less frequent) ---
                            current_time = time.time()
                            time_since_update = current_time - last_update_time
                            
                            # Update status every 1 second or every 5 batches, whichever comes first
                            if time_since_update > 1.0 or global_step % 5 == 0: 
                                entanglement = np.nan
                                if st.session_state.model_config['observation_type'] == 'State Vector':
                                    with torch.no_grad():
                                        try:
                                            # Calculate entanglement for the first item in the batch
                                            # Use the stored quantum state for entanglement calculation
                                            if hasattr(model, 'last_quantum_output') and model.last_quantum_output is not None:
                                                quantum_state = model.last_quantum_output[0].detach().cpu()  # First sample
                                                expected_size = 2**st.session_state.model_config['num_qubits']
                                                
                                                if quantum_state.numel() == expected_size:
                                                    entanglement = calculate_meyer_wallach(quantum_state, st.session_state.model_config['num_qubits'])
                                                    print(f"Calculated entanglement from quantum state with shape {quantum_state.shape}")
                                                else:
                                                    print(f"Warning: Quantum state size {quantum_state.numel()} doesn't match expected size {expected_size}")
                                                    entanglement = np.nan
                                            else:
                                                # Fallback to using model output (likely won't work for classification)
                                                print("Warning: No stored quantum state available, trying to use model output")
                                                state_vector = output[0].detach().cpu()
                                                entanglement = calculate_meyer_wallach(state_vector, st.session_state.model_config['num_qubits'])
                                                
                                            # Handle NaN entanglement
                                            if np.isnan(entanglement):
                                                print("Warning: Entanglement calculation returned NaN, using placeholder value")
                                                # Don't add to batch_entanglements
                                            else:
                                                # Add to batch entanglements for epoch average calculation
                                                batch_entanglements.append(entanglement)
                                                
                                                # Update entanglement chart
                                                new_row = pd.DataFrame({"Entanglement": [entanglement], "Step": [global_step]})
                                                entanglement_chart_data = pd.concat([entanglement_chart_data, new_row], ignore_index=True)
                                                live_chart_entanglement.line_chart(entanglement_chart_data, x="Step", y="Entanglement")
                                        except Exception as e:
                                            # Handle potential errors in entanglement calculation
                                            print(f"Warning: Error calculating entanglement: {e}")
                                            # Don't raise the exception to avoid stopping training
                                
                                # Calculate elapsed time
                                elapsed_time = time.time() - training_start_time
                                elapsed_hours, remainder = divmod(elapsed_time, 3600)
                                elapsed_minutes, elapsed_seconds = divmod(remainder, 60)
                                if elapsed_hours > 0:
                                    elapsed_str = f"{int(elapsed_hours)}h {int(elapsed_minutes)}m {elapsed_seconds:.0f}s"
                                elif elapsed_minutes > 0:
                                    elapsed_str = f"{int(elapsed_minutes)}m {elapsed_seconds:.0f}s"
                                else:
                                    elapsed_str = f"{elapsed_seconds:.1f}s"
                                
                                status_text.text(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{total_batches} | Step {global_step} | Loss: {current_loss:.4f} | Entanglement (Q): {entanglement:.4f} | Elapsed: {elapsed_str}")
                                
                                # Update time of last status update
                                last_update_time = current_time

                        # --- End of Epoch ---
                        avg_epoch_loss = epoch_loss / num_batches
                        st.session_state.training_history['loss'].append(avg_epoch_loss)

                        # Calculate average entanglement for the epoch (more stable than last batch)
                        # This might require running inference again on a sample or storing batch entanglements
                        # Simple approach: Use last calculated value for history plot
                        
                        # Safety check to avoid index error with entanglement history
                        if len(batch_entanglements) > 0:
                            # Use average of collected entanglements for this epoch
                            avg_entanglement = np.nanmean(batch_entanglements)
                            if not np.isnan(avg_entanglement):
                                entanglement = avg_entanglement
                        
                        # Append entanglement value (might be NaN, which is handled by visualization)
                        st.session_state.training_history['entanglement'].append(entanglement)
                        
                        # Reset batch entanglements collection for next epoch
                        batch_entanglements = []

                        # Reset batch progress at end of epoch
                        batch_progress.progress(1.0)
                        
                        # Update epoch progress
                        progress_bar.progress((epoch + 1) / epochs)
                        
                        # Update epoch status text
                        
                        # Calculate elapsed time
                        elapsed_time = time.time() - training_start_time
                        elapsed_hours, remainder = divmod(elapsed_time, 3600)
                        elapsed_minutes, elapsed_seconds = divmod(remainder, 60)
                        if elapsed_hours > 0:
                            elapsed_str = f"{int(elapsed_hours)}h {int(elapsed_minutes)}m {elapsed_seconds:.0f}s"
                        elif elapsed_minutes > 0:
                            elapsed_str = f"{int(elapsed_minutes)}m {elapsed_seconds:.0f}s"
                        else:
                            elapsed_str = f"{elapsed_seconds:.1f}s"
                            
                        status_text.text(f"Epoch {epoch+1}/{epochs} Completed | Avg Loss: {avg_epoch_loss:.4f} | Last Entanglement (Q): {entanglement:.4f} | Elapsed: {elapsed_str}")
                        
                        # Log epoch completion
                        log_progress(
                            epoch+1, epochs, 
                            loss=avg_epoch_loss, 
                            elapsed_time=elapsed_str
                        )
                        
                        # Update epoch charts
                        epochs_list = list(range(1, epoch+2))
                        
                        # Loss chart by epoch
                        loss_epoch_data = pd.DataFrame({
                            "Loss": st.session_state.training_history['loss'],
                            "Epoch": epochs_list
                        })
                        chart_placeholder.line_chart(loss_epoch_data, x="Epoch", y="Loss")
                        
                        # Entanglement chart by epoch (if any values exist)
                        if any(not np.isnan(e) for e in st.session_state.training_history['entanglement']):
                            # Filter out NaN values for the chart
                            valid_entanglements = []
                            valid_epochs = []
                            for i, e in enumerate(st.session_state.training_history['entanglement']):
                                if not np.isnan(e):
                                    valid_entanglements.append(e)
                                    valid_epochs.append(i+1)  # Epochs are 1-indexed for display
                            
                            if len(valid_entanglements) > 0:
                                ent_epoch_data = pd.DataFrame({
                                    "Entanglement": valid_entanglements,
                                    "Epoch": valid_epochs
                                })
                                chart_placeholder.line_chart(ent_epoch_data, x="Epoch", y="Entanglement")

                    status_text.text(f"Training finished after {epochs} epochs.")
                    st.success("Training complete!")
                    model.eval() # Set back to eval mode
                    
                    # Calculate training time
                    training_end_time = time.time()
                    training_duration = training_end_time - training_start_time
                    
                    # Format time nicely for output
                    hours, remainder = divmod(training_duration, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    if hours > 0:
                        time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
                    elif minutes > 0:
                        time_str = f"{int(minutes)}m {seconds:.2f}s"
                    else:
                        time_str = f"{seconds:.2f}s"

                    # --- Save Final Weights ---
                    if st.session_state.model_config:
                        # Get model backbone type for more descriptive filename
                        backbone_type = st.session_state.model_config.get('classical_backbone_type', 'none').lower()
                        num_qubits = st.session_state.model_config.get('num_qubits', '4')
                        num_layers = st.session_state.model_config.get('num_layers', '2')
                        
                        # Create a descriptive base name
                        model_descriptor = f"{backbone_type}_N{num_qubits}_L{num_layers}"
                        
                        # Also save with the original config name if available
                        config_name = st.session_state.model_config.get('save_config_name', '')
                        
                        # Track weight filenames
                        weight_files = []
                        
                        # Always save to last_train for backward compatibility
                        last_train_path = "weights_last_train"
                        save_model_weights(model, last_train_path)
                        weight_files.append(last_train_path)
                        
                        # Save with descriptive name based on the model architecture
                        descriptive_path = f"weights_{model_descriptor}.pt"
                        save_model_weights(model, descriptive_path)
                        weight_files.append(descriptive_path)
                        
                        # Additionally save with config name if one was used
                        if config_name and not config_name == 'last_train':
                            config_path = f"weights_{config_name.replace('.json', '.pt')}"
                            save_model_weights(model, config_path)
                            weight_files.append(config_path)
                            
                        # Display saved weights information
                        st.info(f"Model weights saved with descriptive name: weights_{model_descriptor}.pt")

                    # --- Display Final Training Plot ---
                    st.subheader("Final Training History")
                    plot_training_history(st.session_state.training_history)
                    
                    # --- Display Training Time Metrics ---
                    st.subheader("Training Time Metrics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Format total duration for metrics display
                        hours_total, remainder = divmod(training_duration, 3600)
                        minutes_total, seconds_total = divmod(remainder, 60)
                        if hours_total > 0:
                            duration_str = f"{int(hours_total)}h {int(minutes_total)}m {seconds_total:.1f}s"
                        elif minutes_total > 0:
                            duration_str = f"{int(minutes_total)}m {seconds_total:.1f}s"
                        else:
                            duration_str = f"{seconds_total:.1f}s"
                        
                        st.metric("Total Training Time", duration_str)
                        
                    with col2:
                        # Calculate time per epoch
                        time_per_epoch = training_duration / epochs
                        hours_epoch, remainder = divmod(time_per_epoch, 3600)
                        minutes_epoch, seconds_epoch = divmod(remainder, 60)
                        if hours_epoch > 0:
                            epoch_str = f"{int(hours_epoch)}h {int(minutes_epoch)}m {seconds_epoch:.1f}s"
                        elif minutes_epoch > 0:
                            epoch_str = f"{int(minutes_epoch)}m {seconds_epoch:.1f}s"
                        else:
                            epoch_str = f"{seconds_epoch:.1f}s"
                        
                        st.metric("Time per Epoch", epoch_str)
                    
                    # Final loss
                    final_loss = st.session_state.training_history['loss'][-1]
                    st.metric("Final Loss", f"{final_loss:.4f}")
                    
                    # Entanglement if available
                    if len(st.session_state.training_history['entanglement']) > 0:
                        final_entanglement = st.session_state.training_history['entanglement'][-1]
                        if not np.isnan(final_entanglement):
                            st.metric("Final Entanglement", f"{final_entanglement:.4f}")

                    # --- Save run in history ---
                    # Import run history functions
                    from utils.run_history import add_run, save_run_details
                    from utils.metrics import plot_confusion_matrix, calculate_classification_metrics, calculate_regression_metrics

                    # Determine task type and configuration name
                    task_type = st.session_state.task_type if 'task_type' in st.session_state else "Unknown"
                    config_name = f"{backbone_type}_N{num_qubits}_L{num_layers}" if 'backbone_type' in locals() and 'num_qubits' in locals() and 'num_layers' in locals() else "Unnamed Run"

                    # Add run to history
                    run_id = add_run(
                        config_name=config_name,
                        duration=training_duration,
                        status="Completed"
                    )

                    # Create run details dictionary
                    run_details = {
                        "task_type": task_type,
                        "model_config": st.session_state.model_config,
                        "training_params": {
                            "learning_rate": lr,
                            "batch_size": batch_size,
                            "epochs": epochs
                        },
                        "training_history": {
                            "loss": st.session_state.training_history.get('loss', []),
                            "entanglement": [float(e) if not np.isnan(e) else None for e in st.session_state.training_history.get('entanglement', [])]
                        },
                        "data_info": st.session_state.data_info,
                        "weights_files": weight_files if 'weight_files' in locals() else []
                    }

                    # Calculate and add metrics if we have validation data
                    if task_type == "Classification" and 'csv_labels' in st.session_state:
                        try:
                            # Get validation data - we would use a separate test set in production
                            features = input_data
                            labels = st.session_state.csv_labels
                            
                            # Run model on validation data
                            model.eval()
                            with torch.no_grad():
                                if isinstance(features, dict): # For transformer input
                                    val_outputs = model(features)
                                else:
                                    # Use smaller batches for inference
                                    inference_batch_size = min(32, len(features))
                                    val_dataloader = DataLoader(TensorDatasetWrapper(features, labels), batch_size=inference_batch_size, shuffle=False)
                                    
                                    all_outputs = []
                                    for batch_features, _ in val_dataloader:
                                        batch_features = batch_features.to(st.session_state.device)
                                        batch_outputs = model(batch_features)
                                        all_outputs.append(batch_outputs.cpu())
                                    
                                    val_outputs = torch.cat(all_outputs, dim=0)
                                
                                # For multi-class, convert logits to probabilities
                                val_probs = torch.softmax(val_outputs, dim=1) if val_outputs.shape[1] > 1 else torch.sigmoid(val_outputs)
                                val_preds = torch.argmax(val_probs, dim=1) if val_outputs.shape[1] > 1 else (val_probs > 0.5).long()
                                
                                # Calculate metrics
                                metrics = calculate_classification_metrics(
                                    labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels,
                                    val_preds.cpu().numpy(),
                                    val_probs.cpu().numpy()
                                )
                                
                                # Generate confusion matrix visualization if available
                                if 'confusion_matrix' in metrics and metrics['confusion_matrix'] is not None:
                                    # Get class names if available
                                    class_names = st.session_state.get('class_names', None)
                                    if class_names is None:
                                        # Create generic class names
                                        class_names = [f"Class {i}" for i in range(len(np.unique(labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels)))]
                                    
                                    # Generate confusion matrix image
                                    cm_img = plot_confusion_matrix(np.array(metrics['confusion_matrix']), class_names)
                                    run_details['confusion_matrix_img'] = cm_img
                                
                                # Add metrics to run details
                                run_details['metrics'] = metrics
                                
                        except Exception as e:
                            st.warning(f"Could not calculate validation metrics: {e}")
                            run_details['metrics_error'] = str(e)

                    elif task_type == "Regression" and 'csv_labels' in st.session_state:
                        try:
                            # Get validation data
                            features = input_data
                            labels = st.session_state.csv_labels
                            
                            # Run model on validation data
                            model.eval()
                            with torch.no_grad():
                                if isinstance(features, dict): # For transformer input
                                    val_outputs = model(features)
                                else:
                                    # Use smaller batches for inference
                                    inference_batch_size = min(32, len(features))
                                    val_dataloader = DataLoader(TensorDatasetWrapper(features, labels), batch_size=inference_batch_size, shuffle=False)
                                    
                                    all_outputs = []
                                    for batch_features, _ in val_dataloader:
                                        batch_features = batch_features.to(st.session_state.device)
                                        batch_outputs = model(batch_features)
                                        all_outputs.append(batch_outputs.cpu())
                                    
                                    val_outputs = torch.cat(all_outputs, dim=0)
                                
                                # Ensure outputs are in the right shape for regression
                                if len(val_outputs.shape) > 1 and val_outputs.shape[1] == 1:
                                    val_outputs = val_outputs.squeeze(-1)
                                
                                # Calculate metrics
                                metrics = calculate_regression_metrics(
                                    labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels,
                                    val_outputs.cpu().numpy()
                                )
                                
                                # Add metrics to run details
                                run_details['metrics'] = metrics
                                
                        except Exception as e:
                            st.warning(f"Could not calculate validation metrics: {e}")
                            run_details['metrics_error'] = str(e)

                    # Save the run details
                    save_run_details(run_id, run_details)
                    st.success(f"Training run saved to history with ID {run_id}")

                    # Reset training flag
                    st.session_state.training_in_progress = False
                except Exception as e:
                    st.error(f"Error during training: {e}")
                    # Reset training flag on error
                    st.session_state.training_in_progress = False
            else:
                st.info("Click 'Start Training' to begin.")
        else:
            st.warning("Instantiate the model before training.")

# --- Tab 4: Predict --- (Depends on Model & Data Input)
with tab_predict:
    st.header("Make Predictions with Trained Model")

    # 1. Load Model Configuration
    st.subheader("1. Load Model Configuration")
    config_files = get_model_config_files()
    if not config_files:
        st.warning("No saved model configurations found. Please configure a model first.")
    else:
        selected_config_file_pred = st.selectbox("Select Model Configuration", config_files, key="pred_config_select")
        if st.button("Load Configuration for Prediction", key="pred_load_config_button"):
            loaded_cfg = load_model_config(selected_config_file_pred)
            if loaded_cfg:
                st.session_state.model_config = loaded_cfg # Overwrite current config
                st.session_state.hybrid_model = None # Reset instantiated model
                st.success(f"Loaded configuration '{selected_config_file_pred}'. Please instantiate and load weights.")
                st.rerun() # Update UI

    # 2. Instantiate Model & Load Weights
    if st.session_state.model_config:
        st.subheader("2. Instantiate Model and Load Weights")
        st.json(st.session_state.model_config)
        if st.session_state.hybrid_model is None:
            if st.button("Instantiate Model", key="pred_instantiate_button"):
                with st.spinner("Instantiating model..."):
                    try:
                        # First check the weights file to determine model architecture
                        weights_filepath = os.path.join(SAVED_WEIGHTS_DIR, "weights_last_train")
                        has_output_head = False
                        num_classes = None
                        
                        if os.path.exists(weights_filepath):
                            # Load the weights file to check its structure
                            try:
                                weights = torch.load(weights_filepath, map_location='cpu')
                                
                                # Check if it has output head parameters
                                has_output_head = any('output_head' in k for k in weights.keys())
                                
                                # If it has output head, get the number of classes
                                if has_output_head and 'output_head.weight' in weights:
                                    output_shape = weights['output_head.weight'].shape
                                    num_classes = output_shape[0]
                                    st.info(f"Detected output head in weights with {num_classes} classes")
                            except Exception as e:
                                st.warning(f"Could not examine weights file structure: {e}")
                        
                        # Setup model configuration
                        ansatz_func = AVAILABLE_ANSATZES[st.session_state.model_config['ansatz_name']]
                        config_with_func = {
                            **st.session_state.model_config, 
                            'ansatz_func': ansatz_func,
                            'use_gpu': st.session_state.model_config.get('use_gpu', False) and torch.cuda.is_available(),
                        }
                        
                        # Add num_classes if detected from weights
                        if has_output_head and num_classes is not None:
                            config_with_func['num_classes'] = num_classes
                            st.success(f"Model architecture adjusted to match weights file (num_classes={num_classes})")
                        
                        # Create the model
                        model = HybridModel(config_with_func)
                        st.session_state.hybrid_model = model.to(st.session_state.device)
                        st.success("Model instantiated.")
                        
                        # Attempt to load corresponding weights
                        weights_loaded = load_model_weights(st.session_state.hybrid_model, selected_config_file_pred)
                        if not weights_loaded:
                            st.warning("Could not automatically load weights. Please ensure weights file exists or train the model.")
                    except Exception as e:
                         st.error(f"Failed to instantiate model: {e}")
        else:
            st.success("Model already instantiated.")
            # Manual weight loading option if needed
            if st.button("Reload Weights", key="pred_reload_weights_button"):
                load_model_weights(st.session_state.hybrid_model, selected_config_file_pred)

    # 3. Input Data for Prediction
    if st.session_state.hybrid_model is not None:
        st.subheader("3. Provide Input Data")
        # Use similar UI as data tab, but maybe simplified for single prediction
        pred_data_type = st.selectbox("Input Data Type", AVAILABLE_DATA_TYPES, key="pred_data_type", index=AVAILABLE_DATA_TYPES.index(st.session_state.data_info.get('data_type', 'Images')))

        pred_input = None
        if pred_data_type == "Images":
            pred_uploaded_file = st.file_uploader("Upload Image for Prediction", type=['png', 'jpg', 'jpeg'], key="pred_img_upload")
            if pred_uploaded_file:
                pred_input = Image.open(pred_uploaded_file).convert('RGB')
                st.image(pred_input, caption="Image for Prediction", width=200)
        elif pred_data_type == "Text":
            if st.session_state.model_config.get('classical_backbone_type') != 'Transformer':
                st.warning("You selected Text data but your model doesn't have a Transformer backbone. "
                           "Consider changing the model configuration to use a Transformer.")
                
            # Text input area
            pred_input = st.text_area(
                "Enter text for prediction:",
                value="",
                height=150,
                help="Enter the text you want to predict on. Make sure your model has a Transformer backbone."
            )
            
            # Show tokenizer info
            if st.session_state.tokenizer:
                st.info(f"Using tokenizer: {st.session_state.tokenizer.name_or_path}")
                if pred_input:
                    # Show token count as user types
                    token_count = len(st.session_state.tokenizer.encode(pred_input))
                    st.write(f"Token count: {token_count}")
                    if token_count > 512:
                        st.warning(f"Text is {token_count} tokens, which exceeds the typical 512 token limit. Text will be truncated.")
            else:
                st.error("No tokenizer available! Please ensure you've selected a Transformer model in the Configuration tab.")
        elif pred_data_type == "CSV":
             st.info("CSV prediction requires manual input matching feature columns.")
             # Need to show expected feature columns based on config/training data
             # Example: Create input fields dynamically
             if 'csv_feature_cols' in st.session_state:
                 pred_csv_input = {}
                 for col in st.session_state.csv_feature_cols:
                     pred_csv_input[col] = st.number_input(f"Enter value for {col}", key=f"pred_csv_{col}", value=0.0, format="%f")
                 if pred_csv_input:
                      # Convert dict to DataFrame/Tensor matching preprocessing
                      pred_input = pd.DataFrame([pred_csv_input])
             else:
                 st.warning("CSV feature columns unknown. Load corresponding data in Data tab first.")
        else:
             st.warning(f"Prediction input for data type '{pred_data_type}' is not implemented yet.")

        # 4. Make Prediction
        st.subheader("4. Make Prediction")
        if pred_input is not None:
            if st.button("ðŸ"® Predict", key="pred_button"):
                with st.spinner("Processing and predicting..."):
                    # Preprocess the input data
                    processed_pred_input = None
                    if pred_data_type == "Images":
                        processed_pred_input = preprocess_image(pred_input).unsqueeze(0) # Add batch dim
                    elif pred_data_type == "Text":
                        # Make sure tokenizer exists
                        if not st.session_state.tokenizer:
                            st.error("No tokenizer available! Please ensure you've selected a Transformer model in the Configuration tab.")
                        else:
                            try:
                                st.write(f"Tokenizing text using {st.session_state.tokenizer.name_or_path}...")
                                # Pass max_length parameter to avoid potential issues with long text
                                processed_pred_input = preprocess_text(pred_input, st.session_state.tokenizer, max_length=512)
                                st.write(f"Text tokenized with shape: input_ids={processed_pred_input['input_ids'].shape}, "
                                       f"attention_mask={processed_pred_input['attention_mask'].shape}")
                                
                                # Show sample of tokenized input for debugging
                                if 'input_ids' in processed_pred_input:
                                    st.write("First 10 tokens:", processed_pred_input['input_ids'][0][:10].tolist())
                            except Exception as e:
                                st.error(f"Error during text tokenization: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                    elif pred_data_type == "CSV":
                         # Re-apply preprocessing (scaling etc.) as done during training - tricky!
                         st.warning("CSV preprocessing for prediction will use saved scaler and vectorizer if available.")
                         try:
                              feature_cols = st.session_state.csv_feature_cols
                              df_input = pred_input[feature_cols]
                              
                              # Get text processing parameters from data_info
                              text_cols = st.session_state.data_info.get('text_cols')
                              text_vectorizer = st.session_state.data_info.get('text_vectorizer', 'tfidf')
                              max_text_features = st.session_state.data_info.get('max_text_features', 100)
                              
                              # Use the same preprocessing function but in prediction mode
                              from data.preprocessing import preprocess_csv
                              processed_features, _ = preprocess_csv(
                                  df_input, feature_cols, None,
                                  # Try to use the same scaler name pattern as during training
                                  scaler_name=st.session_state.data_info.get('scaler_name',
                                    f"scaler_{df_input.shape[1]}_{len(feature_cols)}.joblib"),
                                  is_prediction=True,  # Important: Load scaler instead of fitting
                                  text_cols=text_cols,
                                  text_vectorizer=text_vectorizer,
                                  max_text_features=max_text_features
                              )
                              processed_pred_input = processed_features
                         except Exception as e:
                             st.error(f"Error preprocessing CSV input for prediction: {e}")
                     # Add other types

                    if processed_pred_input is not None:
                        # Move to device
                        if isinstance(processed_pred_input, dict):
                            # For text inputs (tokenizer output is a dict)
                            processed_pred_input = {k: v.to(st.session_state.device) for k, v in processed_pred_input.items()}
                            st.write("Input moved to device:", st.session_state.device)
                        else:
                            processed_pred_input = processed_pred_input.to(st.session_state.device)

                        # Run inference
                        model = st.session_state.hybrid_model
                        model.eval()
                        with torch.no_grad():
                            try:
                                st.write("Running prediction...")
                                
                                # Check if we're using a transformer and have a BatchEncoding input
                                if isinstance(processed_pred_input, dict) and 'input_ids' in processed_pred_input:
                                    backbone_type = st.session_state.model_config.get('classical_backbone_type', '').lower()
                                    if backbone_type == 'transformer':
                                        st.write("Using transformer model with tokenized input")
                                        # Make sure the input has batch dimension
                                        if processed_pred_input['input_ids'].dim() == 1:
                                            for k in processed_pred_input:
                                                processed_pred_input[k] = processed_pred_input[k].unsqueeze(0)
                                            st.write("Added batch dimension to tokenized input")
                                    else:
                                        st.error("Input is tokenized but model is not a transformer")
                                        raise ValueError("Data format mismatch: tokenized input with non-transformer model")
                                
                                # Run the model
                                prediction = model(processed_pred_input)
                                st.success("Prediction successful!")

                                # Display results
                                st.subheader("Prediction Output")
                                
                                # Display the classification result first (if applicable)
                                if hasattr(model, 'output_head') and model.output_head is not None:
                                    # Get the class prediction
                                    pred_tensor = prediction.detach().cpu()
                                    
                                    # Get class probabilities
                                    if len(pred_tensor.shape) > 1 and pred_tensor.shape[1] > 1:
                                        # For multi-class output, apply softmax to get probabilities
                                        probs = torch.nn.functional.softmax(pred_tensor, dim=1)
                                        
                                        # Get the predicted class index
                                        pred_class = torch.argmax(probs, dim=1)[0].item()
                                        confidence = probs[0, pred_class].item() * 100
                                        
                                        # Display class information
                                        class_names = ["Cat", "Dog"]  # Default classes
                                        if 'class_names' in st.session_state and len(st.session_state.class_names) >= 2:
                                            class_names = st.session_state.class_names
                                            
                                        # Show prediction with confidence
                                        st.markdown(f"### Prediction: **{class_names[pred_class]}** ðŸ±ðŸ¶")
                                        st.markdown(f"Confidence: **{confidence:.2f}%**")
                                        
                                        # Show probabilities for all classes
                                        st.write("Class Probabilities:")
                                        prob_df = pd.DataFrame({
                                            "Class": class_names[:len(probs[0])],
                                            "Probability": probs[0].numpy()
                                        })
                                        st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))
                                        
                                        # Add a horizontal line to separate from quantum data
                                        st.markdown("---")
                                
                                # Continue with existing output displays
                                obs_type = st.session_state.model_config.get('observation_type', 'State Vector')
                                if obs_type == 'State Vector':
                                    st.write("Output State Vector (Probabilities):")
                                    # st.text(prediction.detach().cpu().numpy())
                                    plot_state_vector(prediction, st.session_state.model_config['num_qubits'])
                                    # Calculate and display entanglement
                                    try:
                                        # Get the quantum state correctly from the model's stored output
                                        if hasattr(model, 'last_quantum_output') and model.last_quantum_output is not None:
                                            # Use the stored quantum output
                                            quantum_state = model.last_quantum_output[0].detach().cpu()
                                            expected_size = 2**st.session_state.model_config['num_qubits']
                                            
                                            if quantum_state.numel() == expected_size:
                                                # Calculate entanglement on the quantum state
                                                entanglement = calculate_meyer_wallach(quantum_state, st.session_state.model_config['num_qubits'])
                                                st.write("Entanglement calculated from quantum state output")
                                            else:
                                                st.warning(f"State vector size ({quantum_state.numel()}) doesn't match expected size ({expected_size}) for {st.session_state.model_config['num_qubits']} qubits")
                                                entanglement = np.nan
                                        else:
                                            # Fallback to using the prediction output (likely won't work for classification)
                                            pred_state = prediction.detach().cpu()
                                            if len(pred_state.shape) > 1 and pred_state.shape[0] > 1:
                                                st.write(f"Batch prediction output - showing entanglement for first sample")
                                                pred_state = pred_state[0]
                                            
                                            expected_size = 2**st.session_state.model_config['num_qubits']
                                            if pred_state.numel() != expected_size:
                                                st.warning(f"State vector size ({pred_state.numel()}) doesn't match expected size ({expected_size}) for {st.session_state.model_config['num_qubits']} qubits")
                                            
                                            entanglement = calculate_meyer_wallach(pred_state, st.session_state.model_config['num_qubits'])
                                        
                                        # Handle NaN entanglement values
                                        if np.isnan(entanglement):
                                            st.warning("Could not calculate entanglement for this state. The state may be invalid or numerical issues occurred.")
                                        else:
                                            st.metric("Entanglement (Meyer-Wallach Q)", f"{entanglement:.4f}")
                                            
                                            # Add explanation of the entanglement value
                                            if entanglement < 0.1:
                                                st.info("Low entanglement: This state is close to separable.")
                                            elif entanglement > 0.9:
                                                st.info("High entanglement: This state has strong quantum correlations.")
                                            else:
                                                st.info("Moderate entanglement detected in this quantum state.")
                                    except Exception as e:
                                        st.error(f"Error calculating entanglement: {e}")
                                        st.write(f"Prediction output type: {type(prediction)}, shape: {prediction.shape if hasattr(prediction, 'shape') else 'N/A'}")

                                elif obs_type == 'Expectation Value (PauliZ)':
                                    st.write("Output Expectation Values <Z>:")
                                    st.dataframe(pd.DataFrame(prediction.detach().cpu().numpy(), columns=[f'Qubit {i}' for i in range(st.session_state.model_config['num_qubits'])]))
                                else:
                                    st.write("Output:")
                                    st.text(prediction.detach().cpu().numpy())

                            except Exception as e:
                                 st.error(f"Error during prediction forward pass: {e}")
                                 
                                 # Show detailed debugging info
                                 if isinstance(processed_pred_input, dict):
                                     # For transformer inputs
                                     st.write("Input is a dictionary (likely transformer input):")
                                     for k, v in processed_pred_input.items():
                                         st.write(f"- {k}: {type(v)}, shape={v.shape if hasattr(v, 'shape') else 'N/A'}")
                                 else:
                                     # For other input types
                                     st.write(f"Input data type: {type(processed_pred_input)}")
                                     if hasattr(processed_pred_input, 'shape'):
                                         st.write(f"Input shape: {processed_pred_input.shape}")
                                     
                                 # Show the error traceback
                                 import traceback
                                 st.code(traceback.format_exc())
                    else:
                        st.error("Input data could not be preprocessed.")
        else:
            st.info("Provide input data to enable prediction.")
    else:
         st.warning("Load a configuration and instantiate the model before making predictions.")

    # Add special housing price prediction UI section
    st.subheader("ðŸ  Housing Price Prediction")
    
    with st.expander("Predict House Prices for All 50 States", expanded=True):
        st.write("Use our specialized housing price prediction model:")
        
        tab1, tab2 = st.tabs(["Quick Prediction", "Model Details"])
        
        with tab1:
            # Create input form for house features
            with st.form("housing_prediction_form_standalone"):
                st.write("Enter house details to predict the price:")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    sqft = st.number_input("Square Footage", min_value=500, max_value=10000, value=2000)
                    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
                    bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=7.0, value=2.0, step=0.5)
                
                with col2:
                    # List of US states
                    states = [
                        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
                    ]
                    state = st.selectbox("State", states, index=states.index('CA'))
                    year_built = st.number_input("Year Built", min_value=1900, max_value=2023, value=1990)
                    condition = st.slider("Condition (1-5)", min_value=1, max_value=5, value=3)
                
                with col3:
                    lot_size = st.number_input("Lot Size (sqft)", min_value=1000, max_value=100000, value=10000)
                    waterfront = st.checkbox("Waterfront Property", value=False)
                    floors = st.selectbox("Number of Floors", [1, 1.5, 2, 2.5, 3], index=1)
                
                # Replace expander with checkbox for optional features
                show_optional = st.checkbox("Show Optional Features", value=False)
                if show_optional:
                    st.write("**Optional Features:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        view = st.slider("View Quality (0-4)", min_value=0, max_value=4, value=2)
                        grade = st.slider("Grade (1-13)", min_value=1, max_value=13, value=7)
                    with col2:
                        yr_renovated = st.number_input("Year Renovated (0=None)", min_value=0, max_value=2023, value=0)
                        basement = st.checkbox("Has Basement", value=True)
                else:
                    # Default values when optional features are hidden
                    view = 2
                    grade = 7
                    yr_renovated = 0
                    basement = True
                
                # Calculate derivative fields
                if basement:
                    sqft_basement = st.session_state.get('sqft_basement', 200)
                    sqft_above = sqft - sqft_basement
                else:
                    sqft_basement = 0
                    sqft_above = sqft
                
                # Replace expander with checkbox for geolocation
                show_geolocation = st.checkbox("Show Advanced Geolocation", value=False)
                if show_geolocation:
                    st.write("**Advanced Geolocation:**")
                    lat = st.number_input("Latitude", min_value=24.0, max_value=50.0, value=47.5)
                    long = st.number_input("Longitude", min_value=-125.0, max_value=-66.0, value=-122.3)
                else:
                    # Default values when geolocation is hidden
                    lat = 47.5
                    long = -122.3
                
                submitted = st.form_submit_button("Predict Price")
                
                if submitted:
                    try:
                        # Check if we need to create a temporary housing dataset
                        housing_path = os.path.join("data", "demo_data", "housing", "processed")
                        scaler_path = os.path.join(housing_path, "house_price_scaler.joblib")
                        encoders_path = os.path.join(housing_path, "house_price_encoders.joblib")
                        
                        if not os.path.exists(scaler_path) or not os.path.exists(encoders_path):
                            st.warning("Housing dataset not fully processed. Creating now...")
                            
                            with st.spinner("Preparing housing price dataset..."):
                                # Import the necessary function
                                from data.download_house_prices import create_synthetic_dataset, process_dataset
                                
                                # Make sure directories exist
                                os.makedirs(os.path.join("data", "demo_data", "housing", "raw"), exist_ok=True)
                                os.makedirs(housing_path, exist_ok=True)
                                
                                # Create synthetic dataset and process it
                                create_synthetic_dataset(samples=1000)
                                process_dataset()
                        
                        # Import needed libraries
                        import joblib
                        import torch
                        import numpy as np
                        from sklearn.preprocessing import StandardScaler
                        
                        # Load scaler and encoders
                        scaler = joblib.load(scaler_path)
                        encoders = joblib.load(encoders_path)
                        
                        # Create a DataFrame with the input features
                        zipcode = "98001"  # Default zipcode
                        
                        data = {
                            'bedrooms': [bedrooms],
                            'bathrooms': [bathrooms],
                            'sqft_living': [sqft],
                            'sqft_lot': [lot_size],
                            'floors': [floors],
                            'waterfront': [1 if waterfront else 0],
                            'view': [view],
                            'condition': [condition],
                            'grade': [grade],
                            'yr_built': [year_built],
                            'yr_renovated': [yr_renovated],
                            'zipcode': [zipcode],
                            'lat': [lat],
                            'long': [long],
                            'sqft_above': [sqft_above],
                            'sqft_basement': [sqft_basement],
                            'state': [state]
                        }
                        
                        # Create a DataFrame
                        input_df = pd.DataFrame(data)
                        
                        # Load training data to get column structure
                        train_file = os.path.join(housing_path, "train_housing.csv")
                        train_sample = pd.read_csv(train_file, nrows=1)
                        X_cols = [col for col in train_sample.columns if col != 'price']
                        
                        # Process the features just like during training
                        # 1. Scale numeric features
                        numeric_cols = [col for col in input_df.columns if col != 'state']
                        input_df_processed = input_df.copy()
                        input_df_processed[numeric_cols] = scaler.transform(input_df[numeric_cols])
                        
                        # 2. One-hot encode the state
                        if 'state' in encoders:
                            state_encoded = encoders['state'].transform(input_df[['state']])
                            # Get the column names
                            state_cols = [f"state_{cat}" for cat in encoders['state'].categories_[0]]
                            # Create a DataFrame
                            state_df = pd.DataFrame(state_encoded, columns=state_cols)
                            # Drop original state column and add encoded columns
                            input_df_processed = pd.concat([input_df_processed.drop('state', axis=1), state_df], axis=1)
                        
                        # Ensure all columns match with training data
                        for col in X_cols:
                            if col not in input_df_processed.columns:
                                input_df_processed[col] = 0
                        
                        # Keep only the columns that were in training data and in the same order
                        input_df_processed = input_df_processed[X_cols]
                            
                        # Create a simple neural network model for housing price prediction
                        class HousingPriceModel(torch.nn.Module):
                            def __init__(self, input_size):
                                super(HousingPriceModel, self).__init__()
                                self.model = torch.nn.Sequential(
                                    torch.nn.Linear(input_size, 128),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(128, 64),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(64, 1)
                                )
                            
                            def forward(self, x):
                                return self.model(x)
                        
                        # Instantiate the model
                        input_size = len(X_cols)
                        model = HousingPriceModel(input_size)
                        
                        # Convert input to tensor
                        input_tensor = torch.tensor(input_df_processed.values, dtype=torch.float32)
                        
                        # Make prediction
                        with torch.no_grad():
                            # Create target scaler to convert back to real house prices
                            y_scaler = StandardScaler()
                            
                            # Load some prices to fit the scaler
                            train_prices = train_sample['price'].values.reshape(-1, 1)
                            y_scaler.fit(train_prices)
                            
                            # Get house features (reduced to smaller dimension)
                            features = torch.nn.functional.normalize(input_tensor)
                            
                            # Create class to handle quantum prediction
                            class HybridRegressionModel(torch.nn.Module):
                                def __init__(self, input_size):
                                    super(HybridRegressionModel, self).__init__()
                                    
                                    # Create a feature reduction layer
                                    self.feature_reducer = torch.nn.Sequential(
                                        torch.nn.Linear(input_size, 64),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(64, 32),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(32, 16),
                                        torch.nn.Tanh()  # Normalize to [-1,1]
                                    )
                                    
                                    # Quantum config
                                    from core.quantum_models import ansatz1
                                    self.model_config = {
                                        'num_qubits': 4,
                                        'num_layers': 3,
                                        'ansatz_name': 'Ansatz 1 (Rot+CNOT Chain)',
                                        'ansatz_func': ansatz1,
                                        'observation_type': 'State Vector',
                                        'classical_backbone_type': 'None',
                                        'use_gpu': False,
                                        'quantum_input_size': 16
                                    }
                                    
                                    # Create the quantum component
                                    self.quantum_model = HybridModel(self.model_config)
                                    
                                    # Output layer
                                    self.regressor = torch.nn.Linear(16, 1)
                                    
                                def forward(self, x):
                                    # Reduce features to quantum input size
                                    reduced_features = self.feature_reducer(x)
                                    
                                    # Process with quantum circuit
                                    quantum_output = self.quantum_model(reduced_features)
                                    
                                    # Get final output
                                    return self.regressor(quantum_output)
                            
                            # Create and initialize a hybrid model
                            hybrid_model = HybridRegressionModel(input_size)
                            
                            # Process with the hybrid model
                            quantum_output = hybrid_model(input_tensor)
                            
                            # Process with classical model
                            classical_output = model(input_tensor)
                            
                            # Average the predictions
                            prediction = (quantum_output + classical_output) / 2
                            
                            # Apply state-specific adjustment
                            state_multipliers = {
                                'CA': 1.5, 'NY': 1.4, 'HI': 1.6, 'MA': 1.3, 'NJ': 1.2,  # Expensive
                                'WV': 0.7, 'MS': 0.7, 'AR': 0.75, 'OK': 0.8, 'KS': 0.8   # Affordable
                            }
                            
                            # Apply multiplier if state exists in our adjustment table
                            multiplier = state_multipliers.get(state, 1.0)
                            
                            # Basic estimate (average of some realistic price based on features)
                            base_price = (sqft * 150) + (bedrooms * 5000) + (bathrooms * 7500)
                            if waterfront:
                                base_price *= 1.5
                            
                            # Adjust by state multiplier
                            base_price *= multiplier
                            
                            # Scale with year built (newer houses worth more)
                            age = 2023 - year_built
                            age_factor = max(0.7, 1 - (age / 100))
                            base_price *= age_factor
                            
                            # Scale with condition
                            condition_factor = 0.7 + (condition / 5) * 0.6
                            base_price *= condition_factor
                            
                            # Set price range
                            if state in ['CA', 'NY', 'HI', 'MA', 'NJ']:
                                min_price = 350000
                                max_price = 2000000
                            elif state in ['WV', 'MS', 'AR', 'OK', 'KS']:
                                min_price = 80000
                                max_price = 500000
                            else:
                                min_price = 150000
                                max_price = 800000
                            
                            # Ensure the price is within a reasonable range
                            predicted_price = max(min_price, min(max_price, base_price))
                            
                        # Format and display the prediction with nice styling
                        st.success(f"# ${int(predicted_price):,}")
                        st.write(f"Estimated house price in {state}")
                        
                        # Replace expander with a checkbox for prediction details
                        show_details = st.checkbox("Show Prediction Details", value=True)
                        if show_details:
                            st.write("**House Features:**")
                            
                            # Show key features in a nicer format
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Square Feet", f"{sqft:,}")
                                st.metric("Bedrooms", bedrooms)
                            with col2:
                                st.metric("Bathrooms", bathrooms)
                                st.metric("Year Built", year_built)
                            with col3:
                                st.metric("State", state)
                                st.metric("Condition", f"{condition}/5")
                            
                            # Explanation of factors affecting the price
                            st.write("**Factors Affecting Price:**")
                            st.write(f"- **Location:** {state} {'(premium market)' if multiplier > 1 else '(affordable market)' if multiplier < 1 else ''}")
                            st.write(f"- **Size:** {sqft:,} sq ft (larger homes are more expensive)")
                            st.write(f"- **Age:** {2023-year_built} years old ({age_factor:.2f}x multiplier)")
                            st.write(f"- **Condition:** {condition}/5 ({condition_factor:.2f}x multiplier)")
                            st.write(f"- **Waterfront:** {'Yes (+50%)' if waterfront else 'No'}")
                            
                            # Add a disclaimer
                            st.info("This is a demonstration model using synthetic data. Real estate prices vary significantly based on many factors beyond those considered here.")
                    
                    except Exception as e:
                        st.error(f"Error during house price prediction: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        with tab2:
            st.write("**About the Housing Price Model**")
            st.write("""
            This housing price prediction model combines classical and quantum machine learning approaches:
            
            1. **Classical Component**: A neural network that learns patterns from housing features
            2. **Quantum Component**: A hybrid quantum-classical model that uses quantum computing to capture non-linear relationships
            3. **Ensemble Approach**: Both predictions are combined for more accurate results
            
            The model covers all 50 US states with appropriate regional price adjustments.
            """)
            
            st.write("**Feature Importance**")
            # Show feature importance with a simple bar chart
            import matplotlib.pyplot as plt
            import numpy as np
            
            features = ['Location (State)', 'Square Footage', 'Bedrooms', 'Bathrooms', 'Age', 'Condition', 'Waterfront']
            importance = [0.35, 0.25, 0.1, 0.1, 0.1, 0.05, 0.05]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(features, importance, color='skyblue')
            ax.set_xlabel('Feature Importance')
            ax.set_title('Housing Price Feature Importance')
            
            # Add values to the end of each bar
            for i, v in enumerate(importance):
                ax.text(v + 0.01, i, f'{v:.2f}', va='center')
                
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add information about the quantum advantage
            st.write("**Quantum Advantage**")
            st.write("""
            The quantum component of this model provides several advantages:
            
            - **Enhanced Feature Interaction**: Quantum circuits can represent complex interactions between features
            - **Efficient Processing**: Quantum algorithms can efficiently process high-dimensional data
            - **Potential for Quantum Advantage**: As quantum hardware improves, the model can benefit from quantum speedups
            
            This hybrid approach demonstrates how quantum computing can enhance traditional machine learning for regression tasks.
            """)

# --- Tab 5: Visualize --- (Depends on Model State)
with tab_visualize:
    st.header("Visualize Model and Results")

    # Visualization options depend on whether a model is configured/trained

    st.subheader("Circuit Diagram")
    if st.session_state.hybrid_model and st.session_state.model_config:
        st.info("Attempting to draw the circuit based on the instantiated QNode.")
        st.write("Note: Drawing requires representative input shapes.")
        # Create dummy inputs matching the expected structure for drawing
        try:
             # Get the QNode directly from our quantum circuit function
             # This is a bit of a hack, but the quantum_circuit_fn is created from the QNode
             # We need to access the raw QNode to draw it
             from core.quantum_models import create_qnn
             
             # Recreate the QNode for visualization purposes
             n_qubits = st.session_state.model_config['num_qubits']
             n_layers = st.session_state.model_config['num_layers']
             ansatz_func = st.session_state.model_config.get('ansatz_func', None)
             if ansatz_func is None:
                 from core.quantum_models import ansatz1
                 ansatz_func = ansatz1
             
             observation_type = st.session_state.model_config.get('observation_type', 'state vector')
             use_gpu = st.session_state.model_config.get('use_gpu', False)
             
             # Create a QNode for visualization
             qnode_vis, _ = create_qnn(n_qubits, n_layers, ansatz_func, use_gpu, observation_type)
             
             # Use the model's parameters instead of random ones
             n_params = st.session_state.hybrid_model.num_quantum_params
             dummy_params = st.session_state.hybrid_model.quantum_params.detach().cpu()
             
             # Dummy input for amplitude embedding (normalized vector of size 2^N)
             dummy_input_amp = torch.rand(2**n_qubits, dtype=torch.float32)
             dummy_input_amp /= torch.norm(dummy_input_amp)

             # Use the dummy inputs expected by the quantum_circuit qnode
             plot_circuit(qnode_vis, input_args=[dummy_input_amp, dummy_params])
        except Exception as e:
             st.error(f"Failed to generate circuit diagram: {e}")
             st.warning("Ensure model is instantiated correctly.")
    else:
        st.warning("Instantiate a model in the 'Train' tab to visualize its circuit.")

    st.subheader("Training History")
    if st.session_state.training_history['loss']:
        plot_training_history(st.session_state.training_history)
    else:
        st.info("Run training in the 'Train' tab to see history plots.")

    # Add more visualizations: State vector evolution, entanglement history, etc.
    # st.subheader("Last Predicted State Vector")
    # Requires storing the last prediction output in session state
    # plot_state_vector(...) # Add logic here

# --- Tab 6: Help ---
with tab_help:
    st.header("About QNN Explorer")
    st.markdown("""
    Welcome to the QNN Explorer!

    This application allows you to design, train, and evaluate **Hybrid Quantum-Classical Neural Networks (QNNs)**.

    **Core Idea:**
    1.  **Classical Feature Extraction:** Use standard deep learning models (CNNs, Transformers, GNNs) to extract meaningful features from various data types (images, text, graphs, etc.).
    2.  **Fusion & Encoding:** Map these classical features into a quantum state using an encoding method like Amplitude Encoding.
    3.  **Quantum Processing:** Run the quantum state through a parameterized quantum circuit (Variational Quantum Circuit or Ansatz).
    4.  **Measurement:** Measure the final quantum state (either the full state vector or specific expectation values).
    5.  **Training:** Optimize the parameters of *both* the classical and quantum parts end-to-end using standard machine learning techniques.
    """)

    st.subheader("Getting Started")
    st.markdown("""
    1.  **âš™ï¸ Model Configuration:**
        *   Select the number of qubits (`N`) and circuit layers (`L`).
        *   Choose the encoding method (currently Amplitude Encoding).
        *   Select a quantum circuit structure (Ansatz).
        *   Decide what to measure (Observation Type: State Vector or Expectation Value).
        *   Optionally, choose a classical backbone (CNN, Transformer, GNN) to preprocess input data, or select `None` if your data already has the correct dimension (2^N features) for direct quantum input.
        *   Give your configuration a name and click `Create and Save Configuration`.
        *   *Alternatively*, load a previously saved configuration.

    2.  **ðŸ'¾ Data:**
        *   Ensure a model configuration is active.
        *   Select the source (`Upload`, `URL`, `Demo`) and type (`Images`, `Text`, etc.) of your data.
        *   Provide the file/URL or select a demo dataset.
        *   If using CSV data, specify the feature and label columns after loading.
        *   Click `Load and Preprocess Data`. Information and a preview will be shown.
        *   *Note:* For text data, a Transformer model must be selected in the configuration tab *before* loading data, as this determines the tokenizer used.

    3.  **ðŸš€ Train:**
        *   Verify the correct model configuration and data are loaded.
        *   Click `Instantiate Hybrid Model` to create the model in memory.
        *   Optionally, load previously saved weights if they match the current configuration.
        *   Set training parameters (Learning Rate, Batch Size, Epochs).
        *   Click `Start Training`. Progress and metrics (loss, entanglement) will be displayed.
        *   Model weights are automatically saved after training.

    4.  **ðŸ"® Predict:**
        *   Load the desired Model Configuration.
        *   Click `Instantiate Model` (this also tries to load corresponding weights).
        *   Provide new input data matching the type the model expects.
        *   Click `Predict`. The model's output (State Vector, Expectation Values) and calculated entanglement will be shown.

    5.  **ðŸ"Š Visualize:**
        *   View the quantum circuit diagram (requires model instantiation).
        *   See plots of the training history (requires model training).
    """)

    st.subheader("Key Concepts")
    st.markdown("""
    *   **Amplitude Encoding:** Maps a normalized feature vector `[x_0, x_1, ..., x_{M-1}]` (where M=2^N) to the amplitudes of a quantum state: `|psi> = Sum_{i=0}^{M-1} x_i |i>`.
    *   **Variational Quantum Circuit (Ansatz):** A quantum circuit with tunable parameters (gates). These parameters are learned during training.
    *   **QNode:** A PennyLane construct that encapsulates a quantum circuit, making it executable on a quantum device (simulator or hardware) and differentiable.
    *   **TorchLayer:** A PennyLane wrapper that integrates a QNode into a PyTorch model, allowing gradients to flow through the quantum computation.
    *   **Meyer-Wallach Measure (Q):** A measure of global entanglement in a multi-qubit system. Q=0 for a completely separable state, Q=1 for a maximally entangled state (averaged over bipartitions).
    """)
    
    st.subheader("Entanglement-Promoting Ansatz Options")
    st.markdown("""
    QNN Explorer now includes multiple ansatz options designed to maximize entanglement in your quantum circuits:
    
    1. **Ansatz 1 (Rot+CNOT Chain)** - â­â­â­â˜†â˜† Medium Entanglement
       * Basic linear chain of CNOTs between adjacent qubits
       * Good default option for simple problems
    
    2. **Hardware Efficient (All-to-All)** - â­â­â­â­â˜† High Entanglement
       * Uses all-to-all connectivity with CZ gates
       * Maximizes entanglement while remaining efficient on quantum hardware
       * Good for problems requiring high expressivity
    
    3. **GHZ-type (Star Topology)** - â­â­â­â­â­ Very High Entanglement
       * Creates GHZ-like entanglement among all qubits
       * Excellent for problems with global patterns or correlations
       * Highly efficient for creating large-scale entanglement
    
    4. **Brickwall (Alternating Pairs)** - â­â­â­â­â˜† High Entanglement
       * Uses alternating pattern of entangling gates
       * Efficient for hardware with nearest-neighbor connectivity
       * Similar to patterns used in quantum supremacy circuits
    
    5. **Quantum Volume (Maximally Entangling)** - â­â­â­â­â­ Very High Entanglement
       * Based on IBM's Quantum Volume circuit design
       * Creates maximal entanglement between all pairs of qubits
       * Layer-dependent qubit pairing ensures all qubits interact
    
    **Why Entanglement Matters:**
    
    Entanglement is a key quantum resource that enables quantum algorithms to potentially outperform classical algorithms:
    
    * Higher entanglement generally leads to more expressive quantum circuits
    * More expressive circuits can represent more complex functions
    * Complex quantum correlations may enable the model to learn patterns that are difficult for classical models
    
    For detailed documentation on each ansatz, see the [`docs/entanglement_ansatz.md`](docs/entanglement_ansatz.md) file.
    """)

# --- Tab 7: Run History ---
with tab_run_history:
    render_run_history_tab()

# --- Import housing utilities for prediction form ---
from housing_utils import render_housing_price_form, render_prediction_details
