import streamlit as st
import pandas as pd
from PIL import Image
import requests
from io import BytesIO, StringIO
import os
import torch
import numpy as np
import glob
import io
from collections import defaultdict
from torchvision import transforms

from .preprocessing import preprocess_image, preprocess_text, preprocess_graph, preprocess_video, preprocess_csv
from .path_utils import ensure_path_sep, make_path
# Need to import tokenizer for text preprocessing if handling text
# from transformers import AutoTokenizer

# Assume tokenizer is loaded globally or passed appropriately for text
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') # Example

# --- Demo Data Paths (Relative to project root) ---
DEMO_DATA_DIR = os.path.join("data", "demo_data")
DEMO_DATASETS = {
    # --- Image classification ---
    "Cats vs Dogs (Image Classification)": {
        "type": "Images",
        "path": os.path.join(DEMO_DATA_DIR, "cats_dogs"),
    },
    "MNIST Digits (Image Classification)": {
        "type": "Images",
        "path": os.path.join(DEMO_DATA_DIR, "mnist"),
    },
    # --- CSV: regression / classification ---
    "Kaggle Housing Prices (Regression)": {
        "type": "CSV",
        "path": ensure_path_sep(os.path.join(DEMO_DATA_DIR, "kaggle", "Housing.csv")),
        "description": ("Real housing prices (545 rows, 13 columns) for regression: "
                        "area, bedrooms, bathrooms, stories, parking and yes/no amenities"),
        "features": ["area", "bedrooms", "bathrooms", "stories", "parking"],
        "target": "price",
        "model_type": "Regression"
    },
    "Kaggle Housing Bands (Binary Classification)": {
        "type": "CSV",
        "path": ensure_path_sep(os.path.join(DEMO_DATA_DIR, "kaggle", "Housing_banded.csv")),
        "description": ("The same 545 rows with prices split at the median into two classes "
                        "(price_band). A logistic regression scores 86.1% on this target, "
                        "whereas the three-band version only reaches 73.7%."),
        "features": ["area", "bedrooms", "bathrooms", "stories", "parking"],
        "target": "price_band",
        "model_type": "Classification"
    },
    "USA Housing Prices (Regression)": {
        "type": "CSV",
        "path": ensure_path_sep(os.path.join(DEMO_DATA_DIR, "housing", "processed")),
        "description": "Housing price data across all 50 US states for regression analysis",
        "features": ["bedrooms", "bathrooms", "sqft_living", "state", "yr_built", "condition"],
        "target": "price",
        "model_type": "Regression"
    },
    "ArXiv Scientific Papers (CSV Classification)": {
        "type": "CSV",
        "path": ensure_path_sep(os.path.join(DEMO_DATA_DIR, "csv", "arxiv_sample.csv")),
        "description": ("Sample of scientific paper metadata (title, authors, summary) "
                        "for CSV/text classification. Target column: 'category'."),
        "target": "category",
    },
}

# --- Utility functions for file validation ---
def validate_file_type(file, expected_type):
    """
    Validates if a file matches the expected data type.
    
    Args:
        file: The uploaded file object
        expected_type: The expected data type (Images, Text, CSV, Video, Graph)
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if file is None:
        return False, "No file uploaded"
    
    file_type = file.type
    file_name = file.name.lower()
    
    # Validate based on expected type
    if expected_type == "Images":
        if not file_type.startswith('image/'):
            return False, f"Expected image file, got {file_type}"
    
    elif expected_type == "Text":
        if not (file_type.startswith('text/') or file_type == 'application/octet-stream'):
            # Check extension as fallback for .txt, .md etc.
            if not file_name.endswith(('.txt', '.md', '.log')):
                return False, f"Expected text file, got {file_type}"
    
    elif expected_type == "CSV":
        if not (file_type == 'text/csv' or file_type == 'application/vnd.ms-excel'):
            # Check extension as fallback
            if not file_name.endswith('.csv'):
                return False, f"Expected CSV file, got {file_type}"
    
    elif expected_type == "Video":
        if not file_type.startswith('video/'):
            return False, f"Expected video file, got {file_type}"
    
    elif expected_type == "Graph":
        # Graph formats vary widely (.pdb, .sdf, .edge, .txt, .csv, .json...)
        # For now, accept common graph file extensions
        if not file_name.endswith(('.pdb', '.sdf', '.edge', '.graphml', '.gml')):
            # This is just a warning since graph formats vary
            st.warning(f"File extension may not be a standard graph format: {file_name}")
    
    return True, None

def get_sample_files(data_type, dir_path=None):
    """
    Get a list of sample files of the specified type from the demo directory.
    
    Args:
        data_type: Type of data to look for (Images, Text, CSV, Video, Graph)
        dir_path: Optional directory path, defaults to demo data directory
        
    Returns:
        list: List of file paths matching the data type
    """
    if dir_path is None:
        dir_path = DEMO_DATA_DIR
    
    # Ensure directory exists
    if not os.path.exists(dir_path):
        print(f"Directory does not exist: {dir_path}")
        return []
    
    # Define file extensions for each data type
    extensions = {
        "Images": ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"],
        "Text": ["*.txt", "*.md", "*.log"],
        "CSV": ["*.csv"],
        "Video": ["*.mp4", "*.avi", "*.mov", "*.mkv"],
        "Graph": ["*.pdb", "*.sdf", "*.edge", "*.graphml", "*.gml"]
    }
    
    # Get files matching the data type
    if data_type not in extensions:
        print(f"Unknown data type: {data_type}")
        return []
    
    # Search for files with matching extensions
    matching_files = []
    for ext in extensions[data_type]:
        pattern = os.path.join(dir_path, "**", ext)
        matching_files.extend(glob.glob(pattern, recursive=True))
    
    return matching_files

# --- Data Loading Functions ---

def load_data(source, data_type, uploaded_file=None, url=None, demo_dataset_name=None):
    """Loads data based on source and type, then preprocesses it."""
    raw_data = None
    processed_data = None
    preview_data = None # For displaying in Streamlit
    data_info = {} # To store metadata like filename, shape etc.

    try:
        # 1. Get Raw Data
        if source == "Upload":
            if uploaded_file is not None:
                raw_data = uploaded_file
                data_info['filename'] = uploaded_file.name
                # Read content based on type for preview/preprocessing

                # --- File Type Validation ---
                file_type = uploaded_file.type # e.g., 'image/png', 'text/csv', 'video/mp4'
                type_match = True
                expected_prefix = ""
                if data_type == "Images" and not file_type.startswith('image/'):
                    type_match = False
                    expected_prefix = "image/"
                elif data_type == "Text" and not (file_type.startswith('text/') or file_type == 'application/octet-stream'):
                    # Check extension as fallback? .txt, .md etc.
                    # For now, broader check
                    if not uploaded_file.name.lower().endswith(('.txt', '.md', '.log')):
                        type_match = False
                        expected_prefix = "text/"
                elif data_type == "CSV" and not (file_type == 'text/csv' or file_type == 'application/vnd.ms-excel'):
                    # Check extension as fallback?
                    if not uploaded_file.name.lower().endswith('.csv'):
                        type_match = False
                        expected_prefix = "text/csv"
                elif data_type == "Video" and not file_type.startswith('video/'):
                    type_match = False
                    expected_prefix = "video/"
                elif data_type == "Graph":
                    # Graph formats vary widely (.pdb, .sdf, .edge, .txt, .csv, .json...)
                    # No reliable MIME type check. Skip validation for graphs for now.
                    pass

                if not type_match:
                    st.error(f"Type mismatch: Selected data type is '{data_type}', but uploaded file type is '{file_type}'. Expected prefix: '{expected_prefix}'.", icon="🚨")
                    # Return error info but clear other data
                    st.session_state.preview_data = None  # Reset preview data
                    return None, None, {"filename": uploaded_file.name, "error": "File type mismatch"}
                # --- End Validation ---

                if data_type in ["Images", "Video"]:
                    # Keep as file-like object for now
                    # Let preprocessing handle opening from buffer/path
                    pass  # No special handling needed, preprocessing will handle

                elif data_type == "Text":
                    # Assuming single text file or process multiple later
                    raw_data = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
                elif data_type == "CSV":
                    raw_data = pd.read_csv(uploaded_file)
                elif data_type == "Graph":
                    # Handle graph file upload (e.g., edge list, adjacency matrix)
                    print("[Placeholder] Graph file upload handling needed.")
                    # raw_data = uploaded_file.getvalue().decode("utf-8") # Example
                    pass # Keep raw_data as file object for now
            else:
                st.error("Please upload a file.")
                return None, None, None

        elif source == "URL":
            if url:
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    data_info['url'] = url
                    if data_type == "Images":
                        raw_data = Image.open(BytesIO(response.content))
                    elif data_type == "Text":
                        raw_data = response.text
                    elif data_type == "CSV":
                        raw_data = pd.read_csv(StringIO(response.text))
                    elif data_type == "Video":
                        # Downloading large videos might be slow/memory intensive
                        # Consider saving to a temporary file
                        print("[Warning] Downloading video from URL. May be slow.")
                        # Save to temp file for cv2.VideoCapture
                        temp_video_path = "temp_video_from_url.mp4"
                        with open(temp_video_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        raw_data = temp_video_path # Pass path to preprocessor
                        data_info['temp_file'] = temp_video_path
                    elif data_type == "Graph":
                         print("[Placeholder] Graph data loading from URL needed.")
                         # raw_data = response.text # Example
                    else:
                         st.error(f"Data type '{data_type}' not supported for URL source yet.")
                         return None, None, None

                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching data from URL: {e}")
                    return None, None, None
                except Exception as e:
                    st.error(f"Error processing data from URL: {e}")
                    return None, None, None
            else:
                st.error("Please enter a URL.")
                return None, None, None

        elif source == "Demo":
            if demo_dataset_name and demo_dataset_name in DEMO_DATASETS:
                demo_info = DEMO_DATASETS[demo_dataset_name]
                if data_type != demo_info["type"]:
                    st.error(f"Selected demo dataset '{demo_dataset_name}' is of type '{demo_info['type']}', not '{data_type}'.")
                    return None, None, None
                file_path = demo_info["path"]
                # Ensure path has consistent separators
                file_path = ensure_path_sep(file_path)
                data_info['demo_name'] = demo_dataset_name
                data_info['path'] = file_path
                if not os.path.exists(file_path):
                    st.error(f"Demo data file not found: {file_path}. Please ensure demo data is placed correctly.")
                    # Optionally provide instructions to download/create demo data
                    return None, None, None

                # Load demo data based on type
                if data_type == "Images":
                    # Assume MNIST sample is a directory of images
                    # For simplicity, load the first image found
                    try:
                        # Special handling for MNIST dataset
                        if demo_dataset_name == "MNIST Digits (Image Classification)":
                            from data.mnist_loader import load_mnist_dataset, download_mnist_dataset
                            
                            # Check if the dataset exists, download if not
                            mnist_path = file_path
                            
                            # Print debug info
                            st.info(f"Looking for MNIST data in: {mnist_path}")
                            if os.path.exists(os.path.join(mnist_path, "train")):
                                st.info(f"Found train directory: {os.path.join(mnist_path, 'train')}")
                                
                                # Check if any of the digit folders exist
                                for digit in range(10):
                                    digit_dir = os.path.join(mnist_path, "train", str(digit))
                                    if os.path.exists(digit_dir):
                                        num_files = len([f for f in os.listdir(digit_dir) if f.endswith('.png')])
                                        st.info(f"Digit {digit} folder has {num_files} images")
                                
                                # Load the dataset
                                images, labels = load_mnist_dataset(mnist_path)
                                if images is not None:
                                    # Set both raw_data and processed_data directly
                                    raw_data = images  # Skip normal preprocessing
                                    processed_data = images  # Already processed by load_mnist_dataset
                                    preview_data = f"MNIST dataset: {len(images)} images, 10 classes"
                                    # Store labels directly
                                    st.session_state['csv_labels'] = labels
                                    return processed_data, preview_data, data_info
                                else:
                                    st.error("Failed to load MNIST images. Check console for details.")
                                    return None, None, None
                            else:
                                st.warning(f"MNIST dataset structure not found. Expected train directory at {os.path.join(mnist_path, 'train')}")
                                st.info("Attempting to download MNIST dataset...")
                                download_mnist_dataset(mnist_path)
                                st.info("Download complete. Please refresh and try loading again.")
                                return None, None, None
                        else:
                            # Regular image loading for other datasets
                            first_image = [f for f in os.listdir(file_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][0]
                            raw_data = os.path.join(file_path, first_image) # Pass path to preprocessor
                    except IndexError:
                        st.error(f"No image files found in demo directory: {file_path}")
                        return None, None, None
                elif data_type == "Text":
                    # Assume IMDB is CSV with a text column
                    df = pd.read_csv(file_path)
                    # Take a sample text for preview/processing (e.g., first review)
                    raw_data = df['review'].iloc[0] # Assuming 'review' column
                    preview_data = df.head() # Preview the dataframe
                elif data_type == "CSV":
                    # Handle housing dataset special case - path points to directory, not file
                    if demo_dataset_name == "USA Housing Prices (Regression)":
                        # Use the train file from the processed directory
                        housing_file = os.path.join(file_path, "train_housing.csv")
                        if os.path.exists(housing_file):
                            raw_data = pd.read_csv(housing_file)
                        else:
                            # Try to create the dataset if it doesn't exist
                            try:
                                from data.download_house_prices import create_synthetic_dataset, process_dataset
                                # Make sure directories exist
                                os.makedirs(file_path, exist_ok=True)
                                os.makedirs(os.path.join(DEMO_DATA_DIR, "housing", "raw"), exist_ok=True)
                                # Create and process the dataset
                                create_synthetic_dataset(samples=1000)
                                process_dataset()
                                # Now try to load the file again
                                if os.path.exists(housing_file):
                                    raw_data = pd.read_csv(housing_file)
                                else:
                                    st.error(f"Failed to create housing dataset: {housing_file}")
                                    return None, None, data_info
                            except Exception as e:
                                st.error(f"Error creating housing dataset: {e}")
                                return None, None, data_info
                    else:
                        # Regular CSV file loading
                        raw_data = pd.read_csv(file_path)
                elif data_type == "Video":
                    raw_data = file_path # Pass path to preprocessor
                elif data_type == "Graph":
                    print(f"[Placeholder] Loading demo graph data from {file_path}")
                    raw_data = file_path # Pass path or content to preprocessor
            else:
                st.error("Please select a valid demo dataset.")
                return None, None, None

        # 2. Preprocess Data
        st.write(f"Preprocessing data (Type: {data_type})...")
        if raw_data is None:
             st.error("No raw data loaded.")
             return None, None, None

        # --- Handle CSV Loading Separately ---
        if data_type == "CSV":
            # For CSV, load_data only loads the raw DataFrame for preview and column selection.
            # Preprocessing happens later in app.py after columns are selected.
            preview_data = raw_data # raw_data is already the DataFrame
            
            # Important fix: Create an initial tensor representation of numeric data
            # This ensures the Train tab recognizes data is loaded even before preprocessing
            if isinstance(raw_data, pd.DataFrame):
                numeric_cols = raw_data.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    temp_data = raw_data[numeric_cols].values
                    processed_data = torch.tensor(temp_data, dtype=torch.float32)
                    data_info['needs_preprocessing'] = True
                    data_info['available_columns'] = raw_data.columns.tolist()
                    data_info['initial_representation'] = True
                    st.info("CSV loaded. Please select feature and label columns below, then click Preprocess.")
                else:
                    st.warning("No numeric columns found in CSV. You'll need to preprocess data to extract features.")
                    # Create a minimal tensor to ensure loaded_data is not None
                    processed_data = torch.zeros((len(raw_data), 1), dtype=torch.float32)
                    data_info['needs_preprocessing'] = True
                    data_info['available_columns'] = raw_data.columns.tolist()
                    data_info['initial_representation'] = True
            else:
                processed_data = None # Only if raw_data is not a DataFrame
                st.info("CSV loaded. Please select feature and label columns below, then click Preprocess.")

        elif data_type == "Images":
            # Handle single image or batch later
            processed_data = preprocess_image(raw_data) # raw_data can be path or PIL image
            preview_data = raw_data if isinstance(raw_data, Image.Image) else Image.open(raw_data)
        elif data_type == "Text":
            # Requires tokenizer - handle state management in app.py
            tokenizer = st.session_state.get('tokenizer') # Get tokenizer from session state
            if tokenizer:
                 processed_data = preprocess_text(raw_data, tokenizer) # raw_data is string
                 preview_data = raw_data[:500] + "..." if isinstance(raw_data, str) else raw_data
            else:
                 st.error("Tokenizer not available for text preprocessing. Select a Transformer model first?")
                 return None, None, None
        elif data_type == "CSV":
            # Need to know which columns are features/labels - get from UI
            feature_cols = st.session_state.get('csv_feature_cols')
            label_col = st.session_state.get('csv_label_col')
            if feature_cols:
                processed_data, labels = preprocess_csv(raw_data, feature_cols, label_col)
                # Combine features and labels for dataset? Return separately?
                # For now, returning features tensor. Labels handled in training loop.
                st.session_state['csv_labels'] = labels # Store labels
                preview_data = raw_data.head()
            else:
                st.error("CSV feature columns not selected.")
                return None, None, None
        elif data_type == "Video":
            processed_data = preprocess_video(raw_data) # raw_data is path
            preview_data = f"Video file: {data_info.get('filename') or data_info.get('path') or data_info.get('temp_file')}"
            # Clean up temp file if created from URL
            if 'temp_file' in data_info and os.path.exists(data_info['temp_file']):
                os.remove(data_info['temp_file'])
                print(f"Removed temporary video file: {data_info['temp_file']}")
        elif data_type == "Graph":
            processed_data = preprocess_graph(raw_data) # raw_data is path or content
            preview_data = f"Graph data source: {data_info.get('filename') or data_info.get('path')}"

        if processed_data is None and data_type not in ["Graph", "CSV"]: # Graph is placeholder, CSV processed later
             st.error(f"Preprocessing failed for {data_type}.")
             return None, None, data_info

        # Store info
        data_info['processed_shape'] = processed_data.shape if hasattr(processed_data, 'shape') else "N/A" # May be N/A for CSV initially
        data_info['data_type'] = data_type

        st.success(f"Data source loaded successfully! ({data_type})")
        return processed_data, preview_data, data_info

    except Exception as e:
        st.error(f"An unexpected error occurred during data loading/preprocessing: {e}")
        # Clean up temp file in case of error
        if 'temp_file' in data_info and os.path.exists(data_info['temp_file']):
           try:
               os.remove(data_info['temp_file'])
               print(f"Removed temporary video file after error: {data_info['temp_file']}")
           except OSError as ose:
               print(f"Error removing temp file {data_info['temp_file']}: {ose}")
        return None, None, data_info 