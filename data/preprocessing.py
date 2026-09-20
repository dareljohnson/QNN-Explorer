import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2 # For video
import os
import joblib # For saving/loading scalers
from sklearn.preprocessing import StandardScaler # For consistent scaling
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer # For text features
import json # For metadata
import pandas as pd # For creating feature subsets
# from transformers import AutoTokenizer # If needed directly here
# from torch_geometric.data import Data # If needed directly here

# --- Path for saved preprocessing artifacts ---
SCALERS_DIR = os.path.join("saved_models", "scalers")
VECTORIZERS_DIR = os.path.join("saved_models", "vectorizers")
os.makedirs(SCALERS_DIR, exist_ok=True)
os.makedirs(VECTORIZERS_DIR, exist_ok=True)

# --- Text tokenization length ---
# Text is truncated/padded to this length. Defined once so the app's tokenizer
# call and preprocess_text() cannot drift apart (they disagreed: 128 vs 512).
MAX_TEXT_LENGTH = 128

# --- Image Preprocessing ---
def preprocess_image(image_input, target_size=(224, 224)):
    """Preprocesses a single image (PIL Image or path)."""
    if isinstance(image_input, str):
        try:
            img = Image.open(image_input).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_input}")
            return None
        except Exception as e:
            print(f"Error opening image {image_input}: {e}")
            return None
    elif isinstance(image_input, Image.Image):
        img = image_input.convert('RGB')
    else:
        # Add handling for other inputs if necessary (e.g., numpy array)
        print("Unsupported image input type.")
        return None

    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        # Normalize using ImageNet stats, common for pretrained models
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(img)

# --- Text Preprocessing ---
def preprocess_text(text_input, tokenizer, max_length=MAX_TEXT_LENGTH):
    """Preprocesses text using a Hugging Face tokenizer."""
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided for text preprocessing.")

    # Tokenize the text
    encoding = tokenizer(text_input,
                         add_special_tokens=True, # Add [CLS] and [SEP]
                         max_length=max_length,
                         padding='max_length',    # Pad to max_length
                         truncation=True,         # Truncate if longer
                         return_attention_mask=True,
                         return_tensors='pt')      # Return PyTorch tensors

    # The encoding dict contains 'input_ids' and 'attention_mask'
    return encoding

# --- Graph Preprocessing ---
def preprocess_graph(graph_data):
    """
    Placeholder for graph preprocessing.
    This would typically involve:
    - Loading graph data (e.g., from PDB, SMILES, network files).
    - Extracting node features, edge connections.
    - Creating a torch_geometric Data or Batch object.
    Args:
        graph_data: Raw graph data (format depends on source).
    Returns:
        A torch_geometric.data.Data object or similar representation.
    """
    print("[Placeholder] Graph preprocessing needs implementation based on data format.")
    # Example structure (highly dependent on input)
    # node_features = ...
    # edge_index = ...
    # data = Data(x=node_features, edge_index=edge_index)
    # return data
    return None

# --- Video Preprocessing ---
def preprocess_video(video_path, num_frames=16, target_size=(112, 112)):
    """Samples frames from a video and preprocesses them."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames:
            print(f"Warning: Video has only {total_frames} frames, less than requested {num_frames}")
            indices = np.arange(total_frames)
        else:
            # Simple uniform sampling
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        count = 0
        frame_idx = 0
        while cap.isOpened() and len(frames) < len(indices):
            ret, frame = cap.read()
            if not ret:
                break
            if count == indices[frame_idx]:
                # Convert frame from BGR (OpenCV) to RGB (PIL/PyTorch)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                processed_frame = preprocess_image(img, target_size=target_size) # Use image preprocessor
                frames.append(processed_frame)
                frame_idx += 1
            count += 1

        cap.release()

        if not frames:
            print(f"Error: No frames could be extracted from {video_path}")
            return None

        # Stack frames into a tensor (C, T, H, W) or (T, C, H, W)
        # Common format for video models is (C, T, H, W)
        video_tensor = torch.stack(frames).permute(1, 0, 2, 3) # T,C,H,W -> C,T,H,W
        return video_tensor

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        if cap.isOpened():
            cap.release()
        return None

# --- CSV Preprocessing ---
def preprocess_csv(df, feature_cols, label_col=None, scaler_name="scaler.joblib", is_prediction=False, text_cols=None, text_vectorizer="tfidf", max_text_features=100):
    """
    Preprocess CSV data for model input.
    
    Args:
        df: DataFrame containing the data
        feature_cols: List of column names to use as features
        label_col: Column name to use as label (or None for no labels)
        scaler_name: Name to use when saving/loading the scaler
        is_prediction: Whether this is for prediction (load scaler) or training (fit scaler)
        text_cols: List of columns containing text data to be vectorized
        text_vectorizer: Type of text vectorizer to use ('tfidf' or 'count')
        max_text_features: Maximum number of features to extract from each text column
        
    Returns:
        Tuple of (features_tensor, labels_tensor)
    """
    if df is None or len(df) == 0:
        print("Warning: Empty DataFrame provided for preprocessing")
        return None, None
    
    # Default path to save/load scalers
    scaler_dir = os.path.join("data", "scalers")
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_path = os.path.join(scaler_dir, scaler_name)
    
    # Storage for vectorizers
    vectorizer_dir = os.path.join("data", "vectorizers")
    os.makedirs(vectorizer_dir, exist_ok=True)
    
    try:
        # Make a copy of the input dataframe
        working_df = df.copy()
        # Do not mutate the caller's feature column list.
        feature_cols = list(feature_cols) if feature_cols is not None else []
        
        # Handle text columns if specified
        if text_cols and len(text_cols) > 0:
            # Check if all text columns exist in the dataframe
            missing_cols = set(text_cols) - set(working_df.columns)
            if missing_cols:
                print(f"Warning: Text columns {missing_cols} not found in dataframe")
                text_cols = [col for col in text_cols if col in working_df.columns]
            
            if len(text_cols) > 0:
                print(f"Processing text columns: {text_cols}")
                
                for col in text_cols:
                    # Skip non-text columns
                    if col not in working_df.columns:
                        continue
                    
                    if pd.api.types.is_numeric_dtype(working_df[col]):
                        print(f"Skipping numeric column {col} that was specified as text")
                        continue
                    
                    # Convert column to string in case it contains non-string objects
                    working_df[col] = working_df[col].astype(str)
                    
                    # Define vectorizer path
                    vectorizer_path = os.path.join(vectorizer_dir, f"{text_vectorizer}_{col}_{max_text_features}.joblib")
                    
                    if is_prediction and os.path.exists(vectorizer_path):
                        # Load existing vectorizer for prediction
                        vectorizer = joblib.load(vectorizer_path)
                        print(f"Loaded existing {text_vectorizer} vectorizer for column {col}")
                    else:
                        # Create and fit new vectorizer for training
                        if text_vectorizer == "tfidf":
                            vectorizer = TfidfVectorizer(max_features=max_text_features)
                        else:  # "count"
                            vectorizer = CountVectorizer(max_features=max_text_features)
                        
                        # Fit the vectorizer
                        vectorizer.fit(working_df[col])
                        
                        # Save the vectorizer
                        joblib.dump(vectorizer, vectorizer_path)
                        print(f"Fitted and saved {text_vectorizer} vectorizer for column {col}")
                    
                    # Transform the text to feature matrix
                    try:
                        text_features = vectorizer.transform(working_df[col]).toarray()
                        
                        # Create new feature columns for the text features
                        for i in range(text_features.shape[1]):
                            feat_name = f"{col}_text_{i}"
                            working_df[feat_name] = text_features[:, i]
                            
                            # Register the new text features for BOTH training and
                            # prediction. Previously they were only registered for
                            # prediction, so text columns were silently ignored
                            # during training.
                            if feat_name not in feature_cols:
                                feature_cols.append(feat_name)
                        
                        print(f"Added {text_features.shape[1]} features for text column {col}")
                    except Exception as e:
                        print(f"Error vectorizing text column {col}: {e}")
        
        # Extract features and convert to numpy array
        numeric_feature_cols = [col for col in feature_cols 
                              if col in working_df.columns and pd.api.types.is_numeric_dtype(working_df[col])]
        
        # Check if we have any usable numeric columns
        if not numeric_feature_cols:
            print("Warning: No numeric feature columns found or specified.")
            # Try to extract any numeric columns from dataframe
            all_numeric_cols = working_df.select_dtypes(include=['number']).columns.tolist()
            if all_numeric_cols:
                print(f"Using available numeric columns: {all_numeric_cols}")
                numeric_feature_cols = all_numeric_cols
            else:
                return None, None
        
        # Use only numeric columns for features
        X = working_df[numeric_feature_cols].copy()
        
        # Handle missing values in features
        X.fillna(0, inplace=True)  # Simple imputation with zeros
        
        # Check for infinite values and replace with large numbers
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        
        # Convert to numpy array
        features_array = X.values
        
        # Scale features
        if is_prediction and os.path.exists(scaler_path):
            # Load existing scaler for prediction
            scaler = joblib.load(scaler_path)
            print(f"Loaded existing scaler from {scaler_path}")
        else:
            # Create and fit new scaler
            scaler = StandardScaler()
            scaler.fit(features_array)
            
            # Save the scaler
            joblib.dump(scaler, scaler_path)
            print(f"Fitted and saved scaler to {scaler_path}")
        
        # Transform the features
        scaled_features = scaler.transform(features_array)
        
        # Process labels if provided
        labels_tensor = None
        if label_col and label_col in working_df.columns:
            y = working_df[label_col].copy()
            
            # Check if labels are numeric (for regression) or categorical (for classification)
            if pd.api.types.is_numeric_dtype(y):
                # For regression: scale the targets to prevent very large loss values
                # Create a separate scaler for target values
                target_scaler_path = os.path.join(scaler_dir, f"target_{scaler_name}")
                
                if is_prediction and os.path.exists(target_scaler_path):
                    # Load existing target scaler for prediction
                    target_scaler = joblib.load(target_scaler_path)
                    print(f"Loaded existing target scaler from {target_scaler_path}")
                else:
                    # Create and fit new target scaler
                    target_scaler = StandardScaler()
                    y_array = y.values.reshape(-1, 1)
                    target_scaler.fit(y_array)
                    
                    # Save the target scaler
                    joblib.dump(target_scaler, target_scaler_path)
                    print(f"Fitted and saved target scaler to {target_scaler_path}")
                
                # Scale the target values
                y_array = y.values.reshape(-1, 1)
                scaled_labels = target_scaler.transform(y_array).flatten()
                
                # Convert to tensor
                labels_tensor = torch.tensor(scaled_labels, dtype=torch.float32)
            else:
                # For classification: factorize categorical labels
                labels, _ = pd.factorize(y)
                labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Convert features to tensor
        features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        
        return features_tensor, labels_tensor
    
    except Exception as e:
        import traceback
        print(f"Error preprocessing CSV data: {e}")
        print(traceback.format_exc())
        return None, None

# --- Utility to Load a Saved Scaler ---
def get_saved_scalers():
    """Lists available saved scalers."""
    scalers = [f for f in os.listdir(SCALERS_DIR) if f.endswith('.joblib')]
    return scalers

def delete_scaler(scaler_name):
    """Deletes a saved scaler."""
    scaler_path = os.path.join(SCALERS_DIR, scaler_name)
    meta_path = os.path.join(SCALERS_DIR, scaler_name.replace('.joblib', '_meta.json'))
    
    if os.path.exists(scaler_path):
        os.remove(scaler_path)
        print(f"Deleted scaler: {scaler_path}")
    
    if os.path.exists(meta_path):
        os.remove(meta_path)
        print(f"Deleted scaler metadata: {meta_path}")

# --- Text Extraction from CSV ---
def encode_classification_labels(labels):
    """Turn a column of labels into class indices.

    Categorical labels (e.g. 'cs.CV') must be factorized: ``torch.tensor()`` on
    raw strings fails with "too many dimensions 'str'". Numeric labels are
    passed through unchanged so regression targets keep their values. Mirrors
    what ``preprocess_csv`` does for the vector path.

    Classes are sorted (``sort=True``) so the index assigned to each class is
    deterministic. With the default order-of-appearance mapping, extracting the
    same data twice - or saving a model and predicting on freshly extracted data
    - could silently permute the classes and turn a 93% model into a 42% one.

    Args:
        labels: sequence of labels (strings, ints or floats)

    Returns:
        (labels_tensor, class_names) -- class_names is None for numeric labels
    """
    series = pd.Series(labels)
    if pd.api.types.is_numeric_dtype(series):
        return torch.tensor(labels), None

    codes, class_names = pd.factorize(series, sort=True)
    return torch.tensor(codes, dtype=torch.long), [str(name) for name in class_names]


def extract_text_from_csv(df, text_column, label_column=None, max_samples=None):
    """
    Extract text from a specified column in a DataFrame.
    
    Args:
        df: DataFrame containing the text
        text_column: Name of the column containing text
        label_column: Optional name of the column containing labels
        max_samples: Maximum number of samples to extract
        
    Returns:
        texts: List of extracted text strings
        labels: List of corresponding labels (or None if no label_column)
    """
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found in DataFrame")
        return None, None
        
    # Filter out missing values in the text column
    valid_df = df.dropna(subset=[text_column])
    
    # Apply max_samples limit if specified
    if max_samples is not None and max_samples < len(valid_df):
        # Select random subset of rows
        valid_df = valid_df.sample(max_samples, random_state=42)
    
    # Extract texts
    texts = valid_df[text_column].astype(str).tolist()
    
    # Extract labels if label_column is provided
    labels = None
    if label_column is not None and label_column in df.columns:
        labels = valid_df[label_column].tolist()
        
    print(f"Extracted {len(texts)} texts from column '{text_column}'")
    if labels:
        print(f"Extracted {len(labels)} labels from column '{label_column}'")
        
    return texts, labels 