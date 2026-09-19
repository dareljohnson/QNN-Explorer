#!/usr/bin/env python
"""
Download and prepare USA Housing Price dataset for QNN Explorer.

This script downloads the USA housing dataset from Kaggle (if available locally),
or creates a synthetic demo dataset if the Kaggle dataset cannot be accessed.

The dataset includes housing data across 50 states with features like:
- Square footage
- Number of bedrooms/bathrooms
- Location (state, zip code)
- Year built
- Various other features
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import zipfile
import requests
from io import BytesIO
import json
import shutil

# Define paths
DATA_DIR = os.path.join("data", "demo_data", "housing")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
INFO_FILE = os.path.join(DATA_DIR, "dataset_info.json")

# Create directories
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def download_kaggle_dataset():
    """
    Attempt to download USA house prices dataset from Kaggle.
    Requires kaggle API credentials to be set up.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Import kaggle (optional dependency)
        import kaggle
        
        print("Attempting to download dataset from Kaggle...")
        
        # Download the dataset
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'fratzcan/usa-house-prices', 
            path=RAW_DATA_DIR, 
            unzip=True
        )
        
        # Verify the dataset was downloaded
        expected_file = os.path.join(RAW_DATA_DIR, "USA-house-prices.csv")
        if os.path.exists(expected_file):
            print(f"Successfully downloaded dataset to {expected_file}")
            return True
        else:
            print(f"Download appeared successful but file {expected_file} not found")
            return False
        
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        print("This may happen if:")
        print("1. The kaggle package is not installed (pip install kaggle)")
        print("2. Kaggle API credentials are not set up")
        print("3. Internet connection issues or dataset no longer exists")
        return False

def download_from_url():
    """
    Attempt to download the dataset from a direct URL.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # URL to dataset (this is a placeholder - replace with actual URL)
        url = "https://raw.githubusercontent.com/haptork/csaransh/master/data/houses.csv"
        
        print(f"Downloading dataset from URL: {url}")
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            output_file = os.path.join(RAW_DATA_DIR, "USA-house-prices.csv")
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded dataset to {output_file}")
            return True
        else:
            print(f"Failed to download from URL. Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error downloading from URL: {e}")
        return False

def create_synthetic_dataset(samples=5000):
    """
    Create a synthetic housing dataset when real data is unavailable.
    
    Args:
        samples: Number of samples to generate
        
    Returns:
        bool: True if successful
    """
    print(f"Creating synthetic housing dataset with {samples} samples...")
    
    # List of 50 US states
    states = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
    ]
    
    # Create plausible ranges for features
    np.random.seed(42)  # For reproducibility
    
    # Generate random data
    data = {
        'price': np.random.normal(250000, 150000, samples),  # House prices with mean $250k
        'bedrooms': np.random.randint(1, 6, samples),  # 1-5 bedrooms
        'bathrooms': np.random.randint(1, 5, samples) + np.random.choice([0, 0.5], samples),  # 1-4.5 bathrooms
        'sqft_living': np.random.normal(2000, 1000, samples),  # Square feet living area
        'sqft_lot': np.random.normal(10000, 5000, samples),  # Square feet lot size
        'floors': np.random.choice([1, 1.5, 2, 2.5, 3], samples),  # Number of floors
        'waterfront': np.random.choice([0, 1], samples, p=[0.95, 0.05]),  # Waterfront property
        'view': np.random.randint(0, 5, samples),  # View rating 0-4
        'condition': np.random.randint(1, 6, samples),  # Condition rating 1-5
        'grade': np.random.randint(1, 14, samples),  # Grade rating 1-13
        'yr_built': np.random.randint(1900, 2023, samples),  # Year built
        'yr_renovated': np.zeros(samples),  # Year renovated (0 means not renovated)
        'zipcode': np.random.randint(10000, 99999, samples),  # Random ZIP codes
        'lat': np.random.uniform(25, 49, samples),  # US latitudes
        'long': np.random.uniform(-124, -67, samples),  # US longitudes
        'sqft_above': np.zeros(samples),  # Will calculate after
        'sqft_basement': np.zeros(samples),  # Will calculate after
        'state': np.random.choice(states, samples)  # Random US state
    }
    
    # Calculate derived fields
    for i in range(samples):
        # Some homes have been renovated
        if np.random.random() < 0.2:  # 20% chance of renovation
            data['yr_renovated'][i] = np.random.randint(data['yr_built'][i], 2023)
            
        # Calculate above/basement square footage
        basement_prob = 0.3  # 30% chance of having a basement
        has_basement = np.random.random() < basement_prob
        if has_basement:
            data['sqft_basement'][i] = np.random.uniform(0.1, 0.5) * data['sqft_living'][i]
            data['sqft_above'][i] = data['sqft_living'][i] - data['sqft_basement'][i]
        else:
            data['sqft_above'][i] = data['sqft_living'][i]
    
    # Create state-specific price adjustments
    state_multipliers = {
        'CA': 1.5, 'NY': 1.4, 'HI': 1.6, 'MA': 1.3, 'NJ': 1.2,  # Expensive states
        'WV': 0.7, 'MS': 0.7, 'AR': 0.75, 'OK': 0.8, 'KS': 0.8   # Less expensive states
    }
    
    # Adjust prices based on state, and add some realistic relationships
    for i in range(samples):
        # State adjustment
        state = data['state'][i]
        multiplier = state_multipliers.get(state, 1.0)
        
        # Other factor adjustments (simplified model)
        size_factor = data['sqft_living'][i] / 2000  # Relative to 2000 sqft
        bedroom_factor = 1 + (data['bedrooms'][i] - 3) * 0.1  # Relative to 3BR
        bathroom_factor = 1 + (data['bathrooms'][i] - 2) * 0.15  # Relative to 2BA
        waterfront_factor = 1.3 if data['waterfront'][i] else 1.0
        age_factor = 1 - (2023 - data['yr_built'][i]) / 200  # Newer houses worth more
        condition_factor = 0.8 + (data['condition'][i] / 5) * 0.4
        
        # Combine all factors and apply some randomness
        combined_factor = (multiplier * size_factor * bedroom_factor * bathroom_factor * 
                          waterfront_factor * age_factor * condition_factor)
        
        # Add randomness (±20%)
        random_factor = np.random.uniform(0.8, 1.2)
        
        # Apply to price
        data['price'][i] = data['price'][i] * combined_factor * random_factor
        
        # Ensure prices are positive and realistic
        data['price'][i] = max(50000, min(5000000, data['price'][i]))
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Fix dtypes
    df['price'] = df['price'].astype(int)
    df['sqft_living'] = df['sqft_living'].astype(int)
    df['sqft_lot'] = df['sqft_lot'].astype(int)
    df['sqft_above'] = df['sqft_above'].astype(int)
    df['sqft_basement'] = df['sqft_basement'].astype(int)
    df['zipcode'] = df['zipcode'].astype(str)
    
    # Save to CSV
    output_file = os.path.join(RAW_DATA_DIR, "USA-house-prices.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved synthetic dataset to {output_file}")
    
    # Create metadata
    with open(INFO_FILE, 'w') as f:
        json.dump({
            "name": "USA House Prices Dataset (Synthetic)",
            "samples": samples,
            "features": list(df.columns),
            "description": "Synthetic housing dataset with features similar to real USA housing data",
            "source": "Synthetically generated",
            "mean_price": float(df['price'].mean()),
            "median_price": float(df['price'].median()),
            "price_range": [float(df['price'].min()), float(df['price'].max())]
        }, f, indent=2)
    
    return True

def process_dataset():
    """
    Process the raw dataset and split into train/test sets.
    Handle categorical variables, missing values, and scaling.
    
    Returns:
        bool: True if successful
    """
    dataset_path = os.path.join(RAW_DATA_DIR, "USA-house-prices.csv")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Raw dataset not found at {dataset_path}")
        return False
        
    print(f"Processing dataset: {dataset_path}")
    
    try:
        # Load the dataset
        df = pd.read_csv(dataset_path)
        
        print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        print("Columns:", df.columns.tolist())
        
        # If the dataset has different column names than expected, try to identify them
        if 'price' not in df.columns:
            # Try to identify price column (usually has large values with currency)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].mean() > 10000:  # Likely a price column
                    print(f"Renaming column '{col}' to 'price'")
                    df.rename(columns={col: 'price'}, inplace=True)
                    break
        
        # Handle missing values
        for col in df.columns:
            # Different strategies based on column type
            if col in ['price', 'sqft_living', 'sqft_lot', 'bathrooms', 'bedrooms']:
                # For important numeric features, use mean
                if df[col].dtype in [np.float64, np.int64] and df[col].isna().any():
                    mean_val = df[col].mean()
                    df[col].fillna(mean_val, inplace=True)
                    print(f"Filled {df[col].isna().sum()} missing values in '{col}' with mean: {mean_val:.2f}")
            else:
                # For other columns, use mode or 0
                if df[col].isna().any():
                    if df[col].dtype in [np.float64, np.int64]:
                        df[col].fillna(0, inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    print(f"Filled missing values in '{col}'")
        
        # Handle categorical variables (like state)
        categorical_cols = []
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'zipcode':  # Skip zipcode
                categorical_cols.append(col)
                
        print(f"Categorical columns: {categorical_cols}")
        
        # Create train-test split
        X = df.drop('price', axis=1)
        y = df['price']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Handle categorical encoding and scaling
        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        
        # Create standard scaler for numeric features
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Scale numeric features
        X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
        X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
        
        # Handle categorical features
        encoders = {}
        
        for col in categorical_cols:
            if col in X_train.columns:
                encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_train = encoders[col].fit_transform(X_train[[col]])
                encoded_test = encoders[col].transform(X_test[[col]])
                
                # Create column names for one-hot encoded features
                categories = encoders[col].categories_[0].tolist()
                encoded_cols = [f"{col}_{cat}" for cat in categories]
                
                # Convert to DataFrame
                train_encoded_df = pd.DataFrame(encoded_train, columns=encoded_cols, index=X_train.index)
                test_encoded_df = pd.DataFrame(encoded_test, columns=encoded_cols, index=X_test.index)
                
                # Add to main DataFrames
                X_train_scaled = pd.concat([X_train_scaled.drop(col, axis=1), train_encoded_df], axis=1)
                X_test_scaled = pd.concat([X_test_scaled.drop(col, axis=1), test_encoded_df], axis=1)
                
                print(f"Encoded categorical column '{col}' into {len(encoded_cols)} features")
                
        # Handle non-numeric columns that weren't categorical (e.g., zipcode)
        for col in X_train.columns:
            if col not in X_train_scaled.columns and col not in categorical_cols:
                print(f"Dropping column '{col}' as it's not numeric or categorical")
                
        # Save train/test sets
        train_file = os.path.join(PROCESSED_DATA_DIR, "train_housing.csv")
        test_file = os.path.join(PROCESSED_DATA_DIR, "test_housing.csv")
        
        # Save X_train_scaled with y_train
        train_df = X_train_scaled.copy()
        train_df['price'] = y_train.values
        train_df.to_csv(train_file, index=False)
        
        # Save X_test_scaled with y_test
        test_df = X_test_scaled.copy()
        test_df['price'] = y_test.values
        test_df.to_csv(test_file, index=False)
        
        print(f"Saved processed train data to {train_file}")
        print(f"Saved processed test data to {test_file}")
        
        # Save scaler and encoders
        import joblib
        
        scaler_file = os.path.join(PROCESSED_DATA_DIR, "house_price_scaler.joblib")
        joblib.dump(scaler, scaler_file)
        
        encoders_file = os.path.join(PROCESSED_DATA_DIR, "house_price_encoders.joblib")
        joblib.dump(encoders, encoders_file)
        
        print(f"Saved scaler to {scaler_file}")
        print(f"Saved encoders to {encoders_file}")
        
        # Update metadata
        try:
            with open(INFO_FILE, 'r') as f:
                info = json.load(f)
                
            info.update({
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "processed_features": X_train_scaled.shape[1],
                "categorical_columns": categorical_cols,
                "numeric_columns": numeric_features.tolist(),
                "price_mean": float(y_train.mean()),
                "price_median": float(y_train.median()),
                "price_std": float(y_train.std())
            })
            
            with open(INFO_FILE, 'w') as f:
                json.dump(info, f, indent=2)
                
            print(f"Updated dataset info in {INFO_FILE}")
        except Exception as e:
            print(f"Error updating metadata: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error processing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to download and process the dataset."""
    print("=" * 50)
    print("USA Housing Price Dataset Download & Processing")
    print("=" * 50)
    
    # Check if the dataset already exists
    if (os.path.exists(os.path.join(PROCESSED_DATA_DIR, "train_housing.csv")) and 
        os.path.exists(os.path.join(PROCESSED_DATA_DIR, "test_housing.csv"))):
        print("Processed dataset already exists. Skipping download and processing.")
        print(f"Data available at: {PROCESSED_DATA_DIR}")
        return
    
    # Try each method in order
    if not download_kaggle_dataset() and not download_from_url():
        print("Could not download real dataset. Creating synthetic data instead.")
        create_synthetic_dataset()
    
    # Process the dataset
    if process_dataset():
        print("\nDataset download and processing complete!")
        print(f"Processed data available in: {PROCESSED_DATA_DIR}")
    else:
        print("\nError occurred during dataset processing.")

if __name__ == "__main__":
    main() 