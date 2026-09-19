import os
import sys
import torch
import streamlit as st

# Add current directory to path for imports
sys.path.append(os.path.abspath('.'))

# Import from app.py
from app import load_model_weights, SAVED_WEIGHTS_DIR

# Create a mock model
class MockModel:
    def __init__(self):
        self.loaded = False
    
    def load_state_dict(self, state_dict):
        self.loaded = True
        print(f"Model loaded with state dict containing {len(state_dict)} parameters")
    
    def eval(self):
        print("Model set to evaluation mode")

# Set device in session state
if 'device' not in st.session_state:
    st.session_state.device = 'cpu'

# Check that the weights file exists
weights_file = os.path.join(SAVED_WEIGHTS_DIR, "weights_last_train")
if os.path.exists(weights_file):
    print(f"✅ Found weights file: {weights_file}")
    print(f"File size: {os.path.getsize(weights_file) / (1024 * 1024):.2f} MB")
else:
    print(f"❌ Weights file not found: {weights_file}")

# Create mock model and attempt to load weights
model = MockModel()

# Try loading with a non-existent config name, which should still find the weights_last_train file
result = load_model_weights(model, "config_N4_L6_transformer.json")

# Check the result
print(f"Load result: {result}")
print(f"Model loaded: {model.loaded}")

print("\nTest completed successfully!") 