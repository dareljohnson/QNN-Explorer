# QNN Explorer - Operational Guide

This document provides a detailed explanation of the Quantum Neural Network (QNN) Explorer application's operation, focusing on both the user interface interactions and the underlying processes. It is intended for developers and QA engineers to understand the complete workflow of the application.

## Table of Contents

1. [Application Overview](#1-application-overview)
2. [Application Startup](#2-application-startup)
3. [Model Configuration Flow](#3-model-configuration-flow)
4. [Data Loading and Preprocessing](#4-data-loading-and-preprocessing)
5. [Training Process](#5-training-process)
6. [Prediction Workflow](#6-prediction-workflow)
7. [Common Error Scenarios](#7-common-error-scenarios)
8. [Performance Considerations](#8-performance-considerations)
9. [Testing Guidelines](#9-testing-guidelines)

## 1. Application Overview

The QNN Explorer is a Streamlit-based web application that enables users to configure, train, and evaluate hybrid quantum-classical neural networks. The application integrates classical machine learning components (CNN, Transformer, GNN) with quantum computing components through PennyLane.

### Key Components Interaction

```
User → Streamlit UI → Core Logic Modules → PyTorch/PennyLane → Output Visualization
```

- **Streamlit UI**: Provides interactive interface for all operations
- **Core Modules**: Handle model creation, data processing, and training
- **PyTorch/PennyLane**: Execute classical and quantum operations
- **Visualization**: Render results and metrics for user interpretation

## 2. Application Startup

### UI Initialization

When a user runs `streamlit run app.py`, the following sequence occurs:

1. **Streamlit Server Start**:
   - Launches a local web server (default: port 8501)
   - Initializes the application's state
   - Creates the tabbed interface structure

2. **Session State Initialization**:
   - Behind the scenes: `st.session_state` is initialized with empty values for:
     - `model_config`: Stores model configuration
     - `hybrid_model`: Holds the instantiated PyTorch model
     - `loaded_data`: Contains preprocessed data
     - `training_history`: Tracks metrics during training
     - `device`: Determines computation device (CPU/GPU)

3. **GPU Detection**:
   - Behind the scenes: `check_gpu()` from `utils/helpers.py` checks for available CUDA devices
   - Sets appropriate device in session state
   - UI displays GPU availability information in sidebar

4. **Directory Validation**:
   - Creates necessary directories if they don't exist (saved_models/configs, saved_models/weights)
   - Ensures data directories are available

### Technical Details

- The application employs Streamlit's caching mechanisms to optimize repetitive operations
- Tab switching doesn't trigger full application re-execution, but state is maintained
- Page refresh causes complete reinitialization of the application state

## 3. Model Configuration Flow

### UI Interaction

When a user interacts with the Model Configuration tab:

1. **Backbone Selection**:
   - User selects a classical backbone type (CNN, Transformer, GNN, None)
   - UI dynamically updates to show relevant options for the selected backbone

2. **Quantum Parameter Configuration**:
   - User sets quantum parameters: number of qubits, layers, observation type
   - Behind the scenes: UI validates parameter combinations for feasibility

3. **Configuration Saving**:
   - User enters configuration name and clicks save
   - Behind the scenes: Configuration is serialized to JSON and saved to `saved_models/configs/`

### Behind the Scenes

When a backbone is selected:
```python
# Pseudocode from app.py
if backbone_type == "CNN":
    display_cnn_options()
    # Load available CNN models
    model_names = AVAILABLE_CNN_MODELS
elif backbone_type == "Transformer":
    display_transformer_options()
    # Load available transformer models
    model_names = AVAILABLE_TRANSFORMER_MODELS
# ...and so on
```

When configuration is saved:
```python
# Pseudocode from app.py
def save_model_config(config, filename):
    filepath = os.path.join(SAVED_CONFIG_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
```

### Technical Details

- Configuration files use a standard format:
  ```json
  {
    "classical_backbone_type": "CNN",
    "classical_model_name": "resnet18",
    "num_qubits": 4,
    "num_layers": 2,
    "ansatz_name": "hardware_efficient_ansatz",
    "use_gpu": true,
    "observation_type": "state",
    "save_config_name": "config_N4_L2_cnn.json"
  }
  ```
- The application maintains backward compatibility with older configuration files
- Configuration validation prevents invalid parameter combinations

## 4. Data Loading and Preprocessing

### UI Interaction

1. **Data Type Selection**:
   - User selects data type (Images, Text, CSV, Graph, Video)
   - UI updates to show appropriate source options

2. **Source Selection**:
   - User chooses data source (Upload, Demo, URL)
   - For uploads: File uploader appears
   - For demo: Demo dataset dropdown appears
   - For URL: URL input field appears

3. **Data Loading**:
   - User clicks "Load and Preprocess Data" button
   - Progress indicator appears during loading
   - Data preview is displayed when complete

### Behind the Scenes (MNIST Example)

When a user selects the MNIST demo dataset:

1. **Dataset Path Verification**:
   ```python
   # Pseudocode from app.py
   mnist_path = os.path.join(DEMO_DATA_DIR, "mnist")
   if not os.path.exists(os.path.join(mnist_path, "train")):
       # Show download option if not found
       if st.button("Download MNIST Dataset Now"):
           with st.spinner("Downloading MNIST dataset..."):
               download_mnist_dataset(mnist_path)
   ```

2. **Image Loading and Conversion**:
   ```python
   # From mnist_loader.py
   # Load images with channel conversion
   transform = transforms.Compose([
       transforms.Resize(img_size),
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
       # Convert grayscale to 3 channels for CNN compatibility
       transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
   ])
   ```

3. **Session State Update**:
   ```python
   # Pseudocode from app.py
   st.session_state.loaded_data = images
   st.session_state.csv_labels = labels
   st.session_state.data_info = {
       "data_type": "Images",
       "data_source": "Demo - MNIST Digits",
       "samples": len(images),
       "classes": 10
   }
   ```

### Technical Details

- Data loaders in `data/loader.py` handle different data types and sources
- Special handling for MNIST and Cats vs Dogs datasets with dedicated loaders
- CSV preprocessing supports both numeric and text columns
- Error handling includes:
  - File format validation
  - Size limit checks
  - Data type compatibility verification

## 5. Training Process

### UI Interaction

1. **Model Instantiation**:
   - User clicks "Instantiate Hybrid Model" button
   - Progress indicator appears during model creation
   - Component summary is displayed when complete

2. **Training Parameter Configuration**:
   - User sets learning rate, batch size, epochs
   - Selects task type (Classification, Unsupervised)
   - Chooses optimizer (Adam, SGD)

3. **Training Execution**:
   - User clicks "Start Training" button
   - Real-time progress display shows:
     - Current epoch/batch
     - Loss values
     - Entanglement metrics (if applicable)
     - Time estimates

4. **Training Completion**:
   - Final metrics and plots are displayed
   - Model weights are automatically saved
   - Training time summary is shown

### Behind the Scenes

1. **Model Instantiation**:
   ```python
   # Pseudocode from app.py
   config_with_func = {
       **st.session_state.model_config, 
       'ansatz_func': ansatz_func,
       'use_gpu': use_gpu_flag and torch.cuda.is_available(),
   }
   
   # Create model instance
   model = HybridModel(config_with_func)
   model = model.to(st.session_state.device)
   ```

2. **DataLoader Creation**:
   ```python
   # Pseudocode from app.py
   if data_info["data_type"] == "Images":
       dataset = ImageDatasetWrapper(loaded_data, labels)
   elif data_info["data_type"] == "Text":
       dataset = TextDatasetWrapper(loaded_data, labels)
   # ...
   
   dataloader = DataLoader(
       dataset, 
       batch_size=batch_size,
       shuffle=True
   )
   ```

3. **Training Loop**:
   ```python
   # Pseudocode from app.py
   for epoch in range(epochs):
       for batch_idx, (data, target) in enumerate(dataloader):
           # Move data to device
           data, target = data.to(device), target.to(device)
           
           # Forward pass
           output = model(data)
           
           # Calculate loss
           loss = criterion(output, target)
           
           # Backward pass
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           # Update metrics
           update_progress(epoch, batch_idx, loss.item())
   ```

4. **Weight Saving**:
   ```python
   # Pseudocode from app.py
   # Get model backbone type for more descriptive filename
   backbone_type = st.session_state.model_config.get('classical_backbone_type', 'none').lower()
   num_qubits = st.session_state.model_config.get('num_qubits', '4')
   num_layers = st.session_state.model_config.get('num_layers', '2')
   
   # Create a descriptive base name
   model_descriptor = f"{backbone_type}_N{num_qubits}_L{num_layers}"
   
   # Always save to last_train for backward compatibility
   save_model_weights(model, "weights_last_train")
   
   # Save with descriptive name based on the model architecture
   save_model_weights(model, f"weights_{model_descriptor}.pt")
   ```

### Technical Details

- The training loop implemented in `app.py` handles both classification and unsupervised tasks
- For quantum operations, batch processing has special handling:
  - Each sample is processed individually due to limitations with broadcasting in PennyLane
  - Results are recombined into batched tensors
- Weight files are stored with multiple naming patterns for better organization:
  - `weights_last_train`: Always contains most recent training
  - `weights_{backbone}_N{qubits}_L{layers}.pt`: Descriptive architecture-based name
  - `weights_{config_name}.pt`: Configuration-based name

## 6. Prediction Workflow

### UI Interaction

1. **Model Loading**:
   - User selects model configuration
   - Clicks "Load Configuration for Prediction"
   - Clicks "Instantiate Model" (loads weights automatically)

2. **Input Preparation**:
   - User selects input data type (matching model configuration)
   - Based on type, UI presents appropriate input method:
     - Image upload or URL
     - Text input area
     - CSV file uploader
     - etc.

3. **Prediction Execution**:
   - User clicks "Predict" button
   - Progress indicator appears during prediction
   - Results are displayed with:
     - Prediction output
     - Confidence scores (for classification)
     - Quantum state visualization (if applicable)
     - Entanglement metrics

### Behind the Scenes

1. **Model Loading**:
   ```python
   # Pseudocode from app.py
   # Try multiple possible file paths with different naming patterns
   filepath_patterns = [
       os.path.join(SAVED_WEIGHTS_DIR, filename.replace('.json', '.pt')),
       os.path.join(SAVED_WEIGHTS_DIR, f"weights_{filename.replace('.json', '.pt')}"),
       os.path.join(SAVED_WEIGHTS_DIR, "weights_last_train")
   ]
   
   # Try each filepath pattern
   for filepath in filepath_patterns:
       if os.path.exists(filepath):
           model.load_state_dict(torch.load(filepath), strict=False)
           return True
   ```

2. **Input Processing**:
   ```python
   # Pseudocode from app.py
   if input_type == "Images":
       # Process uploaded image
       input_tensor = preprocess_image(uploaded_file)
   elif input_type == "Text":
       # Process text input
       input_tensor = preprocess_text(text_input, tokenizer)
   # ...
   
   # Move to device
   input_tensor = input_tensor.to(st.session_state.device)
   ```

3. **Prediction**:
   ```python
   # Pseudocode from app.py
   with torch.no_grad():
       # Forward pass
       output = model(input_tensor)
       
       # For classification
       if task_type == "Classification":
           probabilities = torch.nn.functional.softmax(output, dim=1)
           predicted_class = torch.argmax(probabilities, dim=1).item()
       
       # Get quantum state
       quantum_state = model.last_quantum_output
       
       # Calculate entanglement
       entanglement = calculate_meyer_wallach(quantum_state)
   ```

### Technical Details

- Model loading tries multiple file patterns for better user experience
- Prediction operates in evaluation mode (`model.eval()`) with `torch.no_grad()`
- For quantum models:
  - `last_quantum_output` is stored during forward pass for analysis
  - Visualization uses specialized quantum state renderers
- Error handling accounts for:
  - Missing model weights
  - Input format mismatches
  - Quantum state conversion errors

## 7. Common Error Scenarios

### Channel Mismatch for CNN Models

**Symptom**: Error message: "Expected input to have 3 channels, but got 1 channel"

**Underlying Issue**:
- CNN models like ResNet18 expect 3-channel RGB images
- MNIST and similar datasets provide 1-channel grayscale images

**Resolution Flow**:
1. **Detection**: Error occurs during forward pass in CNN model
2. **Behind the Scenes**: MNIST loader includes channel conversion:
   ```python
   # From mnist_loader.py
   transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
   ```
3. **UI Feedback**: User sees error message with suggestion to check channel conversion

### GPU Acceleration Issues

**Symptom**: Error message: "Device default.qubit.torch does not exist"

**Underlying Issue**: 
- PennyLane GPU plugins not installed or CUDA unavailable

**Resolution Flow**:
1. **Detection**: During model instantiation, quantum device creation fails
2. **Behind the Scenes**: Graceful fallback to CPU:
   ```python
   # From quantum_models.py
   try:
       return qml.device('default.qubit.torch', wires=num_qubits)
   except Exception as e:
       print(f"Error creating GPU device: {e}")
       print("To enable GPU acceleration for PennyLane, you need:")
       print("  1. PyTorch with CUDA support")
       print("  2. For faster simulation: pennylane-lightning-gpu")
       print("Falling back to CPU device.")
       return qml.device('default.qubit', wires=num_qubits)
   ```
3. **UI Feedback**: Warning message with installation instructions

### Dimension Mismatch

**Symptom**: Error: "Mat1 and mat2 shapes cannot be multiplied"

**Underlying Issue**:
- Input feature dimension doesn't match quantum circuit input size

**Resolution Flow**:
1. **Detection**: First forward pass discovers mismatch
2. **Behind the Scenes**: Adaptive fusion layer creates:
   ```python
   # From fusion.py
   # Create a new fusion layer when needed
   if x.shape[1] != self.quantum_input_size:
       self.fusion_layer = nn.Linear(x.shape[1], self.quantum_input_size)
       # Initialize with small values to avoid large gradients
       nn.init.xavier_uniform_(self.fusion_layer.weight, gain=0.1)
       nn.init.zeros_(self.fusion_layer.bias)
   ```
3. **UI Feedback**: Information message about adaptive layer creation

### PennyLane Broadcasting Issues

**Symptom**: Error: "Broadcasting with MottonenStatePreparation is not supported"

**Underlying Issue**:
- PennyLane quantum operations don't support batched inputs directly

**Resolution Flow**:
1. **Detection**: Error occurs during quantum circuit execution
2. **Behind the Scenes**: Individual sample processing:
   ```python
   # From fusion.py pseudocode
   # Process each sample individually
   for i in range(batch_size):
       sample = x[i:i+1]  # Keep batch dimension
       try:
           # Process single sample
           result = self.process_quantum_sample(sample)
           results.append(result)
       except Exception as e:
           print(f"Error processing single sample: {e}")
           # Use dummy output for error cases
           results.append(torch.zeros(1, output_size))
   
   # Combine results
   combined_results = torch.cat(results, dim=0)
   ```
3. **UI Feedback**: Warning about potential performance impact

## 8. Performance Considerations

### Quantum Simulation Performance

- **Scaling Factors**:
  - Number of qubits: Simulation complexity doubles with each additional qubit
  - Number of layers: Linear scaling with circuit depth
  - Batch size: Linear scaling when samples are processed individually

- **GPU Acceleration**:
  - PennyLane with PyTorch integration provides GPU acceleration
  - Additional speedup with `pennylane-lightning-gpu` plugin
  - CPU fallback when GPU acceleration unavailable

### UI Responsiveness

- **Streamlit Limitations**:
  - Full page reloads occur when state changes
  - No background processing without workarounds
  
- **Long-Running Operations**:
  - Training progress updates use Streamlit's progress mechanisms
  - Large dataset handling uses chunked processing
  - GPU memory management for CNN/Transformer models

### Memory Management

- **Dataset Handling**:
  - Large datasets use DataLoader with appropriate batch sizes
  - Image datasets have potential resizing to control memory usage
  - MNIST datasets limited to 1,000 training images (100 per class)

- **Model Optimization**:
  - Observation_type = 'expval' more memory efficient than 'state'
  - Gradient checkpointing for transformer models
  - Memory cleanup after training loops

## 9. Testing Guidelines

### Functional Testing Priorities

1. **Configuration Testing**:
   - Validate all parameter combinations
   - Test configuration save/load functionality
   - Verify compatibility with different backends

2. **Data Handling Testing**:
   - Test all data types and sources
   - Verify preprocessing for each data type
   - Test error handling for invalid inputs

3. **Training Testing**:
   - Verify training with different batch sizes
   - Test different observation types
   - Validate metric calculation and visualization

4. **Prediction Testing**:
   - Test different input methods
   - Verify correct loading of models
   - Validate prediction results visualization

### Automated Testing

Key test files include:

- `test_fusion_fixes.py`: Tests dimension, type, and observation issues
- `test_mnist_channels.py`: Tests for proper channel conversion
- `test_quantum_batch.py`: Tests for batch processing in quantum operations

Run tests with:
```bash
# Run specific tests
python tests/test_mnist_channels.py

# Run all tests
python -m pytest
```

### Manual Testing Checklist

1. **Installation Testing**:
   - Fresh install on different platforms
   - GPU vs. CPU configurations
   - Virtual environment setup

2. **UI Flow Testing**:
   - Complete workflow from configuration to prediction
   - Error handling and recovery
   - Form validation and feedback

3. **Performance Testing**:
   - Training with different qubit counts
   - Memory profiling during longer runs
   - UI responsiveness with large datasets

4. **Integration Testing**:
   - PennyLane version compatibility
   - PyTorch version compatibility
   - Streamlit version compatibility

---

This operational guide provides a detailed understanding of how the QNN Explorer application functions, both from the UI perspective and behind the scenes. It should help developers and QA engineers understand the application's operation, troubleshoot issues, and extend its functionality. 