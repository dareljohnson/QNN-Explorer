# QNN Explorer

A Python application with a Streamlit UI to configure, train, and evaluate hybrid quantum-classical models (Quantum Neural Networks - QNNs).

## Features

*   Supports various data types (Text, Images, Graphs, Video) via classical feature extractors (Transformers, CNNs, GNNs).
*   Amplitude encoding for mapping classical features to quantum states.
*   Configurable variational quantum circuits using PennyLane.
*   Calculation of quantum metrics like Meyer-Wallach entanglement.
*   GPU acceleration via PyTorch and compatible PennyLane devices.
*   Interactive UI for model configuration, data loading, training, and prediction.
*   Robust run history tracking with automated error recovery.

## Run History Management

QNN Explorer includes a comprehensive run history tracking system that maintains a record of all your training sessions, making it easy to compare results and restore previous models.

### Key Features

1. **Run History Tracking**
   - Automatically logs all training runs with configuration details, metrics, and results
   - Stores and organizes training artifacts like model weights and performance metrics
   - Provides a unified interface to browse, compare, and analyze past experiments

2. **Error Resilience and Recovery**
   - Multi-layered error handling to protect against data corruption
   - Automatic fallback mechanisms if JSON files become corrupted
   - Partial data recovery to salvage information from damaged files
   - Backup file creation to minimize data loss

3. **User Interface**
   - Easy browsing of all previous training runs
   - Detailed view of run configuration, parameters, and results
   - Visual indicators for recovered runs with repair options
   - One-click restoration of previous model configurations

### Using Run History

1. **Viewing Run History**
   - Navigate to the "Run History" tab in the application
   - Browse the table of all recorded training runs
   - Select a run to view its detailed information

2. **Understanding Run Details**
   - Configuration: Model architecture, hyperparameters, and settings
   - Training Information: Duration, epochs, batch size, learning rate
   - Performance Metrics: Accuracy, loss curves, and quantum metrics
   - Dataset Information: Data type, feature columns, and sample count

3. **Recovering from Errors**
   - If data corruption is detected, the system automatically attempts recovery
   - A warning message indicates if a run was recovered from corruption
   - Use the "Repair Run Data" button to create a clean record when needed
   - Delete corrupted runs that can't be recovered

4. **Managing Run History**
   - Update run status (Completed, Failed, In Progress)
   - Delete unwanted or obsolete runs
   - Load model weights and configuration from a previous run
   - Export run details for external analysis

The run history system ensures you never lose your training progress, even in case of application crashes or file system errors, providing a robust foundation for experimental tracking in quantum machine learning development.

## Strategies for Increasing Model Accuracy

To achieve accuracy of 83% or higher with hybrid quantum-classical models, consider implementing these strategies:

### 1. Enhance Quantum Circuit Architecture
- **Increase circuit layers**: Try 4-8 layers instead of 2 to create a more expressive quantum model
- **Optimize qubit count**: For classification tasks, 4-6 qubits often works well (balance between expressivity and trainability)
- **Try different ansatz designs**: Test ansatz with more entangling gates or different entanglement patterns

### 2. Improve Classical Components
- **Add a stronger classical backbone**: Use ResNet or EfficientNet for images, or a more powerful Transformer for text
- **Tune the fusion layer**: Add multiple fusion layers with non-linearities instead of a single linear layer
- **Enhance output head**: Add more complexity to the output classification head

### 3. Optimize Training Process
- **Learning rate scheduling**: Start with 1e-3 and decrease by 0.1 every few epochs
- **Increase training epochs**: Train for 20-30 epochs to allow convergence
- **Batch size optimization**: Try 16, 32, and 64 to find optimal trade-off
- **Gradient clipping**: Set max_norm=1.0 to improve training stability

### 4. Data Processing Improvements
- **Feature engineering**: Add domain-specific features relevant to your task
- **Normalize data properly**: Ensure features are normalized to mean=0, std=1
- **Data augmentation**: For images, use random crops, flips, and rotations
- **Feature selection**: Use only the most informative features

### 5. Quantum-Specific Optimizations
- **Parameter initialization**: Initialize quantum parameters in [-π/10, π/10] range
- **Optimize embedding**: Use angle encoding instead of amplitude encoding for classification
- **Mixed precision**: Use single precision for classical and quantum components
- **Entanglement engineering**: Design specific entanglement patterns based on your data structure

### 6. Try Ensemble Methods
- **Train multiple models**: Combine 3-5 models with different initializations
- **Majority voting**: For classification, use majority voting from multiple models

Start by implementing 2-3 of these suggestions at a time, then evaluate the impact before making additional changes.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd qnn-explorer
    ```
2.  **Create a Python environment:** (e.g., using Conda or venv)
    ```bash
    conda create -n quantum_env python=3.10 -y
    conda activate quantum_env
    ```
    *Ensure you have a CUDA-enabled PyTorch version if using GPU.*
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

```bash
streamlit run app.py
```

## Understanding Training Outputs

During training, you'll see output lines like this:

```
Epoch 1/5 | Batch 710/2500 | Step 710 | Loss: -1.6810 | Entanglement (Q): 1.6294
```

These metrics provide insight into training progress:

* **Epoch X/Y**: Current epoch out of total training epochs
* **Batch X/Y**: Current batch out of total batches per epoch
* **Step X**: Total number of optimization steps performed so far
* **Loss**: The value of the loss function being minimized
  * When using entanglement optimization, loss is negative because we're maximizing entanglement (minimizing negative entanglement)
* **Entanglement (Q)**: Meyer-Wallach entanglement measure (between 0 and 2)
  * 0 = No entanglement (separable state)
  * Values > 1 indicate strong quantum correlation across qubits
  * 2 = Maximum possible entanglement
  
A decreasing loss value and increasing entanglement value typically indicate successful training when optimizing for quantum entanglement.

## Optimizing Training Speed

To accelerate training in the QNN Explorer, consider adjusting these settings:

### Essential Speed Optimizations

1. **Reduce Quantum Circuit Size**
   - **Fewer Qubits**: Each additional qubit doubles the state vector size and significantly increases computation time
   - **Fewer Circuit Layers**: Reducing layers from 16 to 8 or 4 can cut training time by 50-75% with minimal accuracy loss

2. **Use Expectation Values**
   - Change "Observation Type" to "Expectation Value (PauliZ)" instead of "State Vector"
   - Much faster computation and enables hardware-compatible gradient methods
   - Essential for training on actual quantum hardware

3. **Increase Batch Size**
   - Default is 16 - try 32 or 64 for faster training
   - For CNN/image models, larger batches provide better GPU utilization
   - Adjust based on available memory

4. **Choose Simpler Classical Backbone**
   - "None" is fastest if input features already match 2^N (qubit count)
   - For images, smaller CNNs like ResNet18 are faster than deeper models
   - For text, DistilBERT is much faster than full BERT or RoBERTa

### Additional Optimizations

- **Enable GPU** if available (checkbox in configuration)
- **Reduce Input Size** for images (smaller resolution = faster processing)
- **Use Higher Learning Rate** (0.003-0.005 instead of default 0.001)
- **Limit Training Data** - use a subset for initial experiments before scaling up

For the fastest possible training, start with the smallest working configuration (4 qubits, 2 layers) and expectation value measurements. This will train 10-20x faster than a larger state vector model, allowing for rapid prototyping before scaling up to more complex models.

## Making Predictions

The Predict tab allows you to test your trained model on new data. The process differs slightly depending on your model type:

### Step 1: Load Configuration and Model
1. Select a previously saved configuration from the dropdown menu
2. Click "Load Configuration for Prediction"
3. Click "Instantiate Model" (this automatically loads the corresponding weights if available)

### Step 2: Select Input Data Type and Provide Data

#### For CNN Models (Images)
1. Select "Images" as the Input Data Type
2. Upload an image file or use the image URL option
3. The image will be preprocessed automatically using the same pipeline as during training
4. Click "Predict" to see the quantum state output and entanglement measure

#### For Transformer Models (Text)
1. Select "Text" as the Input Data Type
2. Enter your text in the provided text area
3. The text will be tokenized using the model's tokenizer (e.g., BERT, RoBERTa)
4. Click "Predict" to process the text through the hybrid quantum-classical model
5. The token count and tokenization details are displayed for reference

#### For GNN Models (Graphs)
1. Select "Graph" as the Input Data Type
2. Upload a graph file in a supported format (e.g., GraphML, JSON)
3. The graph will be converted to the appropriate format for your model
4. Click "Predict" to process the graph data

#### For Direct Quantum Input
1. If your model has no classical backbone (configured as "None"), select "CSV" 
2. Upload a CSV with exactly 2^N features (where N is the number of qubits)
3. Click "Predict" to directly encode the input into the quantum circuit

### Step 3: Interpret Results

The prediction results show:
- The quantum state vector visualization (if using State Vector observation type)
- Expectation values (if using Expectation Value observation type)
- Meyer-Wallach entanglement measure of the output state
- Additional metrics based on your model configuration

## Running Tests

```bash
pytest
```

## Project Structure

```
qnn_explorer/
├── app.py                  # Main Streamlit application
├── core/
│   ├── __init__.py
│   ├── classical_models.py # CNN, GNN, Transformer definitions
│   ├── quantum_models.py   # QNode, AmplitudeEncoding, Ansatz definitions
│   ├── fusion.py           # Hybrid model assembly, fusion layer
│   └── metrics.py          # Meyer-Wallach, KL Divergence functions
├── data/
│   ├── __init__.py
│   ├── loader.py           # Data loading functions
│   ├── preprocessing.py    # Preprocessing pipelines
│   └── demo_data/          # Small demo datasets
├── utils/
│   ├── __init__.py
│   ├── visualization.py    # Plotting functions
│   ├── run_history.py      # Run history tracking and error recovery
│   ├── run_history_tab.py  # UI for run history visualization
│   └── helpers.py          # Misc utility functions
├── tests/
│   ├── __init__.py
│   ├── test_classical.py
│   ├── test_quantum.py
│   ├── test_fusion.py
│   ├── test_data.py
│   ├── test_run_history.py # Tests for run history functionality
│   └── test_qnn_e2e.py     # Integration tests
├── saved_models/           # To store configurations and weights
├── requirements.txt
└── README.md
```

## Tutorials

### MNIST Handwritten Digits Classification

The QNN Explorer includes support for the MNIST handwritten digits dataset, which is a standard benchmark for image classification.

#### About MNIST

The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9). It includes:

- 60,000 training images (reduced to 1,000 in our implementation for quantum computing)
- 10,000 test images (reduced to 200 in our implementation)
- 10 classes (digits 0-9)

#### Downloading MNIST Dataset

You can download the MNIST dataset using the provided script:

```bash
# Activate your environment
python download_mnist.py
```

This will:
1. Download the official MNIST dataset
2. Extract and save a subset of images (100 per class for training, 20 per class for testing)
3. Organize images into class-based folders
4. Prepare the dataset for use with QNN Explorer

#### Using MNIST with QNN Explorer

1. **Configure a Model for MNIST**:
   - Set Classical Backbone Type to "CNN"
   - Choose a CNN model like "resnet18"
   - Set Number of Qubits to 4-6
   - Set Number of Layers to 2-4
   - Choose "State Vector" for Observation Type

2. **Load the MNIST Dataset**:
   - Select "Images" as the data type
   - Choose "Demo Dataset" as the source
   - Select "MNIST Digits (Image Classification)"
   - Click "Load Data"

3. **Train Your Model**:
   - Learning rate: 0.001-0.003
   - Batch size: 16 or 32
   - Epochs: 5-10
   - Task type: Classification

4. **Make Predictions**:
   - Load your trained model
   - Upload an image of a handwritten digit
   - The model will predict the digit class

#### Tips for MNIST Classification

- With 4 qubits and 2 layers, a well-configured hybrid model can achieve 80-90% accuracy
- Training on 1,000 images takes approximately 10-20 minutes on a CPU
- Try different CNN backbones and ansatz designs for better performance

For more details, see the full tutorial in `docs/mnist_tutorial.md`.

### Image Processing with CNN Models

This tutorial demonstrates how to use QNN Explorer to train a hybrid model on image data and make predictions.

#### 1. Model Configuration

In the **Model Configuration** tab:
- Select "CNN" as the Classical Backbone Type
- Choose a CNN model like "resnet18"
- Set the number of qubits (4-6 is a good start)
- Choose "State Vector" as the Observation Type
- Save your configuration with a descriptive name

![Model Configuration for CNN]()

#### 2. Data Preparation

In the **Data** tab:
- Select "Images" as the Data Type
- Choose your data source:
  - **Upload**: Upload a folder of images from your computer
  - **URL**: Provide URLs to images
  - **Demo**: Use the built-in demo image dataset (e.g., MNIST Sample)
- The system will automatically preprocess the images to the correct size for the CNN
- For classification tasks, ensure your images are organized in labeled folders

![Data Preparation for CNN]()

#### 3. Training

In the **Train** tab:
- Click "Instantiate Hybrid Model"
- Configure training parameters:
  - Set learning rate (~0.001)
  - Set batch size (16-32) 
  - Set number of epochs (5-10)
- Select the appropriate task type:
  - **Classification**: For labeled image data
  - **Unsupervised (Metric Optimization)**: To optimize for quantum properties like entanglement
- Click "Start Training" and monitor the progress

The training metrics will show:
- Loss values
- Entanglement measures (for State Vector observation)
- Training time information

![Training CNN Model]()

#### 4. Making Predictions

In the **Predict** tab:
- Load your model configuration
- Click "Instantiate Model" (this loads the trained weights)
- Select "Images" as the Input Data Type
- Upload a new image for prediction
- Click "Predict" to see:
  - The output state vector visualization
  - Entanglement metrics
  - Classification results (if applicable)

![Making Predictions with CNN]()

#### Tips for Image Processing

- **Image Size**: The images are automatically resized, but consistent image dimensions work best
- **Data Augmentation**: Currently not supported in the UI but can be implemented in custom preprocessing
- **Transfer Learning**: The CNN models use ImageNet pre-trained weights for better feature extraction
- **Batch Size**: Adjust based on your hardware - lower for limited memory
- **Performance**: GPU acceleration significantly speeds up CNN training

For more advanced use cases, you can modify the preprocessing pipeline in `data/preprocessing.py`. 

# Create a script in your project root named download_cats_dogs.py

import os
import tensorflow as tf
import tensorflow_datasets as tfds
import shutil
from PIL import Image
import numpy as np

# Create directories for the data
os.makedirs('data/demo_data/cats_dogs/train/cat', exist_ok=True)
os.makedirs('data/demo_data/cats_dogs/train/dog', exist_ok=True)
os.makedirs('data/demo_data/cats_dogs/test/cat', exist_ok=True)
os.makedirs('data/demo_data/cats_dogs/test/dog', exist_ok=True)

print("Downloading cats_vs_dogs dataset...")
# Download and prepare the dataset
ds, info = tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[80%:]'], with_info=True)
train_ds, test_ds = ds

# Process and save images
def save_images(dataset, split='train', max_per_class=1000):
    cat_count, dog_count = 0, 0
    for i, example in enumerate(dataset):
        image = example['image'].numpy()
        label = example['label'].numpy()
        
        # Convert label to string (0=cat, 1=dog)
        class_name = 'cat' if label == 0 else 'dog'
        
        # Check counts
        if class_name == 'cat' and cat_count >= max_per_class:
            continue
        if class_name == 'dog' and dog_count >= max_per_class:
            continue
            
        # Update counts
        if class_name == 'cat':
            cat_count += 1
        else:
            dog_count += 1
            
        # Save the image
        img = Image.fromarray(image)
        filename = f'data/demo_data/cats_dogs/{split}/{class_name}/{class_name}_{i}.jpg'
        img.save(filename)
        
        # Print progress
        if (i+1) % 100 == 0:
            print(f"Processed {i+1} images ({cat_count} cats, {dog_count} dogs)")
            
        # Stop if we have enough images of each class
        if cat_count >= max_per_class and dog_count >= max_per_class:
            break

print("Processing training images...")
save_images(train_ds, 'train', max_per_class=500)  # Limit to 500 per class for faster training

print("Processing test images...")
save_images(test_ds, 'test', max_per_class=100)  # Limit to 100 per class for testing

print("Dataset preparation complete!") 

def load_cats_dogs_dataset(data_dir='data/demo_data/cats_dogs', split='train', img_size=(224, 224)):
    """
    Load cats vs dogs dataset for QNN Explorer
    
    Args:
        data_dir: Path to cats_dogs dataset directory
        split: 'train' or 'test'
        img_size: Image size for resizing
        
    Returns:
        images: Tensor of images [N, C, H, W]
        labels: Tensor of labels [N]
    """
    from torchvision import transforms
    from PIL import Image
    import os
    import torch
    
    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize lists for images and labels
    images = []
    labels = []
    
    # Process cat images (label 0)
    cat_dir = os.path.join(data_dir, split, 'cat')
    if os.path.exists(cat_dir):
        for filename in os.listdir(cat_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(cat_dir, filename)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)
                    images.append(img_tensor)
                    labels.append(0)  # Cat label
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    # Process dog images (label 1)
    dog_dir = os.path.join(data_dir, split, 'dog')
    if os.path.exists(dog_dir):
        for filename in os.listdir(dog_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(dog_dir, filename)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)
                    images.append(img_tensor)
                    labels.append(1)  # Dog label
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    # Convert lists to tensors
    if images:
        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        print(f"Loaded {len(images)} images with shape {images_tensor.shape}")
        return images_tensor, labels_tensor
    else:
        print("No images found!")
        return None, None 

# Add to DEMO_DATASETS dictionary
DEMO_DATASETS = {
    # ... existing datasets
    "Cats vs Dogs (Image Classification)": {
        "type": "Images", 
        "path": os.path.join(DEMO_DATA_DIR, "cats_dogs")
    },
} 

# In the data loading section where you handle demo datasets:
elif selected_demo == "Cats vs Dogs (Image Classification)":
    with st.spinner("Loading Cats vs Dogs dataset..."):
        try:
            from data.cats_dogs_loader import load_cats_dogs_dataset
            images, labels = load_cats_dogs_dataset()
            if images is not None:
                st.session_state.loaded_data = images
                st.session_state.csv_labels = labels
                st.session_state.data_info = {
                    "data_type": "Images",
                    "data_source": "Demo - Cats vs Dogs",
                    "samples": len(images)
                }
                st.success(f"Loaded {len(images)} images with {len(torch.unique(labels))} classes")
            else:
                st.error("Failed to load Cats vs Dogs dataset")
        except Exception as e:
            st.error(f"Error loading dataset: {e}") 

## GPU Acceleration

To enable GPU-accelerated quantum simulations with PennyLane, you need:

1. **PyTorch with CUDA support**:
   ```
   # Install PyTorch with CUDA
   pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. **PennyLane with GPU support**:
   ```
   # Install PennyLane's Lightning GPU plugin (for faster simulation)
   pip install pennylane-lightning-gpu
   
   # Or use the built-in PyTorch integration
   # This is already included with the standard PennyLane installation
   ```

When creating a model, set `use_gpu: true` in the configuration:

```python
config = {
    'num_qubits': 4,
    'num_layers': 2,
    'use_gpu': True,  # Enable GPU acceleration
    'observation_type': 'state'
}
```

The system will automatically try to use GPU-accelerated devices, falling back to CPU if GPU support is not available. 

## Model Weights Management

When training models, the application saves weights in multiple formats for flexibility:

### Weight File Naming Patterns

1. **Descriptive Architecture-Based Names**: 
   - Format: `weights_[backbone]_N[qubits]_L[layers].pt`
   - Example: `weights_cnn_N4_L2.pt` (CNN backbone, 4 qubits, 2 layers)
   - Example: `weights_transformer_N6_L4.pt` (Transformer backbone, 6 qubits, 4 layers)
   - Example: `weights_none_N8_L2.pt` (No classical backbone, 8 qubits, 2 layers)

2. **Configuration-Based Names**:
   - Format: `weights_[config_name].pt`
   - Example: `weights_config_N4_L2_cnn.pt` (if that's the saved config name)

3. **Latest Training Fallback**:
   - Filename: `weights_last_train`
   - Always contains the most recently trained model, regardless of configuration

### Loading Weights

When loading a model, the application tries multiple file patterns in this order:
1. First checks for weights that match the configuration name
2. Then checks for descriptive architecture-based names
3. Finally falls back to the most recent training weights

This system ensures that:
- You can easily identify which weights belong to which model architecture
- Multiple model configurations can coexist with their own weights
- The latest training is always available as a fallback 

## Quantum Circuit Visualization

QNN Explorer provides robust quantum circuit visualization to help you understand the structure and operations of your quantum models.

### Features:

- **Interactive Circuit Diagrams**: Visualize the quantum circuit structure with gates and operations
- **Automatic Circuit Execution**: The visualization system automatically executes the circuit with representative inputs to generate accurate diagrams
- **Error Handling**: Clear error messages and fallbacks when circuit visualization encounters issues
- **Integration with Model Parameters**: Uses the actual trained parameters of your model for realistic circuit representation

### Using Circuit Visualization:

1. **In the Visualize Tab**:
   - After instantiating a model, the circuit diagram will be automatically displayed
   - The diagram shows all quantum gates and operations in your model's ansatz

2. **Programmatically**:
   ```python
   from utils.visualization import plot_circuit
   
   # Option 1: Execute the circuit first, then visualize
   circuit(input_data)  # Execute with valid inputs
   plot_circuit(circuit)  # Visualize the executed circuit
   
   # Option 2: Provide input_args for automatic execution and visualization
   plot_circuit(circuit, input_args=input_data)
   
   # Option 3: Provide multiple input arguments (e.g., amplitudes and parameters)
   plot_circuit(circuit, input_args=[amplitudes, parameters])
   ```

### Interpreting Circuit Diagrams:

- **Horizontal Lines**: Represent qubits (quantum bits)
- **Boxes with Labels**: Represent quantum gates (operations)
- **Vertical Lines**: Represent multi-qubit operations (like CNOT or controlled operations)
- **Left-to-Right Flow**: The circuit executes from left to right, showing the sequence of operations

For more information on circuit visualization, see the [Circuit Visualization Documentation](docs/circuit_visualization_improvements.md). 