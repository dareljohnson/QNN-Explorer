# Quantum Neural Network Explorer - Software Requirements Document

## 1. Project Overview

The Quantum Neural Network Explorer (QNN Explorer) is a Python application with a Streamlit UI that allows users to configure, train, and evaluate hybrid quantum-classical models. It combines classical machine learning components (CNN, Transformer, GNN) with quantum computing components through PennyLane, enabling exploration of quantum neural networks for various data types.

## 2. System Architecture

### 2.1 High-Level Architecture

The QNN Explorer follows a modular architecture:

```
QNN Explorer
├── UI Layer (Streamlit)
├── Core Components
│   ├── Classical Models (CNN, Transformer, GNN)
│   ├── Quantum Models (Circuit definitions, Ansatz designs)
│   ├── Fusion Module (Hybrid model implementation)
│   └── Metrics (Quantum metrics, Loss functions)
├── Data Processing
│   ├── Loaders (Image, Text, CSV, Graph)
│   ├── Preprocessing (Type-specific pipelines)
│   └── Dataset Wrappers
└── Utilities
    ├── Visualization
    ├── Helpers
    └── Model Management
```

### 2.2 Key Components

1. **Hybrid Model**: Combines classical feature extractors with quantum circuits
2. **Fusion Layer**: Adapts classical outputs to quantum inputs
3. **Quantum Circuit**: Processes data with quantum operations
4. **Observation Types**: Supports state vectors and expectation values
5. **Dataset Integrations**: MNIST, Cats vs Dogs, and custom data

## 3. Technical Requirements

### 3.1 Dependencies

- **Python**: 3.10 or higher
- **Core Libraries**:
  - PyTorch (1.13.0+)
  - PennyLane (0.30.0+)
  - Streamlit (1.20.0+)
  - NumPy (1.23.0+)
  - Pandas (1.5.0+)
  - PIL/Pillow (9.0.0+)
  - Transformers (4.26.0+)

- **Optional Libraries**:
  - torch_geometric (for GNN support)
  - pennylane-lightning-gpu (for GPU acceleration)
  - matplotlib (for visualizations)

### 3.2 Hardware Requirements

- **Minimum**: 8GB RAM, quad-core CPU
- **Recommended**: 16GB RAM, 8-core CPU, CUDA-compatible GPU
- **Storage**: At least 2GB for the application and demo datasets

### 3.3 Platform Support

- Windows, macOS, and Linux are supported
- CUDA support for GPU acceleration (optional)

## 4. Features and Components

### 4.1 User Interface

The application provides a tabbed interface with:

1. **Model Configuration**: Configure classical and quantum components
2. **Data**: Load and preprocess data from various sources
3. **Train**: Train hybrid models with visualization of metrics
4. **Predict**: Make predictions with trained models
5. **Metrics**: Visualize quantum metrics like entanglement

### 4.2 Model Configuration

#### 4.2.1 Classical Components

- **Backbone Types**:
  - CNN: Support for ResNet18 and similar architectures
  - Transformer: Support for BERT, DistilBERT
  - GNN: Support for GCN
  - None: Direct quantum input without classical preprocessing

- **Required Parameters**:
  - `classical_backbone_type`: The type of classical model
  - `classical_model_name`: Specific model architecture
  - `num_classes`: Number of output classes (for classification)

#### 4.2.2 Quantum Components

- **Circuit Configuration**:
  - `num_qubits`: Number of qubits (4-10 recommended)
  - `num_layers`: Number of circuit layers (2-8 recommended)
  - `ansatz_func`: Quantum circuit architecture
  - `observation_type`: "State Vector" or "Expectation Value"
  - `use_gpu`: Boolean flag for GPU acceleration

### 4.3 Data Processing

#### 4.3.1 Supported Data Types

- **Images**: PNG, JPG (single files or directories)
- **Text**: Raw text, CSV with text columns
- **CSV**: Tabular data with numeric and text features
- **Graph**: Various graph formats
- **Video**: Basic video processing

#### 4.3.2 Data Sources

- **Upload**: Direct file uploads
- **Demo**: Built-in datasets (MNIST, Cats vs Dogs)
- **URL**: Loading data from URLs

#### 4.3.3 Demo Datasets

- **MNIST**: 1,000 training images (100 per digit), 200 test images
- **Cats vs Dogs**: 1,000 training images, 200 test images

### 4.4 Training

- **Parameters**:
  - Learning rate
  - Batch size
  - Number of epochs
  - Task type (Classification, Unsupervised)
  - Optimizer selection

- **Features**:
  - Real-time training metrics
  - Visualization of loss curves
  - Quantum entanglement tracking
  - Training time estimation

### 4.5 Prediction

- **Input Methods**:
  - Image upload/URL
  - Text input
  - CSV upload
  - Graph data

- **Output**:
  - Prediction results
  - Quantum state visualization
  - Confidence scores
  - Entanglement metrics

### 4.6 Model Management

- **Save/Load**:
  - Configuration files (JSON)
  - Model weights (PyTorch format)
  - Multiple naming patterns for weights

## 5. UI Design Specifications

### 5.1 Overall Layout

The QNN Explorer uses a Streamlit-based interface with a consistent layout structure:

- **Header**: Application title and version
- **Sidebar**: Global controls and session state information
- **Main Content Area**: Tabbed interface for primary functionality
- **Footer**: Additional information and links

![UI Layout Diagram](ui_layout_diagram.png) *(Optional visual representation)*

### 5.2 Tab-Specific Designs

#### 5.2.1 Model Configuration Tab

**Layout**:
- Left column: Configuration inputs
- Right column: Visual representation of configured model

**Components**:
- Dropdown for classical backbone selection
- Input fields for quantum parameters
- Ansatz selection with visual representation
- Configuration naming and save controls
- Configuration loading selector
- Model structure visualization

**Interactions**:
- Dynamic updates to model visualization based on parameter changes
- Validation of parameter combinations
- Auto-suggestion for optimal configurations

#### 5.2.2 Data Tab

**Layout**:
- Top section: Data type and source selection
- Middle section: Data upload/selection controls
- Bottom section: Data preview and preprocessing options

**Components**:
- Data type selector (Images, Text, CSV, Graph, Video)
- Source selector (Upload, Demo, URL)
- File uploader or URL input field
- Dataset preview panel
- Feature/label selection for CSV data
- Preprocessing options panel
- Data loading button

**Interactions**:
- Dynamic form changes based on data type selection
- Automatic format validation
- Preview generation upon load
- Preprocessing progress indicators

#### 5.2.3 Train Tab

**Layout**:
- Left section: Training parameters
- Right section: Training visualization

**Components**:
- Model instantiation button
- Training parameter inputs (learning rate, batch size, epochs)
- Task type selector
- Start/stop training controls
- Real-time loss curve visualization
- Entanglement metric visualization
- Progress indicators
- Training time estimation
- Model saving controls

**Interactions**:
- Real-time metric updates during training
- Interactive pause/resume functionality
- Dynamic plotting of training metrics
- Automatic model saving on completion

#### 5.2.4 Predict Tab

**Layout**:
- Left section: Model loading and input controls
- Right section: Prediction results and visualizations

**Components**:
- Model configuration selector
- Model loading button
- Input type selector matching model configuration
- Data input controls (file upload, text input)
- Prediction button
- Results display
- Quantum state visualization
- Confidence score visualization for classification
- Entanglement metrics display

**Interactions**:
- Input validation before prediction
- Dynamic loading of appropriate input controls
- Interactive quantum state visualization

#### 5.2.5 Metrics Tab

**Layout**:
- Selection controls at top
- Visualization area below

**Components**:
- Metric type selector
- Comparison controls
- Interactive visualizations
- Data export options

**Interactions**:
- Interactive metric exploration
- Zoom and pan controls for visualizations
- Selection of data points for detailed view

### 5.3 Visualization Components

#### 5.3.1 Quantum State Visualization

**Requirements**:
- Vector representation of quantum states
- Complex amplitude visualization
- Phase visualization with color coding
- Interactive elements for state exploration
- Support for both statevector and expectation value representations

**Implementation**:
- Use Bloch sphere representations for single-qubit states
- Bar charts for amplitude visualization
- Heatmaps for multi-qubit state visualization
- Color gradients for phase representation

#### 5.3.2 Training Metrics Visualization

**Requirements**:
- Real-time plotting of loss values
- Entanglement metric tracking
- Comparison with baseline models
- Epoch and batch-level granularity

**Implementation**:
- Line charts for loss and accuracy trends
- Heatmaps for entanglement visualization
- Progress bars for training completion
- Summary statistics display

#### 5.3.3 Circuit Visualization

**Requirements**:
- Quantum circuit representation
- Gate-level visualization
- Layer structure representation
- Parameter visualization

**Implementation**:
- Standard quantum circuit notation
- Color coding for different gate types
- Collapsible layer visualization for complex circuits

### 5.4 Responsive Design

**Layout Adaptation**:
- Fluid layouts that adapt to screen width
- Column stacking on smaller screens
- Reduced visualization size on mobile
- Touch-friendly controls for mobile users

**Breakpoints**:
- Desktop: >1200px (full experience)
- Tablet: 768px-1200px (adapted layouts)
- Mobile: <768px (simplified interface)

**Component Adjustments**:
- Collapsed menus on smaller screens
- Simplified visualizations on mobile
- Touch-optimized controls
- Reduced information density

### 5.5 Accessibility Requirements

**Standards Compliance**:
- WCAG 2.1 AA compliance target
- Section 508 compliance for government use

**Specific Requirements**:
- Sufficient color contrast (minimum 4.5:1 for normal text)
- Keyboard navigation for all functions
- Screen reader compatibility with ARIA attributes
- Text alternatives for all visualizations
- Focus indicators for interactive elements
- Error messages that are clearly identified
- No reliance on color alone for conveying information

**Testing**:
- Regular accessibility audits
- Screen reader testing
- Keyboard-only navigation testing
- Color contrast verification

## 6. Specific Requirements

### 6.1 MNIST Dataset Support

- **Download and Preprocessing**:
  - Automated download via `download_mnist.py`
  - Conversion from grayscale to 3-channel format for CNN compatibility
  - Normalization with MNIST-specific values
  - Organization into class-specific folders

- **Configuration**:
  - CNN backbone (ResNet18 recommended)
  - 4-6 qubits
  - 2-4 layers
  - State Vector observation type

### 6.2 Fusion Module

- **Adaptive Fusion Layer**:
  - Automatic dimension adjustment for mismatched features
  - Support for various input sizes
  - Batch processing for quantum operations

- **Batch Processing Fix**:
  - Individual sample processing for quantum operations
  - Support for various observation types
  - Graceful handling of complex quantum states

### 6.3 GPU Acceleration

- **Requirements**:
  - PyTorch with CUDA support
  - Optional: pennylane-lightning-gpu plugin
  - Graceful fallback to CPU when GPU is unavailable

## 7. Installation and Setup

### 7.1 Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd quantum_qnn2

# Create and activate virtual environment
python -m venv quantum_env
# On Windows
quantum_env\Scripts\Activate.ps1
# On Unix/macOS
source quantum_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 7.2 GPU Support (Optional)

```bash
# Install PyTorch with CUDA
pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install PennyLane GPU plugin (optional, for faster simulation)
pip install pennylane-lightning-gpu
```

### 7.3 Dataset Preparation

```bash
# Prepare MNIST dataset
python download_mnist.py

# Prepare Cats vs Dogs dataset (optional)
python download_cats_dogs.py
```

## 8. Usage

### 8.1 Starting the Application

```bash
streamlit run app.py
```

### 8.2 Workflow

1. **Configure Model**:
   - Select backbone type, quantum parameters
   - Save configuration

2. **Load Data**:
   - Choose data type and source
   - Load and preprocess data

3. **Train Model**:
   - Set training parameters
   - Start training
   - Monitor progress

4. **Make Predictions**:
   - Load trained model
   - Input data
   - View results

## 9. Common Issues and Solutions

### 9.1 Channel Mismatch for CNN Models

- **Issue**: "Expected input to have 3 channels, but got 1 channel"
- **Solution**: Ensure grayscale images are converted to 3-channel format using the Lambda transform

### 9.2 GPU Acceleration Issues

- **Issue**: "Device default.qubit.torch does not exist"
- **Solution**: Install required PyTorch CUDA and PennyLane GPU plugins, or disable GPU in configuration

### 9.3 Dimension Mismatch

- **Issue**: "Mat1 and mat2 shapes cannot be multiplied"
- **Solution**: The adaptive fusion layer should handle this automatically by creating a properly sized linear layer

### 9.4 PennyLane Broadcasting Issues

- **Issue**: "Broadcasting with MottonenStatePreparation is not supported"
- **Solution**: Process each sample individually in the batch, then combine results

## 10. Testing

### 10.1 Test Suite

The project includes tests for various components:

- `test_fusion_fixes.py`: Tests for dimension, type, and observation issues
- `test_mnist_channels.py`: Tests for proper channel conversion
- `test_quantum_batch.py`: Tests for batch processing in quantum operations

### 10.2 Running Tests

```bash
# Run specific tests
python tests/test_mnist_channels.py

# Run all tests
python -m pytest
```

## 11. Development Guidelines

### 11.1 Code Structure

```
quantum_qnn2/
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
│   ├── mnist_loader.py     # MNIST-specific functions
│   └── demo_data/          # Demo datasets
├── utils/
│   ├── __init__.py
│   ├── visualization.py    # Plotting functions
│   └── helpers.py          # Misc utility functions
├── tests/
│   ├── __init__.py
│   ├── test_classical.py
│   ├── test_quantum.py
│   ├── test_fusion.py
│   ├── test_mnist_channels.py
│   └── test_quantum_batch.py
├── docs/
│   ├── fusion_fixes.md
│   ├── mnist_tutorial.md
│   └── software_requirements.md
├── download_mnist.py       # MNIST dataset downloader
├── requirements.txt
└── README.md
```

### 11.2 Contributing Guidelines

- Follow existing code style and patterns
- Add tests for new features
- Document changes in appropriate files
- Use TDD approach for bug fixes

## 12. References

- [PennyLane Documentation](https://pennylane.ai/docs)
- [PyTorch Documentation](https://pytorch.org/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Transformers Documentation](https://huggingface.co/docs/transformers) 