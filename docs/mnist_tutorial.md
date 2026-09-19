# MNIST Tutorial for QNN Explorer

This tutorial explains how to use the MNIST handwritten digits dataset with QNN Explorer for image classification using hybrid quantum-classical models.

## About MNIST

The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9). It's a standard benchmark dataset for machine learning, especially for image classification tasks. The dataset includes:

- 60,000 training images (reduced to 1,000 in our implementation for quantum computing)
- 10,000 test images (reduced to 200 in our implementation)
- 10 classes (digits 0-9)

## Downloading the MNIST Dataset

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

## Using MNIST with QNN Explorer

### Step 1: Download the Dataset

If you haven't already, run the download script:

```bash
python download_mnist.py
```

### Step 2: Configure a Model for MNIST

In the **Model Configuration** tab:

1. Set **Classical Backbone Type** to "CNN"
2. Choose a CNN model like "resnet18" (pre-trained models work well)
3. Set **Number of Qubits** to 4-6 (more qubits = more expressivity but slower)
4. Set **Number of Layers** to 2-4
5. Choose an **Ansatz Function** (hardware_efficient_ansatz works well)
6. Set **Observation Type** to "State Vector" for best results with small datasets
7. Save your configuration

### Step 3: Load the MNIST Dataset

In the **Data** tab:

1. Select **Images** as the data type
2. Choose **Demo Dataset** as the data source
3. Select "MNIST Digits (Image Classification)" from the dropdown
4. Click **Load Data**

The app will load 1,000 training images (100 per digit) and display a sample.

### Step 4: Train the Model

In the **Train** tab:

1. Click **Instantiate Hybrid Model**
2. Set the learning rate (0.001-0.003 works well)
3. Set batch size (16 or 32)
4. Set number of epochs (5-10)
5. Select "Classification" as the task type
6. Click **Start Training**

Training will take some time, especially with more qubits or layers.

### Step 5: Make Predictions

In the **Predict** tab:

1. Load your trained model
2. Select **Images** as input type
3. Upload an image of a handwritten digit or use one of the test images
4. Click **Predict**

The model will predict the digit and display the quantum state and confidence scores.

## Tips for Working with MNIST

1. **Dimensions**: MNIST images are grayscale 28x28. They're automatically converted to 3-channel format (by repeating the grayscale channel 3 times) to be compatible with standard CNN architectures that expect RGB inputs.

2. **Preprocessing**: The images are normalized using MNIST-specific mean and std values.

3. **Training Time**: With 4 qubits and 2 layers, training on 1,000 images should take 10-20 minutes on a CPU.

4. **Accuracy**: A well-configured hybrid model can achieve 80-90% accuracy on MNIST.

5. **Optimization**:
   - Try different CNN backbones (resnet18, mobilenet_v2)
   - Experiment with different ansatz designs
   - Adjust the quantum layer depth
   - Use learning rate scheduling for better convergence

## Example Configuration for MNIST

```python
config = {
    'classical_backbone_type': 'CNN',
    'classical_model_name': 'resnet18',
    'num_qubits': 4,
    'num_layers': 2,
    'ansatz_func': 'hardware_efficient_ansatz',
    'use_gpu': True,  # If available
    'observation_type': 'state'
}
```

This configuration provides a good balance between accuracy and training time for MNIST digit classification. 