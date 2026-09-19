import numpy as np
import warnings
import os
import base64
from io import BytesIO

# Try to import sklearn metrics, but provide fallbacks if it's not available
SKLEARN_AVAILABLE = False
try:
    from sklearn.metrics import (
        accuracy_score, log_loss, roc_auc_score, precision_score, recall_score, 
        f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, 
        r2_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    warnings.warn("scikit-learn not available. Metrics calculations will be limited.", ImportWarning)
    # Provide simple fallback implementations for basic metrics
    def accuracy_score(y_true, y_pred):
        """Simple accuracy implementation."""
        return np.mean(np.array(y_true) == np.array(y_pred))
    
    def mean_absolute_error(y_true, y_pred):
        """Simple MAE implementation."""
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    
    def mean_squared_error(y_true, y_pred):
        """Simple MSE implementation."""
        return np.mean(np.square(np.array(y_true) - np.array(y_pred)))
    
    # Other metrics will return None when sklearn is not available

# Try to import matplotlib/seaborn for plotting, with fallbacks
PLOTTING_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Agg')  # headless backend - avoids requiring a GUI/Tk
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    warnings.warn("matplotlib/seaborn not available. Plotting will be disabled.", ImportWarning)

# Try to import torch, but provide fallbacks if it's not available
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    warnings.warn("PyTorch not available. Tensor handling in metrics will be limited.", ImportWarning)

# Import plot_to_base64 from run_history
try:
    from utils.run_history import plot_to_base64
except ImportError:
    # Fallback implementation if we can't import from run_history
    def plot_to_base64(fig):
        """Convert a matplotlib figure to a base64 encoded string."""
        if not PLOTTING_AVAILABLE:
            return ""
            
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return img_str

def calculate_meyer_wallach(state_vector, num_qubits=None):
    """
    Calculate the Meyer-Wallach entanglement measure for a quantum state.
    This is a number between 0 (no entanglement) and 1 (maximal entanglement),
    which we scale to [0,2] for better visualization.
    
    Args:
        state_vector: Quantum state vector
        num_qubits: Number of qubits (can be inferred from state_vector)
        
    Returns:
        Entanglement measure E(|ψ⟩) ∈ [0,2]
    """
    # If we have a batched input, take the first sample
    if hasattr(state_vector, 'shape') and len(state_vector.shape) > 1 and state_vector.shape[0] > 1:
        state_vector = state_vector[0]
    
    # Convert to numpy for calculation if it's a tensor
    if TORCH_AVAILABLE and isinstance(state_vector, torch.Tensor):
        if state_vector.is_complex():
            # For complex tensors, we need to handle real and imaginary parts
            state_vector = state_vector.detach().cpu()
            # Convert to numpy and keep complex values
            state_vector_np = state_vector.numpy()
            if hasattr(state_vector_np, 'dtype') and np.issubdtype(state_vector_np.dtype, np.complexfloating):
                # It's already complex
                pass
            else:
                # Combine real and imaginary parts if stored separately
                if state_vector.shape[-1] == 2:  # Real and imaginary parts
                    real_part = state_vector[..., 0].numpy()
                    imag_part = state_vector[..., 1].numpy()
                    state_vector_np = real_part + 1j * imag_part
        else:
            # For real tensors
            state_vector_np = state_vector.detach().cpu().numpy()
    else:
        # Already numpy or not a tensor
        state_vector_np = state_vector
    
    # Flatten if needed
    if hasattr(state_vector_np, 'shape') and len(state_vector_np.shape) > 1:
        state_vector_np = state_vector_np.flatten()
    
    # Compute purity-based measure (simplified for demo)
    # In a real implementation, we would calculate proper reduced density matrices
    entanglement = 0.0
    
    # Simple approximation for demo: use amplitude variations as a proxy
    if hasattr(state_vector_np, 'shape'):  # Check if it's array-like
        amplitudes = np.abs(state_vector_np) ** 2
        entropy = -np.sum(amplitudes * np.log2(amplitudes + 1e-10))
        max_entropy = np.log2(len(amplitudes))
        
        # Normalize to [0,2] range
        entanglement = 2.0 * (entropy / max_entropy)
    
    return entanglement

def calculate_classification_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for positive class (for binary) or all classes
        class_names: Optional list of class names for better visualization
        
    Returns:
        dict: Dictionary of metrics
    """
    # Convert tensors to numpy arrays if needed
    if TORCH_AVAILABLE and isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if TORCH_AVAILABLE and isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if y_prob is not None and TORCH_AVAILABLE and isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()
    
    metrics = {}
    
    # Basic metrics that only need true and predicted labels
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Handle multiclass case for precision, recall, f1
    if SKLEARN_AVAILABLE:
        try:
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        except Exception as e:
            metrics['precision'] = None
            metrics['recall'] = None
            metrics['f1'] = None
            print(f"Error calculating precision/recall/f1: {e}")
    else:
        metrics['precision'] = None
        metrics['recall'] = None
        metrics['f1'] = None
    
    # Confusion matrix - store as numpy array, not list
    if SKLEARN_AVAILABLE:
        try:
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
            
            # Generate confusion matrix visualization
            if PLOTTING_AVAILABLE:
                if class_names is None:
                    # Create generic class names
                    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
                    class_names = [f"Class {i}" for i in range(len(unique_classes))]
                
                # Generate and store the confusion matrix image
                cm_img = plot_confusion_matrix(metrics['confusion_matrix'], class_names)
                metrics['confusion_matrix_img'] = cm_img
        except Exception as e:
            metrics['confusion_matrix'] = None
            print(f"Error calculating confusion matrix: {e}")
    else:
        metrics['confusion_matrix'] = None
    
    # Calculate AUC and log_loss if probabilities are provided
    if y_prob is not None and SKLEARN_AVAILABLE:
        try:
            # Check if y_prob is 1D (single class probabilities) or 2D (multiclass)
            if len(y_prob.shape) == 1 or y_prob.shape[1] == 1:
                # For binary classification with single probability column
                metrics['auc'] = roc_auc_score(y_true, y_prob.flatten())
                # For log loss, we need probabilities for both classes
                y_prob_binary = np.column_stack([1 - y_prob.flatten(), y_prob.flatten()])
                metrics['log_loss'] = log_loss(y_true, y_prob_binary)
            elif y_prob.shape[1] == 2:
                # Binary classification with two probability columns
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                metrics['log_loss'] = log_loss(y_true, y_prob)
            else:
                # Multiclass
                metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
                metrics['log_loss'] = log_loss(y_true, y_prob)
        except Exception as e:
            metrics['auc'] = None
            metrics['log_loss'] = None
            print(f"Error calculating AUC/log_loss: {e}")
    else:
        metrics['auc'] = None
        metrics['log_loss'] = None
    
    return metrics

def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        dict: Dictionary of metrics
    """
    # Convert tensors to numpy arrays if needed
    if TORCH_AVAILABLE and isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if TORCH_AVAILABLE and isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    metrics = {}
    
    # Calculate basic regression metrics
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Calculate R-squared
    if SKLEARN_AVAILABLE:
        metrics['r2'] = r2_score(y_true, y_pred)
    else:
        metrics['r2'] = None
    
    # Calculate RMSLE if data is positive
    try:
        if np.all(y_true > 0) and np.all(y_pred > 0):
            rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
            metrics['rmsle'] = rmsle
        else:
            metrics['rmsle'] = None
    except Exception as e:
        metrics['rmsle'] = None
        print(f"Error calculating RMSLE: {e}")
    
    return metrics

def calculate_clustering_metrics(X, labels, true_labels=None):
    """
    Calculate clustering metrics.
    
    Args:
        X: Feature matrix
        labels: Predicted cluster labels
        true_labels: True labels (if available)
        
    Returns:
        dict: Dictionary of metrics
    """
    # If sklearn is not available, return empty metrics
    if not SKLEARN_AVAILABLE:
        return {'silhouette': None, 'adjusted_rand': None, 'adjusted_mutual_info': None}
    
    # Convert tensors to numpy arrays if needed
    if TORCH_AVAILABLE and isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if TORCH_AVAILABLE and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if true_labels is not None and TORCH_AVAILABLE and isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.detach().cpu().numpy()
    
    # Ensure X is 2D
    if hasattr(X, 'shape') and len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    
    metrics = {}
    
    # Calculate silhouette score
    try:
        metrics['silhouette'] = silhouette_score(X, labels)
    except Exception as e:
        metrics['silhouette'] = None
        print(f"Error calculating silhouette score: {e}")
    
    # If true labels are available, calculate more metrics
    if true_labels is not None:
        try:
            metrics['adjusted_rand'] = adjusted_rand_score(true_labels, labels)
            metrics['adjusted_mutual_info'] = adjusted_mutual_info_score(true_labels, labels)
        except Exception as e:
            metrics['adjusted_rand'] = None
            metrics['adjusted_mutual_info'] = None
            print(f"Error calculating supervised clustering metrics: {e}")
    else:
        # Set these metrics to None when true_labels are not provided
        metrics['adjusted_rand'] = None
        metrics['adjusted_mutual_info'] = None
    
    return metrics

def plot_confusion_matrix(cm, class_names=None):
    """
    Plot confusion matrix and return as a base64 encoded image.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        
    Returns:
        str: Base64 encoded image
    """
    # Return empty string if plotting is not available
    if not PLOTTING_AVAILABLE:
        return ""
    
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    # Convert the plot to base64
    return plot_to_base64(plt.gcf())

def plot_training_accuracy(epochs, accuracies, title="Training Accuracy over Time"):
    """
    Plot training accuracy over epochs and return the figure.
    
    Args:
        epochs: List of epoch numbers
        accuracies: List of accuracy values
        title: Optional title for the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Return None if plotting is not available
    if not PLOTTING_AVAILABLE:
        return None
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, marker='o')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    
    return fig

def plot_regression_predictions(y_true, y_pred):
    """
    Plot true vs predicted values for regression and return the figure.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Return None if plotting is not available
    if not PLOTTING_AVAILABLE:
        return None
    
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    
    return fig

def get_metric_descriptions():
    """Return descriptions of all available metrics."""
    return {
        # Classification metrics
        "accuracy": "Proportion of correct predictions among the total number of predictions",
        "precision": "Ratio of true positives to the total predicted positives (weighted average across classes)",
        "recall": "Ratio of true positives to the total actual positives (weighted average across classes)",
        "f1": "Harmonic mean of precision and recall (weighted average across classes)",
        "auc": "Area Under the ROC Curve, measures the model's ability to distinguish between classes",
        "log_loss": "Cross-entropy loss, penalizes confident but incorrect predictions more heavily",
        "confusion_matrix": "Table showing the counts of true positive, false positive, true negative, and false negative predictions",
        
        # Regression metrics
        "mae": "Mean Absolute Error - average of absolute differences between predictions and actual values",
        "mse": "Mean Squared Error - average of squared differences between predictions and actual values",
        "rmse": "Root Mean Square Error - square root of MSE, in the same units as the target variable",
        "rmsle": "Root Mean Square Logarithmic Error - useful when targets have a wide range of values",
        "r2": "R-squared coefficient of determination - proportion of variance in the target explained by the model",
        
        # Clustering metrics
        "silhouette": "Measure of how similar an object is to its own cluster compared to other clusters (-1 to 1)",
        "adjusted_rand": "Similarity measure between two clusterings, adjusted for chance (-1 to 1)",
        "adjusted_mutual_info": "Mutual information between two clusterings, adjusted for chance (0 to 1)"
    } 