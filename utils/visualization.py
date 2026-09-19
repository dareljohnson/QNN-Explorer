import matplotlib
matplotlib.use('Agg')  # headless backend - avoids requiring a GUI/Tk
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
from typing import List, Optional, Union, Tuple, Dict
import pandas as pd
import os
import warnings
import sys

# Try to import streamlit, but provide fallbacks if it's not available
STREAMLIT_AVAILABLE = False
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    warnings.warn("Streamlit not available. Visualization functions will be modified for non-Streamlit environment.", ImportWarning)
    # Create a minimal st stub for testing purposes
    class StreamlitStub:
        """A minimal streamlit stub for testing purposes."""
        @staticmethod
        def warning(*args, **kwargs): 
            warnings.warn(str(args[0]))
        @staticmethod
        def info(*args, **kwargs): 
            print(f"INFO: {args[0]}")
        @staticmethod
        def error(*args, **kwargs): 
            print(f"ERROR: {args[0]}")
        @staticmethod
        def success(*args, **kwargs): 
            print(f"SUCCESS: {args[0]}")
    
    # Use the stub instead of real streamlit
    st = StreamlitStub()

# Try to import pennylane, but provide fallbacks if it's not available
PENNYLANE_AVAILABLE = False
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    warnings.warn("PennyLane not available. Circuit visualization will be disabled.", ImportWarning)
    # Create minimal stub for testing
    class PennyLaneStub:
        """A minimal PennyLane stub for testing purposes."""
        pass
    
    qml = PennyLaneStub()

# Try to import torch, but provide fallbacks if it's not available
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    warnings.warn("PyTorch not available. Some visualization features will be limited.", ImportWarning)

# Try to import seaborn, but provide fallbacks if it's not available
SEABORN_AVAILABLE = False
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    warnings.warn("Seaborn not available. Some plot styling will be disabled.", ImportWarning)

def plot_to_base64(fig):
    """
    Convert a matplotlib figure to a base64 encoded string.
    
    Args:
        fig: A matplotlib figure object
        
    Returns:
        A base64 encoded string of the figure
    """
    if fig is None:
        return None
        
    img_stream = io.BytesIO()
    fig.savefig(img_stream, format='png', bbox_inches='tight')
    img_stream.seek(0)
    img_data = base64.b64encode(img_stream.read()).decode('utf-8')
    img_stream.close()
    plt.close(fig)  # Close the figure to free up memory
    return img_data

def plot_circuit(circuit, wire_order=None, input_args=None):
    """
    Draw a quantum circuit.
    
    Args:
        circuit: A quantum circuit (e.g., PennyLane QNode)
        wire_order: Optional custom wire ordering
        input_args: Optional input arguments for the circuit
        
    Returns:
        A base64 encoded string of the circuit diagram
    """
    if not PENNYLANE_AVAILABLE:
        warnings.warn("PennyLane not available. Circuit visualization is disabled.", ImportWarning)
        # Create an error message image
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Circuit visualization requires PennyLane.", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.axis('off')
        return plot_to_base64(fig)
    
    try:
        # If input_args are provided, execute the circuit to create the tape
        if input_args is not None:
            try:
                # Execute circuit with provided input_args
                if isinstance(input_args, list):
                    circuit(*input_args)
                else:
                    circuit(input_args)
                if STREAMLIT_AVAILABLE:
                    st.success("Circuit executed successfully with provided input arguments.")
            except Exception as e:
                if STREAMLIT_AVAILABLE:
                    st.warning(f"Could not execute circuit with provided input arguments: {str(e)}")
                    st.info("Attempting to draw the circuit based on the instantiated QNode.")
                    st.info("Note: Drawing requires representative input shapes.")
                warnings.warn(f"Could not execute circuit with provided input arguments: {str(e)}")
        
        # Try to get the circuit from a QNode
        if hasattr(circuit, 'qtape') and circuit.qtape is not None:
            queue = circuit.qtape.operations + circuit.qtape.measurements
            wires = circuit.qtape.wires
        elif hasattr(circuit, 'queue') and hasattr(circuit, 'wires'):
            queue = circuit.queue
            wires = circuit.wires
        else:
            # If we still don't have a tape but input_args are provided, try drawing directly
            if input_args is not None:
                try:
                    fig, ax = qml.draw_mpl(circuit)(*input_args)
                    return plot_to_base64(fig)
                except Exception as e:
                    warnings.warn(f"Direct circuit drawing failed: {str(e)}")
            
            # Generic fallback - create an informative error message
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "Circuit visualization requires executing the circuit first.\n" +
                              "Try providing valid input_args parameter or execute the circuit before visualization.", 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.axis('off')
            return plot_to_base64(fig)
        
        # Use pennylane's qml.draw_mpl
        fig, ax = qml.draw_mpl(queue, wire_order=wire_order, show_all_wires=True)(wires)
        
        # Return as base64 encoded string
        return plot_to_base64(fig)
    except Exception as e:
        # If circuit visualization fails, create an error message image
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"Circuit visualization failed: {str(e)}", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.axis('off')
        return plot_to_base64(fig)

def plot_state_vector(state, num_qubits=None, max_states=16):
    """
    Plot a quantum state vector as bar chart of probabilities.
    
    Args:
        state: Quantum state vector or tensor
        num_qubits: Number of qubits, inferred from state if not provided
        max_states: Maximum number of states to show
        
    Returns:
        A matplotlib figure object
    """
    if state is None:
        warnings.warn("No state vector available to plot.")
        return None
    
    try:
        # Convert tensor to numpy if needed
        if TORCH_AVAILABLE and isinstance(state, torch.Tensor):
            state_data = state.detach().cpu().numpy()
        else:
            state_data = state
        
        # If this is a batch of state vectors, take just the first one
        if len(state_data.shape) > 1 and state_data.shape[0] > 1:
            state_data = state_data[0]
            if STREAMLIT_AVAILABLE:
                st.info("Showing state vector for the first item in the batch.")
            else:
                print("INFO: Showing state vector for the first item in the batch.")
        
        # Flatten if needed (e.g., for complex state with real/imag parts)
        state_data = state_data.flatten()
        
        # Calculate probabilities
        if np.iscomplexobj(state_data):
            probs = np.abs(state_data) ** 2
        else:
            # If not complex, square the values (assuming normalized state)
            probs = state_data ** 2
        
        # If too many states, keep only the largest probabilities
        if len(probs) > max_states:
            # Create indices for all states, then sort by probability
            indices = np.argsort(probs)[::-1]  # Descending order
            # Keep only top states
            indices = indices[:max_states]
            probs_reduced = probs[indices]
            
            # Create labels based on binary representation
            state_labels = [f"|{i:0{int(np.log2(len(probs)))}b}⟩" for i in indices]
            
            # Plot the reduced set
            probs = probs_reduced
        else:
            # Use all states
            num_qubits = num_qubits or int(np.log2(len(probs)))
            state_labels = [f"|{i:0{num_qubits}b}⟩" for i in range(len(probs))]
        
        # Create a new figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the probabilities
        bars = ax.bar(state_labels, probs)
        
        # Color the bars
        for bar, prob in zip(bars, probs):
            bar.set_color(plt.cm.viridis(prob))
        
        # Add labels and title
        ax.set_xlabel('Quantum State')
        ax.set_ylabel('Probability')
        ax.set_title('Quantum State Probabilities')
        
        # Rotate x-tick labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add grid lines and tight layout
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        warnings.warn(f"Error plotting state vector: {e}")
        if STREAMLIT_AVAILABLE:
            st.error(f"Error plotting state vector: {e}")
        return None

def plot_training_history(history, metrics=None, window_size=1):
    """
    Plot training history from a dictionary of metrics.
    
    Args:
        history: Dictionary with keys as metric names and values as lists of values
        metrics: Optional list of specific metrics to plot
        window_size: Optional smoothing window size
        
    Returns:
        A matplotlib figure object
    """
    if not history:
        st.info("No training history available yet.")
        return None
    
    # If metrics not specified, use all numeric metrics in history
    if metrics is None:
        metrics = [k for k, v in history.items() 
                  if isinstance(v, (list, np.ndarray)) and 
                  len(v) > 0 and
                  not isinstance(v[0], str)]
    
    # Filter out metrics with no data
    metrics = [m for m in metrics if m in history and len(history[m]) > 0]
    
    if not metrics:
        return None
    
    # Create one plot per metric
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3*n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]  # Make it a list for consistent indexing
    
    for i, metric in enumerate(metrics):
        values = history[metric]
        
        # Convert to numpy array
        if isinstance(values, list):
            # Filter out None/NaN values
            values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
            values = np.array(values)
        
        if len(values) == 0:
            continue
            
        epochs = np.arange(1, len(values) + 1)
        
        # Apply smoothing if window_size > 1
        if window_size > 1 and len(values) > window_size:
            smooth_values = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            # Padding at the beginning to match original length
            padding = np.full(window_size-1, np.nan)
            smooth_values = np.concatenate((padding, smooth_values))
            
            # Plot both raw and smoothed data
            axes[i].plot(epochs, values, 'o-', alpha=0.5, label=f'Raw {metric}')
            axes[i].plot(epochs, smooth_values, 'r-', linewidth=2, label=f'Smoothed {metric}')
            axes[i].legend()
        else:
            # Just plot the raw data
            axes[i].plot(epochs, values, 'o-', label=metric)
        
        axes[i].set_ylabel(metric.capitalize())
        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].set_title(f'{metric.capitalize()} over Training')
    
    axes[-1].set_xlabel('Epoch')
    plt.tight_layout()
    
    return fig

def plot_entanglement(entanglement_values, title="Quantum Entanglement during Training"):
    """
    Plot the evolution of quantum entanglement during training.
    
    Args:
        entanglement_values: List of entanglement measurements
        title: Title for the plot
        
    Returns:
        A matplotlib figure object
    """
    if not entanglement_values:
        st.info("No entanglement history available.")
        return None
    
    # Filter out None/NaN values
    valid_values = []
    valid_epochs = []
    
    for i, e in enumerate(entanglement_values):
        if e is not None and not (isinstance(e, float) and np.isnan(e)):
            valid_values.append(e)
            valid_epochs.append(i + 1)  # 1-indexed epochs
    
    if not valid_values:
        return None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the entanglement values
    ax.plot(valid_epochs, valid_values, 'o-', color='purple', linewidth=2)
    
    # Style the plot
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Entanglement Measure (Q)')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add reference lines for interpretation
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5)
    ax.axhline(y=1.5, color='b', linestyle='--', alpha=0.5)
    
    # Add annotations for reference lines
    ax.text(valid_epochs[-1] + 0.5, 0.5, 'Low', verticalalignment='center', color='r')
    ax.text(valid_epochs[-1] + 0.5, 1.0, 'Medium', verticalalignment='center', color='g')
    ax.text(valid_epochs[-1] + 0.5, 1.5, 'High', verticalalignment='center', color='b')
    
    plt.tight_layout()
    
    return fig

def plot_training_accuracy(epochs, accuracies, title="Training Accuracy"):
    """
    Plot the evolution of model accuracy during training.
    
    Args:
        epochs: List of epoch numbers
        accuracies: List of accuracy values (0-1)
        title: Title for the plot
        
    Returns:
        A matplotlib figure object or None if no valid data
    """
    if not epochs or not accuracies or len(epochs) != len(accuracies):
        return None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the accuracy values
    ax.plot(epochs, accuracies, 'o-', color='green', linewidth=2)
    
    # Style the plot
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis limits with some padding
    y_min = max(0, min(accuracies) - 0.05)
    y_max = min(1, max(accuracies) + 0.05)
    ax.set_ylim(y_min, y_max)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Add a trend line if we have enough data points
    if len(epochs) > 2:
        try:
            z = np.polyfit(epochs, accuracies, 1)
            p = np.poly1d(z)
            ax.plot(epochs, p(epochs), "--", color='blue', alpha=0.7, 
                    label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
            ax.legend(loc='lower right')
        except:
            # Skip trend line if there's an error
            pass
    
    plt.tight_layout()
    
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
    # Check if required libraries are available
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        warnings.warn("matplotlib or numpy not available. Plotting will be disabled.", ImportWarning)
        return None
    
    # Convert tensors to numpy arrays if needed
    if 'torch' in sys.modules and hasattr(sys.modules['torch'], 'Tensor'):
        import torch
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('True vs Predicted Values')
    
    plt.tight_layout()
    
    return fig 