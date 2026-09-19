"""
Run History Tab Module

This module provides Streamlit UI components for viewing, managing, and recovering
run history data. It includes visualizations of training metrics, model configurations,
and dataset information, along with error recovery features for handling corrupted
run data.

The module is designed to work with or without Streamlit available, falling back
to a minimal stub implementation for testing environments.
"""

# Add conditional import for Streamlit
import pandas as pd
import numpy as np
import warnings
import os
import time
from datetime import datetime

# Try to import streamlit, but provide a fallback if it's not available
STREAMLIT_AVAILABLE = False
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    warnings.warn("Streamlit not available. Run History Tab visualization will be disabled.", ImportWarning)
    # Create a minimal st stub for testing purposes
    class StreamlitStub:
        """A minimal streamlit stub for testing purposes.
        
        This class provides stub implementations of Streamlit functions to allow
        the module to be imported and tested even when Streamlit is not available.
        Each method is a no-op that mimics the expected return values.
        """
        @staticmethod
        def header(*args, **kwargs): pass
        @staticmethod
        def subheader(*args, **kwargs): pass
        @staticmethod
        def write(*args, **kwargs): pass
        @staticmethod
        def info(*args, **kwargs): pass
        @staticmethod
        def success(*args, **kwargs): pass
        @staticmethod
        def warning(*args, **kwargs): pass
        @staticmethod
        def error(*args, **kwargs): pass
        @staticmethod
        def dataframe(*args, **kwargs): pass
        @staticmethod
        def selectbox(*args, **kwargs): return args[1][0] if args[1] else None
        @staticmethod
        def button(*args, **kwargs): return False
        @staticmethod
        def metric(*args, **kwargs): pass
        @staticmethod
        def markdown(*args, **kwargs): pass
        @staticmethod
        def code(*args, **kwargs): pass
        @staticmethod
        def columns(n): 
            class ColumnStub:
                def __enter__(self): return StreamlitStub()
                def __exit__(self, *args): pass
            return [ColumnStub() for _ in range(n)]
        @staticmethod
        def radio(*args, **kwargs): return args[1][0] if args[1] else None
        @staticmethod
        def line_chart(*args, **kwargs): pass
        @staticmethod
        def image(*args, **kwargs): pass
        @staticmethod
        def text(*args, **kwargs): pass
        @staticmethod
        def rerun(*args, **kwargs): pass
        @staticmethod
        def container(*args, **kwargs):
            class ContainerStub:
                def __enter__(self): return StreamlitStub()
                def __exit__(self, *args): pass
            return ContainerStub()
        
        class column_config:
            """Stub for column_config in Streamlit."""
            @staticmethod
            class Column:
                def __init__(self, *args, **kwargs): pass
        
        class session_state:
            """Mock session state."""
            _dict = {}  # Use a class variable to store state
            
            @classmethod
            def __getattr__(cls, name):
                return cls._dict.get(name, None)
            
            @classmethod
            def __setattr__(cls, name, value):
                cls._dict[name] = value
            
            @classmethod
            def get(cls, key, default=None):
                return cls._dict.get(key, default)
            
            @classmethod
            def reset(cls):
                cls._dict = {}
    
    # Use the stub
    st = StreamlitStub()

from utils.run_history import (
    load_run_history, 
    add_run, 
    get_run, 
    update_run_status, 
    save_run_details, 
    load_run_details,
    delete_run,
    RUN_DETAILS_PATH
)

# Import metric descriptions
try:
    from utils.metrics import get_metric_descriptions
    METRICS_DESCRIPTIONS = get_metric_descriptions()
except ImportError:
    METRICS_DESCRIPTIONS = {}

def format_metric_value(value, metric_name):
    """Format a metric value based on the metric type."""
    if value is None:
        return "N/A"
    
    if metric_name in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'silhouette', 'adjusted_rand', 'adjusted_mutual_info']:
        # These metrics are typically between 0 and 1
        return f"{value:.4f}" if isinstance(value, (float, int)) else str(value)
    elif metric_name in ['r2']:
        # R² can be negative or <= 1
        return f"{value:.4f}" if isinstance(value, (float, int)) else str(value)
    elif metric_name in ['mae', 'mse', 'rmse', 'rmsle', 'log_loss']:
        # Error metrics - typically larger values
        return f"{value:.4f}" if isinstance(value, (float, int)) else str(value)
    else:
        # Default formatting
        return f"{value:.4f}" if isinstance(value, (float, int)) else str(value)

def render_run_history_tab():
    """Render the Run History tab in the Streamlit interface.
    
    This function creates a complete UI for browsing, viewing, and managing
    training run history. It includes:
    
    - A table of all training runs
    - Detailed view of selected run information
    - Visualizations of training metrics and model performance
    - Error recovery UI for handling corrupted run data
    - Options to update run status or delete runs
    
    The UI adapts to different types of models (classification, regression,
    clustering) and displays appropriate metrics for each. It also includes
    robust error handling to prevent UI crashes when data is missing or corrupted.
    
    Returns:
        None: This function modifies the Streamlit UI directly
    """
    if not STREAMLIT_AVAILABLE:
        print("Run History Tab requires Streamlit to be installed.")
        return
        
    st.header("Training Run History")
    
    # Load run history
    try:
        run_history = load_run_history()
    except Exception as e:
        st.error(f"Failed to load run history: {e}")
        return
    
    # Initialize session state for selected run if needed
    if "selected_run_id" not in st.session_state:
        st.session_state.selected_run_id = None
    
    # Show the history table if not empty
    if len(run_history) > 0:
        # Format duration as a readable string
        run_history['duration_str'] = run_history['duration'].apply(
            lambda x: f"{int(x // 3600)}h {int((x % 3600) // 60)}m {int(x % 60)}s" if x >= 3600 else 
                     f"{int(x // 60)}m {int(x % 60)}s" if x >= 60 else 
                     f"{int(x)}s"
        )
        
        # Create a copy for display with selected columns
        display_df = run_history[['run_id', 'timestamp', 'config_name', 'duration_str', 'status']].copy()
        display_df.columns = ['ID', 'Timestamp', 'Configuration', 'Duration', 'Status']
        
        # Apply highlighting based on the status
        def highlight_status(s):
            return ['background-color: #a1f5a1' if 'Pass' in str(x) else 
                    'background-color: #f5a1a1' if 'Fail' in str(x) else 
                    '' for x in s]
        
        # Add a hint to the user about how to select runs
        st.write("**Run History:**")
        
        # Display the table with full width and highlighting
        st.dataframe(
            display_df.style.apply(lambda x: highlight_status(x) if x.name == 'Status' else [''] * len(x), axis=0),
            use_container_width=True,
            hide_index=True
        )
        
        # Create a dropdown for run selection
        run_options = [{'label': f"Run {row['ID']}: {row['Configuration']} ({row['Timestamp']})", 
                        'value': row['ID']} 
                      for _, row in display_df.iterrows()]
        run_options.insert(0, {'label': 'Select a run...', 'value': None})
        
        selected_option = st.selectbox(
            "Select a run to view details:",
            options=[opt['value'] for opt in run_options],
            format_func=lambda x: next((opt['label'] for opt in run_options if opt['value'] == x), 'Select a run...'),
            key="run_selector"
        )
        
        # Update session state when dropdown selection changes
        if selected_option is not None and selected_option != st.session_state.get('selected_run_id'):
            st.session_state['selected_run_id'] = selected_option
            st.rerun()
        
        # Run details section
        if st.session_state.selected_run_id is not None:
            st.markdown("---")
            st.subheader("Run Details")
            
            # Show which run is selected
            try:
                selected_run_info = run_history[run_history['run_id'] == st.session_state.selected_run_id].iloc[0]
                st.info(f"Viewing Run {st.session_state.selected_run_id} - {selected_run_info['config_name']} ({selected_run_info['timestamp']})")
                
                # Load details for the selected run
                try:
                    run_details = load_run_details(st.session_state.selected_run_id)
                except Exception as e:
                    st.error(f"Error loading run details: {e}")
                    st.info("Attempting to repair run data...")
                    
                    # Check if files exist
                    run_id = st.session_state.selected_run_id
                    json_path = os.path.join(RUN_DETAILS_PATH, f"run_{run_id}.json")
                    pkl_path = os.path.join(RUN_DETAILS_PATH, f"run_{run_id}.pkl")
                    
                    if os.path.exists(json_path):
                        st.info(f"JSON file exists at {json_path} but may be corrupted")
                    else:
                        st.warning(f"JSON file not found at {json_path}")
                        
                    if os.path.exists(pkl_path):
                        st.info(f"Pickle file exists at {pkl_path}, will try to use as fallback")
                    else:
                        st.warning(f"Pickle file not found at {pkl_path}")
                    
                    # Try to get minimal details
                    run_details = {
                        'recovery_note': f"This run has corrupted data files. Error: {str(e)}",
                        'run_id': run_id,
                        'config_name': selected_run_info['config_name'],
                        'status': selected_run_info['status'],
                        'timestamp': selected_run_info['timestamp'],
                    }
                
                if run_details:
                    # Display recovery note if this is recovered data
                    if 'recovery_note' in run_details:
                        st.warning(run_details['recovery_note'])
                    
                    # Display run information
                    run_info = run_history[run_history['run_id'] == st.session_state.selected_run_id].iloc[0]
                    
                    # Create columns for run info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Configuration", run_info['config_name'])
                    with col2:
                        st.metric("Duration", run_info['duration_str'])
                    with col3:
                        # Style status based on value
                        status_color = ""
                        if run_info['status'] == "Completed":
                            status_color = "🟢"
                        elif run_info['status'] == "Failed":
                            status_color = "🔴"
                        elif run_info['status'] == "In Progress":
                            status_color = "🟡"
                        elif "Pass" in run_info['status']:
                            status_color = "✅"
                        elif "Fail" in run_info['status']:
                            status_color = "❌"
                        
                        st.metric("Status", f"{status_color} {run_info['status']}")
                    
                    # Display validation notes if present for Pass/Fail statuses
                    if ("Pass" in run_info['status'] or "Fail" in run_info['status']) and 'validation_note' in run_details:
                        st.markdown("#### Validation Notes")
                        note_style = "background-color:#e6f7e6;padding:10px;border-radius:5px;" if "Pass" in run_info['status'] else "background-color:#f7e6e6;padding:10px;border-radius:5px;"
                        st.markdown(f"<div style='{note_style}'>{run_details['validation_note']}</div>", unsafe_allow_html=True)
                    
                    # Add quantum model configuration details
                    if 'model_config' in run_details:
                        model_config = run_details['model_config']
                        st.subheader("Quantum Model Configuration")
                        
                        # Create a multi-column layout for configuration parameters
                        cols = st.columns(3)
                        
                        # Column 1: Quantum parameters
                        with cols[0]:
                            st.markdown("**Quantum Parameters**")
                            if 'num_qubits' in model_config:
                                st.metric("Number of Qubits", model_config['num_qubits'])
                            if 'num_layers' in model_config:
                                st.metric("Number of Layers", model_config['num_layers'])
                            if 'ansatz_name' in model_config:
                                st.metric("Ansatz", model_config['ansatz_name'])
                            if 'encoding_method' in model_config:
                                st.metric("Encoding", model_config['encoding_method'])
                            if 'observation_type' in model_config:
                                st.metric("Observation", model_config['observation_type'])
                        
                        # Column 2: Classical parameters
                        with cols[1]:
                            st.markdown("**Classical Parameters**")
                            if 'classical_backbone_type' in model_config:
                                backbone_type = model_config['classical_backbone_type']
                                st.metric("Backbone Type", backbone_type)
                            if 'classical_model_name' in model_config and model_config['classical_model_name']:
                                st.metric("Model Name", model_config['classical_model_name'])
                            if 'quantum_input_size' in model_config:
                                st.metric("Input Size", model_config['quantum_input_size'])
                        
                        # Column 3: Training parameters
                        with cols[2]:
                            st.markdown("**Hardware Settings**")
                            if 'use_gpu' in model_config:
                                st.metric("GPU Used", "Yes" if model_config['use_gpu'] else "No")
                        
                        # Add training parameters if available
                        if 'training_params' in run_details:
                            training_params = run_details['training_params']
                            st.markdown("**Training Parameters**")
                            
                            train_cols = st.columns(3)
                            with train_cols[0]:
                                if 'learning_rate' in training_params:
                                    st.metric("Learning Rate", f"{training_params['learning_rate']:.4f}")
                            with train_cols[1]:
                                if 'batch_size' in training_params:
                                    st.metric("Batch Size", training_params['batch_size'])
                            with train_cols[2]:
                                if 'epochs' in training_params:
                                    st.metric("Epochs", training_params['epochs'])
                    
                    # Add weights file information if available
                    if 'weights_files' in run_details and run_details['weights_files']:
                        st.subheader("Model Weights")
                        weight_files = run_details['weights_files']
                        
                        # Create a formatted string with all weight files
                        weight_files_str = "\n".join(f"- {weight_file}" for weight_file in weight_files)
                        st.code(weight_files_str, language="markdown")
                        
                        # Add a button to copy the primary weight filename to clipboard
                        if weight_files:
                            primary_weight = weight_files[1] if len(weight_files) > 1 else weight_files[0]  # Get descriptive name if available
                            st.code(primary_weight, language="bash")
                    
                    # Show metrics based on task type
                    if 'task_type' in run_details:
                        task_type = run_details['task_type']
                        
                        # Classification metrics
                        if task_type == 'Classification':
                            st.subheader("Classification Metrics")
                            
                            if 'metrics' in run_details:
                                metrics = run_details['metrics']
                                
                                # Create a nice table to display all metrics
                                metric_data = []
                                
                                # Main metrics in the first row
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    value = format_metric_value(metrics.get('accuracy'), 'accuracy')
                                    st.metric("Accuracy", value)
                                    metric_data.append({
                                        "Metric": "Accuracy", 
                                        "Value": value,
                                        "Description": METRICS_DESCRIPTIONS.get('accuracy', '')
                                    })
                                with col2:
                                    value = format_metric_value(metrics.get('precision'), 'precision')
                                    st.metric("Precision", value)
                                    metric_data.append({
                                        "Metric": "Precision", 
                                        "Value": value,
                                        "Description": METRICS_DESCRIPTIONS.get('precision', '')
                                    })
                                with col3:
                                    value = format_metric_value(metrics.get('recall'), 'recall')
                                    st.metric("Recall", value)
                                    metric_data.append({
                                        "Metric": "Recall", 
                                        "Value": value,
                                        "Description": METRICS_DESCRIPTIONS.get('recall', '')
                                    })
                                with col4:
                                    value = format_metric_value(metrics.get('f1'), 'f1')
                                    st.metric("F1 Score", value)
                                    metric_data.append({
                                        "Metric": "F1 Score", 
                                        "Value": value,
                                        "Description": METRICS_DESCRIPTIONS.get('f1', '')
                                    })
                                
                                # Second row of metrics
                                col1, col2 = st.columns(2)
                                with col1:
                                    value = format_metric_value(metrics.get('auc'), 'auc')
                                    st.metric("AUC", value)
                                    metric_data.append({
                                        "Metric": "Area Under Curve", 
                                        "Value": value,
                                        "Description": METRICS_DESCRIPTIONS.get('auc', '')
                                    })
                                with col2:
                                    value = format_metric_value(metrics.get('log_loss'), 'log_loss')
                                    st.metric("Log Loss", value)
                                    metric_data.append({
                                        "Metric": "Logarithmic Loss", 
                                        "Value": value,
                                        "Description": METRICS_DESCRIPTIONS.get('log_loss', '')
                                    })
                                
                                # Show all metrics in a table for reference
                                with st.expander("All Metrics Details"):
                                    st.dataframe(pd.DataFrame(metric_data))
                                
                                # Add a section for final metrics with gauges
                                if any(f"final_{m}" in run_details for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']):
                                    st.subheader("Final Performance Metrics")
                                    
                                    # Create 2 rows of 3 gauges each for the metrics
                                    gauge_cols1 = st.columns(3)
                                    gauge_cols2 = st.columns(3)
                                    
                                    # Row 1: Accuracy, Precision, Recall
                                    with gauge_cols1[0]:
                                        if 'final_accuracy' in run_details:
                                            value = run_details['final_accuracy']
                                            st.markdown(f"### Accuracy")
                                            st.progress(min(1.0, float(value)))
                                            st.markdown(f"**{value:.4f}**")
                                        elif 'accuracy' in metrics and metrics['accuracy'] is not None:
                                            value = metrics['accuracy']
                                            st.markdown(f"### Accuracy")
                                            st.progress(min(1.0, float(value)))
                                            st.markdown(f"**{value:.4f}**")
                                    
                                    with gauge_cols1[1]:
                                        if 'final_precision' in run_details:
                                            value = run_details['final_precision']
                                            st.markdown(f"### Precision")
                                            st.progress(min(1.0, float(value)))
                                            st.markdown(f"**{value:.4f}**")
                                        elif 'precision' in metrics and metrics['precision'] is not None:
                                            value = metrics['precision']
                                            st.markdown(f"### Precision")
                                            st.progress(min(1.0, float(value)))
                                            st.markdown(f"**{value:.4f}**")
                                    
                                    with gauge_cols1[2]:
                                        if 'final_recall' in run_details:
                                            value = run_details['final_recall']
                                            st.markdown(f"### Recall")
                                            st.progress(min(1.0, float(value)))
                                            st.markdown(f"**{value:.4f}**")
                                        elif 'recall' in metrics and metrics['recall'] is not None:
                                            value = metrics['recall']
                                            st.markdown(f"### Recall")
                                            st.progress(min(1.0, float(value)))
                                            st.markdown(f"**{value:.4f}**")
                                    
                                    # Row 2: F1, AUC
                                    with gauge_cols2[0]:
                                        if 'final_f1' in run_details:
                                            value = run_details['final_f1']
                                            st.markdown(f"### F1 Score")
                                            st.progress(min(1.0, float(value)))
                                            st.markdown(f"**{value:.4f}**")
                                        elif 'f1' in metrics and metrics['f1'] is not None:
                                            value = metrics['f1']
                                            st.markdown(f"### F1 Score")
                                            st.progress(min(1.0, float(value)))
                                            st.markdown(f"**{value:.4f}**")
                                    
                                    with gauge_cols2[1]:
                                        if 'final_auc' in run_details:
                                            value = run_details['final_auc']
                                            st.markdown(f"### AUC")
                                            st.progress(min(1.0, float(value)))
                                            st.markdown(f"**{value:.4f}**")
                                        elif 'auc' in metrics and metrics['auc'] is not None:
                                            value = metrics['auc']
                                            st.markdown(f"### AUC")
                                            st.progress(min(1.0, float(value)))
                                            st.markdown(f"**{value:.4f}**")
                                
                                # Show confusion matrix
                                if 'confusion_matrix' in metrics and metrics['confusion_matrix'] is not None:
                                    st.subheader("Confusion Matrix")
                                    
                                    # If we have a base64 image, display it
                                    if 'confusion_matrix_img' in run_details:
                                        st.image(f"data:image/png;base64,{run_details['confusion_matrix_img']}")
                                    elif 'confusion_matrix_img' in metrics:
                                        st.image(f"data:image/png;base64,{metrics['confusion_matrix_img']}")
                                    else:
                                        # Otherwise, display as text
                                        st.text(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
                        
                        # Regression metrics
                        elif task_type == 'Regression':
                            st.subheader("Regression Metrics")
                            
                            if 'metrics' in run_details:
                                metrics = run_details['metrics']
                                
                                # Create a nice table to display all metrics
                                metric_data = []
                                
                                # First row of metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    value = format_metric_value(metrics.get('mae'), 'mae')
                                    st.metric("MAE", value)
                                    metric_data.append({
                                        "Metric": "Mean Absolute Error", 
                                        "Value": value,
                                        "Description": METRICS_DESCRIPTIONS.get('mae', '')
                                    })
                                with col2:
                                    value = format_metric_value(metrics.get('mse'), 'mse')
                                    st.metric("MSE", value)
                                    metric_data.append({
                                        "Metric": "Mean Squared Error", 
                                        "Value": value,
                                        "Description": METRICS_DESCRIPTIONS.get('mse', '')
                                    })
                                with col3:
                                    value = format_metric_value(metrics.get('rmse'), 'rmse')
                                    st.metric("RMSE", value)
                                    metric_data.append({
                                        "Metric": "Root Mean Square Error", 
                                        "Value": value,
                                        "Description": METRICS_DESCRIPTIONS.get('rmse', '')
                                    })
                                
                                # Second row of metrics
                                col1, col2 = st.columns(2)
                                with col1:
                                    value = format_metric_value(metrics.get('rmsle'), 'rmsle')
                                    st.metric("RMSLE", value)
                                    metric_data.append({
                                        "Metric": "Root Mean Square Logarithmic Error", 
                                        "Value": value,
                                        "Description": METRICS_DESCRIPTIONS.get('rmsle', '')
                                    })
                                with col2:
                                    value = format_metric_value(metrics.get('r2'), 'r2')
                                    st.metric("R² Score", value)
                                    metric_data.append({
                                        "Metric": "R-Squared Score", 
                                        "Value": value,
                                        "Description": METRICS_DESCRIPTIONS.get('r2', '')
                                    })
                                
                                # Show all metrics in a table for reference
                                with st.expander("All Metrics Details"):
                                    st.dataframe(pd.DataFrame(metric_data))
                                
                                # Add a section for final regression metrics with gauges
                                if any(f"final_{m}" in run_details for m in ['mae', 'mse', 'rmse', 'r2', 'rmsle']):
                                    st.subheader("Final Performance Metrics")
                                    
                                    # Create 2 rows of 3 gauges each for the metrics
                                    gauge_cols1 = st.columns(3)
                                    gauge_cols2 = st.columns(2)
                                    
                                    # For error metrics like MAE, MSE, RMSE, invert the scale since lower is better
                                    # We'll scale them between 0-1 based on reasonable thresholds
                                    
                                    # Row 1: MAE, MSE, RMSE
                                    with gauge_cols1[0]:
                                        if 'final_mae' in run_details:
                                            value = float(run_details['final_mae'])
                                            # Scale inversely from 0-10 for display (lower is better)
                                            scaled = max(0, min(1, 1 - value/10)) if value <= 10 else 0
                                            st.markdown(f"### MAE")
                                            st.progress(scaled)
                                            st.markdown(f"**{value:.4f}**")
                                        elif 'mae' in metrics and metrics['mae'] is not None:
                                            value = float(metrics['mae'])
                                            scaled = max(0, min(1, 1 - value/10)) if value <= 10 else 0
                                            st.markdown(f"### MAE")
                                            st.progress(scaled)
                                            st.markdown(f"**{value:.4f}**")
                                    
                                    with gauge_cols1[1]:
                                        if 'final_mse' in run_details:
                                            value = float(run_details['final_mse'])
                                            # Scale inversely from 0-100 for display (lower is better)
                                            scaled = max(0, min(1, 1 - value/100)) if value <= 100 else 0
                                            st.markdown(f"### MSE")
                                            st.progress(scaled)
                                            st.markdown(f"**{value:.4f}**")
                                        elif 'mse' in metrics and metrics['mse'] is not None:
                                            value = float(metrics['mse'])
                                            scaled = max(0, min(1, 1 - value/100)) if value <= 100 else 0
                                            st.markdown(f"### MSE")
                                            st.progress(scaled)
                                            st.markdown(f"**{value:.4f}**")
                                    
                                    with gauge_cols1[2]:
                                        if 'final_rmse' in run_details:
                                            value = float(run_details['final_rmse'])
                                            # Scale inversely from 0-10 for display (lower is better)
                                            scaled = max(0, min(1, 1 - value/10)) if value <= 10 else 0
                                            st.markdown(f"### RMSE")
                                            st.progress(scaled)
                                            st.markdown(f"**{value:.4f}**")
                                        elif 'rmse' in metrics and metrics['rmse'] is not None:
                                            value = float(metrics['rmse'])
                                            scaled = max(0, min(1, 1 - value/10)) if value <= 10 else 0
                                            st.markdown(f"### RMSE")
                                            st.progress(scaled)
                                            st.markdown(f"**{value:.4f}**")
                                    
                                    # Row 2: R2, RMSLE
                                    with gauge_cols2[0]:
                                        if 'final_r2' in run_details:
                                            value = float(run_details['final_r2'])
                                            # R2 is already in 0-1 range (higher is better)
                                            scaled = max(0, min(1, value))
                                            st.markdown(f"### R² Score")
                                            st.progress(scaled)
                                            st.markdown(f"**{value:.4f}**")
                                        elif 'r2' in metrics and metrics['r2'] is not None:
                                            value = float(metrics['r2'])
                                            scaled = max(0, min(1, value))
                                            st.markdown(f"### R² Score")
                                            st.progress(scaled)
                                            st.markdown(f"**{value:.4f}**")
                                    
                                    with gauge_cols2[1]:
                                        if 'final_rmsle' in run_details:
                                            value = float(run_details['final_rmsle'])
                                            # Scale inversely from 0-2 for display (lower is better)
                                            scaled = max(0, min(1, 1 - value/2)) if value <= 2 else 0
                                            st.markdown(f"### RMSLE")
                                            st.progress(scaled)
                                            st.markdown(f"**{value:.4f}**")
                                        elif 'rmsle' in metrics and metrics['rmsle'] is not None:
                                            value = float(metrics['rmsle'])
                                            scaled = max(0, min(1, 1 - value/2)) if value <= 2 else 0
                                            st.markdown(f"### RMSLE")
                                            st.progress(scaled)
                                            st.markdown(f"**{value:.4f}**")
                                
                                # Show regression prediction plot if available
                                if 'regression_plot' in run_details:
                                    st.subheader("Prediction vs Actual Values")
                                    st.image(f"data:image/png;base64,{run_details['regression_plot']}")
                                    st.caption("Scatter plot of predicted values vs actual values. The diagonal line represents perfect predictions.")
                        
                        # Clustering metrics
                        elif task_type == 'Clustering':
                            st.subheader("Clustering Metrics")
                            
                            if 'metrics' in run_details:
                                metrics = run_details['metrics']
                                
                                # Create a nice table to display all metrics
                                metric_data = []
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    value = format_metric_value(metrics.get('silhouette'), 'silhouette')
                                    st.metric("Silhouette Score", value)
                                    metric_data.append({
                                        "Metric": "Silhouette Score", 
                                        "Value": value,
                                        "Description": METRICS_DESCRIPTIONS.get('silhouette', '')
                                    })
                                with col2:
                                    value = format_metric_value(metrics.get('adjusted_rand'), 'adjusted_rand')
                                    st.metric("Adjusted Rand Score", value)
                                    metric_data.append({
                                        "Metric": "Adjusted Rand Score", 
                                        "Value": value,
                                        "Description": METRICS_DESCRIPTIONS.get('adjusted_rand', '')
                                    })
                                with col3:
                                    value = format_metric_value(metrics.get('adjusted_mutual_info'), 'adjusted_mutual_info')
                                    st.metric("Adjusted Mutual Info", value)
                                    metric_data.append({
                                        "Metric": "Adjusted Mutual Information", 
                                        "Value": value,
                                        "Description": METRICS_DESCRIPTIONS.get('adjusted_mutual_info', '')
                                    })
                                
                                # Show all metrics in a table for reference
                                with st.expander("All Metrics Details"):
                                    st.dataframe(pd.DataFrame(metric_data))
                    
                    # Add training history visualization if available
                    if 'training_history' in run_details:
                        history = run_details['training_history']
                        st.subheader("Training History")

                        # Plot training loss if available
                        if 'loss' in history and len(history['loss']) > 0:
                            try:
                                loss_data = history['loss']
                                epochs = list(range(1, len(loss_data) + 1))
                                
                                # Create a DataFrame for plotting
                                loss_df = pd.DataFrame({
                                    "Epoch": epochs,
                                    "Loss": loss_data
                                })
                                
                                # Plot the loss curve
                                st.line_chart(loss_df.set_index('Epoch'))
                            except Exception as e:
                                st.warning(f"Could not plot loss history: {e}")
                        
                        # Plot accuracy if available
                        if 'accuracy' in history and len(history['accuracy']) > 0:
                            try:
                                # Filter out None or NaN values
                                acc_data = []
                                acc_epochs = []
                                
                                for i, acc in enumerate(history['accuracy']):
                                    if acc is not None and not (isinstance(acc, float) and np.isnan(acc)):
                                        acc_data.append(acc)
                                        acc_epochs.append(i + 1)  # Epochs are 1-indexed for display
                                
                                if acc_data:
                                    # Create a DataFrame for plotting
                                    acc_df = pd.DataFrame({
                                        "Epoch": acc_epochs,
                                        "Accuracy": acc_data
                                    })
                                    
                                    # Plot the accuracy curve
                                    st.subheader("Training Accuracy")
                                    st.line_chart(acc_df.set_index('Epoch'))
                            except Exception as e:
                                st.warning(f"Could not plot accuracy history: {e}")
                        
                        # Plot entanglement if available
                        if 'entanglement' in history and len(history['entanglement']) > 0:
                            try:
                                # Filter out None or NaN values
                                ent_data = []
                                ent_epochs = []
                                
                                for i, e in enumerate(history['entanglement']):
                                    if e is not None and not (isinstance(e, float) and np.isnan(e)):
                                        ent_data.append(e)
                                        ent_epochs.append(i + 1)  # Epochs are 1-indexed for display
                                
                                if ent_data:
                                    # Create a DataFrame for plotting
                                    ent_df = pd.DataFrame({
                                        "Epoch": ent_epochs,
                                        "Entanglement": ent_data
                                    })
                                    
                                    # Plot the entanglement curve
                                    st.subheader("Quantum Entanglement")
                                    st.line_chart(ent_df.set_index('Epoch'))
                            except Exception as e:
                                st.warning(f"Could not plot entanglement history: {e}")
                        
                        # Display batch-level training info if available
                        if 'batch_info' in run_details:
                            batch_info = run_details['batch_info']
                            
                            # Show example batch stats 
                            with st.expander("Batch-Level Training Information"):
                                if isinstance(batch_info, list) and len(batch_info) > 0:
                                    if isinstance(batch_info[0], dict):
                                        # Convert to DataFrame for display
                                        batch_df = pd.DataFrame(batch_info)
                                        st.dataframe(batch_df)
                                    else:
                                        st.text("\n".join(str(b) for b in batch_info))
                                else:
                                    st.write(batch_info)
                    
                    # Add dataset information if available
                    if 'data_info' in run_details:
                        data_info = run_details['data_info']
                        st.subheader("Dataset Information")
                        
                        # Create columns for data info
                        data_cols = st.columns(3)
                        
                        with data_cols[0]:
                            if 'data_type' in data_info:
                                st.metric("Data Type", data_info['data_type'])
                            if 'samples' in data_info:
                                st.metric("Number of Samples", data_info['samples'])
                        
                        with data_cols[1]:
                            if 'feature_columns' in data_info:
                                feature_cols = data_info['feature_columns']
                                if isinstance(feature_cols, list):
                                    st.metric("Features Used", len(feature_cols))
                                    if len(feature_cols) <= 5:  # Only show if not too many
                                        st.text("Feature columns: " + ", ".join(feature_cols))
                                else:
                                    st.metric("Features Used", feature_cols)
                            elif 'processed_shape' in data_info:
                                shape = data_info['processed_shape']
                                if isinstance(shape, (list, tuple)) and len(shape) > 1:
                                    st.metric("Feature Dimensions", shape[1])
                        
                        with data_cols[2]:
                            if 'label_column' in data_info and data_info['label_column'] is not None:
                                st.metric("Label Column", data_info['label_column'])
                            if 'classes' in data_info:
                                classes = data_info['classes']
                                if isinstance(classes, list):
                                    st.metric("Number of Classes", len(classes))
                                    if len(classes) <= 10:  # Only show if not too many
                                        st.text("Classes: " + ", ".join(str(c) for c in classes))
                                else:
                                    st.metric("Number of Classes", classes)
                    
                    # Edit run status
                    st.subheader("Edit Run")
                    
                    # Add options for "Pass" and "Fail" status
                    status_options = ["Completed", "Failed", "In Progress", "Pass - Validated", "Fail - Not Validated"]
                    current_status = run_info['status']
                    default_idx = status_options.index(current_status) if current_status in status_options else 0
                    
                    new_status = st.radio(
                        "Set run status", 
                        options=status_options, 
                        index=default_idx
                    )
                    
                    # Add note field for Pass/Fail statuses
                    note = ""
                    if "Pass" in new_status or "Fail" in new_status:
                        note = st.text_area(
                            "Validation Notes", 
                            value=run_details.get("validation_note", ""), 
                            help="Add notes about why this run passed or failed validation"
                        )
                    
                    if st.button("Update Status"):
                        update_run_status(st.session_state.selected_run_id, new_status)
                        
                        # If there's a note, update the run details too
                        if note:
                            run_details["validation_note"] = note
                            save_run_details(st.session_state.selected_run_id, run_details)
                            
                        st.success(f"Run status updated to {new_status}")
                        st.rerun()
                    
                    # Delete run button
                    if st.button("Delete Run", type="primary", use_container_width=True):
                        run_id_to_delete = st.session_state.selected_run_id  # Store before clearing
                        delete_run(run_id_to_delete)
                        st.success(f"Run {run_id_to_delete} deleted")
                        st.session_state.selected_run_id = None
                        st.rerun()
                    
                    # Show repair button if recovery note is present
                    if 'recovery_note' in run_details:
                        if st.button("Repair Run Data", type="secondary"):
                            # Save the minimal details again to create clean files
                            save_run_details(st.session_state.selected_run_id, run_details)
                            st.success("Run data files recreated. The data has been simplified but should be valid now.")
                            st.rerun()
                else:
                    st.error("Failed to load any details for this run.")
                    
                    # Offer to reset this run
                    if st.button("Create Empty Run Details"):
                        minimal_details = {
                            'run_id': st.session_state.selected_run_id,
                            'config_name': selected_run_info['config_name'],
                            'status': selected_run_info['status'],
                            'timestamp': selected_run_info['timestamp'],
                            'recovery_note': 'This is a newly created empty run record created due to data corruption.'
                        }
                        save_run_details(st.session_state.selected_run_id, minimal_details)
                        st.success("Created new minimal run details. Please reload the page.")
                        st.rerun()
            except Exception as e:
                st.error(f"Error displaying run {st.session_state.selected_run_id}: {e}")
                st.warning("The run data may be corrupted. Try selecting a different run.")
        else:
            st.info("Select a run from the table above to view details.")
    else:
        st.info("No training runs found. Train a model to see the history.") 