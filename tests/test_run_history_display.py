"""
Unit tests for the run history display functionality.
Tests that all model configurations and metrics are correctly displayed in the Run Details.
"""

import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call
import sys

# Add the parent directory to the path to make relative imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.run_history import add_run, load_run_history, save_run_details

# Patch the STREAMLIT_AVAILABLE variable
import utils.run_history_tab
utils.run_history_tab.STREAMLIT_AVAILABLE = True

class TestRunHistoryDetailsDisplay(unittest.TestCase):
    """Test case for the run history details display functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for the test history
        self.test_dir = os.path.join("tests", "test_history")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Path for the test history file
        self.test_history_path = os.path.join(self.test_dir, "test_run_history.csv")
        self.test_details_path = os.path.join(self.test_dir, "run_details")
        os.makedirs(self.test_details_path, exist_ok=True)
        
        # Create some test run history data
        self.test_runs = pd.DataFrame({
            'run_id': [1],
            'timestamp': ['2023-05-01'],
            'config_name': ['Test Config 1'],
            'duration': [60],
            'status': ['Completed']
        })
        
        # Save the test data
        self.test_runs.to_csv(self.test_history_path, index=False)
        
        # Create comprehensive run details with all possible fields
        self.complete_run_details = {
            'run_id': 1,
            'config_name': 'Test Config 1',
            'task_type': 'Classification',
            'model_config': {
                'num_qubits': 4,
                'num_layers': 2,
                'ansatz_name': 'Test Ansatz',
                'encoding_method': 'amplitude',
                'observation_type': 'expectation',
                'classical_backbone_type': 'CNN',
                'classical_model_name': 'resnet18',
                'quantum_input_size': 64,
                'use_gpu': True
            },
            'training_params': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 10
            },
            'metrics': {
                'accuracy': 0.92,
                'precision': 0.91,
                'recall': 0.90,
                'f1': 0.905,
                'auc': 0.95,
                'log_loss': 0.24,
                'confusion_matrix': np.array([[45, 5], [3, 47]]).tolist()
            },
            'weights_files': [
                'saved_models/weights_last_train.pt',
                'saved_models/weights_cnn_N4_L2.pt'
            ],
            'training_history': {
                'loss': [0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.28, 0.25],
                'entanglement': [0.5, 0.7, 0.9, 1.1, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
                'accuracy': [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.88, 0.9, 0.92]
            },
            'data_info': {
                'data_type': 'Images',
                'samples': 100,
                'feature_columns': None,
                'processed_shape': [100, 64],
                'classes': ['class_0', 'class_1'],
                'label_column': 'label'
            }
        }
        
        # Save the details to a file
        details_file = os.path.join(self.test_details_path, f"run_{self.complete_run_details['run_id']}.json")
        with open(details_file, 'w') as f:
            import json
            json.dump(self.complete_run_details, f)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up test files
        if os.path.exists(self.test_history_path):
            os.remove(self.test_history_path)
        
        # Clean up test details files
        details_file = os.path.join(self.test_details_path, f"run_{self.complete_run_details['run_id']}.json")
        if os.path.exists(details_file):
            os.remove(details_file)
        
        # Remove test directories
        if os.path.exists(self.test_details_path):
            os.rmdir(self.test_details_path)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
    
    def _setup_mock_streamlit(self, mock_st, run_id=1):
        """Set up streamlit mocks with session state and selectbox."""
        # Set up session state mock
        session_state_dict = {'selected_run_id': run_id}
        
        class MockSessionState:
            def __init__(self):
                self.selected_run_id = run_id
                
            def get(self, key, default=None):
                return session_state_dict.get(key, default)
                
            def __getitem__(self, key):
                return session_state_dict.get(key)
                
            def __setitem__(self, key, value):
                session_state_dict[key] = value
                if key == 'selected_run_id':
                    self.selected_run_id = value
                
            def __contains__(self, key):
                return key in session_state_dict
        
        mock_st.session_state = MockSessionState()
        
        # Mock the selectbox to return our run_id
        def mock_selectbox_side_effect(*args, **kwargs):
            if 'key' in kwargs and kwargs['key'] == 'run_selector':
                return run_id
            return None
        
        mock_st.selectbox.side_effect = mock_selectbox_side_effect
        
        # Create a MagicMock for metrics tracking
        mock_metric = MagicMock()
        mock_st.metric = mock_metric
        
        # Create a list to store metric calls for easier verification
        mock_st.captured_metrics = []
        
        # Set up a side effect to capture metric calls
        def metric_side_effect(*args, **kwargs):
            print(f"Captured metric: {args[0]} = {args[1]}")
            mock_st.captured_metrics.append((args, kwargs))
            
        mock_metric.side_effect = metric_side_effect
        
        # Mock radio
        mock_st.radio.return_value = "Completed"
        
        # Mock columns to properly handle context manager
        def columns_side_effect(sizes):
            mocks = []
            for _ in range(len(sizes) if isinstance(sizes, list) else sizes):
                mock_col = MagicMock()
                
                # Make the mock column work as a context manager
                mock_col.__enter__ = MagicMock(return_value=mock_col)
                mock_col.__exit__ = MagicMock(return_value=None)
                
                # Add a metric method to the column
                col_metric = MagicMock()
                col_metric.side_effect = metric_side_effect
                mock_col.metric = col_metric
                
                mocks.append(mock_col)
            return mocks
            
        mock_st.columns.side_effect = columns_side_effect
        
        # Return the configured mock
        return mock_st

    @patch('utils.run_history_tab.RUN_DETAILS_PATH', new="tests/test_history/run_details")
    @patch('utils.run_history_tab.st')
    @patch('utils.run_history_tab.load_run_history')
    @patch('utils.run_history_tab.load_run_details')
    def test_model_config_display(self, mock_load_details, mock_load_history, mock_st):
        """Test that all model configuration fields are displayed correctly."""
        # Configure the mocks to return our test data
        mock_load_history.return_value = self.test_runs
        mock_load_details.return_value = self.complete_run_details
        
        # The run history DataFrame must have a duration_str column
        run_df = mock_load_history.return_value
        run_df['duration_str'] = run_df['duration'].apply(
            lambda x: f"{int(x // 3600)}h {int((x % 3600) // 60)}m {int(x % 60)}s" if x >= 3600 else 
                     f"{int(x // 60)}m {int(x % 60)}s" if x >= 60 else 
                     f"{int(x)}s"
        )
        
        # Set up streamlit mocks
        self._setup_mock_streamlit(mock_st)
        
        # Print info for debugging
        print("\nTest data:")
        print(f"Run history: {run_df}")
        
        # Call the function that renders the run history tab
        from utils.run_history_tab import render_run_history_tab
        render_run_history_tab()
        
        # For debugging
        print("\nCaptured metrics:")
        if hasattr(mock_st, 'captured_metrics'):
            for args, _ in mock_st.captured_metrics:
                if len(args) >= 2:
                    print(f"{args[0]}: {args[1]}")
        
        # Get all metric labels
        all_labels = []
        for call in mock_st.captured_metrics:
            args = call[0]
            if len(args) >= 1:
                all_labels.append(args[0])
                
        print(f"All labels: {all_labels}")
        
        # Verify key model configuration parameters are displayed
        self.assertIn("Configuration", all_labels)
        self.assertIn("Number of Qubits", all_labels)
        self.assertIn("Number of Layers", all_labels)
        self.assertIn("Ansatz", all_labels)
        self.assertIn("Encoding", all_labels)
        self.assertIn("GPU Used", all_labels)

    @patch('utils.run_history_tab.RUN_DETAILS_PATH', new="tests/test_history/run_details")
    @patch('utils.run_history_tab.st')
    @patch('utils.run_history_tab.load_run_history')
    @patch('utils.run_history_tab.load_run_details')
    def test_classification_metrics_display(self, mock_load_details, mock_load_history, mock_st):
        """Test that all classification metrics are displayed correctly."""
        # Configure the mocks to return our test data
        mock_load_history.return_value = self.test_runs
        mock_load_details.return_value = self.complete_run_details
        
        # The run history DataFrame must have a duration_str column
        run_df = mock_load_history.return_value
        run_df['duration_str'] = run_df['duration'].apply(
            lambda x: f"{int(x // 3600)}h {int((x % 3600) // 60)}m {int(x % 60)}s" if x >= 3600 else 
                     f"{int(x // 60)}m {int(x % 60)}s" if x >= 60 else 
                     f"{int(x)}s"
        )
        
        # Set up streamlit mocks
        self._setup_mock_streamlit(mock_st)
        
        # Call the function that renders the run history tab
        from utils.run_history_tab import render_run_history_tab
        render_run_history_tab()
        
        # Get all metric labels
        all_labels = []
        for call in mock_st.captured_metrics:
            args = call[0]
            if len(args) >= 1:
                all_labels.append(args[0])
        
        # Create a dict of all metrics for value checking
        metrics_dict = {}
        for call in mock_st.captured_metrics:
            args = call[0]
            if len(args) >= 2:
                metrics_dict[args[0]] = args[1]
        
        # Check that all expected classification metrics are displayed
        self.assertIn("Accuracy", all_labels)
        self.assertIn("Precision", all_labels)
        self.assertIn("Recall", all_labels)
        self.assertIn("F1 Score", all_labels)
        self.assertIn("AUC", all_labels)
        self.assertIn("Log Loss", all_labels)
        
        # Check the values match our test data (formatted as strings)
        self.assertEqual(metrics_dict.get("Accuracy"), "0.9200")
        self.assertEqual(metrics_dict.get("Precision"), "0.9100")
        self.assertEqual(metrics_dict.get("Recall"), "0.9000")
        self.assertEqual(metrics_dict.get("F1 Score"), "0.9050")
        self.assertEqual(metrics_dict.get("AUC"), "0.9500")
        self.assertEqual(metrics_dict.get("Log Loss"), "0.2400")

    @patch('utils.run_history_tab.RUN_DETAILS_PATH', new="tests/test_history/run_details")
    @patch('utils.run_history_tab.st')
    @patch('utils.run_history_tab.load_run_history')
    @patch('utils.run_history_tab.load_run_details')
    def test_training_params_display(self, mock_load_details, mock_load_history, mock_st):
        """Test that all training parameters are displayed correctly."""
        # Configure the mocks to return our test data
        mock_load_history.return_value = self.test_runs
        mock_load_details.return_value = self.complete_run_details
        
        # The run history DataFrame must have a duration_str column
        run_df = mock_load_history.return_value
        run_df['duration_str'] = run_df['duration'].apply(
            lambda x: f"{int(x // 3600)}h {int((x % 3600) // 60)}m {int(x % 60)}s" if x >= 3600 else 
                     f"{int(x // 60)}m {int(x % 60)}s" if x >= 60 else 
                     f"{int(x)}s"
        )
        
        # Set up streamlit mocks
        self._setup_mock_streamlit(mock_st)
        
        # Call the function that renders the run history tab
        from utils.run_history_tab import render_run_history_tab
        render_run_history_tab()
        
        # Get all metric labels
        all_labels = []
        for call in mock_st.captured_metrics:
            args = call[0]
            if len(args) >= 1:
                all_labels.append(args[0])
        
        # Create a dict of all metrics for value checking
        metrics_dict = {}
        for call in mock_st.captured_metrics:
            args = call[0]
            if len(args) >= 2:
                metrics_dict[args[0]] = args[1]
        
        # Check that all expected training parameters are displayed
        self.assertIn("Learning Rate", all_labels)
        self.assertIn("Batch Size", all_labels)
        self.assertIn("Epochs", all_labels)
        
        # Check for specific values
        self.assertEqual(metrics_dict.get("Learning Rate"), "0.0010")
        self.assertEqual(metrics_dict.get("Batch Size"), 32)
        self.assertEqual(metrics_dict.get("Epochs"), 10)

    @patch('utils.run_history_tab.RUN_DETAILS_PATH', new="tests/test_history/run_details")
    @patch('utils.run_history_tab.st')
    @patch('utils.run_history_tab.load_run_history')
    @patch('utils.run_history_tab.load_run_details')
    def test_dataset_info_display(self, mock_load_details, mock_load_history, mock_st):
        """Test that all dataset information is displayed correctly."""
        # Configure the mocks to return our test data
        mock_load_history.return_value = self.test_runs
        mock_load_details.return_value = self.complete_run_details
        
        # The run history DataFrame must have a duration_str column
        run_df = mock_load_history.return_value
        run_df['duration_str'] = run_df['duration'].apply(
            lambda x: f"{int(x // 3600)}h {int((x % 3600) // 60)}m {int(x % 60)}s" if x >= 3600 else 
                     f"{int(x // 60)}m {int(x % 60)}s" if x >= 60 else 
                     f"{int(x)}s"
        )
        
        # Set up streamlit mocks
        self._setup_mock_streamlit(mock_st)
        
        # Call the function that renders the run history tab
        from utils.run_history_tab import render_run_history_tab
        render_run_history_tab()
        
        # Get all metric labels
        all_labels = []
        for call in mock_st.captured_metrics:
            args = call[0]
            if len(args) >= 1:
                all_labels.append(args[0])
        
        # Create a dict of all metrics for value checking
        metrics_dict = {}
        for call in mock_st.captured_metrics:
            args = call[0]
            if len(args) >= 2:
                metrics_dict[args[0]] = args[1]
        
        # Check that all expected dataset information is displayed
        self.assertIn("Data Type", all_labels)
        self.assertIn("Number of Samples", all_labels)
        self.assertIn("Label Column", all_labels)
        
        # Check for specific values
        self.assertEqual(metrics_dict.get("Data Type"), "Images")
        self.assertEqual(metrics_dict.get("Number of Samples"), 100)
        self.assertEqual(metrics_dict.get("Label Column"), "label")

    @patch('utils.run_history_tab.RUN_DETAILS_PATH', new="tests/test_history/run_details")
    @patch('utils.run_history_tab.st')
    @patch('utils.run_history_tab.load_run_history')
    @patch('utils.run_history_tab.load_run_details')
    def test_regression_metrics_display(self, mock_load_details, mock_load_history, mock_st):
        """Test that all regression metrics are displayed correctly."""
        # Create regression run details
        regression_run_details = self.complete_run_details.copy()
        regression_run_details['task_type'] = 'Regression'
        regression_run_details['metrics'] = {
            'mae': 0.324,
            'mse': 0.156,
            'rmse': 0.395,
            'rmsle': 0.187,
            'r2': 0.876
        }
        
        # Configure the mocks to return our test data
        mock_load_history.return_value = self.test_runs
        mock_load_details.return_value = regression_run_details
        
        # The run history DataFrame must have a duration_str column
        run_df = mock_load_history.return_value
        run_df['duration_str'] = run_df['duration'].apply(
            lambda x: f"{int(x // 3600)}h {int((x % 3600) // 60)}m {int(x % 60)}s" if x >= 3600 else 
                     f"{int(x // 60)}m {int(x % 60)}s" if x >= 60 else 
                     f"{int(x)}s"
        )
        
        # Set up streamlit mocks
        self._setup_mock_streamlit(mock_st)
        
        # Call the function that renders the run history tab
        from utils.run_history_tab import render_run_history_tab
        render_run_history_tab()
        
        # Get all metric labels
        all_labels = []
        for call in mock_st.captured_metrics:
            args = call[0]
            if len(args) >= 1:
                all_labels.append(args[0])
        
        # Create a dict of all metrics for value checking
        metrics_dict = {}
        for call in mock_st.captured_metrics:
            args = call[0]
            if len(args) >= 2:
                metrics_dict[args[0]] = args[1]
        
        # Check that all expected regression metrics are displayed
        self.assertIn("MAE", all_labels)
        self.assertIn("MSE", all_labels)
        self.assertIn("RMSE", all_labels)
        self.assertIn("RMSLE", all_labels)
        self.assertIn("R² Score", all_labels)
        
        # Check for specific values
        self.assertEqual(metrics_dict.get("MAE"), "0.3240")
        self.assertEqual(metrics_dict.get("MSE"), "0.1560")
        self.assertEqual(metrics_dict.get("RMSE"), "0.3950")
        self.assertEqual(metrics_dict.get("RMSLE"), "0.1870")
        self.assertEqual(metrics_dict.get("R² Score"), "0.8760")

    @patch('utils.run_history_tab.RUN_DETAILS_PATH', new="tests/test_history/run_details")
    @patch('utils.run_history_tab.st')
    @patch('utils.run_history_tab.load_run_history')
    @patch('utils.run_history_tab.load_run_details')
    def test_clustering_metrics_display(self, mock_load_details, mock_load_history, mock_st):
        """Test that all clustering metrics are displayed correctly."""
        # Create clustering run details
        clustering_run_details = self.complete_run_details.copy()
        clustering_run_details['task_type'] = 'Clustering'
        clustering_run_details['metrics'] = {
            'silhouette': 0.75,
            'adjusted_rand': 0.82,
            'adjusted_mutual_info': 0.78
        }
        
        # Configure the mocks to return our test data
        mock_load_history.return_value = self.test_runs
        mock_load_details.return_value = clustering_run_details
        
        # The run history DataFrame must have a duration_str column
        run_df = mock_load_history.return_value
        run_df['duration_str'] = run_df['duration'].apply(
            lambda x: f"{int(x // 3600)}h {int((x % 3600) // 60)}m {int(x % 60)}s" if x >= 3600 else 
                     f"{int(x // 60)}m {int(x % 60)}s" if x >= 60 else 
                     f"{int(x)}s"
        )
        
        # Set up streamlit mocks
        self._setup_mock_streamlit(mock_st)
        
        # Call the function that renders the run history tab
        from utils.run_history_tab import render_run_history_tab
        render_run_history_tab()
        
        # Get all metric labels
        all_labels = []
        for call in mock_st.captured_metrics:
            args = call[0]
            if len(args) >= 1:
                all_labels.append(args[0])
        
        # Create a dict of all metrics for value checking
        metrics_dict = {}
        for call in mock_st.captured_metrics:
            args = call[0]
            if len(args) >= 2:
                metrics_dict[args[0]] = args[1]
        
        # Check that all expected clustering metrics are displayed
        self.assertIn("Silhouette Score", all_labels)
        self.assertIn("Adjusted Rand Score", all_labels)
        self.assertIn("Adjusted Mutual Info", all_labels)
        
        # Check for specific values
        self.assertEqual(metrics_dict.get("Silhouette Score"), "0.7500")
        self.assertEqual(metrics_dict.get("Adjusted Rand Score"), "0.8200")
        self.assertEqual(metrics_dict.get("Adjusted Mutual Info"), "0.7800")

if __name__ == '__main__':
    unittest.main() 