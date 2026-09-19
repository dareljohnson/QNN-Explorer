"""Simple test to check imports."""
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch not found")

try:
    import pennylane as qml
    print(f"PennyLane version: {qml.__version__}")
    # Correctly list available device plugins
    from pennylane import plugin
    print(f"Available device plugins: {plugin.plugin_devices}")
except ImportError:
    print("PennyLane not found") 