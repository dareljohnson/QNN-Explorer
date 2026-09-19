# Miscellaneous helper functions can be added here.

import torch

def check_gpu():
    """Checks if GPU is available and prints info."""
    if torch.cuda.is_available():
        print(f"GPU found: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        return torch.device("cuda")
    else:
        print("No GPU found, using CPU.")
        return torch.device("cpu") 