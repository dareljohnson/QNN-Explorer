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

def resolve_device(use_gpu_requested: bool) -> torch.device:
    """Pick the device for the model and for batches.

    The per-configuration "Use GPU if available" setting decides this. It used
    to be ignored: models were placed on check_gpu()'s device unconditionally
    while the flag only steered the quantum simulator. That put the classical
    backbone on cuda and the quantum layer on cpu, and validation then failed
    with "Expected all tensors to be on the same device, but found at least two
    devices, cuda:0 and cpu".

    Args:
        use_gpu_requested: value of the configuration's use_gpu flag

    Returns:
        torch.device("cuda") when requested and available, else torch.device("cpu")
    """
    if use_gpu_requested and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")