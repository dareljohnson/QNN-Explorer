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

# Backbone learning-rate factor by backbone type. Pretrained transformers need a
# much smaller step than CNNs: at the previous lr*0.1 (=5e-4 at the default
# lr of 0.005) a DistilBERT backbone's features are destroyed as fast as the
# head learns, and the loss never leaves chance (ln(3) for 3 classes). Measured
# on the ArXiv demo: 5e-4 -> flat loss at ~1.09 after 300 steps, 5e-5 and 5e-6
# -> loss 0.0000 and 100% batch accuracy within 100 steps.
TRANSFORMER_BACKBONE_LR_FACTOR = 0.01
DEFAULT_BACKBONE_LR_FACTOR = 0.1


def backbone_learning_rate(lr: float, backbone_type: str | None) -> float:
    """Learning rate for the classical backbone's parameter group.

    Args:
        lr: the learning rate configured in the UI, used at full strength for
            the fusion layer, quantum circuit and output head
        backbone_type: 'transformer', 'cnn', 'gnn', 'regression' or None

    Returns:
        The learning rate to use for the pretrained backbone.
    """
    if (backbone_type or "").lower() == "transformer":
        return lr * TRANSFORMER_BACKBONE_LR_FACTOR
    return lr * DEFAULT_BACKBONE_LR_FACTOR


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