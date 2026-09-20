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


def build_optimizer(model, lr: float, finetune_backbone: bool = False):
    """Adam over a hybrid model, with the pretrained backbone handled correctly.

    Freezing is the default because fine-tuning overfits small text datasets: on
    the ArXiv demo a frozen backbone reached 92% held-out, a 2-epoch fine-tune
    94.5%, and the old 15-epoch fine-tune 40% (100% train).

    Args:
        model: HybridModel
        lr: learning rate for the fusion layer, quantum circuit and output head
        finetune_backbone: whether the pretrained backbone may be updated

    Returns:
        (optimizer, backbone_params, fine_tuning) -- backbone_params is None when
        the backbone is frozen, and fine_tuning says whether a step cap applies.
    """
    backbone = getattr(model, 'classical_backbone', None)
    has_backbone = backbone is not None and len(list(backbone.parameters())) > 0
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith('classical_backbone.')]

    if not has_backbone:
        return torch.optim.Adam(model.parameters(), lr=lr), None, False

    if not finetune_backbone:
        for param in backbone.parameters():
            param.requires_grad_(False)
        return torch.optim.Adam(head_params, lr=lr, weight_decay=1e-4), None, False

    backbone_lr = backbone_learning_rate(lr, getattr(model, 'classical_backbone_type', None))
    optimizer = torch.optim.Adam([
        {'params': list(backbone.parameters()), 'lr': backbone_lr},
        {'params': head_params, 'lr': lr, 'weight_decay': 1e-4},
    ])
    return optimizer, list(backbone.parameters()), True


def freeze_params(params) -> None:
    """Stop a parameter group from being updated any further."""
    for param in params or []:
        param.requires_grad_(False)


def forward_in_batches(model, features, batch_size: int = 32):
    """Run a model over features in batches on the model's own device.

    Accepts a tensor or a tokenizer dict (input_ids/attention_mask) and returns
    cpu tensors. The validation path used to hand the entire cpu dataset to a
    cuda model in one call, which failed with "Expected all tensors to be on the
    same device, but found at least two devices, cuda:0 and cpu".
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        # A module without parameters (e.g. a frozen feature transform) is fine;
        # its inputs still have to live somewhere.
        device = torch.device("cpu")
    if isinstance(features, dict):
        total = len(next(iter(features.values())))
        def sliced(start):
            return {k: v[start:start + batch_size] for k, v in features.items()}
    else:
        total = len(features)
        def sliced(start):
            return features[start:start + batch_size]

    outputs = []
    for start in range(0, total, max(1, batch_size)):
        batch = sliced(start)
        if isinstance(batch, dict):
            batch = {k: v.to(device) for k, v in batch.items()}
        else:
            batch = batch.to(device)
        outputs.append(model(batch).cpu())
    return torch.cat(outputs, dim=0)


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