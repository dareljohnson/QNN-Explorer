# Miscellaneous helper functions can be added here.

import numpy as np
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

    Accepts a tensor, a tokenizer dict (input_ids/attention_mask), or a list of
    per-sample tensors (image datasets), and returns cpu tensors. The validation path used to hand the entire cpu dataset to a
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
        elif isinstance(batch, torch.Tensor):
            batch = batch.to(device)
        else:
            # A list/tuple of per-sample tensors, as image datasets produce.
            # These used to reach `.to(device)` and fail with
            # "AttributeError: 'list' object has no attribute 'to'".
            items = [item.to(device) if isinstance(item, torch.Tensor) else item
                     for item in batch]
            try:
                batch = torch.stack(items)
            except (TypeError, RuntimeError):
                batch = items
        outputs.append(model(batch).cpu())
    return torch.cat(outputs, dim=0)


def split_train_val(features, labels, val_fraction: float = 0.2, seed: int = 0):
    """Split features and labels into ((train), (val)).

    Handles a tensor, a list, or a tokenizer dict (input_ids/attention_mask).
    The validation split is what makes held-out metrics and best-checkpoint
    selection possible; without it the app measured accuracy on the data it had
    just trained on.

    Args:
        features: tensor, list, or dict of tensors indexed by row
        labels: tensor or list aligned with features
        val_fraction: share of rows to hold out (0 disables the split)
        seed: shuffle seed, so a run is reproducible

    Returns:
        ((train_features, train_labels), (val_features, val_labels)); the val half
        is (None, None) when val_fraction is 0 or there is only one row.
    """
    total = len(next(iter(features.values()))) if isinstance(features, dict) else len(features)
    n_val = 0 if val_fraction <= 0 else int(round(total * val_fraction))
    n_val = min(max(n_val, 0), max(total - 1, 0))
    if n_val == 0:
        return (features, labels), (None, None)

    perm = torch.randperm(total, generator=torch.Generator().manual_seed(seed))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def take(idx):
        if isinstance(features, dict):
            taken_features = {k: v[idx] for k, v in features.items()}
        elif isinstance(features, torch.Tensor):
            taken_features = features[idx]
        else:
            taken_features = [features[i] for i in idx.tolist()]
        if labels is None:
            taken_labels = None
        elif isinstance(labels, torch.Tensor):
            taken_labels = labels[idx]
        else:
            taken_labels = [labels[i] for i in idx.tolist()]
        return taken_features, taken_labels

    return take(train_idx), take(val_idx)


def evaluate_split(model, features, labels, batch_size: int = 32):
    """Loss-free accuracy/target summary for a held-out split.

    Returns (loss, accuracy); accuracy is None for regression-style targets.
    """
    if features is None or labels is None:
        return None, None
    outputs = forward_in_batches(model, features, batch_size=batch_size)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    if labels.dim() > 1:
        labels = labels.squeeze(-1)
    if labels.dtype.is_floating_point:
        loss = torch.nn.functional.mse_loss(outputs.squeeze(-1), labels).item()
        return loss, None
    loss = torch.nn.functional.cross_entropy(outputs, labels.long()).item()
    accuracy = (outputs.argmax(-1) == labels.long()).float().mean().item()
    return loss, accuracy


def is_nan(value) -> bool:
    """True when a value is missing or NaN.

    ``np.isnan(None)`` raises TypeError, and the app records unmeasurable
    entanglement as None -- so the run-history block crashed while reading its
    own training history: "ufunc 'isnan' not supported for the input types".
    That happened after "Training complete!", so runs got a history row but no
    details file, and the training-in-progress flag was never reset.
    """
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


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