"""Regression tests for Transformer text training on the app's data path.

Two defects made Transformer text classification impossible, both found by
driving the real UI end to end:

1. ``app.py`` did ``torch.tensor(labels)`` on the extracted label column. With
   categorical labels (e.g. 'cs.CV') that raises::

       Error during tokenization: too many dimensions 'str'

   The vector path (``preprocess_csv``) had always factorized labels; the text
   path never did. ``encode_classification_labels`` now does it for both.

2. ``ensure_real`` cast *every* tensor to float32, including token ids, so the
   Transformer's ``nn.Embedding`` received floats::

       Expected tensor for argument #1 'indices' to have one of the following
       scalar types: Long, Int; but got torch.cuda.FloatTensor instead

   Integral/bool tensors are indices and masks now and are left alone.

3. ``st.session_state.device`` came from ``check_gpu()``, ignoring the
   configuration's ``use_gpu`` flag, which put the backbone on cuda and the
   quantum simulator on cpu and broke validation with "Expected all tensors to
   be on the same device ... cuda:0 and cpu". ``resolve_device`` makes the flag
   authoritative.
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.preprocessing import encode_classification_labels  # noqa: E402
from utils.helpers import (build_optimizer, forward_in_batches, freeze_params,
                           resolve_device)  # noqa: E402
from utils.tensor_utils import ensure_real  # noqa: E402


# --------------------------------------------------------------- labels
def test_string_labels_are_factorized_to_class_indices():
    labels = ["cs.CV", "cs.RO", "cs.CV", "cs.CR", "cs.CV", "cs.RO"]
    tensor, class_names = encode_classification_labels(labels)

    assert tensor.dtype == torch.long, "class indices must be Long for CrossEntropyLoss"
    assert class_names == ["cs.CR", "cs.CV", "cs.RO"], "classes must be sorted"
    assert tensor.tolist() == [1, 2, 1, 0, 1, 2]
    assert len(torch.unique(tensor)) == len(class_names) == 3


def test_class_indices_do_not_depend_on_row_order():
    """Appearance-order mapping silently permuted classes: 93% held-out became 42%."""
    a, names_a = encode_classification_labels(["cs.CV", "cs.RO", "cs.CR"])
    b, names_b = encode_classification_labels(["cs.RO", "cs.CR", "cs.CV"])

    assert names_a == names_b == ["cs.CR", "cs.CV", "cs.RO"]
    # each class must land on the same index whichever order the rows arrive in
    assert dict(zip(["cs.CV", "cs.RO", "cs.CR"], a.tolist())) ==            dict(zip(["cs.RO", "cs.CR", "cs.CV"], b.tolist())) ==            {"cs.CR": 0, "cs.CV": 1, "cs.RO": 2}


def test_string_labels_no_longer_crash_torch_tensor():
    """The exact failure mode: torch.tensor() on strings."""
    with pytest.raises(Exception):
        torch.tensor(["cs.CV", "cs.RO"])

    tensor, names = encode_classification_labels(["cs.CV", "cs.RO"])
    assert names == ["cs.CV", "cs.RO"], "only these two classes are present, sorted"
    assert tensor.tolist() == [0, 1]


def test_numeric_labels_pass_through_unchanged():
    tensor, class_names = encode_classification_labels([1.5, 2.25, 3.0])

    assert class_names is None, "numeric labels are regression targets, not classes"
    assert torch.allclose(tensor, torch.tensor([1.5, 2.25, 3.0]))
    assert tensor.dtype.is_floating_point


def test_integer_labels_are_not_refactored():
    tensor, class_names = encode_classification_labels([10, 20, 20, 10])

    assert class_names is None
    assert tensor.tolist() == [10, 20, 20, 10], "index labels must keep their values"


def test_three_class_demo_labels_after_factorization():
    """The ArXiv demo's category_code column, as used by the smoke run."""
    labels = ["cs.CV"] * 400 + ["cs.CR"] * 400 + ["cs.RO"] * 400
    tensor, class_names = encode_classification_labels(labels)

    assert class_names == ["cs.CR", "cs.CV", "cs.RO"]
    assert len(torch.unique(tensor)) == 3
    assert len(class_names) == 3


# --------------------------------------------------------------- dtypes
def test_ensure_real_preserves_token_ids_and_masks():
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    mask = torch.ones(1, 3, dtype=torch.long)

    out = ensure_real({"input_ids": ids, "attention_mask": mask})

    assert out["input_ids"].dtype == torch.long
    assert out["attention_mask"].dtype == torch.long
    assert out["input_ids"] is ids, "integral tensors should be returned untouched"


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.int32, torch.int64])
def test_ensure_real_leaves_every_integral_dtype_alone(dtype):
    tensor = torch.zeros(4, dtype=dtype)
    assert ensure_real(tensor) is tensor


def test_ensure_real_still_casts_floats():
    tensor = torch.randn(4, dtype=torch.float64)
    out = ensure_real(tensor)

    assert out.dtype == torch.float32
    assert isinstance(out, torch.Tensor)


# --------------------------------------------------------------- device
def test_resolve_device_honours_the_flag():
    assert resolve_device(False) == torch.device("cpu")

    expected = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert resolve_device(True) == expected

    if not torch.cuda.is_available():
        # Requesting a GPU without one must not raise; it must fall back.
        assert resolve_device(True) == torch.device("cpu")


# --------------------------------------------------- backbone learning rate
def test_transformer_backbone_uses_a_much_smaller_step():
    """lr*0.1 (=5e-4) leaves the loss at chance; lr*0.01 trains cleanly."""
    from utils.helpers import backbone_learning_rate

    assert backbone_learning_rate(0.005, "Transformer") == pytest.approx(0.00005)
    assert backbone_learning_rate(0.005, "transformer") == pytest.approx(0.00005)
    assert backbone_learning_rate(0.005, "Transformer") < 0.005 * 0.1


def test_non_transformer_backbones_keep_the_old_step():
    from utils.helpers import backbone_learning_rate

    for backbone in ("CNN", "GNN", "Regression", None, ""):
        assert backbone_learning_rate(0.005, backbone) == pytest.approx(0.0005)


# --------------------------------------------------- optimiser / batching
class _DummyHybrid(torch.nn.Module):
    """Minimal stand-in with a classical_backbone attribute."""

    def __init__(self):
        super().__init__()
        self.classical_backbone = torch.nn.Linear(4, 4)
        self.classical_backbone_type = "Transformer"
        self.head = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.classical_backbone(x))


def test_backbone_is_frozen_by_default():
    model = _DummyHybrid()
    optimizer, backbone_params, fine_tuning = build_optimizer(model, 0.005, finetune_backbone=False)

    assert backbone_params is None and fine_tuning is False
    assert all(not p.requires_grad for p in model.classical_backbone.parameters())
    tuned = [p for group in optimizer.param_groups for p in group["params"]]
    assert all(p is not model.classical_backbone.weight for p in tuned), "frozen backbone must not be optimised"
    assert model.head.weight in tuned


def test_finetuning_uses_a_smaller_backbone_rate_and_reports_it():
    model = _DummyHybrid()
    optimizer, backbone_params, fine_tuning = build_optimizer(model, 0.005, finetune_backbone=True)

    assert fine_tuning is True and backbone_params is not None
    rates = sorted(g["lr"] for g in optimizer.param_groups)
    assert rates[0] == pytest.approx(0.00005), "backbone group must use the transformer rate"
    assert rates[1] == pytest.approx(0.005)


def test_freeze_params_stops_updates():
    layer = torch.nn.Linear(3, 3)
    freeze_params(list(layer.parameters()))
    assert all(not p.requires_grad for p in layer.parameters())

    freeze_params(None)  # must tolerate the frozen-by-default case


def test_forward_in_batches_handles_dicts_and_batches():
    calls = []

    class _Recorder(torch.nn.Module):
        def forward(self, x):
            calls.append((type(x).__name__, x["input_ids"].shape[0]))
            return x["input_ids"].float().mean(dim=1, keepdim=True).repeat(1, 2)

    enc = {"input_ids": torch.arange(60).reshape(10, 6),
           "attention_mask": torch.ones(10, 6, dtype=torch.long)}
    out = forward_in_batches(_Recorder(), enc, batch_size=4)

    assert out.shape == (10, 2)
    assert calls == [("dict", 4), ("dict", 4), ("dict", 2)], calls


def test_forward_in_batches_handles_plain_tensors():
    class _Double(torch.nn.Module):
        def forward(self, x):
            return x * 2

    features = torch.arange(20).reshape(10, 2).float()
    out = forward_in_batches(_Double(), features, batch_size=3)
    assert torch.equal(out, features * 2)
