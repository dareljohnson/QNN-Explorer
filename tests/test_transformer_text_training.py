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
                           resolve_device, split_train_val, evaluate_split)  # noqa: E402
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


# --------------------------------------------------- train/validation split
def test_split_train_val_holds_out_a_fraction():
    features = {"input_ids": torch.arange(100).reshape(20, 5),
                "attention_mask": torch.ones(20, 5, dtype=torch.long)}
    labels = torch.arange(20, dtype=torch.long)

    (tr_f, tr_y), (va_f, va_y) = split_train_val(features, labels, val_fraction=0.25, seed=0)

    assert len(va_y) == 5 and len(tr_y) == 15
    assert va_f["input_ids"].shape == (5, 5)
    # the two halves must be disjoint (an overlapping split silently leaks)
    assert set(tr_y.tolist()).isdisjoint(set(va_y.tolist()))
    assert set(tr_y.tolist()) | set(va_y.tolist()) == set(range(20))


def test_split_train_val_works_on_plain_tensors():
    features = torch.arange(40).reshape(10, 4).float()
    labels = torch.arange(10, dtype=torch.long)

    (tr_f, tr_y), (va_f, va_y) = split_train_val(features, labels, val_fraction=0.2, seed=1)

    assert tr_f.shape == (8, 4) and va_f.shape == (2, 4)
    assert len(tr_y) == 8 and len(va_y) == 2


def test_split_train_val_can_be_disabled_and_never_empties_the_training_set():
    features = torch.zeros(5, 2)
    labels = torch.zeros(5, dtype=torch.long)

    (tr_f, _), (va_f, va_y) = split_train_val(features, labels, val_fraction=0.0)
    assert va_f is None and va_y is None
    assert tr_f is features

    # a fraction that would consume everything must still leave training rows
    (tr_f2, _), (va_f2, _) = split_train_val(features, labels, val_fraction=1.0)
    assert len(tr_f2) >= 1 and len(va_f2) == 4


def test_split_is_reproducible_for_a_seed():
    features = torch.arange(30).reshape(10, 3).float()
    labels = torch.arange(10, dtype=torch.long)

    _, (a_f, a_y) = split_train_val(features, labels, 0.3, seed=7)
    _, (b_f, b_y) = split_train_val(features, labels, 0.3, seed=7)

    assert torch.equal(a_f, b_f) and torch.equal(a_y, b_y)


# --------------------------------------------------- image-style batches
def test_forward_in_batches_accepts_a_list_of_tensors():
    """Image datasets arrive as a list of per-sample tensors, not a stacked tensor.

    This used to raise "AttributeError: 'list' object has no attribute 'to'"
    inside the validation split, which the training error handler then masked.
    """
    class _Flatten(torch.nn.Module):
        def forward(self, x):
            return x.reshape(x.shape[0], -1)[:, :2]

    images = [torch.randn(3, 8, 8) for _ in range(6)]
    out = forward_in_batches(_Flatten(), images, batch_size=2)

    assert out.shape == (6, 2)


def test_forward_in_batches_accepts_a_list_with_unstackable_items():
    class _Count(torch.nn.Module):
        def forward(self, batch):
            return torch.zeros(len(batch), 2)

    out = forward_in_batches(_Count(), [[1, 2], [3, 4]], batch_size=1)
    assert out.shape == (2, 2)


def test_evaluate_split_accepts_plain_list_labels():
    class _Constant(torch.nn.Module):
        def forward(self, x):
            batch = len(x) if not isinstance(x, torch.Tensor) else x.shape[0]
            return torch.stack([torch.tensor([0.1, 0.9])] * batch)

    features = [torch.randn(3, 8, 8) for _ in range(4)]
    labels = [1, 1, 1, 1]
    loss, accuracy = evaluate_split(_Constant(), features, labels)

    assert accuracy == pytest.approx(1.0)
    assert loss is not None


# ------------------------------------------- scoring a head-less model
def test_evaluate_split_refuses_to_score_a_headless_model():
    """The reported crash: mse_loss((109, 16), (109,)).

    2^N state vectors have 16 columns; 109 float labels broadcast as 16 vs 109.
    Unsupervised tasks have no output head, so there is no prediction to score.
    """
    class _StateVector(torch.nn.Module):
        """Stands in for HybridModel with output_head = None."""

        output_head = None

        def forward(self, x):
            batch = len(x) if not isinstance(x, torch.Tensor) else x.shape[0]
            return torch.randn(batch, 16)  # 2^N for N=4

    features = torch.randn(109, 109)
    labels = torch.rand(109)  # numeric label column -> float dtype

    loss, accuracy = evaluate_split(_StateVector(), features, labels)

    assert (loss, accuracy) == (None, None), "must report 'not scorable', not raise"


def test_evaluate_split_scores_a_real_regression_head():
    class _Scalar(torch.nn.Module):
        output_head = object()  # a head exists

        def forward(self, x):
            batch = len(x) if not isinstance(x, torch.Tensor) else x.shape[0]
            return torch.zeros(batch, 1)

    # zeros against a zero prediction: MSE is exactly 0
    loss, accuracy = evaluate_split(_Scalar(), torch.randn(8, 4), torch.zeros(8))

    assert accuracy is None and loss == pytest.approx(0.0)


def test_evaluate_split_skips_output_width_that_cannot_match_labels():
    class _Wide(torch.nn.Module):
        output_head = object()

        def forward(self, x):
            batch = len(x) if not isinstance(x, torch.Tensor) else x.shape[0]
            return torch.randn(batch, 2)  # 2 logits

    # labels go up to 5, so 2 logits cannot be compared against them
    loss, accuracy = evaluate_split(_Wide(), torch.randn(6, 4), torch.tensor([0, 5, 1, 2, 3, 4]))
    assert (loss, accuracy) == (None, None)

    # a healthy 3-class case still scores
    class _Three(torch.nn.Module):
        output_head = object()

        def forward(self, x):
            batch = len(x) if not isinstance(x, torch.Tensor) else x.shape[0]
            out = torch.zeros(batch, 3)
            out[:, 0] = 1.0
            return out

    loss, accuracy = evaluate_split(_Three(), torch.randn(6, 4), torch.tensor([0, 0, 0, 0, 0, 0]))
    assert accuracy == pytest.approx(1.0)
