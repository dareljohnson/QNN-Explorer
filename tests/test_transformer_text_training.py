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
from utils.helpers import resolve_device  # noqa: E402
from utils.tensor_utils import ensure_real  # noqa: E402


# --------------------------------------------------------------- labels
def test_string_labels_are_factorized_to_class_indices():
    labels = ["cs.CV", "cs.RO", "cs.CV", "cs.CR", "cs.CV", "cs.RO"]
    tensor, class_names = encode_classification_labels(labels)

    assert tensor.dtype == torch.long, "class indices must be Long for CrossEntropyLoss"
    assert tensor.tolist() == [0, 1, 0, 2, 0, 1]
    assert class_names == ["cs.CV", "cs.RO", "cs.CR"]
    assert len(torch.unique(tensor)) == len(class_names) == 3


def test_string_labels_no_longer_crash_torch_tensor():
    """The exact failure mode: torch.tensor() on strings."""
    with pytest.raises(Exception):
        torch.tensor(["cs.CV", "cs.RO"])

    tensor, _ = encode_classification_labels(["cs.CV", "cs.RO"])
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
