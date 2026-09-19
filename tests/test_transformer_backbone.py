"""Regression tests for the Transformer classical backbone.

Background (2026-09-19): selecting the Transformer backbone could not complete a
single forward pass. ``HybridModel.forward`` calls its backbone positionally at
``core/fusion.py:255`` -- ``self.classical_backbone(x)`` -- but HuggingFace
``AutoModel`` is keyword-only and returns a ``BaseModelOutput``:

    dict input (what the app's Text path produces)
        TypeError: unhashable type: 'slice'
    tensor input
        AttributeError: 'BaseModelOutput' object has no attribute 'shape'

``HybridModel.transformer_pooling()`` existed to bridge the second half but was
never called. ``TransformerFeatureExtractor`` now does both: dispatch a tokenizer
dict as ``**kwargs``, then pool ``last_hidden_state`` to ``[batch, hidden]``.

These tests are hermetic: they build a tiny BERT from config rather than
downloading a checkpoint, so they run offline and in about a second.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

transformers = pytest.importorskip("transformers")
from transformers import BertConfig, BertModel  # noqa: E402
from transformers.modeling_outputs import BaseModelOutput  # noqa: E402

import core.classical_models as classical_models  # noqa: E402
import core.fusion as fusion  # noqa: E402
from core.classical_models import TransformerFeatureExtractor  # noqa: E402
from core.quantum_models import ansatz1  # noqa: E402

HIDDEN = 32


def _tiny_bert():
    """A 1-layer BERT built from config -- no download, no checkpoint."""
    config = BertConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    return BertModel(config)


def _batch(batch_size=2, seq_len=8):
    ids = torch.randint(0, 100, (batch_size, seq_len))
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


# --------------------------------------------------------------------------
# pooling semantics, pinned deterministically with a fake model
# --------------------------------------------------------------------------
class _FakeHFModel(nn.Module):
    """Returns a BaseModelOutput with known values, exactly like AutoModel."""

    def __init__(self, hidden_size):
        super().__init__()
        self.config = type("C", (), {"hidden_size": hidden_size})()

    def forward(self, input_ids=None, attention_mask=None):
        batch, seq = input_ids.shape
        values = torch.arange(batch * seq * self.config.hidden_size, dtype=torch.float32)
        return BaseModelOutput(last_hidden_state=values.reshape(batch, seq, self.config.hidden_size))


def test_pooling_uses_cls_token():
    wrapper = TransformerFeatureExtractor(_FakeHFModel(4))
    out = wrapper(_batch(batch_size=2, seq_len=2))

    assert out.shape == (2, 4)
    hidden = _FakeHFModel(4).forward(**_batch(batch_size=2, seq_len=2)).last_hidden_state
    assert torch.equal(out, hidden[:, 0]), "expected the [CLS] token representation"

    mean_wrapper = TransformerFeatureExtractor(_FakeHFModel(4), pool="mean")
    assert torch.allclose(mean_wrapper(_batch(batch_size=2, seq_len=2)), hidden.mean(dim=1))


def test_returns_a_tensor_not_a_model_output():
    out = TransformerFeatureExtractor(_FakeHFModel(4))(_batch())
    assert isinstance(out, torch.Tensor)
    assert not isinstance(out, BaseModelOutput)


def test_rejects_unsupported_input_type():
    wrapper = TransformerFeatureExtractor(_FakeHFModel(4))
    with pytest.raises(TypeError):
        wrapper(["not", "a", "tensor"])


# --------------------------------------------------------------------------
# the real factory must return something that survives being called
# --------------------------------------------------------------------------
def test_get_transformer_model_wraps_automodel(monkeypatch):
    tiny = _tiny_bert()
    monkeypatch.setattr(
        classical_models.AutoModel, "from_pretrained", lambda *a, **k: tiny
    )

    backbone, feature_size = classical_models.get_transformer_model("tiny-bert")

    assert isinstance(backbone, TransformerFeatureExtractor)
    assert feature_size == HIDDEN

    # The regression: a tokenizer dict must not be passed positionally.
    out = backbone(_batch())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, HIDDEN)


# --------------------------------------------------------------------------
# end to end through HybridModel: forward, shapes, and backprop
# --------------------------------------------------------------------------
def _transformer_hybrid(monkeypatch, num_classes=2):
    """Build HybridModel through the REAL get_transformer_model factory.

    Only AutoModel.from_pretrained is replaced (with a config-built tiny BERT),
    so the production code path -- including the wrapper that get_transformer_model
    is responsible for applying -- is what gets exercised here.
    """
    tiny = _tiny_bert()
    monkeypatch.setattr(
        classical_models.AutoModel, "from_pretrained", lambda *a, **k: tiny
    )
    return fusion.HybridModel(
        {
            "classical_backbone_type": "Transformer",
            "classical_model_name": "tiny-bert",
            "num_qubits": 4,
            "num_layers": 2,
            "ansatz_func": ansatz1,
            "use_gpu": False,
            "observation_type": "State Vector",
            "encoding_method": "amplitude",
            "num_classes": num_classes,
        }
    )


def test_hybrid_forward_accepts_tokenizer_dict(monkeypatch):
    model = _transformer_hybrid(monkeypatch)
    model.eval()

    with torch.no_grad():
        out = model(_batch())

    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()


def test_hybrid_backprop_reaches_backbone_fusion_and_quantum(monkeypatch):
    model = _transformer_hybrid(monkeypatch)
    model.train()

    out = model(_batch())
    out.sum().backward()

    assert model.fusion_layer.module.weight.grad is not None, "fusion layer got no gradient"
    assert model.quantum_params.grad is not None, "quantum params got no gradient"
    backbone_with_grads = [p for p in model.classical_backbone.parameters() if p.grad is not None]
    assert backbone_with_grads, "transformer backbone got no gradients (fine-tuning impossible)"

    # The residual/skip path must carry the pooled transformer features into the head.
    assert model.skip_size == HIDDEN
