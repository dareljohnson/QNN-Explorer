"""Regression tests for the Visualize tab in app.py.

Background (2026-09-19): the Visualize tab rendered no plots. Cause: every
``plot_*`` function in ``utils/visualization.py`` *returns* its artwork --
``plot_circuit`` a base64 PNG string, ``plot_training_history`` /
``plot_state_vector`` a matplotlib Figure -- and that module contains no
``st.image`` / ``st.pyplot`` call of its own. Display is the caller's job, and
four call sites in app.py discarded the return value:

    app.py:2876  Visualize > Circuit Diagram      (base64 string dropped)
    app.py:2885  Visualize > Training History     (Figure dropped)
    app.py:2376  Predict  > Output State Vector   (Figure dropped)
    app.py:1857  Train    > Final Training History(Figure dropped)

These tests run the real app inside Streamlit's own runtime (AppTest, not a
mock) and assert that images actually reach the front end.

Each test chdirs into a tmp dir first: the app writes run-history/scaler files
under relative paths on load, and that must not land in the repository.
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot use

import pytest  # noqa: E402

from streamlit.testing.v1 import AppTest  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
APP_PATH = REPO_ROOT / "app.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CIRCUIT_CAPTION_PREFIX = "Quantum circuit"

# Mirrors the values the app's own config tab produces: HybridModel lowercases
# these internally, and app.py:248 feeds 'observation_type' straight into a
# selectbox whose options are AVAILABLE_OBSERVATIONS.
MODEL_CONFIG = {
    "classical_backbone_type": "None",
    "classical_model_name": "linear",
    "num_qubits": 4,
    "num_layers": 2,
    "use_gpu": False,
    "observation_type": "State Vector",
    "encoding_method": "amplitude",
}


def _images(at):
    """Return [(url, caption)] for every image element the app emitted."""
    out = []
    for element in at.get("imgs"):
        for img in getattr(element.proto, "imgs", []):
            out.append((img.url or "", img.caption))
    return out


def _run_app(tmp_path, monkeypatch, session_state=None):
    """Run app.py in a real Streamlit runtime, isolated in tmp_path."""
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(APP_PATH), default_timeout=900)
    for key, value in (session_state or {}).items():
        at.session_state[key] = value
    at.run()
    return at


def test_visualize_renders_nothing_extra_without_a_model(tmp_path, monkeypatch):
    """No model -> the tab must say so, and must not emit a circuit image."""
    at = _run_app(tmp_path, monkeypatch)

    assert not at.exception, [str(e.value) for e in at.exception]

    warnings = [str(w.value) for w in at.warning]
    assert any("visualize its circuit" in w for w in warnings), (
        f"expected the 'instantiate a model' warning, got: {warnings}"
    )

    circuit_images = [i for i in _images(at) if i[1].startswith(CIRCUIT_CAPTION_PREFIX)]
    assert circuit_images == [], f"unexpected circuit image(s): {circuit_images}"


def test_visualize_renders_circuit_and_training_history(tmp_path, monkeypatch):
    """With a model + history, both Visualize sub-sections must render images.

    plot_circuit() -> base64 PNG handed to st.image  => data URI + caption
    plot_training_history() -> Figure handed to st.pyplot => non-data-URI media
    """
    from core.fusion import HybridModel
    from core.quantum_models import ansatz1

    config = dict(MODEL_CONFIG, ansatz_func=ansatz1)
    at = _run_app(
        tmp_path,
        monkeypatch,
        session_state={
            "model_config": config,
            "hybrid_model": HybridModel(config),
            "training_history": {
                "loss": [1.0, 0.5, 0.3],
                "accuracy": [0.4, 0.7, 0.86],
            },
        },
    )

    assert not at.exception, [str(e.value) for e in at.exception]

    images = _images(at)
    captions = [c for _, c in images]

    # Circuit diagram: a real base64 PNG, not a placeholder.
    pngs = [url for url, _ in images if url.startswith("data:image/png;base64,")]
    assert pngs, f"no base64 PNG emitted; images were: {images}"
    assert any(c.startswith(CIRCUIT_CAPTION_PREFIX) for c in captions), (
        f"circuit image has no caption; captions: {captions}"
    )

    # Training history: st.pyplot emits a media reference (no data: URI prefix).
    pyplot_images = [url for url, _ in images if not url.startswith("data:image/")]
    assert len(pyplot_images) >= 2, (
        "expected the st.pyplot training-history figure in addition to the app's "
        f"pre-existing media element; images were: {images}"
    )


def test_circuit_diagram_renders_from_the_configuration_alone(tmp_path, monkeypatch):
    """The diagram needs only the configuration, not an instantiated model.

    ``st.session_state.hybrid_model`` lives in session state and is cleared by a
    server restart, which left the Visualize tab showing "instantiate a model
    first" even though the circuit's structure is fully described by the config.
    """
    from core.quantum_models import ansatz1

    at = _run_app(tmp_path, monkeypatch,
                  session_state={"model_config": dict(MODEL_CONFIG, ansatz_func=ansatz1)})

    assert not at.exception, [str(e.value) for e in at.exception]

    images = _images(at)
    circuit = [c for _, c in images if c.startswith(CIRCUIT_CAPTION_PREFIX)]
    assert circuit, f"no circuit diagram rendered from the configuration; images: {images}"
