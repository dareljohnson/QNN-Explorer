"""Regression tests for run-history persistence.

A run's ``model_config`` carries ``ansatz_func``, a callable the app injects when
a model is instantiated. ``json.dump`` cannot serialise it, and it raised *after*
writing part of the file, so both ``run_N.json`` and its ``.bak`` were left
truncated:

    {
      ... "quantum_input_size": 16,
      "ansatz_func":            <- end of file

The app's repair subsystem then kept archiving and rebuilding these files. Two
fixes: sanitise non-JSON values, and write through a temp file + rename so no
failure can truncate the destination.
"""

import json
import math
import pathlib
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils.run_history as run_history  # noqa: E402
from utils.run_history import atomic_json_dump, json_safe, save_run_details  # noqa: E402


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


# ------------------------------------------------------------------ json_safe
def test_callables_become_their_name():
    """The exact value that broke every fresh-config run."""
    safe = json_safe({"model_config": {"ansatz_func": _sigmoid, "num_qubits": 4}})

    assert safe["model_config"]["ansatz_func"] == "_sigmoid"
    json.dumps(safe)  # must not raise


def test_numpy_tensors_and_scalars_are_converted():
    safe = json_safe({
        "array": np.array([1, 2, 3]),
        "scalar": np.float32(1.5),
        "integer": np.int64(7),
        "tensor": torch.ones(2, 2),
    })

    assert safe["array"] == [1, 2, 3]
    assert safe["scalar"] == pytest.approx(1.5)
    assert safe["integer"] == 7
    assert safe["tensor"] == [[1.0, 1.0], [1.0, 1.0]]
    json.dumps(safe)


def test_nan_and_infinity_become_null():
    """JSON has no NaN/Infinity; writing them produced files other parsers reject."""
    safe = json_safe({"history": [1.0, float("nan"), float("inf"), -float("inf")]})

    assert safe["history"] == [1.0, None, None, None]
    assert json.loads(json.dumps(safe))["history"] == [1.0, None, None, None]


def test_nested_structures_and_unknown_objects():
    class Opaque:
        def __repr__(self):
            return "Opaque()"

    safe = json_safe({"a": [{"b": (_sigmoid, np.array([1]))}], "c": Opaque(), "d": {1, 2}})

    assert safe["a"][0]["b"] == ["_sigmoid", [1]]
    assert safe["c"] == "Opaque()"
    assert sorted(safe["d"]) == [1, 2]
    json.dumps(safe)


# ------------------------------------------------------------ atomic writing
def test_atomic_write_leaves_no_partial_file_on_failure(tmp_path):
    target = tmp_path / "run_1.json"

    class Unserialisable:
        pass

    ok = atomic_json_dump({"bad": Unserialisable()}, str(target), indent=2)

    assert ok is False
    assert not target.exists(), "a failed write must not leave a partial file"
    assert not (tmp_path / "run_1.json.tmp").exists(), "temp file must be cleaned up"


def test_atomic_write_succeeds_normally(tmp_path):
    target = tmp_path / "run_2.json"
    assert atomic_json_dump({"a": 1}, str(target))
    assert json.loads(target.read_text()) == {"a": 1}


# ------------------------------------------------------- save_run_details
def test_save_run_details_survives_a_callable_in_model_config(tmp_path, monkeypatch):
    monkeypatch.setattr(run_history, "RUN_DETAILS_PATH", str(tmp_path))

    details = {
        "task_type": "Classification",
        "model_config": {
            "num_qubits": 4,
            "classical_model_name": "distilbert-base-uncased",
            "ansatz_func": _sigmoid,
            "use_gpu": True,
        },
        "training_history": {"loss": [1.0, float("nan")]},
    }

    assert save_run_details(1, details) is True, "the save should now succeed"

    written = tmp_path / "run_1.json"
    backup = tmp_path / "run_1.json.bak"
    assert written.exists() and backup.exists()

    loaded = json.loads(written.read_text())
    assert loaded["model_config"]["ansatz_func"] == "_sigmoid"
    assert loaded["training_history"]["loss"] == [1.0, None]
    json.loads(backup.read_text()), "the backup must be valid JSON too"


def test_old_writer_would_have_truncated_the_file(tmp_path):
    """Documents the original failure: streaming json.dump to the final path."""
    target = tmp_path / "old_behaviour.json"
    with pytest.raises(TypeError):
        with open(target, "w") as handle:
            json.dump({"ansatz_func": _sigmoid, "tail": list(range(500))}, handle, indent=2)

    content = target.read_text()
    assert content and not content.rstrip().endswith("}"), "file is left truncated"
    with pytest.raises(json.JSONDecodeError):
        json.loads(content)


# ------------------------------------------------------------------- is_nan
def test_is_nan_handles_none_and_nan():
    """np.isnan(None) raises; the run-history block died on exactly this."""
    from utils.helpers import is_nan

    with pytest.raises(TypeError):
        np.isnan(None)

    assert is_nan(None) is True
    assert is_nan(float("nan")) is True
    assert is_nan(0.5) is False
    assert is_nan(0) is False
    assert is_nan("text") is False, "non-numeric values must not raise"


def test_entanglement_history_with_none_survives_a_save(tmp_path, monkeypatch):
    """A None entanglement value used to abort the whole run-history save."""
    monkeypatch.setattr(run_history, "RUN_DETAILS_PATH", str(tmp_path))

    details = {
        "training_history": {
            "entanglement": [0.5, None, float("nan"), 0.7],  # as the app records it
            "loss": [1.0, 0.9],
            "accuracy": [0.5, 0.6],
        }
    }

    assert save_run_details(2, details) is True
    loaded = json.loads((tmp_path / "run_2.json").read_text())
    assert loaded["training_history"]["entanglement"] == [0.5, None, None, 0.7]
