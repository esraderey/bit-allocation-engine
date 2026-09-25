"""Offline tests for ``scripts/evaluate_pythia_410m.py``.

The Hugging Face download is replaced by a stub that points at synthetic
safetensors checkpoints written into ``tmp_path``; no network is used.
Skipped when the ``eval`` extra (safetensors) is not installed.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

safetensors_numpy = pytest.importorskip("safetensors.numpy")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_pythia_410m.py"


@pytest.fixture
def evaluator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Load the script with ``huggingface_hub`` replaced by a local stub."""
    fake_hub = types.ModuleType("huggingface_hub")

    def snapshot_download(model_id: str, cache_dir=None, allow_patterns=None) -> str:
        return str(tmp_path / model_id)

    fake_hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    spec = importlib.util.spec_from_file_location("evaluate_pythia_410m", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_checkpoint(
    root: Path, model_id: str, shards: list[dict[str, np.ndarray]]
) -> None:
    directory = root / model_id
    directory.mkdir(parents=True, exist_ok=True)
    for index, tensors in enumerate(shards):
        safetensors_numpy.save_file(
            tensors, str(directory / f"model-{index:05d}.safetensors")
        )


def test_sharded_checkpoint_is_evaluated_as_a_whole(evaluator, tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    shard_a = {"a": rng.normal(0, 0.02, (64, 100)).astype(np.float16)}
    shard_b = {"b": rng.normal(0, 0.02, (32, 100)).astype(np.float32)}
    _write_checkpoint(tmp_path, "sharded", [shard_a, shard_b])

    report = evaluator.evaluate("sharded", 4096, tmp_path)

    assert report["weights"]["considered_tensors"] == 2
    assert report["weights"]["considered_parameters"] == 6400 + 3200
    assert sum(report["allocation"]["elements"].values()) == 9600


def test_no_eligible_tensors_raises_clear_error(evaluator, tmp_path: Path) -> None:
    only_ints_and_vectors = {
        "ids": np.arange(12, dtype=np.int64).reshape(3, 4),
        "v": np.ones(5, dtype=np.float32),
    }
    _write_checkpoint(tmp_path, "empty", [only_ints_and_vectors])

    with pytest.raises(RuntimeError, match="nothing to evaluate"):
        evaluator.evaluate("empty", 4096, tmp_path)


def test_missing_safetensors_raises(evaluator, tmp_path: Path) -> None:
    (tmp_path / "nofiles").mkdir()
    with pytest.raises(RuntimeError, match="No safetensors"):
        evaluator.evaluate("nofiles", 4096, tmp_path)


def test_allocated_sse_is_bracketed_by_extremes(evaluator, tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    tensors = {
        "w": rng.normal(0, 0.02, (64, 100)).astype(np.float16),
        "small": rng.normal(0, 1, (3, 5)).astype(np.float32),
    }
    _write_checkpoint(tmp_path, "single", [tensors])

    report = evaluator.evaluate("single", 4096, tmp_path)

    reference = report["quantization_reference"]
    assert reference["INT16"]["mse"] <= reference["allocated"]["mse"]
    assert reference["allocated"]["mse"] <= reference["INT2"]["mse"]
