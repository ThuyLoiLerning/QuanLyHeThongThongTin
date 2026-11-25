"""
TDD tests for Ditto (PVLDB 2020) baseline runner.
Run RED before run.py exists, GREEN after.
"""
import os
import importlib.util
import pytest

DITTO_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "03_ditto_pvldb2020")
)
RUNNER_PATH = os.path.join(DITTO_DIR, "run.py")


def _load_runner():
    spec = importlib.util.spec_from_file_location("ditto_run", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ditto_runner_exists():
    assert os.path.exists(RUNNER_PATH), f"Ditto runner missing: {RUNNER_PATH}"


def test_ditto_adapter_structure():
    mod = _load_runner()
    assert hasattr(mod, "run_ditto_experiment"), "run_ditto_experiment function not found"


def test_evaluate_ditto_exists():
    mod = _load_runner()
    assert hasattr(mod, "evaluate_ditto"), "evaluate_ditto function not found"


def test_evaluate_ditto_quick():
    """1 epoch, 200 samples: must return valid metrics dict with F1 > 0."""
    mod = _load_runner()
    metrics = mod.evaluate_ditto(max_epochs=2, max_samples=200, save_metrics=False)
    for key in ("precision", "recall", "f1", "runtime_sec"):
        assert key in metrics, f"Missing key: {key}"
    assert 0.0 <= metrics["f1"] <= 1.0, f"F1 out of range: {metrics['f1']}"
    assert metrics["f1"] > 0.0, f"F1 is zero — model not learning"


def test_metrics_csv_created(tmp_path):
    """After save_metrics=True, CSV must exist with correct columns."""
    mod = _load_runner()
    metrics_path = str(tmp_path / "metrics.csv")
    mod.evaluate_ditto(max_epochs=2, max_samples=200, save_metrics=True, output_metrics_path=metrics_path)
    assert os.path.exists(metrics_path), f"metrics.csv not created at {metrics_path}"

    import csv
    with open(metrics_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) >= 1
    for col in ("model", "dataset", "precision", "recall", "f1", "runtime_sec"):
        assert col in rows[0], f"Missing column: {col}"
