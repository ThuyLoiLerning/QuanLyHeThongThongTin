import os
import importlib.util
import pytest

DEEPMATCHER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', '02_deepmatcher_sigmod2018'))
RUNNER_PATH     = os.path.join(DEEPMATCHER_DIR, 'run.py')


def _load_dm_runner():
    spec = importlib.util.spec_from_file_location("dm_run", RUNNER_PATH)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_deepmatcher_runner_file_exists():
    assert os.path.exists(RUNNER_PATH), f"DeepMatcher runner missing: {RUNNER_PATH}"


def test_deepmatcher_evaluation():
    dm = _load_dm_runner()
    assert hasattr(dm, 'evaluate_deepmatcher'), "evaluate_deepmatcher function not found"
    metrics = dm.evaluate_deepmatcher(max_epochs=1)
    for key in ('precision', 'recall', 'f1'):
        assert key in metrics, f"Missing key: {key}"
    assert metrics['f1'] > 0.60, f"DeepMatcher F1 too low: {metrics['f1']}"
