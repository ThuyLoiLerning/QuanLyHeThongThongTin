import os
import importlib.util
import pytest

DEEPMATCHER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', '02_deepmatcher_sigmod2018'))
RUNNER_PATH     = os.path.join(DEEPMATCHER_DIR, 'run.py')


def _load_dm_runner():
    spec = importlib.util.spec_from_file_location("dm_run2", RUNNER_PATH)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_deepmatcher_code_or_runner_exists():
    assert os.path.exists(RUNNER_PATH), f"DeepMatcher runner missing: {RUNNER_PATH}"


def test_deepmatcher_adapter_structure():
    dm = _load_dm_runner()
    assert hasattr(dm, 'run_deepmatcher_experiment'), "run_deepmatcher_experiment not found"
