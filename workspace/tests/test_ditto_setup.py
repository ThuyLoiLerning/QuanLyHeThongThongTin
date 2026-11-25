import os
import importlib.util
import pytest

DITTO_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "03_ditto_pvldb2020")
)
RUNNER_PATH = os.path.join(DITTO_DIR, "run.py")


def _load_ditto_runner():
    spec = importlib.util.spec_from_file_location("ditto_setup_runner", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ditto_runner_exists():
    assert os.path.exists(RUNNER_PATH), f"Ditto runner missing: {RUNNER_PATH}"


def test_ditto_adapter_structure():
    ditto_runner = _load_ditto_runner()
    assert hasattr(ditto_runner, "run_ditto_experiment")
