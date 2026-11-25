import os
import sys
import pandas as pd
import pytest

RUNNER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "01_tfidf_baseline", "run.py")


def test_tfidf_runner_file_exists():
    assert os.path.exists(RUNNER_PATH)


def test_tfidf_training_and_evaluation():
    sys.path.insert(0, os.path.dirname(RUNNER_PATH))
    import run as tfidf_runner

    metrics = tfidf_runner.evaluate_tfidf_baseline(max_features=500, max_iter=50)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["f1"] > 0.50, f"TF-IDF F1 too low: {metrics['f1']}"
