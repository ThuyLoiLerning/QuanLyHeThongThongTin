# workspace/tests/test_benchmark_data.py
import os
import pandas as pd
import pytest

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "shared_data", "Amazon-Google")


def test_amazon_google_splits_exist():
    for split in ["train.csv", "valid.csv", "test.csv"]:
        path = os.path.join(BENCHMARK_DIR, split)
        assert os.path.exists(path), f"Missing split file: {path}"


def test_amazon_google_split_schemas_and_counts():
    train_path = os.path.join(BENCHMARK_DIR, "train.csv")
    valid_path = os.path.join(BENCHMARK_DIR, "valid.csv")
    test_path = os.path.join(BENCHMARK_DIR, "test.csv")

    df_train = pd.read_csv(train_path)
    df_valid = pd.read_csv(valid_path)
    df_test = pd.read_csv(test_path)

    # Schema must contain label and entity pair IDs
    # Real DeepMatcher format: ltable_id, rtable_id, label (no generic "id" column)
    expected_cols = {"ltable_id", "rtable_id", "label"}
    assert expected_cols.issubset(set(df_train.columns)), f"Columns missing in train: {df_train.columns}"
    assert expected_cols.issubset(set(df_test.columns)), f"Columns missing in test: {df_test.columns}"

    # Verify standard DeepMatcher Amazon-Google split sizes
    # Total pairs: 11,460 (train ~6,874, valid ~2,293, test ~2,293)
    total_len = len(df_train) + len(df_valid) + len(df_test)
    assert 11400 <= total_len <= 11500, f"Unexpected total pair count: {total_len}"
    assert (df_test["label"].isin([0, 1])).all(), "Labels must be binary 0 or 1"
