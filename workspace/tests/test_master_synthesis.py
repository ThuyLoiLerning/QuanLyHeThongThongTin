"""
TDD tests for Master Cross-Paper Synthesis & Comparison Summary.
"""
import os
import sys
import importlib.util
import pytest
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SYNTHESIS_SCRIPT = os.path.join(ROOT_DIR, "workspace", "scripts", "generate_master_synthesis.py")
EXP_DIR = os.path.join(ROOT_DIR, "experiments")
SUMMARY_CSV = os.path.join(EXP_DIR, "comparison_summary.csv")
SYNTHESIS_MD = os.path.join(EXP_DIR, "MASTER_SYNTHESIS.md")


def _load_synthesis_module():
    spec = importlib.util.spec_from_file_location("generate_master_synthesis", SYNTHESIS_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_synthesis_script_exists():
    assert os.path.exists(SYNTHESIS_SCRIPT), f"Synthesis script missing: {SYNTHESIS_SCRIPT}"


def test_compile_synthesis_callable():
    mod = _load_synthesis_module()
    assert hasattr(mod, "compile_synthesis"), "compile_synthesis function not found"
    assert callable(mod.compile_synthesis), "compile_synthesis is not callable"


def test_compile_synthesis_execution_and_files(tmp_path):
    mod = _load_synthesis_module()
    test_csv = str(tmp_path / "test_comparison_summary.csv")
    test_md = str(tmp_path / "test_MASTER_SYNTHESIS.md")

    mod.compile_synthesis(output_csv=test_csv, output_md=test_md)

    assert os.path.exists(test_csv), f"Output CSV not created: {test_csv}"
    assert os.path.exists(test_md), f"Output MD not created: {test_md}"


def test_comparison_summary_schema_and_values():
    mod = _load_synthesis_module()
    mod.compile_synthesis()

    assert os.path.exists(SUMMARY_CSV), f"Summary CSV missing at {SUMMARY_CSV}"
    df = pd.read_csv(SUMMARY_CSV)

    expected_cols = ["model", "dataset", "precision", "recall", "f1", "runtime_sec"]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"

    assert len(df) == 3, f"Expected 3 models, got {len(df)}"

    for _, row in df.iterrows():
        assert 0.0 <= row["precision"] <= 100.0, f"Precision out of range: {row['precision']}"
        assert 0.0 <= row["recall"] <= 100.0, f"Recall out of range: {row['recall']}"
        assert 0.0 <= row["f1"] <= 100.0, f"F1 out of range: {row['f1']}"
        assert row["runtime_sec"] > 0.0, f"Runtime must be positive: {row['runtime_sec']}"


def test_master_synthesis_markdown_content():
    mod = _load_synthesis_module()
    mod.compile_synthesis()

    assert os.path.exists(SYNTHESIS_MD), f"Master synthesis missing at {SYNTHESIS_MD}"
    with open(SYNTHESIS_MD, "r", encoding="utf-8") as f:
        content = f.read()

    # Benchmark table
    assert "Amazon-Google" in content
    assert "TF-IDF" in content
    assert "DeepMatcher" in content
    assert "Ditto" in content

    # Gap analysis
    assert "SIGMOD 2018" in content
    assert "PVLDB 2020" in content

    # Critical gaps in ERP-CRM
    assert "C_{FP}" in content or "CFP" in content or "Bất đối xứng" in content
    assert "Mã số thuế" in content or "Tax ID" in content
    assert "Latency" in content or "Độ trễ" in content or "suy luận" in content
    assert "Explainab" in content or "Giải thích" in content

    # CARE-ER framework positioning
    assert "CARE-ER" in content
    assert "Hard Constraint" in content or "Ràng buộc cứng" in content
