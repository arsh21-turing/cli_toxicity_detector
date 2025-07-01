#!/usr/bin/env python3
"""Tests for --json-lines streaming mode (single-text + file)."""

from __future__ import annotations

import json
import os
import sys
from io import StringIO
from typing import List
from unittest.mock import patch, MagicMock

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from categories import ToxicityCategory  # noqa: E402, after path tweak
from main import display_single_text_result  # noqa: E402
from file_processor import process_file, display_results  # noqa: E402


# ---------------------------------------------------------------------------
# Reusable fixtures ---------------------------------------------------------
# ---------------------------------------------------------------------------

@pytest.fixture()
def base_configs():
    cfg_json_lines = {
        "model": {"name": "stub"},
        "thresholds": {c.name: 0.5 for c in ToxicityCategory},
        "display": {"json_lines": True, "json_output": False, "raw_scores": False, "verbosity": "normal"},
    }
    return cfg_json_lines


@pytest.fixture()
def sample_result():
    # Single prediction object as produced by predict_toxicity helper
    return {
        "text": "foo bar",
        "is_toxic": False,
        "most_probable_category": ToxicityCategory.NON_TOXIC,
        "category_results": {
            ToxicityCategory.NON_TOXIC: {"score": 0.9, "above_threshold": True, "threshold": 0.5},
            ToxicityCategory.INSULT: {"score": 0.1, "above_threshold": False, "threshold": 0.5},
        },
    }


# ---------------------------------------------------------------------------
# Single-text mode ----------------------------------------------------------
# ---------------------------------------------------------------------------

@patch("sys.stdout", new_callable=StringIO)
def test_single_text_stream(mock_stdout: StringIO, sample_result, base_configs):
    """First line should be compact JSON when json_lines is on."""
    display_single_text_result(sample_result, base_configs)
    output = mock_stdout.getvalue().strip().split("\n")
    assert output, "Expected some stdout"

    compact = json.loads(output[0])
    assert compact["text"] == sample_result["text"]
    assert "categories" in compact
    # Ensure there is some form of summary afterwards (human)
    assert any("Overall assessment" in line for line in output[1:])


# ---------------------------------------------------------------------------
# File streaming ------------------------------------------------------------
# ---------------------------------------------------------------------------

@patch("sys.stdout", new_callable=StringIO)
@patch("model_loader.predict_toxicity")
def test_process_file_stream(mock_predict, mock_stdout: StringIO, base_configs):
    """Each returned prediction should be emitted as a compact JSON line."""

    # Prepare stub return value: two predictions
    stub_preds: List[dict] = [
        {
            "text": "hello",
            "is_toxic": False,
            "most_probable_category": ToxicityCategory.NON_TOXIC,
            "category_results": {
                ToxicityCategory.NON_TOXIC: {"score": 0.8, "above_threshold": True, "threshold": 0.5},
            },
        },
        {
            "text": "dummy",
            "is_toxic": True,
            "most_probable_category": ToxicityCategory.INSULT,
            "category_results": {
                ToxicityCategory.INSULT: {"score": 0.9, "above_threshold": True, "threshold": 0.5},
            },
        },
    ]

    # predict_toxicity should be called twice (batchsize default 32 but we simulate 2 lines), we can just return stub for each call
    mock_predict.side_effect = [stub_preds]

    # Create temp file
    import tempfile, textwrap

    tmp = tempfile.NamedTemporaryFile("w+", delete=False)
    try:
        tmp.write("hello\n\ndummy\n")
        tmp.flush()
        tmp.close()

        summary = process_file(tmp.name, {"model": None, "tokenizer": None, "device": None}, base_configs, show_progress=False)
        # Assert summary still correct
        assert summary["total_lines"] == 2
    finally:
        os.unlink(tmp.name)

    # Examine stdout
    lines = [ln for ln in mock_stdout.getvalue().split("\n") if ln.strip()]
    # Two compact JSON lines expected
    compact_objs = [json.loads(ln) for ln in lines[:2]]
    assert compact_objs[0]["line_number"] == 1
    assert compact_objs[1]["line_number"] == 2


# ---------------------------------------------------------------------------
# Summary presentation ------------------------------------------------------
# ---------------------------------------------------------------------------

@patch("sys.stdout", new_callable=StringIO)
def test_display_results_after_stream(mock_stdout: StringIO, base_configs):
    """display_results should not include details list when json_lines enabled."""

    summary_stub = {
        "file_path": "dummy.txt",
        "total_lines": 1,
        "statistics": {
            "total_analyzed": 1,
            "total_toxic": 0,
            "percent_toxic": 0.0,
            "categories": {c: {"total_above_threshold": 0, "most_probable_count": 0, "avg_score": 0.1, "threshold": 0.5, "percentage_above_threshold": 0.0, "percentage_most_probable": 0.0} for c in ToxicityCategory},
        },
        "line_results": [],
    }
    display_results(summary_stub, base_configs, json_output=False)
    out = mock_stdout.getvalue()
    # Should contain summary lines
    assert "Total lines analysed" in out
    # Should *not* mention Detailed Line Results (skipped)
    assert "Detailed Line Results" not in out 