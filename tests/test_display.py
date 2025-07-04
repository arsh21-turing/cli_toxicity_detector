#!/usr/bin/env python3
"""Tests for the display functionality (human + JSON)."""

from __future__ import annotations

import json
import os
import sys
from io import StringIO
from unittest.mock import patch

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from categories import ToxicityCategory  # noqa: E402, after path tweak
from main import display_single_text_result  # noqa: E402
from file_processor import display_results  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures ------------------------------------------------------------------
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_objects():
    single_ok = {
        "text": "This is a test text.",
        "is_toxic": False,
        "most_probable_category": ToxicityCategory.NON_TOXIC,
        "category_results": {
            ToxicityCategory.INSULT: {"score": 0.1, "above_threshold": False, "threshold": 0.5},
            ToxicityCategory.HATE: {"score": 0.05, "above_threshold": False, "threshold": 0.5},
            ToxicityCategory.NON_TOXIC: {"score": 0.85, "above_threshold": True, "threshold": 0.5},
        },
        "raw_logits": [-2.3, -2.9, 1.7],
        "sigmoid_scores": [0.1, 0.05, 0.85],
    }

    single_bad = {
        "text": "This is a toxic text.",
        "is_toxic": True,
        "most_probable_category": ToxicityCategory.INSULT,
        "category_results": {
            ToxicityCategory.INSULT: {"score": 0.8, "above_threshold": True, "threshold": 0.5},
            ToxicityCategory.HATE: {"score": 0.3, "above_threshold": False, "threshold": 0.5},
            ToxicityCategory.NON_TOXIC: {"score": 0.2, "above_threshold": False, "threshold": 0.5},
        },
        "raw_logits": [1.4, -0.8, -1.4],
        "sigmoid_scores": [0.8, 0.3, 0.2],
    }

    file_summary = {
        "file_path": "test.txt",
        "total_lines": 3,
        "statistics": {
            "total_analyzed": 3,
            "total_toxic": 1,
            "percent_toxic": 33.33,
            "categories": {
                ToxicityCategory.INSULT: {
                    "total_above_threshold": 1,
                    "percentage_above_threshold": 33.33,
                    "avg_score": 0.33,
                    "threshold": 0.5,
                    "most_probable_count": 1,
                    "percentage_most_probable": 33.33,
                },
                ToxicityCategory.HATE: {
                    "total_above_threshold": 0,
                    "percentage_above_threshold": 0,
                    "avg_score": 0.15,
                    "threshold": 0.5,
                    "most_probable_count": 0,
                    "percentage_most_probable": 0,
                },
                ToxicityCategory.NON_TOXIC: {
                    "total_above_threshold": 2,
                    "percentage_above_threshold": 66.67,
                    "avg_score": 0.53,
                    "threshold": 0.5,
                    "most_probable_count": 2,
                    "percentage_most_probable": 66.67,
                },
            },
        },
        "line_results": [
            {
                "line_number": 1,
                "text": "Non-toxic line 1",
                "is_toxic": False,
                "most_probable_category": ToxicityCategory.NON_TOXIC,
                "category_results": {
                    ToxicityCategory.INSULT: {"score": 0.1, "above_threshold": False, "threshold": 0.5},
                    ToxicityCategory.HATE: {"score": 0.1, "above_threshold": False, "threshold": 0.5},
                    ToxicityCategory.NON_TOXIC: {"score": 0.8, "above_threshold": True, "threshold": 0.5},
                },
            },
            {
                "line_number": 2,
                "text": "Toxic insult line",
                "is_toxic": True,
                "most_probable_category": ToxicityCategory.INSULT,
                "category_results": {
                    ToxicityCategory.INSULT: {"score": 0.8, "above_threshold": True, "threshold": 0.5},
                    ToxicityCategory.HATE: {"score": 0.3, "above_threshold": False, "threshold": 0.5},
                    ToxicityCategory.NON_TOXIC: {"score": 0.2, "above_threshold": False, "threshold": 0.5},
                },
            },
            {
                "line_number": 3,
                "text": "Non-toxic line 2",
                "is_toxic": False,
                "most_probable_category": ToxicityCategory.NON_TOXIC,
                "category_results": {
                    ToxicityCategory.INSULT: {"score": 0.1, "above_threshold": False, "threshold": 0.5},
                    ToxicityCategory.HATE: {"score": 0.05, "above_threshold": False, "threshold": 0.5},
                    ToxicityCategory.NON_TOXIC: {"score": 0.6, "above_threshold": True, "threshold": 0.5},
                },
            },
        ],
    }

    cfg_normal = {"display": {"json_output": False, "verbosity": "normal", "raw_scores": False}}
    cfg_json = {"display": {"json_output": True, "verbosity": "normal", "raw_scores": False}}
    cfg_verbose = {"display": {"json_output": False, "verbosity": "verbose", "raw_scores": True}}

    return single_ok, single_bad, file_summary, cfg_normal, cfg_json, cfg_verbose


# ---------------------------------------------------------------------------
# Single-text mode -----------------------------------------------------------
# ---------------------------------------------------------------------------


@patch("sys.stdout", new_callable=StringIO)
def test_single_text_normal(mock_stdout: StringIO, sample_objects):
    ok, bad, *_ = sample_objects
    cfg = sample_objects[3]

    display_single_text_result(ok, cfg)
    out = mock_stdout.getvalue()
    assert "Overall assessment: NON-TOXIC" in out
    assert "Most probable category: NON_TOXIC" in out

    mock_stdout.truncate(0); mock_stdout.seek(0)
    display_single_text_result(bad, cfg)
    out = mock_stdout.getvalue()
    assert "Overall assessment: TOXIC" in out
    assert "Most probable category: INSULT" in out


@patch("sys.stdout", new_callable=StringIO)
def test_single_text_json(mock_stdout: StringIO, sample_objects):
    ok, *_ = sample_objects
    cfg = sample_objects[4]
    display_single_text_result(ok, cfg)
    obj = json.loads(mock_stdout.getvalue())
    assert obj["most_probable_category"] == "NON_TOXIC"
    assert obj["is_toxic"] is False


# ---------------------------------------------------------------------------
# File summary ---------------------------------------------------------------
# ---------------------------------------------------------------------------


@patch("sys.stdout", new_callable=StringIO)
def test_file_results_normal(mock_stdout: StringIO, sample_objects):
    *_, summary, cfg_normal, _, _ = sample_objects
    display_results(summary, {"display": cfg_normal["display"].copy()}, False)
    txt = mock_stdout.getvalue()
    assert "Results for test.txt" in txt
    assert "Total lines analysed: 3" in txt
    assert "NON_TOXIC: 2 lines" in txt


@patch("sys.stdout", new_callable=StringIO)
def test_file_results_json(mock_stdout: StringIO, sample_objects):
    *_, summary, _, cfg_json, _ = sample_objects
    display_results(summary, {"display": cfg_json["display"].copy()}, True)
    data = json.loads(mock_stdout.getvalue())
    assert data["total_lines"] == 3
    assert data["statistics"]["total_toxic"] == 1


@patch("sys.stdout", new_callable=StringIO)
def test_file_results_verbose(mock_stdout: StringIO, sample_objects):
    *_, summary, _, _, cfg_verbose = sample_objects
    display_results(summary, {"display": cfg_verbose["display"].copy()}, False)
    txt = mock_stdout.getvalue()
    assert "Detailed Line Results:" in txt
    assert "Line 2 [TOXIC]" in txt
    assert "threshold: 0.50" in txt
    assert "[NON-TOXIC]" in txt 