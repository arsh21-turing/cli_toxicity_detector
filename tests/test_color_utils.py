#!/usr/bin/env python3
"""Tests for colour helper utilities and CLI --no-color flag."""

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

from color_utils import colorize_toxic, colorize_percentage, supports_color  # noqa: E402
from categories import ToxicityCategory  # noqa: E402
from main import display_single_text_result  # noqa: E402


@pytest.mark.parametrize("is_toxic, expected_colour", [(True, "\033[91m"), (False, "\033[92m")])
def test_colorize_toxic_basic(is_toxic, expected_colour):
    with patch("color_utils.supports_color", return_value=True):
        out = colorize_toxic(is_toxic, enabled=True)
        assert expected_colour in out


def test_colorize_percentage_threshold():
    with patch("color_utils.supports_color", return_value=True):
        red = colorize_percentage(80.0, threshold=50.0, enabled=True)
        green = colorize_percentage(10.0, threshold=50.0, enabled=True)
        assert "\033[91m" in red and "\033[92m" in green


@patch("sys.stdout", new_callable=StringIO)
def test_display_single_text_respects_color_flag(mock_stdout: StringIO):
    sample = {
        "text": "dummy",
        "is_toxic": True,
        "most_probable_category": ToxicityCategory.INSULT,
        "category_results": {ToxicityCategory.INSULT: {"score": 0.9, "above_threshold": True, "threshold": 0.5}},
    }

    cfg_on = {"display": {"color_output": True}}
    with patch("color_utils.supports_color", return_value=True):
        display_single_text_result(sample, cfg_on)
    assert "\033[91m" in mock_stdout.getvalue()  # red present

    mock_stdout.truncate(0)
    mock_stdout.seek(0)

    cfg_off = {"display": {"color_output": False}}
    with patch("color_utils.supports_color", return_value=True):
        display_single_text_result(sample, cfg_off)
    assert "\033[91m" not in mock_stdout.getvalue() 