#!/usr/bin/env python3
"""tests/test_file_processor.py
Pytest suite for the *analyze_file* helper in ``file_processor.py``.

The tests verify that:
1. Blank/whitespace-only lines are skipped and ``total_lines`` reflects only real text lines.
2. The ``analyzer_func`` is invoked exactly once per non-blank line.
3. Empty files or files containing only blank lines are handled gracefully.
4. Toxic lines are counted, categories aggregated and percentages calculated.
5. Optional logging and progress-indicator side-effects behave as expected (without polluting test output).
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import List

import pytest
from unittest.mock import MagicMock

from pathlib import Path

# Ensure project root is on sys.path so `import file_processor` works when
# running the suite from the *tests* directory or via tools like tox.
ROOT_DIR = Path(__file__).resolve().parent.parent
if ROOT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, ROOT_DIR.as_posix())

import file_processor  # import after stdlib/pytest so we can monkey-patch below


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_logger(monkeypatch):
    """Replace *file_processor.logger* with a ``MagicMock`` so we can assert calls."""
    mock = MagicMock()
    monkeypatch.setattr(file_processor, "logger", mock)
    return mock


# Helper to build a temporary file with given *lines* and return its path

def _tmp_file(lines: List[str]) -> str:
    fh = tempfile.NamedTemporaryFile(mode="w+", delete=False)
    fh.write("\n".join(lines))
    fh.write("\n")  # final newline ensures last line processed
    fh.flush()
    fh.close()
    return fh.name


# ---------------------------------------------------------------------------
# Core behaviour – counting & skipping blank lines
# ---------------------------------------------------------------------------

def test_analyze_file_skips_blank_lines(mock_logger):
    """Non-blank lines should be processed; blank lines ignored."""
    path = _tmp_file([
        "Line 1",
        "",            # blank
        "Line 2",
        "   ",         # whitespace only
        "\t",         # tab only
        "Line 3",
        "", "",        # multiple blanks
        "Line 4",
    ])

    try:
        mock_analyzer = MagicMock(return_value={"is_toxic": False})
        data = file_processor.analyze_file(path, mock_analyzer, show_progress=False)

        assert data["total_lines"] == 4
        assert mock_analyzer.call_count == 4
        assert [r["text"] for r in data["results"]] == [
            "Line 1",
            "Line 2",
            "Line 3",
            "Line 4",
        ]
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Edge-case files
# ---------------------------------------------------------------------------

def test_analyze_file_empty_file(mock_logger):
    path = _tmp_file([])  # no lines
    try:
        mock_analyzer = MagicMock()
        data = file_processor.analyze_file(path, mock_analyzer, show_progress=False)
        assert data["total_lines"] == 0
        mock_analyzer.assert_not_called()
    finally:
        os.unlink(path)


def test_analyze_file_only_blank_lines(mock_logger):
    path = _tmp_file(["", "   ", "\t", "\n"])
    try:
        mock_analyzer = MagicMock()
        data = file_processor.analyze_file(path, mock_analyzer, show_progress=False)
        assert data["total_lines"] == 0
        mock_analyzer.assert_not_called()
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Toxic content counting & category aggregation
# ---------------------------------------------------------------------------

def test_analyze_file_toxic_content(mock_logger):
    path = _tmp_file([
        "This is fine.",
        "This is toxic.",
        "Still fine.",
        "Very toxic indeed.",
    ])

    def mock_analyzer(txt: str):
        return {
            "is_toxic": "toxic" in txt.lower(),
            "category": "test_category" if "toxic" in txt.lower() else None,
        }

    try:
        data = file_processor.analyze_file(path, mock_analyzer, show_progress=False)
        assert data["total_lines"] == 4
        assert data["toxic_count"] == 2
        assert data["categories"] == {"test_category": 2}
        assert data["toxic_percentage"] == 50.0
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Logger side-effects (lightweight checks)
# ---------------------------------------------------------------------------

def test_analyze_file_logging_messages(mock_logger):
    path = _tmp_file(["a", "b"])
    try:
        mock_analyzer = MagicMock(return_value={"is_toxic": False})
        file_processor.analyze_file(path, mock_analyzer, show_progress=False)

        # Expect at least the two main log lines
        calls = [args[0] for args, _ in mock_logger.info.call_args_list]
        assert any(msg.startswith("Analyzing file:") for msg in calls)
        assert any(msg.startswith("Completed file:") for msg in calls)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Progress indicator (uses monkey-patch to silence stdout)
# ---------------------------------------------------------------------------

def test_analyze_file_progress_indicator(mock_logger, monkeypatch):
    path = _tmp_file([f"Line {i}" for i in range(10)])

    # Intercept stdout writes so we can count them without polluting test output
    write_calls = []

    def fake_write(text):
        write_calls.append(text)
        return len(text)

    monkeypatch.setattr(sys.stdout, "write", fake_write)
    monkeypatch.setattr(sys.stdout, "flush", lambda: None)

    try:
        mock_analyzer = MagicMock(return_value={"is_toxic": False})

        # No progress expected when flag is False
        file_processor.analyze_file(path, mock_analyzer, show_progress=False)
        assert len(write_calls) == 0

        # Progress messages expected when flag is True
        file_processor.analyze_file(path, mock_analyzer, show_progress=True)
        assert len(write_calls) > 0
    finally:
        os.unlink(path) 