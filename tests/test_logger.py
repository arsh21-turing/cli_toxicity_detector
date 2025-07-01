#!/usr/bin/env python3
"""Unit-tests for the enhanced *logger.py* module.

Only lightweight checks are performed so the suite runs quickly and without
side-effects on CI systems (no need for real colour output or large files).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict

import pytest

# Ensure project root importable when tests executed from /tests
ROOT_DIR = Path(__file__).resolve().parent.parent
import sys
if ROOT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, ROOT_DIR.as_posix())

from logger import (  # noqa: E402  – inserted after path tweak
    setup_logger,
    get_logger,
    set_log_level,
    JSONLogFormatter,
    TeeHandler,
)


@pytest.fixture()
def temp_dir(tmp_path: Path):
    return tmp_path


# ---------------------------------------------------------------------------
# Core configuration ---------------------------------------------------------
# ---------------------------------------------------------------------------

def _reset_root() -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    # Also reset module-level cached configuration so subsequent tests re-initialise
    import importlib
    import logger as _logger_mod  # reimport alias
    _logger_mod._root_configured = False  # type: ignore[attr-defined]
    _logger_mod._json_handler = None  # type: ignore[attr-defined]


def test_basic_console_logging(temp_dir: Path):
    _reset_root()

    cfg: Dict[str, str] = {"level": "INFO"}
    setup_logger(cfg)

    log = get_logger("unit")
    assert log.getEffectiveLevel() == logging.INFO

    # There should be at least one handler (console)
    assert logging.getLogger().handlers, "root logger has no handlers"


# ---------------------------------------------------------------------------
# JSON tee -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def test_json_tee_file_written(temp_dir: Path):
    _reset_root()
    json_path = temp_dir / "logs.json"
    setup_logger({"level": "INFO"}, str(json_path))
    log = get_logger("jsontest")
    log.info("hello json")

    assert json_path.exists(), "JSON log file not created"
    lines = json_path.read_text().strip().splitlines()
    assert lines, "JSON log file empty"
    first = json.loads(lines[0])
    assert first["message"] == "hello json"
    assert first["level"] == "INFO"


# ---------------------------------------------------------------------------
# Dynamic log-level change ----------------------------------------------------
# ---------------------------------------------------------------------------

def test_set_log_level_runtime(capsys):
    _reset_root()
    setup_logger({"level": "INFO"})
    log = get_logger("dyn")

    log.debug("hidden")
    captured = capsys.readouterr()
    assert "hidden" not in captured.err

    set_log_level("DEBUG")
    log.debug("visible")
    captured = capsys.readouterr()
    assert "visible" in captured.err


# ---------------------------------------------------------------------------
# Formatter sanity -----------------------------------------------------------
# ---------------------------------------------------------------------------

def test_json_formatter_direct():
    fmt = JSONLogFormatter()
    rec = logging.LogRecord(
        name="fmt", level=logging.WARNING, pathname="x", lineno=5, msg="warn", args=(), exc_info=None
    )
    data = json.loads(fmt.format(rec))
    assert data["level"] == "WARNING"
    assert data["message"] == "warn"


# ---------------------------------------------------------------------------
# TeeHandler minimal ---------------------------------------------------------
# ---------------------------------------------------------------------------

def test_tee_handler_emits_json(temp_dir: Path):
    path = temp_dir / "tee.json"
    handler = TeeHandler(str(path))
    rec = logging.LogRecord(
        name="tee", level=logging.INFO, pathname="x", lineno=1, msg="hi", args=(), exc_info=None
    )
    handler.emit(rec)
    handler.close()
    assert path.exists()
    line = path.read_text().strip()
    obj = json.loads(line)
    assert obj["message"] == "hi" 