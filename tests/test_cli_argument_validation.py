#!/usr/bin/env python3
"""Argument-validation tests for the CLI.

These checks make sure exactly one of the mutually–exclusive options
(--text, --file, --create-config) must be supplied.
"""

from __future__ import annotations

import pytest
import os, sys
from pathlib import Path

# Ensure project root is on sys.path so that `import main` works when running pytest from the tests directory
ROOT_DIR = Path(__file__).resolve().parent.parent
if ROOT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, ROOT_DIR.as_posix())

import main  # project's entry module


def _parser():
    """Return a fresh parser instance from the CLI module."""
    return main._build_parser()


def test_no_required_flag_raises():
    """Calling the CLI with no mutually-exclusive flag should abort."""
    with pytest.raises(SystemExit):
        _parser().parse_args([])


def test_multiple_flags_raises():
    """Supplying more than one mutually-exclusive flag must also abort."""
    p = _parser()

    # --text and --file together -----------------------------------------
    with pytest.raises(SystemExit):
        p.parse_args(["--text", "hello", "--file", "sample.txt"])

    # --text and --create-config together --------------------------------
    with pytest.raises(SystemExit):
        p.parse_args(["--text", "hello", "--create-config"])

    # --file and --create-config together --------------------------------
    with pytest.raises(SystemExit):
        p.parse_args(["--file", "sample.txt", "--create-config"]) 