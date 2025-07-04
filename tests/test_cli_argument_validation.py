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


def test_help_includes_groq_flag(capsys):
    """--help text should mention the Groq fallback switch."""
    _parser().print_help()
    captured = capsys.readouterr().out
    assert "--allow-groq-fallback" in captured


@pytest.mark.parametrize(
    "argv",
    [
        ["--text", "hi", "--allow-groq-fallback"],
        ["--file", "sample.txt", "--allow-groq-fallback"],
        ["--create-config", "--allow-groq-fallback"],
    ],
)
def test_groq_flag_valid_with_single_required(argv):
    """Groq fallback should parse cleanly alongside any *single* primary flag."""
    # Should parse without raising
    ns = _parser().parse_args(argv)
    assert getattr(ns, "allow_groq_fallback") is True


def test_groq_flag_does_not_mask_exclusivity():
    """Providing Groq flag must not bypass mutually-exclusive checks."""
    p = _parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--text", "hi", "--file", "x.txt", "--allow-groq-fallback"])


def test_help_includes_groq_cache_stats(capsys):
    """--help text should mention the Groq cache stats switch."""
    _parser().print_help()
    captured = capsys.readouterr().out
    assert "--groq-cache-stats" in captured


def test_groq_cache_stats_exclusive():
    """--groq-cache-stats cannot be combined with analysis flags."""
    p = _parser()

    # Cannot combine with --text
    with pytest.raises(SystemExit):
        p.parse_args(["--groq-cache-stats", "--text", "hi"])

    # Cannot combine with --file
    with pytest.raises(SystemExit):
        p.parse_args(["--groq-cache-stats", "--file", "sample.txt"])

    # Cannot combine with --create-config
    with pytest.raises(SystemExit):
        p.parse_args(["--groq-cache-stats", "--create-config"])



def test_help_includes_groq_tie_policy(capsys):
    """--help should mention the Groq tie policy option and its choices."""
    _parser().print_help()
    captured = capsys.readouterr().out
    assert "--groq-tie-policy" in captured
    assert "prefer-groq" in captured  # choices list


import pytest  # ensure available for parametrised test above

@pytest.mark.parametrize("policy", ["prefer-groq", "prefer-local", "highest-confidence"])
def test_tie_policy_parsed(policy):
    """Parser should accept each valid tie policy choice."""
    ns = _parser().parse_args(["--text", "hi", "--groq-tie-policy", policy])
    assert ns.groq_tie_policy == policy


def test_tie_policy_invalid():
    """Invalid tie policy should raise SystemExit via argparse."""
    with pytest.raises(SystemExit):
        _parser().parse_args(["--text", "hi", "--groq-tie-policy", "not-a-policy"]) 