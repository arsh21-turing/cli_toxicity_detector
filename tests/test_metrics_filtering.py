#!/usr/bin/env python3
"""Integration test for the `--metrics` flag.

It creates a temporary .txt file, invokes the CLI with a restricted metrics
list, captures JSON output and checks that only the requested metrics appear
and that their values are numeric.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Build command so it works no matter where tests are invoked from.
ROOT_DIR = Path(__file__).resolve().parents[1]
CLI_CMD = [sys.executable, str(ROOT_DIR / "main.py")]
REQUESTED = {"accuracy", "f1_micro"}


def _run_cli(file_path: str) -> dict:
    cmd = [
        *CLI_CMD,
        "--file",
        file_path,
        "--metrics",
        ",".join(sorted(REQUESTED)),
        "--json",
        "--quiet",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)


def test_metrics_filtering(tmp_path: Path) -> None:
    # create tiny non-toxic file
    test_file = tmp_path / "sample.txt"
    test_file.write_text("This is a harmless line.\n")

    data = _run_cli(str(test_file))

    assert "metrics" in data, "Output JSON missing 'metrics' key"
    metrics = data["metrics"]
    assert set(metrics.keys()) == REQUESTED, (
        "CLI did not restrict metrics set as requested. "
        f"expected={REQUESTED} got={set(metrics.keys())}"
    )
    for name, val in metrics.items():
        assert isinstance(val, (int, float)), f"Metric {name} is not numeric: {val}" 