#!/usr/bin/env python3
"""tests/test_config_loader.py

Minimal tests for *config_loader.load_config*:
1. Highest-priority path in CONFIG_PATHS wins.
2. Missing keys are inherited from the default configuration.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict

import pytest

# Make sure project root is importable when tests run via 'pytest tests/'
ROOT_DIR = Path(__file__).resolve().parent.parent
if ROOT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, ROOT_DIR.as_posix())

import config_loader  # noqa: E402  (after path tweaking)

DEFAULTS: Dict = config_loader._DEFAULT_CONFIG  # type: ignore (private constant)


@pytest.fixture()
def tmp_config(monkeypatch, tmp_path):
    """Create two temporary config files and patch CONFIG_PATHS accordingly.

    Returns a tuple (hi_path, low_path).
    """
    hi_cfg = tmp_path / "high.json"
    low_cfg = tmp_path / "low.json"

    # High-priority file overrides just the model name.
    hi_cfg.write_text(json.dumps({"model": {"name": "high-priority-model"}}))

    # Low-priority file would be chosen if high didn't exist.
    low_cfg.write_text(json.dumps({"model": {"name": "low-priority-model"}}))

    # Patch search order so our paths are considered first.
    monkeypatch.setattr(
        config_loader,
        "CONFIG_PATHS",
        [hi_cfg, low_cfg],
        raising=True,
    )

    return hi_cfg, low_cfg


def test_load_config_priority(tmp_config):
    """The first existing file in CONFIG_PATHS must be used."""
    cfg = config_loader.load_config()
    assert cfg["model"]["name"] == "high-priority-model"


def test_load_config_merges_missing_keys(monkeypatch, tmp_path):
    """User config with partial data should inherit defaults for missing keys."""
    partial_cfg_path = tmp_path / "partial.json"

    # User provides only model.name override, nothing else.
    partial_cfg_path.write_text(json.dumps({"model": {"name": "partial-model"}}))

    # Patch path list to just this file.
    monkeypatch.setattr(config_loader, "CONFIG_PATHS", [partial_cfg_path], raising=True)

    cfg = config_loader.load_config()

    # Override applied
    assert cfg["model"]["name"] == "partial-model"

    # Key absent in user file should fall back to defaults
    assert cfg["model"]["threshold"] == DEFAULTS["model"]["threshold"]
    assert cfg["categories"] == DEFAULTS["categories"] 