#!/usr/bin/env python3
"""config_loader.py

Utility to locate, load, and validate configuration for Smart CLI Toxicity
Detector. Supports YAML (preferred) and JSON. Provides defaults and helper to
create a template configuration file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Added import ----------------------------------------------------------------
from categories import ToxicityCategory  # single-source of category names

# ---------------------------------------------------------------------------
# Defaults – extended with per-category *thresholds* -------------------------
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS: Dict[str, float] = {
    cat.name: 0.6 for cat in ToxicityCategory
}

# Note: keep legacy "categories" key untouched for backward-compatibility with
# existing unit-tests while introducing the new explicit "thresholds" section.

_DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "name": "unitary/toxic-bert",
        "threshold": 0.6,  # legacy global threshold (maintained for tests)
        "cache_dir": str(Path.home() / ".toxicity_detector"),
    },
    # Legacy structure used by tests -------------------------------------------------
    "categories": {
        "insult": 0.6,
        "hate": 0.6,
        "obscene": 0.6,
        "threat": 0.6,
        "sexual": 0.6,
        "self-harm": 0.6,
    },
    # NEW structure – authoritative per-category thresholds -------------------------
    "thresholds": _DEFAULT_THRESHOLDS.copy(),
    "output": {
        "show_probabilities": True,
        "color_output": True,
    },
    # Groq-related defaults -------------------------------------------------
    "groq": {
        "fallback_enabled": False,
        "lower_bound": 0.4,
        "upper_bound": 0.6,
        "tie_policy": "prefer-groq",
    },
}

# Candidate config paths (ordered by precedence)
CONFIG_PATHS = [
    Path("toxicity_detector.yaml"),
    Path("toxicity_detector.yml"),
    Path("toxicity_detector.json"),
    Path.home() / ".toxicity_detector" / "config.yaml",
    Path.home() / ".toxicity_detector" / "config.yml",
    Path.home() / ".toxicity_detector" / "config.json",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        print("Warning: PyYAML not installed. Falling back to JSON only.")
        return {}

    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception as exc:
        print(f"Warning: failed to read YAML config {path}: {exc}")
        return {}


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"Warning: failed to read JSON config {path}: {exc}")
        return {}


def _deep_merge(dest: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge for dict-of-dicts (two levels)."""
    out = dest.copy()
    for key, val in src.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            nested = out[key].copy()
            nested.update(val)
            out[key] = nested
        else:
            out[key] = val
    return out


def load_config() -> Dict[str, Any]:
    """Search known locations and return merged configuration dict."""
    cfg = _DEFAULT_CONFIG
    for path in CONFIG_PATHS:
        if path.exists():
            user_cfg: Dict[str, Any]
            if path.suffix.lower() in {".yml", ".yaml"}:
                user_cfg = _read_yaml(path)
            else:
                user_cfg = _read_json(path)
            cfg = _deep_merge(cfg, user_cfg)

            # ---------------------------------------------------------------------
            # Ensure every category has a threshold set (fallback to defaults) -----
            # ---------------------------------------------------------------------
            for cat in ToxicityCategory:
                cfg.setdefault("thresholds", {})
                cfg["thresholds"].setdefault(cat.name, _DEFAULT_THRESHOLDS[cat.name])
            # ---------------------------------------------------------------------
            break  # first match wins
    return cfg


def create_default_config(path: str | Path) -> bool:
    """Write the default configuration to *path* in YAML if possible, else JSON."""
    path = Path(path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in {".yml", ".yaml"}:
            try:
                import yaml  # type: ignore
                path.write_text(yaml.dump(_DEFAULT_CONFIG, sort_keys=False))
            except ImportError:
                path.write_text(json.dumps(_DEFAULT_CONFIG, indent=2))
        else:
            path.write_text(json.dumps(_DEFAULT_CONFIG, indent=2))
        return True
    except Exception as exc:
        print(f"Error creating default config at {path}: {exc}")
        return False 