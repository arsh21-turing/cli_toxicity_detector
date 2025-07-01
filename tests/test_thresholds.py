import json
from unittest.mock import MagicMock, patch

import argparse
import sys
from pathlib import Path
from typing import Dict

import pytest

# Ensure project root on path so 'import main' works when running from tests dir
ROOT_DIR = Path(__file__).resolve().parent.parent
if ROOT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, ROOT_DIR.as_posix())

import main  # noqa: E402  (after path manipulation)
import config_loader  # noqa: E402
from categories import ToxicityCategory


# ---------------------------------------------------------------------------
# Helper to build Namespace objects easily
# ---------------------------------------------------------------------------

def _ns(**kwargs):
    return argparse.Namespace(**kwargs)


# ---------------------------------------------------------------------------
# create_threshold_argument ---------------------------------------------------
# ---------------------------------------------------------------------------

def test_create_threshold_args_exist():
    parser = argparse.ArgumentParser()
    main.create_threshold_argument(parser)

    arg_dests = {a.dest for a in parser._actions}

    # global flag
    assert "threshold" in arg_dests

    # per-category flags
    for cat in ToxicityCategory:
        dest = f"threshold_{cat.name.lower().replace('_', '_')}"
        assert dest in arg_dests, f"Missing CLI flag for {cat}"


# ---------------------------------------------------------------------------
# parse_threshold_args --------------------------------------------------------
# ---------------------------------------------------------------------------

def test_parse_global_threshold():
    ns = _ns(threshold=0.75, **{f"threshold_{cat.name.lower().replace('_', '_')}": None for cat in ToxicityCategory})
    mapping = main.parse_threshold_args(ns)
    assert all(v == 0.75 for v in mapping.values()) and len(mapping) == len(ToxicityCategory)


def test_parse_individual_thresholds():
    kwargs = {f"threshold_{cat.name.lower().replace('_', '_')}": (i + 1) / 10 for i, cat in enumerate(ToxicityCategory)}
    ns = _ns(threshold=None, **kwargs)
    mapping = main.parse_threshold_args(ns)
    for cat in ToxicityCategory:
        expected = kwargs[f"threshold_{cat.name.lower().replace('_', '_')}"]
        assert mapping[cat.name] == expected


def test_global_with_overrides():
    # Global default .5 but override HATE and SEXUAL
    base = {f"threshold_{cat.name.lower().replace('_', '_')}": None for cat in ToxicityCategory}
    base.update({"threshold_hate": 0.8, "threshold_sexual": 0.3})
    ns = _ns(threshold=0.5, **base)
    mapping = main.parse_threshold_args(ns)
    assert mapping["HATE"] == 0.8
    assert mapping["SEXUAL"] == 0.3
    assert mapping["INSULT"] == 0.5  # unchanged


# ---------------------------------------------------------------------------
# Config loading – thresholds key --------------------------------------------
# ---------------------------------------------------------------------------

def test_load_config_thresholds(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "cfg.json"
    custom = {"thresholds": {"INSULT": 0.9, "NON_TOXIC": 0.1}}
    cfg_path.write_text(json.dumps(custom))

    monkeypatch.setattr(config_loader, "CONFIG_PATHS", [cfg_path], raising=True)

    cfg = config_loader.load_config()

    # custom values present
    assert cfg["thresholds"]["INSULT"] == 0.9
    assert cfg["thresholds"]["NON_TOXIC"] == 0.1

    # every category present
    for cat in ToxicityCategory:
        assert cat.name in cfg["thresholds"], f"Missing threshold for {cat.name}" 