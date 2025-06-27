#!/usr/bin/env python3
"""Central logger for Smart CLI Toxicity Detector.

Provides a configured ``logger`` instance writing to stderr at INFO level by
default. Import this module to use the shared logger across the codebase.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def _setup(level: int = logging.INFO, fmt: str = _FORMAT) -> logging.Logger:  # pragma: no cover
    logger = logging.getLogger("toxicity_detector")
    if logger.handlers:
        return logger  # already configured

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = _setup() 