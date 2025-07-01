#!/usr/bin/env python3
"""metrics_loader.py – lightweight cache for HuggingFace *evaluate* metrics.

Rationale: `evaluate.load(<metric>)` triggers an HTTP call / local disk lookup the
first time a process needs a metric.  When several metrics are requested (e.g.
precision/recall/f1) the repeated downloads slow startup.  This module exposes
`get_metric(name)` that caches the loaded object so subsequent calls are
instant.
"""
from __future__ import annotations

from typing import Any, Dict

import importlib
import logging

logger = logging.getLogger(__name__)

# Global in-memory cache – simple and effective for single-process CLI usage.
_METRIC_CACHE: Dict[str, Any] = {}


def _ensure_evaluate():
    if importlib.util.find_spec("evaluate") is None:
        raise ImportError(
            "The 'evaluate' library is required but not installed. Add it to your"
            " requirements with `pip install evaluate`."
        )
    import evaluate  # type: ignore – imported for its side effect
    return evaluate


def get_metric(name: str):  # noqa: D401 – helper, not a method
    """Return a cached HuggingFace metric instance.

    Parameters
    ----------
    name
        Metric identifier understood by `evaluate.load`.
    """
    if name in _METRIC_CACHE:
        return _METRIC_CACHE[name]
    evaluate = _ensure_evaluate()
    logger.info("Loading metric '%s' via evaluate.load", name)
    metric = evaluate.load(name)
    _METRIC_CACHE[name] = metric
    return metric


def clear_metric(name: str | None = None) -> None:  # noqa: D401
    """Clear *name* from the cache, or all metrics if *name* is None."""
    if name is None:
        _METRIC_CACHE.clear()
    else:
        _METRIC_CACHE.pop(name, None) 