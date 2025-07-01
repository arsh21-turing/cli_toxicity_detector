"""cache_manager.py

Utility helpers for persisting and validating category-embedding caches.
The cache is keyed by the *model name* and stores:
• embeddings              – Dict[ToxicityCategory, np.ndarray]
• metadata:
    • model_name          – str
    • taxonomy_hash       – str (hash of current ToxicityCategory set)
    • config_hash         – str (hash of per-category threshold map)
    • thresholds          – original threshold map
    • categories          – list[str] for easy inspection

A cache is considered valid only if both hashes match today's values.
When mismatched, the caller is expected to regenerate a fresh cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np  # heavy but only imported when cache used

from categories import ALL_CATEGORIES, ToxicityCategory
from logger import logger

# ---------------------------------------------------------------------------
# Constants -----------------------------------------------------------------
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR = Path.home() / ".toxicity_detector"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILENAME = "category_embeddings.pkl"


# ---------------------------------------------------------------------------
# Helper – hashing -----------------------------------------------------------
# ---------------------------------------------------------------------------

def _md5(obj: str | bytes) -> str:  # pragma: no cover – trivial
    if isinstance(obj, str):
        obj = obj.encode()
    return hashlib.md5(obj).hexdigest()


def taxonomy_hash() -> str:
    names = sorted(cat.name for cat in ALL_CATEGORIES)
    return _md5(json.dumps(names))


def thresholds_hash(thresholds: Dict[str, float]) -> str:
    ordered = json.dumps(sorted(thresholds.items()))
    return _md5(ordered)


# ---------------------------------------------------------------------------
# Path helper ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def cache_path(model_name: str, cache_dir: Optional[str | Path] = None) -> Path:
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    model_hash = _md5(model_name)[:8]
    return cache_dir / f"{model_hash}_{CACHE_FILENAME}"


# ---------------------------------------------------------------------------
# Public API -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def save_embeddings(
    embeddings: Dict[ToxicityCategory, np.ndarray],
    *,
    model_name: str,
    thresholds: Dict[str, float],
    cache_dir: Optional[str | Path] = None,
) -> None:
    """Persist *embeddings* together with metadata."""

    path = cache_path(model_name, cache_dir)
    data = {
        "embeddings": embeddings,
        "metadata": {
            "model_name": model_name,
            "taxonomy_hash": taxonomy_hash(),
            "config_hash": thresholds_hash(thresholds),
            "thresholds": thresholds,
        },
    }
    try:
        with open(path, "wb") as fh:
            pickle.dump(data, fh)
    except Exception as exc:  # pragma: no cover – file system issues
        logger.warning("Failed to write embedding cache %s: %s", path, exc)


def load_embeddings(
    *,
    model_name: str,
    thresholds: Dict[str, float],
    cache_dir: Optional[str | Path] = None,
) -> Optional[Dict[ToxicityCategory, np.ndarray]]:
    """Return cached embeddings if metadata matches; else *None*."""

    path = cache_path(model_name, cache_dir)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        meta = data.get("metadata", {})
        if meta.get("taxonomy_hash") != taxonomy_hash():
            logger.info("Taxonomy changed – rebuilding embedding cache for %s", model_name)
            path.unlink(missing_ok=True)
            return None
        if meta.get("config_hash") != thresholds_hash(thresholds):
            logger.info("Category thresholds changed – rebuilding embedding cache for %s", model_name)
            path.unlink(missing_ok=True)
            return None
        return data.get("embeddings")
    except Exception as exc:  # pragma: no cover – corruption case
        logger.debug("Failed to read embedding cache %s: %s", path, exc)
        path.unlink(missing_ok=True)
        return None


def clear_cache(model_name: str | None = None, *, cache_dir: Optional[str | Path] = None) -> None:
    """Delete cache file(s). If *model_name* None clear all."""

    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    if model_name is None:
        for p in cache_dir.glob(f"*_{CACHE_FILENAME}"):
            p.unlink(missing_ok=True)
    else:
        path = cache_path(model_name, cache_dir)
        path.unlink(missing_ok=True) 