from __future__ import annotations

"""Disk-backed cache for Groq API responses.

Each text is hashed (SHA-256) to produce a stable filename; the payload stored
on disk is a small JSON object with the original API response plus some
metadata (timestamp, preview).  The helper intentionally avoids any heavy
third-party dependencies.
"""

import json
import os
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".toxicity_detector" / "groq_cache"


class GroqCache:
    """Filesystem cache keyed by SHA-256 of the raw text."""

    def __init__(self, cache_dir: str | os.PathLike | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Groq cache initialised at %s", self.cache_dir)

    # ------------------------------------------------------------------
    # Private helpers ---------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    # ------------------------------------------------------------------
    # Public API --------------------------------------------------------
    # ------------------------------------------------------------------
    def get(self, text: str) -> Optional[Dict[str, Any]]:
        fp = self._path(self._key(text))
        if not fp.exists():
            return None
        try:
            with fp.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:  # pragma: no cover – corrupt file
            logger.warning("Failed to read cache entry %s: %s", fp, exc)
            return None

    def set(self, text: str, response: Dict[str, Any]) -> None:
        key = self._key(text)
        fp = self._path(key)
        entry = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "text_preview": text[:100],
            "response": response,
        }
        try:
            with fp.open("w", encoding="utf-8") as fh:
                json.dump(entry, fh, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Unable to write cache entry %s: %s", fp, exc)

    # maintenance -------------------------------------------------------
    def clear(self) -> int:
        removed = 0
        for fp in self.cache_dir.glob("*.json"):
            try:
                fp.unlink()
                removed += 1
            except Exception as exc:
                logger.warning("Unable to delete cache file %s: %s", fp, exc)
        return removed

    # stats -------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        files = list(self.cache_dir.glob("*.json"))
        size_bytes = sum(f.stat().st_size for f in files)
        timestamps = []
        for fp in files:
            try:
                with fp.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                ts_raw = data.get("timestamp")
                if ts_raw:
                    # datetime.fromisoformat will raise if malformed; guard it
                    try:
                        ts = datetime.fromisoformat(ts_raw)
                        timestamps.append(ts)
                    except ValueError:
                        logger.debug("Invalid timestamp in cache file %s: %s", fp, ts_raw)
            except Exception as exc:
                logger.debug("Unable to read timestamp from %s: %s", fp, exc)
                continue

        oldest = min(timestamps).isoformat() if timestamps else None
        newest = max(timestamps).isoformat() if timestamps else None
        size_mb = round(size_bytes / (1024 * 1024), 2)

        # Preserve previous keys (entries, size_bytes, dir) and extend
        return {
            "entries": len(files),
            "size_bytes": size_bytes,
            "size_mb": size_mb,
            "oldest": oldest,
            "newest": newest,
            "dir": str(self.cache_dir),
        }

    # Backwards-compat helper used in tests
    def get_cache_size(self) -> int:  # pragma: no cover
        return len(list(self.cache_dir.glob("*.json"))) 