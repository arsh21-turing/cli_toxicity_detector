#!/usr/bin/env python3
"""Central logging utilities for Smart CLI Toxicity Detector.

The module keeps backwards-compatibility with the previous simple `logger` object
while exposing richer helpers:

* ``setup_logger(config: dict, json_path: str | None = None)`` – configure root
  logging according to *config* dict and optional JSON-log file.
* ``get_logger(name)`` – retrieve a named logger configured with the global
  handlers.
* ``set_log_level(level)`` – change verbosity at runtime (string or int).
* ``get_json_log_path()`` – return JSON-log destination if enabled.

Console output remains colourised and goes to *stderr*.  When *json_path* is
supplied every record is also emitted as newline-delimited JSON to that file
for machine consumption.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Union

# ---------------------------------------------------------------------------
# Colour helpers -------------------------------------------------------------
# ---------------------------------------------------------------------------

_COLOUR_MAP = {
    logging.DEBUG: "\033[36m",    # cyan
    logging.INFO: "\033[32m",     # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",    # red
    logging.CRITICAL: "\033[35m", # magenta
}
_RESET = "\033[0m"


class _ColourFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        base = super().format(record)
        colour = _COLOUR_MAP.get(record.levelno)
        return f"{colour}{base}{_RESET}" if colour else base


class _JsonFormatter(logging.Formatter):
    """Line-delimited JSON for each *record*."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        obj: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            obj["exception"] = self.formatException(record.exc_info)
        # merge extra fields added via ``logger.info("msg", extra={...})``
        for k, v in record.__dict__.items():
            if k.startswith("_") or k in obj:
                continue
            try:
                json.dumps({k: v})  # type: ignore[arg-type] – serialisable?
                obj[k] = v
            except TypeError:
                obj[k] = str(v)
        return json.dumps(obj, ensure_ascii=False)


class _TeeHandler(logging.Handler):
    """Writes *record* to a text IO object (JSON file) in addition to others."""

    def __init__(self, path: str) -> None:  # noqa: D401
        super().__init__()
        self._path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
        atexit.register(self.close)
        self.setFormatter(_JsonFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._fh.write(msg + "\n")
            self._fh.flush()
        except Exception:  # pragma: no cover – never crash app due to logging
            self.handleError(record)

    def close(self) -> None:  # noqa: D401
        if not self._fh.closed:
            self._fh.close()
        super().close()


# ---------------------------------------------------------------------------
# Public helpers -------------------------------------------------------------
# ---------------------------------------------------------------------------

_root_configured = False
_json_handler: Optional[_TeeHandler] = None
_json_path: Optional[str] = None

_DEFAULT_FMT = "%Y-%m-%d %H:%M:%S"
_CONSOLE_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def setup_logger(cfg: Dict[str, Any] | None = None, json_log_path: str | None = None) -> None:
    """Initialise root logger once.

    *cfg* may include keys ``level`` (e.g. "DEBUG") and ``format``.
    """
    global _root_configured, _json_handler, _json_path

    if _root_configured:
        # update JSON handler if requested after initial call
        if json_log_path and not _json_handler:
            _json_handler = _TeeHandler(json_log_path)
            logging.getLogger().addHandler(_json_handler)
            _json_path = json_log_path
        return

    cfg = cfg or {}
    level_name = str(cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    log_fmt = cfg.get("format", _CONSOLE_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler (colour)
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(_ColourFormatter(log_fmt, datefmt=_DEFAULT_FMT))
    ch.setLevel(level)
    root.addHandler(ch)

    # Optional JSON tee
    if json_log_path:
        _json_handler = _TeeHandler(json_log_path)
        _json_handler.setLevel(level)
        root.addHandler(_json_handler)
        _json_path = json_log_path

    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger."""
    if not _root_configured:
        setup_logger()
    return logging.getLogger(name)


def set_log_level(level: Union[str, int]) -> None:
    """Adjust root & handler levels at runtime."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), None)  # type: ignore[assignment]
        if level is None:
            raise ValueError(f"Unknown log level: {level}")
    root = logging.getLogger()
    root.setLevel(level)  # type: ignore[arg-type]
    for h in root.handlers:
        h.setLevel(level)  # type: ignore[arg-type]


def get_json_log_path() -> Optional[str]:
    return _json_path


# ---------------------------------------------------------------------------
# Backwards-compat global ----------------------------------------------------
# ---------------------------------------------------------------------------

logger = get_logger("toxicity_detector")

# Public alias names for tests/backwards-compat --------------------------------
JSONLogFormatter = _JsonFormatter  # type: ignore
TeeHandler = _TeeHandler  # type: ignore