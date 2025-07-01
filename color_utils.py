#!/usr/bin/env python3
"""color_utils.py – small helper for ANSI colour output.

All colour handling for the CLI lives here so the rest of the codebase can
simply import *colorize_toxic* or *colorize_percentage*.
"""
from __future__ import annotations

import os
import sys
from typing import Dict

# Basic 8-bit ANSI colours (bright variants for better contrast)
COLORS: Dict[str, str] = {
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RESET": "\033[0m",
}


def supports_color() -> bool:  # pragma: no cover – platform-specific heuristic
    """Return *True* when the current stdout appears to support ANSI colours."""

    # Respect NO_COLOR https://no-color.org/
    if os.environ.get("NO_COLOR") is not None:
        return False

    # If output redirected (non-TTY) bail out early
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False

    # Minimal TERM sanity check – many modern terms set this
    term = os.environ.get("TERM", "")
    if term == "dumb":
        return False

    # Good enough for the vast majority of Linux/macOS/modern Windows terms
    return True


def _apply(code: str, text: str, enabled: bool) -> str:
    return f"{code}{text}{COLORS['RESET']}" if enabled else text


def colorize(text: str, color: str, enabled: bool = True) -> str:
    """Wrap *text* in ANSI *color* sequence when *enabled* is true."""
    return _apply(COLORS.get(color.upper(), ""), text, enabled and supports_color())


def colorize_toxic(is_toxic: bool, enabled: bool = True) -> str:
    """Return "TOXIC" in red or "NON-TOXIC" in green depending on *is_toxic*."""
    label = "TOXIC" if is_toxic else "NON-TOXIC"
    colour = "RED" if is_toxic else "GREEN"
    return colorize(label, colour, enabled)


def colorize_percentage(value: float, threshold: float = 50.0, *, enabled: bool = True) -> str:
    """Colour the *value* percentage based on *threshold*.

    Percentages >= *threshold* are rendered in red, otherwise green.
    """
    pct_str = f"{value:.2f}%"
    colour = "RED" if value >= threshold else "GREEN"
    return colorize(pct_str, colour, enabled) 