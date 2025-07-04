"""Pytest configuration to ensure project root is on sys.path.

This allows importing top-level modules (e.g. ``groq_cache``) when pytest's
working directory is the tests folder.
"""

import sys
from pathlib import Path

# Add the parent directory (project root) to sys.path so tests can import the
# library modules when pytest changes the working directory to the ``tests``
# folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT)) 