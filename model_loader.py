#!/usr/bin/env python3
"""
model_loader.py

Load a multilingual SentenceTransformer model, build toxicity category
embeddings, and expose an analyse function. Integrates with config_loader so
users can adjust model/thresholds without CLI flags.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

# Third-party imports are optional; fall back if unavailable.
try:
    import numpy as np  # type: ignore

    _NUMPY_OK = True
except ImportError:  # pragma: no cover
    _NUMPY_OK = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    _ST_OK = True
except ImportError:  # pragma: no cover
    _ST_OK = False

from config_loader import load_config

CONFIG = load_config()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES: List[str] = [
    "insult",
    "hate",
    "obscene",
    "threat",
    "sexual",
    "self-harm",
]

_DEFAULT_MODEL = CONFIG["model"]["name"]
_DEFAULT_THRESHOLD = CONFIG["model"].get("threshold", 0.6)
_CACHE_DIR = Path(CONFIG["model"].get("cache_dir", Path.home() / ".toxicity_detector"))
_CACHE_FILE = _CACHE_DIR / "category_embeddings.pkl"


class ModelLoader:
    """Encapsulates model + category embeddings."""

    def __init__(self, model_name: str | None = None, threshold: float | None = None):
        self.model_name = model_name or _DEFAULT_MODEL
        self.threshold = threshold if threshold is not None else _DEFAULT_THRESHOLD
        self.category_thresholds: Dict[str, float] = CONFIG.get("categories", {})

        self.model: "SentenceTransformer | None" = None
        self.embeddings: Dict[str, "np.ndarray"] | None = None

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> bool:
        if not (_ST_OK and _NUMPY_OK):
            return False
        try:
            self.model = SentenceTransformer(self.model_name)
            return True
        except Exception as exc:  # pragma: no cover
            print(f"Error loading model '{self.model_name}': {exc}")
            return False

    def _load_embeddings(self) -> None:
        if _CACHE_FILE.exists():
            try:
                with _CACHE_FILE.open("rb") as fh:
                    cache = pickle.load(fh)
                if cache.get("model_name") == self.model_name:
                    self.embeddings = cache["embeddings"]
                    return
            except Exception:
                pass  # fall through to regenerate
        # need to create
        self._build_embeddings()
        self._cache_embeddings()

    def _build_embeddings(self) -> None:
        assert self.model is not None and _NUMPY_OK
        import numpy as np

        prompt_map: Dict[str, List[str]] = {
            "insult": [
                "insulting language",
                "personal attack",
                "derogatory remarks",
            ],
            "hate": [
                "hate speech",
                "racist comment",
                "xenophobic slur",
            ],
            "obscene": [
                "obscene language",
                "explicit profanity",
                "vulgar words",
            ],
            "threat": [
                "violent threat",
                "intimidation statement",
                "death threat",
            ],
            "sexual": [
                "explicit sexual content",
                "sexual harassment",
                "lewd remarks",
            ],
            "self-harm": [
                "suicidal thought",
                "self-harm encouragement",
                "self-injury content",
            ],
        }
        self.embeddings = {
            cat: np.mean(self.model.encode(prompts), axis=0)
            for cat, prompts in prompt_map.items()
        }

    def _cache_embeddings(self) -> None:
        try:
            with _CACHE_FILE.open("wb") as fh:
                pickle.dump({"model_name": self.model_name, "embeddings": self.embeddings}, fh)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ensure(self) -> bool:
        if self.model is not None and self.embeddings is not None:
            return True
        if not self._load_model():
            return False
        self._load_embeddings()
        return self.embeddings is not None

    def analyse(self, text: str) -> Dict[str, Any]:
        if not (_ST_OK and _NUMPY_OK) or not self.ensure():
            return _placeholder_result(text)
        import numpy as np

        assert self.model is not None and self.embeddings is not None
        vec = self.model.encode(text)
        probs: Dict[str, float] = {}
        toxic: List[tuple[str, float]] = []

        for cat, emb in self.embeddings.items():
            sim = np.dot(vec, emb) / (np.linalg.norm(vec) * np.linalg.norm(emb))
            prob = (sim + 1) / 2
            probs[cat] = prob
            thresh = self.category_thresholds.get(cat, self.threshold)
            if prob > thresh:
                toxic.append((cat, prob))

        if not toxic:
            return {
                "is_toxic": False,
                "category": None,
                "confidence": max(probs.values()),
                "probabilities": probs,
            }

        toxic.sort(key=lambda x: x[1], reverse=True)
        top_cat, conf = toxic[0]
        return {
            "is_toxic": True,
            "category": top_cat,
            "confidence": conf,
            "probabilities": probs,
            "toxic_categories": [c for c, _ in toxic],
        }


# -------------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------------

def get_analyzer(threshold: float | None = None, model_name: str | None = None) -> Callable[[str], Dict[str, Any]]:
    loader = ModelLoader(model_name=model_name, threshold=threshold)
    if loader.ensure():
        return loader.analyse
    return _placeholder_result


def _placeholder_result(text: str) -> Dict[str, Any]:  # type: ignore[override]
    return {
        "is_toxic": False,
        "category": None,
        "confidence": 0.0,
        "probabilities": {cat: 0.0 for cat in CATEGORIES},
    } 