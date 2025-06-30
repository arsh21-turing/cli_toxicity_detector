#!/usr/bin/env python3
"""
Model loader for toxicity detection.

This module handles downloading, caching, and using a pretrained 
toxicity classification model to predict probabilities for 
multiple toxicity categories.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Union, Any

# Optional heavy deps ---------------------------------------------------------
try:
    import torch  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover – allow running tests without heavy deps
    torch = None  # type: ignore
    AutoModelForSequenceClassification = AutoTokenizer = pipeline = None  # type: ignore
    def tqdm(x: Any, **kwargs):  # type: ignore
        return x

# Flag indicates whether real model is available
_HAS_TRANSFORMERS = AutoTokenizer is not None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults -------------------------------------------------------------------
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "unitary/toxic-bert"
DEFAULT_CATEGORIES = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

if not _HAS_TRANSFORMERS:
    class _StubModel:
        def __init__(self, *, categories=DEFAULT_CATEGORIES, **kwargs):
            self.categories = categories
            self.batch_size = 32

        def predict_proba(self, texts: Union[str, List[str]], *, show_progress: bool = False):
            if isinstance(texts, str):
                texts = [texts]
            return [{c: 0.0 for c in self.categories} for _ in texts]

        def predict_batch(self, texts: List[str], *, threshold=0.5, show_progress: bool = False):
            probas = self.predict_proba(texts)
            outputs = []
            for p in probas:
                outputs.append({
                    "is_toxic": False,
                    "categories": {c: False for c in self.categories},
                    "probabilities": p,
                })
            return outputs

        def unload(self):
            pass

    # simplify singleton usage
    _model_singleton = _StubModel()

    def get_model(*args, **kwargs):
        return _model_singleton

    def predict_proba(texts, **kwargs):
        return _model_singleton.predict_proba(texts)

    def analyze_text(text, **kwargs):
        return _model_singleton.predict_batch([text])[0]

    def unload_model():
        pass

    # Exit early because real ToxicityModel defined below relies on transformers
else:
    # ---------------------------------------------------------------------------
    # Core class ------------------------------------------------------------------
    # ---------------------------------------------------------------------------


    class ToxicityModel:
        """Wrapper around a pretrained multi-label toxicity model."""

        def __init__(
            self,
            *,
            model_name: str = DEFAULT_MODEL,
            categories: List[str] = DEFAULT_CATEGORIES,
            cache_dir: Optional[str] = None,
            device: Optional[str] = None,
            batch_size: int = 32,
        ) -> None:
            self.model_name = model_name
            self.categories = categories
            self.cache_dir = cache_dir
            self.batch_size = batch_size

            self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")

            self._model = None
            self._tokenizer = None
            self._classifier = None

            logger.info(
                "Initialised ToxicityModel(model=%s, device=%s)", model_name, self.device
            )

        # ---------------------------------------------------------------------
        # Lazy loading ---------------------------------------------------------
        # ---------------------------------------------------------------------

        def _load(self) -> None:
            if self._classifier is not None:
                return
            logger.info("Loading model %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            ).to(self.device)
            self._classifier = pipeline(
                "text-classification",
                model=self._model,
                tokenizer=self._tokenizer,
                device=0 if self.device == "cuda" else -1,
                top_k=None,  # return all labels
            )
            logger.info("Model loaded and ready.")

        # ---------------------------------------------------------------------
        # Public inference -----------------------------------------------------
        # ---------------------------------------------------------------------

        def predict_proba(self, texts: Union[str, List[str]], *, show_progress: bool = False) -> List[Dict[str, float]]:
            """Return per-label probability maps for *texts*."""
            if isinstance(texts, str):
                texts = [texts]
            if not texts:
                return []
            self._load()

            batches = [
                texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)
            ]
            results: List[Dict[str, float]] = []
            with torch.no_grad():
                iterator = tqdm(batches, disable=not show_progress, desc="inferring")
                for batch in iterator:
                    outputs = self._classifier(batch)
                    for out in outputs:
                        # HF pipeline returns a list[{label, score}, ...] per text
                        if isinstance(out, list):
                            res_map = {item["label"].lower(): float(item["score"]) for item in out}
                        elif isinstance(out, dict):  # single-label fallback
                            res_map = {out.get("label", "unknown").lower(): float(out.get("score", 0.0))}
                        else:
                            res_map = {}
                        # ensure all categories present
                        for cat in self.categories:
                            res_map.setdefault(cat, 0.0)
                        results.append(res_map)
            return results

        def predict_batch(
            self,
            texts: List[str],
            *,
            threshold: Union[float, Dict[str, float]] = 0.5,
            show_progress: bool = False,
        ) -> List[Dict[str, Union[bool, Dict[str, float]]]]:
            """Return verdict + maps for *texts* given *threshold*."""
            probas = self.predict_proba(texts, show_progress=show_progress)
            if isinstance(threshold, float):
                thresholds = {c: threshold for c in self.categories}
            else:
                thresholds = threshold

            outputs: List[Dict[str, Union[bool, Dict[str, float]]]] = []
            for prob_map in probas:
                cat_flags = {
                    cat: prob_map.get(cat, 0.0) >= thresholds.get(cat, 0.5)
                    for cat in self.categories
                }
                outputs.append(
                    {
                        "is_toxic": any(cat_flags.values()),
                        "categories": cat_flags,
                        "probabilities": prob_map,
                    }
                )
            return outputs

        # ---------------------------------------------------------------------
        # Memory management ----------------------------------------------------
        # ---------------------------------------------------------------------

        def unload(self) -> None:
            if self._model is None:
                return
            logger.info("Unloading model from memory")
            del self._model, self._tokenizer, self._classifier
            self._model = self._tokenizer = self._classifier = None
            if torch:
                torch.cuda.empty_cache()  # safe on CPU as well


    # ---------------------------------------------------------------------------
    # Singleton helpers ----------------------------------------------------------
    # ---------------------------------------------------------------------------

    _model_singleton: Optional[ToxicityModel] = None


    def get_model(
        *,
        model_name: str = DEFAULT_MODEL,
        categories: List[str] = DEFAULT_CATEGORIES,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> ToxicityModel:
        global _model_singleton
        if _model_singleton is None:
            _model_singleton = ToxicityModel(
                model_name=model_name,
                categories=categories,
                cache_dir=cache_dir,
                device=device,
                batch_size=batch_size,
            )
        return _model_singleton


    def predict_proba(
        texts: Union[str, List[str]],
        *,
        show_progress: bool = False,
        **model_kwargs: Union[str, int, List[str], None],
    ) -> List[Dict[str, float]]:
        model = get_model(**model_kwargs)
        return model.predict_proba(texts, show_progress=show_progress)


    def analyze_text(
        text: str,
        *,
        threshold: Union[float, Dict[str, float]] = 0.5,
        **model_kwargs: Union[str, int, List[str], None],
    ) -> Dict[str, Union[bool, Dict[str, float]]]:
        model = get_model(**model_kwargs)
        return model.predict_batch([text], threshold=threshold)[0]


    def unload_model() -> None:
        global _model_singleton
        if _model_singleton is not None:
            _model_singleton.unload()
            _model_singleton = None 