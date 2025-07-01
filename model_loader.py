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
import math
from categories import ToxicityCategory, TOXIC_CATEGORIES

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

# ---------------------------------------------------------------------------
# New helpers – sigmoid + rich inference -------------------------------------
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:  # pragma: no cover simple math util
    """Scalar sigmoid to avoid mandatory NumPy dependency in stub path."""

    try:
        import numpy as _np  # type: ignore
        return float(1 / (1 + _np.exp(-x)))
    except Exception:
        return 1 / (1 + math.exp(-x))


# Predict-toxicity API --------------------------------------------------------

DEFAULT_THRESHOLD_MAP: Dict[str, float] = {c.name: 0.5 for c in ToxicityCategory}


def predict_toxicity(
    texts: List[str] | str,
    *,
    thresholds: Dict[str, float] | None = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int | None = None,
    show_progress: bool = False,
) -> List[Dict[str, Any]]:
    """High-level helper that returns structured toxicity results.

    The function keeps the existing lazy singleton model mechanism so it plays
    nicely with the rest of the module.  It DOES NOT replace *predict_proba* /
    *predict_batch* used by legacy code – it simply builds on them.
    """

    if isinstance(texts, str):
        texts = [texts]

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLD_MAP.copy()
    else:
        # ensure every category has an entry
        merged = DEFAULT_THRESHOLD_MAP.copy()
        merged.update({k.upper(): float(v) for k, v in thresholds.items()})
        thresholds = merged

    mdl = get_model(model_name=model_name)
    if batch_size is None:
        batch_size = getattr(mdl, "batch_size", 32)

    # Step 1: obtain logits – we do NOT call existing predict_proba because it
    # already applies a softmax over the model's original label set which might
    # not align exactly with our categories.  Instead, if the real transformers
    # pipeline is available we fetch logits directly; otherwise we emulate with
    # zeros so the function still works in the lightweight test environment.

    results: List[Dict[str, Any]] = []

    if _HAS_TRANSFORMERS:
        import torch  # type: ignore
        from tqdm import tqdm as _tqdm  # type: ignore

        mdl._load()  # make sure underlying HF objects are ready
        device = mdl.device  # type: ignore[attr-defined]
        model = mdl._model  # type: ignore[attr-defined]
        tokenizer = mdl._tokenizer  # type: ignore[attr-defined]

        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        iterator = _tqdm(batches, disable=not show_progress, desc="toxicity-inference")
        for batch in iterator:
            toks = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            toks = {k: v.to(device) for k, v in toks.items()}
            with torch.no_grad():
                logits = model(**toks).logits.cpu().tolist()  # type: ignore[arg-type]

            for text, logit_vec in zip(batch, logits):
                # Apply sigmoid element-wise
                scores = [_sigmoid(v) for v in logit_vec]

                # Map first N scores onto our *toxic* categories; fallback zeros
                cat_map: Dict[ToxicityCategory, float] = {}
                for idx, cat in enumerate(TOXIC_CATEGORIES):
                    cat_map[cat] = scores[idx] if idx < len(scores) else 0.0

                max_toxic = max(cat_map.values()) if cat_map else 0.0
                cat_map[ToxicityCategory.NON_TOXIC] = 1.0 - max_toxic

                # Determine verdicts
                verdict_map: Dict[ToxicityCategory, Dict[str, Any]] = {}
                for c, score in cat_map.items():
                    thr = thresholds.get(c.name, 0.5)
                    verdict_map[c] = {"score": score, "above_threshold": score >= thr, "threshold": thr}

                most_probable = max(cat_map.items(), key=lambda it: it[1])[0]
                is_toxic = any(v["above_threshold"] for c, v in verdict_map.items() if c != ToxicityCategory.NON_TOXIC)

                results.append(
                    {
                        "text": text,
                        "category_results": verdict_map,
                        "most_probable_category": most_probable,
                        "is_toxic": is_toxic,
                        "raw_logits": logit_vec,
                        "sigmoid_scores": scores,
                    }
                )
    else:
        # Lightweight stub – generate deterministic zeros so unit tests run fast
        for text in texts:
            cat_map = {c: 0.0 for c in TOXIC_CATEGORIES}
            cat_map[ToxicityCategory.NON_TOXIC] = 1.0

            verdict_map = {
                c: {"score": s, "above_threshold": False, "threshold": thresholds[c.name]}
                for c, s in cat_map.items()
            }
            results.append(
                {
                    "text": text,
                    "category_results": verdict_map,
                    "most_probable_category": ToxicityCategory.NON_TOXIC,
                    "is_toxic": False,
                    "raw_logits": [0.0] * len(cat_map),
                    "sigmoid_scores": [0.0] * len(cat_map),
                }
            )

    return results 