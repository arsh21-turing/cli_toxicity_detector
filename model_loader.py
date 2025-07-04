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

# ---------------------------------------------------------------------------
# Groq fallback helpers ------------------------------------------------------
# ---------------------------------------------------------------------------

# Confidence band that triggers a second opinion (inclusive bounds)
GRAY_ZONE_MIN = 0.4
GRAY_ZONE_MAX = 0.6

# Disk cache for Groq responses --------------------------------------
from groq_cache import GroqCache

_groq_cache = GroqCache()

def _groq_second_opinion(text: str) -> Optional[Dict[str, float]]:
    """Return a category→probability map from Groq or *None* if unavailable.

    The function is deliberately lightweight: it avoids importing heavy
    dependencies unless really necessary and degrades gracefully when the
    *groq* client or an API key is missing.  A real-world implementation would
    surface the exact error so the caller can react accordingly.
    """

    # Check cache first ------------------------------------------------
    cached = _groq_cache.get(text)
    if cached:
        logger.info("Groq cache hit")
        return cached.get("response")

    try:
        import groq  # type: ignore

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not set – skipping Groq fallback")
            return None

        client = groq.Client(api_key=api_key)  # type: ignore[attr-defined]

        prompt = (
            "Classify the following sentence for toxicity. "
            "Return a JSON object whose keys are category names (insult, hate, obscene, "
            "threat, sexual, self_harm, non_toxic) and whose values are probabilities "
            "between 0 and 1.\nSentence: " + text
        )

        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model="llama3-70b-8192",  # placeholder
            messages=[
                {"role": "system", "content": "You are a helpful toxicity classifier."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=256,
            response_format={"type": "json_object"},  # type: ignore[arg-type]
        )

        import json as _json

        raw = response.choices[0].message.content  # type: ignore[index]
        data = _json.loads(raw)

        # Normalise keys – map to upper enum names
        prob_map: Dict[str, float] = {}
        for key, val in data.items():
            prob_map[key.upper()] = float(val)

        # Save to cache
        _groq_cache.set(text, {k: v for k, v in prob_map.items()})
        return prob_map

    except Exception as exc:
        # Log at debug level to avoid noisy stderr when library missing
        logger.debug("Groq fallback unavailable: %s", exc)
        return None

def predict_toxicity(
    texts: List[str] | str,
    *,
    thresholds: Dict[str, float] | None = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int | None = None,
    show_progress: bool = False,
    allow_groq_fallback: bool = False,
    gray_min: float | None = None,
    gray_max: float | None = None,
    tie_policy: str = "prefer-groq",
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

    # Resolve active gray-zone bounds ------------------------------
    if gray_min is None:
        gray_min = GRAY_ZONE_MIN
    if gray_max is None:
        gray_max = GRAY_ZONE_MAX

    # Validate tie_policy early to avoid repeated checks
    if tie_policy not in {"prefer-groq", "prefer-local", "highest-confidence"}:
        raise ValueError(f"Unknown tie_policy: {tie_policy}")

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
                top_score = cat_map[most_probable]
                is_toxic = any(
                    v["above_threshold"] for c, v in verdict_map.items() if c != ToxicityCategory.NON_TOXIC
                )

                tie_source = "local"  # default unless Groq overrides

                # ------------------------------------------------------
                # Groq fallback if in gray zone ------------------------
                # ------------------------------------------------------
                groq_used = False
                if allow_groq_fallback and gray_min <= top_score <= gray_max:
                    groq_map = _groq_second_opinion(text)
                    if groq_map:
                        groq_used = True

                        # Build verdict map from Groq probabilities
                        groq_verdict_map: Dict[ToxicityCategory, Dict[str, Any]] = {}
                        for cat in ToxicityCategory:
                            groq_score = groq_map.get(cat.name, cat_map.get(cat, 0.0))
                            thr = thresholds.get(cat.name, 0.5) if thresholds else 0.5
                            groq_verdict_map[cat] = {
                                "score": groq_score,
                                "above_threshold": groq_score >= thr,
                                "threshold": thr,
                            }

                        groq_most = max(groq_verdict_map.items(), key=lambda it: it[1]["score"])[0]
                        groq_top_score = groq_verdict_map[groq_most]["score"]
                        groq_is_toxic = any(
                            v["above_threshold"]
                            for c, v in groq_verdict_map.items()
                            if c != ToxicityCategory.NON_TOXIC
                        )

                        # Log disagreements --------------------------------
                        if (groq_is_toxic != is_toxic) or (groq_most != most_probable):
                            logger.warning(
                                "Groq and local model disagree – local: toxic=%s cat=%s | groq: toxic=%s cat=%s",
                                is_toxic,
                                most_probable.name,
                                groq_is_toxic,
                                groq_most.name,
                            )

                        # Decide which result to keep based on *tie_policy*
                        chosen_source = "groq"
                        if tie_policy == "prefer-local":
                            chosen_source = "local"
                        elif tie_policy == "highest-confidence":
                            # Compare distance from 0.5 (confidence centre)
                            local_conf = abs(top_score - 0.5)
                            groq_conf = abs(groq_top_score - 0.5)
                            if local_conf > groq_conf:
                                chosen_source = "local"

                        if chosen_source == "groq":
                            verdict_map = groq_verdict_map
                            most_probable = groq_most
                            top_score = groq_top_score
                            is_toxic = groq_is_toxic
                            groq_used = True
                        else:
                            # Keep local results; flag we consulted Groq
                            groq_used = False

                        tie_source = chosen_source

                results.append(
                    {
                        "text": text,
                        "category_results": verdict_map,
                        "most_probable_category": most_probable,
                        "is_toxic": is_toxic,
                        "raw_logits": logit_vec,
                        "sigmoid_scores": scores,
                        "groq_used": groq_used,
                        "gray_zone_bounds": (gray_min, gray_max),
                        "tie_policy": tie_policy,
                        "tie_source": tie_source,
                    }
                )
    else:
        # Lightweight stub – generate deterministic zeros so unit tests run fast
        for text in texts:
            cat_map = {c: 0.0 for c in TOXIC_CATEGORIES}
            cat_map[ToxicityCategory.NON_TOXIC] = 1.0

            top_score = 0.0  # with stub all toxic scores zero
            verdict_map = {
                c: {"score": s, "above_threshold": False, "threshold": thresholds[c.name]}
                for c, s in cat_map.items()
            }

            most_probable = ToxicityCategory.NON_TOXIC
            is_toxic = False
            groq_used = False
            tie_source = "local"

            # Allow Groq fallback even in stub mode for unit tests
            if allow_groq_fallback and gray_min <= top_score <= gray_max:
                groq_map = _groq_second_opinion(text)
                if groq_map:
                    # Build verdict map from Groq probabilities
                    groq_verdict_map = {}
                    for cat in ToxicityCategory:
                        groq_score = groq_map.get(cat.name, 0.0)
                        thr = thresholds.get(cat.name, 0.5)
                        groq_verdict_map[cat] = {
                            "score": groq_score,
                            "above_threshold": groq_score >= thr,
                            "threshold": thr,
                        }

                    groq_most = max(groq_verdict_map.items(), key=lambda it: it[1]["score"])[0]
                    groq_top_score = groq_verdict_map[groq_most]["score"]
                    groq_is_toxic = any(
                        v["above_threshold"] for c, v in groq_verdict_map.items() if c != ToxicityCategory.NON_TOXIC
                    )

                    # Decide based on tie_policy
                    chosen_source = "groq"
                    if tie_policy == "prefer-local":
                        chosen_source = "local"
                    elif tie_policy == "highest-confidence":
                        if abs(top_score - 0.5) > abs(groq_top_score - 0.5):
                            chosen_source = "local"

                    if chosen_source == "groq":
                        verdict_map = groq_verdict_map
                        most_probable = groq_most
                        top_score = groq_top_score
                        is_toxic = groq_is_toxic
                        groq_used = True
                        tie_source = "groq"
                    else:
                        # remain local
                        groq_used = False
                        tie_source = "local"

            results.append(
                {
                    "text": text,
                    "category_results": verdict_map,
                    "most_probable_category": most_probable,
                    "is_toxic": is_toxic,
                    "raw_logits": [0.0] * len(cat_map),
                    "sigmoid_scores": [0.0] * len(cat_map),
                    "groq_used": groq_used,
                    "gray_zone_bounds": (gray_min, gray_max),
                    "tie_policy": tie_policy,
                    "tie_source": tie_source,
                }
            )

    return results 