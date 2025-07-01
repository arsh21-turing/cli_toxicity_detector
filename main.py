#!/usr/bin/env python3
"""
Main entry point for the toxicity detection tool.

This script provides the command-line interface and orchestrates the workflow
for analyzing text for toxicity.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from logger import logger as log
from config_loader import create_default_config, load_config
from file_processor import process_file
from model_loader import predict_toxicity
from categories import ToxicityCategory
from color_utils import colorize_toxic, supports_color

# ---------------------------------------------------------------------------
# Threshold CLI helpers ------------------------------------------------------
# ---------------------------------------------------------------------------

def create_threshold_argument(parser: argparse.ArgumentParser) -> None:
    """Attach global + per-category threshold flags to *parser*."""

    # Global flag – applies to every category unless overridden
    parser.add_argument(
        "--threshold",
        type=float,
        help="Global probability threshold applied to all categories (overridden by individual flags)",
    )

    # Per-label flags in a separate argument group for nicer --help output
    grp = parser.add_argument_group("category thresholds")
    for cat in ToxicityCategory:
        flag = f"--threshold-{cat.name.lower().replace('_', '-')}"
        # dest uses underscore so we can easily inspect Namespace attributes
        dest = f"threshold_{cat.name.lower().replace('_', '_')}"
        grp.add_argument(
            flag,
            dest=dest,
            type=float,
            help=f"Threshold for {cat.name} category",
        )


def parse_threshold_args(args: argparse.Namespace) -> Dict[str, float]:
    """Return a mapping *category_name* → threshold extracted from *args*."""

    thresholds: Dict[str, float] = {}

    # Global threshold first
    if getattr(args, "threshold", None) is not None:
        for cat in ToxicityCategory:
            thresholds[cat.name] = float(args.threshold)  # ensure float not Decimal etc.

    # Per-category overrides
    for cat in ToxicityCategory:
        attr = f"threshold_{cat.name.lower().replace('_', '_')}"
        val = getattr(args, attr, None)
        if val is not None:
            thresholds[cat.name] = float(val)

    return thresholds


# ---------------------------------------------------------------------------
# Argument parsing -----------------------------------------------------------
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Toxicity Detection Tool – analyse text or files for toxic content",
        prog="toxicity-detector",
    )

    mode_grp = p.add_mutually_exclusive_group(required=True)
    mode_grp.add_argument("--text", help="Analyse a single text string")
    mode_grp.add_argument("--file", help="Analyse a .txt file line-by-line")
    mode_grp.add_argument(
        "--create-config",
        action="store_true",
        help="Create a default configuration file in the current directory",
    )

    # Overrides
    p.add_argument("--model", help="Override model name specified in config")
    # NOTE: per-category threshold flags added below

    # Output behaviour
    p.add_argument("--json", action="store_true", help="Emit JSON instead of coloured text")
    p.add_argument("--json-lines", action="store_true", help="Stream compact one-line JSON objects for each analysed sentence")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colour output")
    p.add_argument("--verbose", "-v", action="store_true", help="Show extra details (per-line, probabilities)")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress progress indicators")
    p.add_argument("--probabilities", "-p", action="store_true", help="Display the full category→probability map for each analysed sentence")
    p.add_argument("--metrics", help="Comma-separated list of evaluation metrics to compute when a labelled file is supplied")

    p.add_argument("--output", "-o", help="Save JSON output to the given file")

    # Inject threshold args after core flags to keep help tidy
    create_threshold_argument(p)

    return p


# ---------------------------------------------------------------------------
# Single-sentence path --------------------------------------------------------
# ---------------------------------------------------------------------------

def _process_single(text: str, *, cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    threshold = args.threshold if args.threshold is not None else cfg["model"]["threshold"]
    model_name = args.model if args.model else cfg["model"]["name"]

    res = predict_toxicity(text=text, threshold=threshold, model_name=model_name)

    if args.probabilities:
        res["raw_probabilities"] = load_model(model_name=model_name)[0]

    if args.json:
        payload: Dict[str, Any] = {
            "text": text,
            "is_toxic": res["is_toxic"],
            "categories": res["categories"],
            "probabilities": res["probabilities"],
            "timestamp": datetime.now().isoformat(),
        }
        if args.probabilities:
            payload["raw_probabilities"] = res["raw_probabilities"]
        _emit_json(payload, args.output)
    else:
        _print_human_single(res, args)
    return res


def _print_human_single(res: Dict[str, Any], args: argparse.Namespace) -> None:
    verdict = "TOXIC" if res["is_toxic"] else "OK"
    print(f"Result: {verdict}")

    if res["is_toxic"]:
        detected = [c for c, v in res["categories"].items() if v]
        print("Detected categories:", ", ".join(detected))

    if args.probabilities:
        for cat, p in res.get("raw_probabilities", {}).items():
            print(f"  {cat}: {p:.4f}")


# ---------------------------------------------------------------------------
# JSON helper ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def _emit_json(obj: Any, path: Optional[str] = None) -> None:
    data = json.dumps(obj, indent=2, ensure_ascii=False)
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(data)
    print(data)


# ---------------------------------------------------------------------------
# Result presentation --------------------------------------------------------
# ---------------------------------------------------------------------------

def display_single_text_result(result: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """Pretty-print *result* coming from model_loader.predict_toxicity().

    The helper respects *cfg["display"]* keys when present but degrades
    gracefully when the dict is missing (so older configs still work).
    """

    display = cfg.get("display", {})
    json_out = bool(display.get("json_output"))
    json_lines = bool(display.get("json_lines"))
    raw_scores = bool(display.get("raw_scores"))
    verbosity = display.get("verbosity", "normal")
    show_probs = bool(display.get("show_probabilities"))

    # First: optional streaming of compact JSON line
    if json_lines:
        import json as _json
        compact = {
            "text": result["text"],
            "is_toxic": result["is_toxic"],
            "most_probable_category": str(result["most_probable_category"].name),
            "categories": {
                cat.name: {
                    "score": data["score"],
                    "above_threshold": data["above_threshold"],
                    "threshold": data["threshold"],
                }
                for cat, data in result["category_results"].items()
            },
        }
        if raw_scores:
            compact["raw_logits"] = result.get("raw_logits")
            compact["sigmoid_scores"] = result.get("sigmoid_scores")
        print(_json.dumps(compact, separators=(",", ":"), ensure_ascii=False))

    # Pretty/legacy JSON output afterwards (to keep "summary")
    if json_out:
        import json as _json
        serialisable = {
            "text": result["text"],
            "is_toxic": result["is_toxic"],
            "most_probable_category": str(result["most_probable_category"].name),
            "category_results": {
                cat.name: data for cat, data in result["category_results"].items()
            },
        }
        if raw_scores:
            serialisable["raw_logits"] = result.get("raw_logits")
            serialisable["sigmoid_scores"] = result.get("sigmoid_scores")
        if show_probs:
            # Fetch raw probabilities from legacy helper for backwards-compat
            try:
                from model_loader import predict_proba  # type: ignore
                prob_map = predict_proba(result["text"])[0]  # type: ignore[index]
            except Exception:
                prob_map = {}
            serialisable["raw_probabilities"] = prob_map
        print(_json.dumps(serialisable, indent=2, ensure_ascii=False))
        return

    # human readable --------------------------------------------------------
    verdict = "TOXIC" if result["is_toxic"] else "NON-TOXIC"
    verdict_col = colorize_toxic(result["is_toxic"], display.get("color_output", True))
    print(f"Analysis for: \"{result['text']}\"")
    print(f"Overall assessment: {verdict_col}")
    print(f"Most probable category: {result['most_probable_category'].name}")
    print("-" * 50)

    for cat, data in sorted(result["category_results"].items(), key=lambda x: x[1]["score"], reverse=True):
        mark = "✓" if data["above_threshold"] else "✗"
        print(f"{cat.name:12} [{mark}] {data['score']:.4f} (thr={data['threshold']:.2f})")
        if verbosity != "normal":
            from categories import get_category_description
            desc = get_category_description(cat)
            if desc:
                print(f"  {desc}")
    if raw_scores:
        print("-" * 50)
        print("Raw logits:", result.get("raw_logits"))
        print("Sigmoid scores:", result.get("sigmoid_scores"))

    # Legacy probability dump ---------------------------------------------
    if show_probs and not raw_scores:
        try:
            from model_loader import predict_proba  # type: ignore
            prob_map = predict_proba(result["text"])[0]  # type: ignore[index]
        except Exception:
            prob_map = {}
        for label, val in prob_map.items():
            print(f"{label}: {val:.3f}")


# ---------------------------------------------------------------------------
# Main -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:  # noqa: C901 – clarity
    parser = _build_parser()

    # Extra display flags ---------------------------------------------------
    parser.add_argument("--raw-scores", action="store_true", help="Show raw logits/sigmoid scores")

    args = parser.parse_args(argv)

    cfg = load_config()
    cfg.setdefault("display", {})  # ensure key exists

    # honour CLI json/raw flags -------------------------------------------
    if args.json:
        cfg["display"]["json_output"] = True
    if getattr(args, "json_lines", False):
        cfg["display"]["json_lines"] = True
    if getattr(args, "no_color", False):
        cfg["display"]["color_output"] = False
    else:
        cfg["display"].setdefault("color_output", supports_color())
    if args.raw_scores:
        cfg["display"]["raw_scores"] = True

    if args.probabilities:
        cfg["display"]["show_probabilities"] = True

    # Parse threshold overrides -------------------------------------------
    overrides = parse_threshold_args(args)
    if overrides:
        cfg.setdefault("thresholds", {}).update(overrides)

    # Metrics list requested by CLI (legacy behaviour)
    if getattr(args, "metrics", None):
        cfg["requested_metrics"] = [m.strip() for m in str(args.metrics).split(",") if m.strip()]

    # ---------------------------------------------------------------------
    if args.text:
        res = predict_toxicity(
            texts=[args.text],
            thresholds=cfg.get("thresholds"),
            model_name=args.model or cfg.get("model", {}).get("name", "unitary/toxic-bert"),
            show_progress=False,
        )[0]
        display_single_text_result(res, cfg)
        return 0

    if args.file:
        from model_loader import get_model
        mdl = get_model(model_name=args.model or cfg.get("model", {}).get("name"))
        summary = process_file(
            args.file,
            mdl,  # currently unused inside but keeps signature stable
            cfg,
            show_progress=not args.quiet,
        )
        from file_processor import display_results  # local import to avoid circular
        display_results(summary, cfg, json_output=args.json)
        return 0

    if args.create_config:
        path = create_default_config(Path("toxicity_detector.yaml"))
        print(f"Default configuration file written to {path}")
        return 0

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main()) 