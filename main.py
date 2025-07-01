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
from typing import Any, Dict, List, Optional
from pathlib import Path

from logger import logger as log
from config_loader import create_default_config, load_config
from file_processor import process_file
from model_loader import analyze_text, predict_proba, unload_model

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
    p.add_argument("--threshold", type=float, help="Override global detection threshold")

    # Output behaviour
    p.add_argument("--json", action="store_true", help="Emit JSON instead of coloured text")
    p.add_argument("--verbose", "-v", action="store_true", help="Show extra details (per-line, probabilities)")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress progress indicators")
    p.add_argument("--probabilities", "-p", action="store_true", help="Display the full category→probability map for each analysed sentence")
    p.add_argument("--metrics", help="Comma-separated list of evaluation metrics to compute when a labelled file is supplied")

    p.add_argument("--output", "-o", help="Save JSON output to the given file")
    return p


# ---------------------------------------------------------------------------
# Single-sentence path --------------------------------------------------------
# ---------------------------------------------------------------------------

def _process_single(text: str, *, cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    threshold = args.threshold if args.threshold is not None else cfg["model"]["threshold"]
    model_name = args.model if args.model else cfg["model"]["name"]

    res = analyze_text(text=text, threshold=threshold, model_name=model_name)

    if args.probabilities:
        res["raw_probabilities"] = predict_proba(texts=text, model_name=model_name)[0]

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
# Main -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:  # noqa: C901 – flow clarity
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_config()

    if args.create_config:
        cfg_path = create_default_config(Path("toxicity_detector.yaml"))
        print(f"Default configuration file written to {cfg_path}")
        return 0

    try:
        if args.text:
            _process_single(args.text, cfg=cfg, args=args)
        elif args.file:
            if not args.file.endswith(".txt"):
                print("Error: only .txt files are supported for --file analysis", file=sys.stderr)
                return 1
            options = {
                "threshold": args.threshold if args.threshold is not None else cfg["model"]["threshold"],
                "model_name": args.model if args.model else cfg["model"]["name"],
                "show_progress": not args.quiet,
                "include_line_content": args.verbose,
                "include_probabilities": args.probabilities,
                "metrics_list": args.metrics,
            }
            summary = process_file(args.file, options=options)
            if args.json:
                _emit_json(summary, args.output)
            else:
                _print_human_file(summary, args)
        return 0
    finally:
        unload_model()


# ---------------------------------------------------------------------------
# Human file summary ---------------------------------------------------------
# ---------------------------------------------------------------------------

def _print_human_file(summary: Dict[str, Any], args: argparse.Namespace) -> None:
    print(f"File Analysis: {summary['total_lines']} lines – {summary['percent_toxic']:.2f}% toxic")
    for cat, cnt in summary.get("category_counts", {}).items():
        if cnt:
            print(f"  {cat}: {cnt}")
    if args.verbose and "line_results" in summary:
        for idx, res in enumerate(summary["line_results"], 1):
            verdict = "TOXIC" if res["is_toxic"] else "OK"
            line_txt = res.get("content", "")
            print(f"[{idx:>4}] {verdict} {line_txt[:60]}")
            if args.probabilities:
                for cat, p in res.get("raw_probabilities", {}).items():
                    print(f"       {cat}: {p:.3f}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main()) 