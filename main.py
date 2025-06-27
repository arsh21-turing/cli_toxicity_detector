#!/usr/bin/env python3
"""Smart CLI Toxicity Detector – entry point.

Supports:
 • single-sentence or batch-file detection
 • configuration file overrides (YAML/JSON)
 • creation of default config template
"""
from __future__ import annotations

import argparse
import os
import sys
import json
from pathlib import Path

from config_loader import CONFIG_PATHS, create_default_config, load_config
from file_processor import analyze_file, display_file_results
from model_loader import get_analyzer
from logger import logger

CFG = load_config()

# ANSI colours
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _print_result(res: dict, show_probs: bool = True) -> None:
    """Pretty-print analysis result to terminal."""
    if res["is_toxic"]:
        verdict = f"{RED}{BOLD}Toxic{RESET}"
        cat_str = f" ({res['category']})"
        if len(res.get("toxic_categories", [])) > 1:
            others = ", ".join(res["toxic_categories"][1:])
            cat_str += f" [also: {others}]"
    else:
        verdict = f"{GREEN}{BOLD}Non-Toxic{RESET}"
        cat_str = ""

    logger.info(f"Verdict: {'Toxic' if res['is_toxic'] else 'Non-Toxic'}{cat_str}")
    print(f"Verdict: {verdict}{cat_str}")
    print(f"Confidence: {res['confidence']:.2f}")

    if show_probs and CFG["output"].get("show_probabilities", True):
        print("\nCategory probabilities:")
        for cat, prob in sorted(res["probabilities"].items(), key=lambda x: x[1], reverse=True):
            colour = GREEN if prob <= 0.4 else YELLOW if prob <= 0.6 else RED
            cat_thresh = CFG["categories"].get(cat, CFG["model"].get("threshold", 0.6))
            marker = "*" if prob > cat_thresh else ""
            print(f"  - {cat}: {colour}{prob:.2f}{RESET}{marker}")
        print("* exceeds category threshold" if any(p > CFG["categories"].get(c, CFG["model"].get("threshold", 0.6)) for c, p in res["probabilities"].items()) else "")


def _cmd_create_config(path: str | None) -> None:
    dest = Path(path or CONFIG_PATHS[0]).expanduser()
    if create_default_config(dest):
        print(f"Created default configuration at {dest}. You may edit it now.")
    else:
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="smart-cli-toxicity-detector", description="Smart CLI Toxicity Detector")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--text", help="Sentence to analyse")
    grp.add_argument("--file", help="Path to .txt file (one sentence per line)")
    grp.add_argument("--create-config", nargs="?", const=str(CONFIG_PATHS[0]), metavar="PATH", help="Create default config file")

    p.add_argument("--verbose", action="store_true", help="Detailed per-line file output")
    p.add_argument("--threshold", type=float, help="Override overall threshold (0-1)")
    p.add_argument("--model", type=str, help="Override model name from config")
    p.add_argument("--no-probabilities", action="store_true", help="Hide probability breakdown")
    p.add_argument("--quiet", action="store_true", help="Suppress progress indicator during file analysis")
    p.add_argument("--json", action="store_true", help="Output analysis as compact JSON (machine-readable)")

    # Optional explicit config path override
    p.add_argument(
        "--config",
        type=str,
        help="Path to a YAML/JSON configuration file to use instead of the default search paths",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()

    # argparse exits with SystemExit on error; wrap to provide generic message if needed
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        # argparse already printed message; just propagate exit code
        raise
    except Exception as exc:
        print(f"Error parsing command-line arguments: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.create_config is not None:
        _cmd_create_config(args.create_config)
        return

    # ------------------------------------------------------------------
    # Custom configuration path validation
    # ------------------------------------------------------------------
    if args.config:
        cfg_path = Path(args.config).expanduser()
        if not cfg_path.exists() or not cfg_path.is_file():
            print(f"Error: invalid configuration file '{cfg_path}'. File does not exist or is not a regular file.", file=sys.stderr)
            sys.exit(1)

        try:
            if cfg_path.suffix.lower() in {".yml", ".yaml"}:
                try:
                    import yaml  # type: ignore
                except ImportError:
                    raise RuntimeError("PyYAML not installed; cannot parse YAML configs.")
                yaml.safe_load(cfg_path.read_text())  # validate only
            else:
                json.loads(cfg_path.read_text())  # validate only
        except Exception as exc:
            print(f"Error: invalid configuration file '{cfg_path}': {exc}", file=sys.stderr)
            sys.exit(1)

    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        print("Error: threshold must be between 0 and 1", file=sys.stderr)
        sys.exit(1)

    analyse = get_analyzer(threshold=args.threshold, model_name=args.model)

    # unknown model detection: if user supplied --model but analyzer fell back to placeholder
    if args.model and analyse.__name__ == "_placeholder_result":
        print(f"Error: unknown model '{args.model}'. Please check the model name.", file=sys.stderr)
        sys.exit(1)

    show_probs = not args.no_probabilities and not args.json
    show_prog = not args.quiet and not args.json

    json_out = args.json

    if args.text:
        result = analyse(args.text)
        if json_out:
            print(json.dumps(result, separators=(',', ':'), ensure_ascii=False))
        else:
            _print_result(result, show_probs)
    else:
        if not os.path.exists(args.file):
            print(f"Error: provided --file target is not a readable regular file – file not found.", file=sys.stderr)
            sys.exit(1)

        if not os.path.isfile(args.file):
            print("Error: provided --file target is not a readable regular file.", file=sys.stderr)
            sys.exit(1)

        if not os.access(args.file, os.R_OK):
            print(f"Error: provided --file target '{args.file}' is not readable.", file=sys.stderr)
            sys.exit(1)

        summary = analyze_file(args.file, analyse, show_progress=show_prog)
        if json_out:
            out = summary if args.verbose else {k: v for k, v in summary.items() if k != 'results'}
            print(json.dumps(out, separators=(',', ':'), ensure_ascii=False))
        else:
            display_file_results(summary, verbose=args.verbose)


if __name__ == "__main__":
    main() 