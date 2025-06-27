#!/usr/bin/env python3
"""
file_processor.py

Utility module for Smart CLI Toxicity Detector to process .txt files line by line
and aggregate placeholder toxicity results.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List

from logger import logger

# Re-use ANSI codes locally for coloured output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

AnalysisResult = Dict[str, Any]
AggregatedResults = Dict[str, Any]


def analyze_file(
    file_path: str,
    analyzer_func: Callable[[str], AnalysisResult],
    *,
    show_progress: bool = True,
) -> AggregatedResults:
    """Analyse *file_path* line-by-line, optionally showing a progress spinner."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    results: List[AnalysisResult] = []
    toxic_count = 0
    categories: Dict[str, int] = {}
    line_count = 0

    total_lines = 0
    if show_progress:
        try:
            with open(file_path, "r", encoding="utf-8") as fh_count:
                total_lines = sum(1 for l in fh_count if l.strip())
            print(f"Analyzing {total_lines} lines from {file_path}…")
        except Exception:
            show_progress = False  # silently disable if counting fails

    spinner_chars = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
    spinner_idx = 0
    update_every = max(1, total_lines // 100) if total_lines else 10

    with open(file_path, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            sentence = line.strip()
            if not sentence:
                continue  # ignore blank lines

            line_count += 1
            # progress update
            if show_progress and line_count % update_every == 0:
                spinner_idx = (spinner_idx + 1) % len(spinner_chars)
                pct = f"{(line_count/total_lines*100):.0f}%" if total_lines else "?"
                sys.stdout.write(
                    f"\r{spinner_chars[spinner_idx]} Processing {line_count}/{total_lines or '?'} lines ({pct}) "
                )
                sys.stdout.flush()

            res = analyzer_func(sentence)
            res = {
                **res,
                "line_number": line_number,
                "text": sentence[:100] + ("…" if len(sentence) > 100 else ""),
            }
            results.append(res)

            if res.get("is_toxic"):
                toxic_count += 1
                cat = (res.get("category") or "unspecified").lower()
                categories[cat] = categories.get(cat, 0) + 1

    # clear progress line
    if show_progress:
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()

    toxic_percentage = (toxic_count / line_count * 100) if line_count else 0.0

    logger.info(f"Analyzing file: {file_path}")
    logger.info(
        f"Completed file: {file_path} | lines={line_count} | toxic={toxic_count} | pct={toxic_percentage:.1f}%"
    )

    return {
        "file_path": file_path,
        "total_lines": line_count,
        "toxic_count": toxic_count,
        "toxic_percentage": toxic_percentage,
        "categories": categories,
        "results": results,
    }


def display_file_results(data: AggregatedResults, verbose: bool = False) -> None:
    """Pretty-print *data* returned by :func:`analyze_file` with ANSI colours."""
    print(f"\n{BOLD}File Analysis Results:{RESET} {data['file_path']}")
    print(f"Total lines processed: {data['total_lines']}")

    perc = data["toxic_percentage"]
    if perc == 0:
        colour = GREEN
    elif perc < 10:
        colour = YELLOW
    else:
        colour = RED

    print(f"Toxic content: {colour}{perc:.1f}%{RESET} ({data['toxic_count']} lines)")

    if data["toxic_count"]:
        print("\nCategory breakdown:")
        for cat, cnt in data["categories"].items():
            print(f"  - {cat}: {cnt} lines")

    if verbose:
        print(f"\n{BOLD}Detailed Results:{RESET}")
        for res in data["results"]:
            if res["is_toxic"]:
                verdict = f"{RED}Toxic{RESET}"
                category = f" ({res['category']})" if res.get("category") else ""
            else:
                verdict = f"{GREEN}Non-Toxic{RESET}"
                category = ""
            print(f"Line {res['line_number']:>4}: {verdict}{category} - '{res['text']}'") 