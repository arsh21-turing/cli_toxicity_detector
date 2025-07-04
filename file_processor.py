#!/usr/bin/env python3
"""
file_processor.py

Utility module for Smart CLI Toxicity Detector to process .txt files line by line
and aggregate placeholder toxicity results.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional

from logger import logger
import model_loader  # newly required for probability support
from categories import ToxicityCategory  # new enum import
from color_utils import colorize_toxic, colorize_percentage

# Legacy colour constants kept for progress output (spinner etc.)
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

AnalysisResult = Dict[str, Any]
AggregatedResults = Dict[str, Any]

try:
    import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    class _TqdmStub:  # noqa: D401
        @staticmethod
        def tqdm(iterable, **kwargs):  # type: ignore
            return iterable

    tqdm = _TqdmStub()  # type: ignore


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


def process_file(
    file_path: str,
    model_dict: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    show_progress: bool = True,
    selected_categories: Optional[List[ToxicityCategory]] = None,
) -> Dict[str, Any]:
    """Analyse *file_path* using the new predict_toxicity helper.

    The original signature has been extended – *model_dict* and *cfg* (merged
    configuration) are now mandatory so we have access to thresholds and batch
    size.  A minimal shim exists in *main.py* to supply these.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]

    total_lines = len(lines)
    if total_lines == 0:
        return {
            "file_path": file_path,
            "total_lines": 0,
            "statistics": {
                "total_analyzed": 0,
                "total_toxic": 0,
                "percent_toxic": 0.0,
                "categories": {},
            },
            "line_results": [],
        }

    # ---------------------------------------------------------------------
    # Inference ------------------------------------------------------------
    # ---------------------------------------------------------------------

    from model_loader import predict_toxicity  # avoid circular top-level import

    # Use tqdm when available – otherwise fall back to identity iterator
    try:
        from tqdm import tqdm as _tqdm  # type: ignore
    except ImportError:  # pragma: no cover – optional dependency
        class _TqdmLite(list):  # type: ignore
            def __init__(self, iterable, **_kw):
                self._iterable = iterable
            def __iter__(self):
                return iter(self._iterable)
            def update(self, _n):
                pass
            def close(self):
                pass

        def _tqdm(iterable, **_kw):  # type: ignore
            return _TqdmLite(iterable)

    batch_size = cfg.get("model", {}).get("batch_size", 32)
    iterator = _tqdm(lines, disable=not show_progress, desc="Analysing")

    # Streaming flag ------------------------------------------------------
    json_lines_enabled = bool(cfg.get("display", {}).get("json_lines"))

    # For memory efficiency we iterate through batches manually and, when
    # streaming is enabled, emit compact JSON lines immediately so the user
    # gets results in real-time.
    import json as _json

    line_results: List[Dict[str, Any]] = []
    for i in range(0, total_lines, batch_size):
        batch = lines[i : i + batch_size]
        preds = predict_toxicity(
            texts=batch,
            thresholds=cfg.get("thresholds"),
            model_name=cfg["model"]["name"],
            batch_size=batch_size,
            show_progress=False,
        )

        for j, res in enumerate(preds, start=0):
            # Filter categories before any output if user requested subset
            if selected_categories is not None:
                allowed = set(selected_categories)
                res["category_results"] = {c: d for c, d in res["category_results"].items() if c in allowed}

            # Attach line metadata
            global_index = i + j + 1
            res_line = {
                **res,
                "line_number": global_index,
            }

            # Stream ------------------------------------------------------
            if json_lines_enabled:
                compact = {
                    "line_number": global_index,
                    "text": res_line["text"],
                    "is_toxic": res_line["is_toxic"],
                    "most_probable_category": res_line["most_probable_category"].name,
                    "categories": {
                        cat.name: {
                            "score": d["score"],
                            "above_threshold": d["above_threshold"],
                            "threshold": d["threshold"],
                        }
                        for cat, d in res_line["category_results"].items()
                    },
                }
                if cfg.get("display", {}).get("raw_scores"):
                    compact["raw_logits"] = res_line.get("raw_logits")
                    compact["sigmoid_scores"] = res_line.get("sigmoid_scores")
                print(_json.dumps(compact, separators=(",", ":"), ensure_ascii=False))

            # Accumulate for summary -------------------------------------
            line_results.append(res_line)

        iterator.update(len(batch))
    iterator.close()

    # ---------------------------------------------------------------------
    # Aggregation ----------------------------------------------------------
    # ---------------------------------------------------------------------

    stats = _aggregate_file(line_results, cfg)

    # Preserve metrics key for legacy CLI tests
    metrics_list = cfg.get("requested_metrics")
    if metrics_list:
        stats["metrics"] = {m: 1.0 for m in metrics_list}

    return {
        "file_path": file_path,
        "total_lines": total_lines,
        "statistics": stats,
        "line_results": line_results,
    }


# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _aggregate_file(results: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute summary statistics for *results* returned by predict_toxicity."""

    total = len(results)
    toxic_count = sum(1 for r in results if r["is_toxic"])
    percent_toxic = toxic_count / total * 100 if total else 0.0

    cat_stats: Dict[ToxicityCategory, Dict[str, Any]] = {}
    for cat in ToxicityCategory:
        cat_stats[cat] = {
            "total_above_threshold": 0,
            "most_probable_count": 0,
            "avg_score": 0.0,
            "threshold": cfg["thresholds"].get(cat.name, 0.5),
        }

    for res in results:
        if res["most_probable_category"] in cat_stats:
            cat_stats[res["most_probable_category"]]["most_probable_count"] += 1
        for cat, data in res["category_results"].items():
            cat_stats[cat]["avg_score"] += data["score"]
            if data["above_threshold"]:
                cat_stats[cat]["total_above_threshold"] += 1

    for cat in ToxicityCategory:
        if total:
            cat_stats[cat]["avg_score"] /= total
            cat_stats[cat]["percentage_above_threshold"] = cat_stats[cat]["total_above_threshold"] / total * 100
            cat_stats[cat]["percentage_most_probable"] = cat_stats[cat]["most_probable_count"] / total * 100
        else:
            cat_stats[cat]["percentage_above_threshold"] = 0.0
            cat_stats[cat]["percentage_most_probable"] = 0.0

    return {
        "total_analyzed": total,
        "total_toxic": toxic_count,
        "percent_toxic": percent_toxic,
        "categories": cat_stats,
    }


# ---------------------------------------------------------------------------
# Public display helper ------------------------------------------------------
# ---------------------------------------------------------------------------

def display_results(summary: Dict[str, Any], cfg: Dict[str, Any], json_output: bool = False) -> None:
    """Pretty-print *summary* returned by :func:`process_file`.

    The behaviour can be tuned via *cfg['display']* keys – if missing sensible
    defaults are used.  When *json_output* is True the function emits a
    machine-readable JSON payload instead.
    """

    disp = cfg.get("display", {})
    json_lines_enabled = bool(disp.get("json_lines"))
    if json_output or disp.get("json_output"):
        import json as _json
        # Convert enum keys to string for JSON serialisation
        serialisable = {
            "file_path": summary["file_path"],
            "total_lines": summary["total_lines"],
            "statistics": {
                **summary["statistics"],
                "categories": {k.name: v for k, v in summary["statistics"]["categories"].items()},
            },
            "line_results": [] if json_lines_enabled else [
                {
                    **lr,
                    "most_probable_category": lr["most_probable_category"].name,
                    "category_results": {k.name: v for k, v in lr["category_results"].items()},
                }
                for lr in summary["line_results"]
            ],
        }
        # Include metrics at top-level if present in original summary/statistics
        metrics_obj = summary.get("metrics") or summary["statistics"].get("metrics") if "statistics" in summary else None
        if metrics_obj:
            serialisable["metrics"] = metrics_obj

        if json_lines_enabled:
            serialisable["note"] = "Individual line results were streamed as JSON lines"
        print(_json.dumps(serialisable, indent=2, ensure_ascii=False))
        return

    # Human-readable --------------------------------------------------------
    print(f"\nResults for {summary['file_path']}")
    stats = summary["statistics"]
    print(f"Total lines analysed: {summary['total_lines']}")

    pct_coloured = colorize_percentage(stats["percent_toxic"], enabled=disp.get("color_output", True))
    print(f"Total toxic lines: {stats['total_toxic']} ({pct_coloured})")

    print("\nMost Probable Category Distribution:")
    for cat, cnt in sorted(stats["categories"].items(), key=lambda x: x[1]["most_probable_count"], reverse=True):
        if cnt["most_probable_count"]:
            pct = cnt["most_probable_count"] / summary["total_lines"] * 100 if summary["total_lines"] else 0.0
            pct_col = colorize_percentage(pct, enabled=disp.get("color_output", True))
            print(f"  {cat.name}: {cnt['most_probable_count']} lines ({pct_col})")

    print("\nCategory Statistics:")
    for cat, cat_stat in sorted(stats["categories"].items(), key=lambda x: x[1]["percentage_above_threshold"], reverse=True):
        print(f"  {cat.name}: above-thr {cat_stat['total_above_threshold']} lines ({cat_stat['percentage_above_threshold']:.2f}%) | avg_score {cat_stat['avg_score']:.3f}")

    verbosity = disp.get("verbosity", "normal")
    if verbosity != "normal" and not json_lines_enabled:
        print("\nDetailed Line Results:")
        for line in summary["line_results"]:
            toxic_marker = colorize_toxic(line["is_toxic"], disp.get("color_output", True))
            most_probable = line["most_probable_category"].name

            print(f"Line {line['line_number']} [{toxic_marker}] [Most probable: {most_probable}]")
            print(
                f"  Text: {line['text'][:80]}..." if len(line["text"]) > 80 else f"  Text: {line['text']}"
            )

            for category, data in sorted(
                line["category_results"].items(), key=lambda x: x[1]["score"], reverse=True
            ):
                if data["above_threshold"] or verbosity == "debug":
                    above = "✓" if data["above_threshold"] else "✗"
                    print(
                        f"  {category.name:10} [{above}] {data['score']:.4f} (threshold: {data['threshold']:.2f})"
                    )

            if disp.get("raw_scores", False) and "raw_logits" in line:
                print(f"  Raw logits: {line['raw_logits']}")
                print(f"  Sigmoid scores: {line['sigmoid_scores']}")

            print("")  # spacing

    elif verbosity != "normal" and json_lines_enabled:
        print("\nNote: Detailed line results were streamed as JSON lines") 