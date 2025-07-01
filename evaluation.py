"""#!/usr/bin/env python3
Evaluation metrics for multi-label toxicity classification.

This module provides functions to evaluate model performance using
various metrics appropriate for multi-label classification tasks while
remaining lightweight and flexible.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

try:
    import evaluate  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The `evaluate` package is required for metric computation. Install it via `pip install evaluate`."
    ) from exc

try:
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore
    from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
except ImportError:  # pragma: no cover – plotting is optional
    plt = sns = None  # type: ignore

# ---------------------------------------------------------------------------
# Metric caching ----------------------------------------------------------------
# ---------------------------------------------------------------------------

_METRIC_CACHE: Dict[str, Any] = {}


def get_metric(name: str) -> Any:
    """Return a HuggingFace *evaluate* metric, cached to avoid re-downloads."""
    if name not in _METRIC_CACHE:
        _METRIC_CACHE[name] = evaluate.load(name)
    return _METRIC_CACHE[name]


# ---------------------------------------------------------------------------
# Default metric set ---------------------------------------------------------
# ---------------------------------------------------------------------------

DEFAULT_METRICS: Set[str] = {
    "accuracy",
    "precision_micro",
    "precision_macro",
    "recall_micro",
    "recall_macro",
    "f1_micro",
    "f1_macro",
    "roc_auc",
    "average_precision",
    "hamming_loss",
}


# ---------------------------------------------------------------------------
# Core computation -----------------------------------------------------------
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float = 0.5,
    category_names: Optional[List[str]] = None,
    metrics_to_compute: Optional[Set[str]] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Compute requested metrics and a concise confusion-matrix summary.

    Parameters
    ----------
    predictions / labels
        Arrays of shape ``(n_samples, n_labels)``.
    threshold
        Decision threshold for probability ‑> binary conversion.
    category_names
        Optional list of label names (length == n_labels).
    metrics_to_compute
        If given, restrict computation to this subset; otherwise *DEFAULT_METRICS*.

    Returns
    -------
    (metrics, cm_summary)
        ``metrics`` is a flat mapping ``metric_name -> score``; ``cm_summary`` is a
        nested dict with per-label TP/FP/FN/TN counts plus a *total* aggregate.
    """
    if predictions.shape != labels.shape:
        raise ValueError("`predictions` and `labels` must share shape.")

    if metrics_to_compute is None:
        metrics_to_compute = DEFAULT_METRICS.copy()

    bin_preds = (predictions >= threshold).astype(int)

    # ---------------------------------------------------------------------
    # Confusion-matrix summary
    # ---------------------------------------------------------------------
    cm_summary: Dict[str, Any] = {}
    totals = dict(tp=0, fp=0, fn=0, tn=0)

    if category_names is None or len(category_names) != labels.shape[1]:
        category_names = [f"label_{i}" for i in range(labels.shape[1])]

    if "confusion" in metrics_to_compute or True:  # always compute summary
        for idx, name in enumerate(category_names):
            cm = confusion_matrix(labels[:, idx], bin_preds[:, idx], labels=[0, 1])
            # shape guaranteed 2x2 when both classes present, else handle gracefully
            if cm.shape == (2, 2):
                tn, fp = cm[0]
                fn, tp = cm[1]
            else:  # one-class edge-case
                tn = fp = fn = tp = 0
                if cm.shape == (1, 1):
                    tn = cm[0, 0]
                elif cm.shape == (1, 2):
                    tn, fp = cm[0]
                elif cm.shape == (2, 1):
                    tn = cm[0, 0]
                    fn = cm[1, 0]
            cm_summary[name] = {
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
            }
            totals["tp"] += tp
            totals["fp"] += fp
            totals["fn"] += fn
            totals["tn"] += tn

    cm_summary["total"] = {
        "true_positives": totals["tp"],
        "false_positives": totals["fp"],
        "false_negatives": totals["fn"],
        "true_negatives": totals["tn"],
    }

    # ---------------------------------------------------------------------
    # Metric computation (conditional)
    # ---------------------------------------------------------------------
    out: Dict[str, float] = {}

    if "accuracy" in metrics_to_compute:
        out["accuracy"] = get_metric("accuracy").compute(
            predictions=bin_preds.tolist(), references=labels.tolist()
        )["accuracy"]

    if any(m in metrics_to_compute for m in {"precision_micro", "precision_macro"}):
        precision_metric = get_metric("precision")
        if "precision_micro" in metrics_to_compute:
            out["precision_micro"] = precision_metric.compute(
                predictions=bin_preds.flatten(), references=labels.flatten(), average="micro"
            )["precision"]
        if "precision_macro" in metrics_to_compute:
            out["precision_macro"] = precision_metric.compute(
                predictions=bin_preds, references=labels, average="macro"
            )["precision"]

    if any(m in metrics_to_compute for m in {"recall_micro", "recall_macro"}):
        recall_metric = get_metric("recall")
        if "recall_micro" in metrics_to_compute:
            out["recall_micro"] = recall_metric.compute(
                predictions=bin_preds.flatten(), references=labels.flatten(), average="micro"
            )["recall"]
        if "recall_macro" in metrics_to_compute:
            out["recall_macro"] = recall_metric.compute(
                predictions=bin_preds, references=labels, average="macro"
            )["recall"]

    if any(m in metrics_to_compute for m in {"f1_micro", "f1_macro"}):
        f1_metric = get_metric("f1")
        if "f1_micro" in metrics_to_compute:
            out["f1_micro"] = f1_metric.compute(
                predictions=bin_preds.flatten(), references=labels.flatten(), average="micro"
            )["f1"]
        if "f1_macro" in metrics_to_compute:
            out["f1_macro"] = f1_metric.compute(
                predictions=bin_preds, references=labels, average="macro"
            )["f1"]

    if "roc_auc" in metrics_to_compute:
        try:
            out["roc_auc"] = get_metric("roc_auc").compute(
                prediction_scores=predictions, references=labels, multi_class="ovr"
            )["roc_auc"]
        except Exception:
            out["roc_auc"] = float("nan")

    if "average_precision" in metrics_to_compute:
        try:
            out["average_precision"] = get_metric("average_precision").compute(
                prediction_scores=predictions, references=labels, average="macro"
            )["average_precision"]
        except Exception:
            out["average_precision"] = float("nan")

    if "hamming_loss" in metrics_to_compute:
        out["hamming_loss"] = float(np.mean(bin_preds != labels))

    # Per-label metrics requested dynamically
    prefix_metrics = {m for m in metrics_to_compute if "_" in m and m.split("_")[0] in category_names}
    if prefix_metrics:
        precision_metric = get_metric("precision")
        recall_metric = get_metric("recall")
        f1_metric = get_metric("f1")
        for idx, name in enumerate(category_names):
            if f"{name}_precision" in prefix_metrics:
                out[f"{name}_precision"] = precision_metric.compute(
                    predictions=bin_preds[:, idx], references=labels[:, idx]
                )["precision"]
            if f"{name}_recall" in prefix_metrics:
                out[f"{name}_recall"] = recall_metric.compute(
                    predictions=bin_preds[:, idx], references=labels[:, idx]
                )["recall"]
            if f"{name}_f1" in prefix_metrics:
                out[f"{name}_f1"] = f1_metric.compute(
                    predictions=bin_preds[:, idx], references=labels[:, idx]
                )["f1"]

    return out, cm_summary


# ---------------------------------------------------------------------------
# Visualisation helpers (optional) ------------------------------------------
# ---------------------------------------------------------------------------

def _check_plotting():
    if plt is None or sns is None:
        raise RuntimeError("matplotlib & seaborn are required for plotting helpers.")


def plot_confusion_matrices(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float = 0.5,
    category_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
    output_file: Optional[str] = "confusion_matrices.png",
) -> Dict[str, np.ndarray]:
    """Plot per-label confusion matrices and return raw matrices."""
    _check_plotting()
    binary_preds = (predictions >= threshold).astype(int)
    if category_names is None or len(category_names) != labels.shape[1]:
        category_names = [f"label_{i}" for i in range(labels.shape[1])]

    n = labels.shape[1]
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=figsize)
    axes = axes.flatten()

    matrices: Dict[str, np.ndarray] = {}
    for idx, (ax, name) in enumerate(zip(axes, category_names)):
        cm = confusion_matrix(labels[:, idx], binary_preds[:, idx])
        matrices[name] = cm
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["non", "toxic"], yticklabels=["non", "toxic"])
        ax.set_title(name)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    plt.close()
    return matrices


def plot_roc_curves(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    category_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    output_file: Optional[str] = "roc_curves.png",
) -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
    """Plot ROC curves and return underlying data."""
    _check_plotting()
    if category_names is None or len(category_names) != labels.shape[1]:
        category_names = [f"label_{i}" for i in range(labels.shape[1])]

    plt.figure(figsize=figsize)
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
    for idx, name in enumerate(category_names):
        fpr, tpr, _ = roc_curve(labels[:, idx], predictions[:, idx])
        auc_val = np.trapz(tpr, fpr)
        roc_data[name] = (fpr, tpr, auc_val)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curves")
    plt.legend()
    plt.grid(True)
    if output_file:
        plt.savefig(output_file)
    plt.close()
    return roc_data


# ---------------------------------------------------------------------------
# Threshold search -----------------------------------------------------------
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str = "f1",
    category_index: Optional[int] = None,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, Dict[int, float]]]:
    """Grid-search the threshold that maximises *metric* (accuracy or F1)."""
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    metric_obj = get_metric(metric)

    def _score(pred_col: np.ndarray, ref_col: np.ndarray, th: float) -> float:
        bin_col = (pred_col >= th).astype(int)
        return metric_obj.compute(predictions=bin_col, references=ref_col)[metric]

    if category_index is not None:
        best_th, best_val = 0.5, -1.0
        for th in thresholds:
            val = _score(predictions[:, category_index], labels[:, category_index], th)
            if val > best_val:
                best_th, best_val = th, val
        return {"threshold": best_th, f"best_{metric}": best_val}

    th_map: Dict[int, float] = {}
    metric_map: Dict[int, float] = {}
    for idx in range(labels.shape[1]):
        best_th, best_val = 0.5, -1.0
        for th in thresholds:
            val = _score(predictions[:, idx], labels[:, idx], th)
            if val > best_val:
                best_th, best_val = th, val
        th_map[idx] = best_th
        metric_map[idx] = best_val
    return {"thresholds": th_map, f"best_{metric}": metric_map}


# ---------------------------------------------------------------------------
# Reporting -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def generate_evaluation_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    *,
    category_names: List[str],
    threshold: float = 0.5,
    metrics_to_compute: Optional[Set[str]] = None,
    output_file: Optional[str] = "evaluation_report.txt",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute metrics + CM summary and optionally write a plain-text report."""
    metrics, cm_summary = compute_metrics(
        predictions,
        labels,
        threshold=threshold,
        category_names=category_names,
        metrics_to_compute=metrics_to_compute,
    )

    opt = find_optimal_threshold(predictions, labels, metric="f1")

    lines = [
        "===== Toxicity Classification Evaluation Report =====",
        f"Samples: {len(predictions)}",
        f"Categories: {', '.join(category_names)}\n",
        "-- Overall metrics --",
    ]
    for k in (
        "accuracy",
        "precision_micro",
        "precision_macro",
        "recall_micro",
        "recall_macro",
        "f1_micro",
        "f1_macro",
        "roc_auc",
        "average_precision",
        "hamming_loss",
    ):
        if k in metrics:
            lines.append(f"{k}: {metrics[k]:.4f}")

    lines += [
        "\n-- Confusion matrix totals --",
        f"TP: {cm_summary['total']['true_positives']}  FP: {cm_summary['total']['false_positives']}  "
        f"FN: {cm_summary['total']['false_negatives']}  TN: {cm_summary['total']['true_negatives']}",
        "\n-- Per-category metrics --",
    ]

    for idx, name in enumerate(category_names):
        lines.append(f"[{name}] tp={cm_summary[name]['true_positives']} fp={cm_summary[name]['false_positives']} "
                     f"fn={cm_summary[name]['false_negatives']} tn={cm_summary[name]['true_negatives']}")
        for suffix in ("precision", "recall", "f1"):
            key = f"{name}_{suffix}"
            if key in metrics:
                lines.append(f"    {suffix}: {metrics[key]:.4f}")
        if 'thresholds' in opt and idx in opt['thresholds']:
            lines.append(f"    optimal_threshold: {opt['thresholds'][idx]:.2f}")

    text = "\n".join(lines)
    print(text)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as fh:
            fh.write(text)
    return metrics, cm_summary


# ---------------------------------------------------------------------------
# Supported metrics introspection -------------------------------------------
# ---------------------------------------------------------------------------

def get_supported_metrics() -> Dict[str, str]:
    """Return a mapping of built-in metric keys to one-liner descriptions."""
    return {
        "accuracy": "Exact-match accuracy across all labels",
        "precision_micro": "Micro-averaged precision",
        "precision_macro": "Macro-averaged precision",
        "recall_micro": "Micro-averaged recall",
        "recall_macro": "Macro-averaged recall",
        "f1_micro": "Micro-averaged F1",
        "f1_macro": "Macro-averaged F1",
        "roc_auc": "Area under the ROC curve (macro)",
        "average_precision": "Area under the PR curve (macro)",
        "hamming_loss": "Fraction of misclassified labels",
    } 