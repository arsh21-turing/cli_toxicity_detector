#!/usr/bin/env python3
"""
Main entry point for the toxicity detection tool.

This script provides the command-line interface and orchestrates the workflow
for analyzing text for toxicity.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from enum import Enum

from logger import logger as log
from config_loader import create_default_config, load_config
from file_processor import process_file
from model_loader import predict_toxicity, get_model as load_model
from categories import ToxicityCategory
from color_utils import colorize_toxic, supports_color, colorize
from batch_processor import batch_process


class SortOrder(Enum):
    """Enum for confidence sort order."""
    HIGHEST_FIRST = "highest"
    LOWEST_FIRST = "lowest"


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


def check_confidence_filter(confidence: float, args: argparse.Namespace) -> tuple[bool, str]:
    """
    Check if a confidence value passes the confidence filtering criteria.
    
    Args:
        confidence: The confidence score to check (0.0-1.0)
        args: Command line arguments containing confidence filter settings
        
    Returns:
        Tuple of (passes_filter: bool, reason: str)
        - passes_filter: True if the confidence passes all filters
        - reason: String describing why it was filtered (empty if passes)
    """
    # Check single confidence filter
    if hasattr(args, 'confidence_filter') and args.confidence_filter is not None:
        try:
            threshold = float(args.confidence_filter)
            if confidence < threshold:
                return False, f"below threshold {threshold:.4f}"
        except (ValueError, TypeError):
            pass  # Treat invalid values as no filter
    
    # Check range filters
    if hasattr(args, 'min_confidence') and args.min_confidence is not None:
        try:
            min_threshold = float(args.min_confidence)
            if confidence < min_threshold:
                return False, f"below minimum {min_threshold:.4f}"
        except (ValueError, TypeError):
            pass
    
    if hasattr(args, 'max_confidence') and args.max_confidence is not None:
        try:
            max_threshold = float(args.max_confidence)
            if confidence > max_threshold:
                return False, f"above maximum {max_threshold:.4f}"
        except (ValueError, TypeError):
            pass
    
    return True, ""


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
# Confidence Explanation Functions ------------------------------------------
# ---------------------------------------------------------------------------

def generate_confidence_explanation(result: Dict[str, Any], text: str) -> Dict[str, Any]:
    """
    Generate a natural language explanation for the model's confidence level.
    
    Args:
        result: Prediction result dictionary
        text: Input text that was analyzed
        
    Returns:
        Dictionary containing explanation, factors, and suggestions
    """
    # Extract key information from the result - handle both formats
    category_results = result.get('category_results', {})
    is_toxic = result.get('is_toxic', False)
    
    # Handle legacy format with categories/probabilities
    if not category_results and 'categories' in result and 'probabilities' in result:
        # Convert legacy format to category_results format
        from unittest.mock import MagicMock
        category_results = {}
        categories = result['categories']
        probabilities = result['probabilities']
        
        for cat_name, is_above_threshold in categories.items():
            if cat_name != 'NON_TOXIC':  # Skip NON_TOXIC category
                mock_cat = MagicMock()
                mock_cat.name = cat_name.lower()
                category_results[mock_cat] = {
                    'score': probabilities.get(cat_name, 0.0),
                    'above_threshold': is_above_threshold
                }
    
    if not category_results:
        return {
            'explanation': 'Unable to generate explanation: insufficient data',
            'confidence_level': 'unknown',
            'primary_category': 'unknown',
            'confidence_score': 0.0,
            'confidence_factors': [],
            'uncertainty_factors': [],
            'improvement_suggestions': []
        }
    
    # Find the category with the highest score
    max_score = 0.0
    max_category = None
    for cat, data in category_results.items():
        score = data.get('score', 0.0)
        if score > max_score:
            max_score = score
            max_category = cat
    
    if max_category is None:
        return {
            'explanation': 'Unable to determine primary category',
            'confidence_level': 'unknown',
            'primary_category': 'unknown',
            'confidence_score': 0.0,
            'confidence_factors': [],
            'uncertainty_factors': [],
            'improvement_suggestions': []
        }
    
    # Determine confidence level
    if max_score >= 0.8:
        confidence_level = "high"
    elif max_score >= 0.6:
        confidence_level = "moderate"  
    elif max_score >= 0.4:
        confidence_level = "borderline"
    else:
        confidence_level = "low"
    
    # Extract category name safely
    primary_category = getattr(max_category, 'name', str(max_category))
    
    # Generate confidence factors and uncertainty factors
    confidence_factors = []
    uncertainty_factors = []
    
    # Check for toxic terms in the primary category
    if primary_category in ['hate', 'insult', 'profanity', 'threat', 'identity_attack', 'sexual']:
        toxic_terms = _find_toxic_terms(text, primary_category)
        if toxic_terms:
            terms_str = '", "'.join(toxic_terms)
            confidence_factors.append(f'The text contains terms strongly associated with {primary_category}: "{terms_str}".')
    
    # Check for explicit language patterns
    if _has_explicit_content(text):
        confidence_factors.append("The text contains explicit language that strongly indicates toxicity.")
    
    # Check for directive/imperative language
    if _has_directive_language(text):
        confidence_factors.append("The text uses directive language or imperatives, which are often associated with toxic content.")
    
    # Check for implicit/coded language (uncertainty factor)
    if _has_implicit_content(text):
        uncertainty_factors.append("The text may contain implicit or coded language that could affect classification confidence.")
    
    # Analyze confidence level for additional factors/uncertainties
    if confidence_level == "high":
        if max_score >= 0.95:
            confidence_factors.append(f"Very high confidence score ({max_score:.2f}) indicates clear toxicity patterns.")
        elif len(confidence_factors) == 0:
            confidence_factors.append("High confidence despite no obvious toxic markers may indicate subtle patterns the model detected.")
    
    elif confidence_level == "moderate":
        uncertainty_factors.append("Medium confidence suggests some ambiguity in the text that makes classification less certain.")
        if max_score < 0.7:
            uncertainty_factors.append("Score is in the moderate range, suggesting mixed signals in the content.")
    
    elif confidence_level == "borderline":
        uncertainty_factors.append("Borderline confidence indicates the text is difficult to classify definitively.")
        uncertainty_factors.append("This prediction would benefit from human review or additional context.")
    
    elif confidence_level == "low":
        uncertainty_factors.append("Low confidence suggests the model is very uncertain about this classification.")
        uncertainty_factors.append("The text may be ambiguous, context-dependent, or contain conflicting signals.")
    
    # Add fallback confidence factors if none found
    if not confidence_factors:
        confidence_factors.append("No specific confidence factors identified.")
        
    # Add fallback uncertainty factors if none found for non-high confidence
    if confidence_level != "high" and not uncertainty_factors:
        uncertainty_factors.append("No specific uncertainty factors identified.")
    
    # Generate improvement suggestions
    improvement_suggestions = []
    
    if confidence_level in ["borderline", "low"]:
        improvement_suggestions.append("Consider using Groq fallback (--allow-groq-fallback) for additional validation.")
        improvement_suggestions.append("Manual review recommended for borderline cases.")
        
    if confidence_level == "moderate":
        improvement_suggestions.append("Additional context or longer text samples might improve classification confidence.")
        
    if not improvement_suggestions:
        improvement_suggestions.append("No specific suggestions available for this prediction.")
    
    # Generate main explanation
    if confidence_level == "high":
        explanation = f"The model has high confidence ({max_score:.2f}) in classifying this content as {primary_category.upper()}."
    elif confidence_level == "moderate":
        explanation = f"The model has moderate confidence ({max_score:.2f}) in classifying this content as {primary_category.upper()}."
    elif confidence_level == "borderline":
        explanation = f"The model has borderline confidence ({max_score:.2f}) in classifying this content as {primary_category.upper()}. Consider using Groq fallback for additional validation."
    else:
        explanation = f"The model has low confidence ({max_score:.2f}) in classifying this content as {primary_category.upper()}. Human review strongly recommended."
    
    return {
        'explanation': explanation,
        'confidence_level': confidence_level,
        'primary_category': primary_category,
        'confidence_score': max_score,
        'confidence_factors': confidence_factors,
        'uncertainty_factors': uncertainty_factors,
        'improvement_suggestions': improvement_suggestions
    }


def _find_toxic_terms(text: str, category: str) -> List[str]:
    """Find terms in the text that contribute to toxicity for a given category."""
    text_lower = text.lower()
    found_terms = []
    
    # Category-specific terms
    category_terms = {
        'hate': ['hate', 'racist', 'bigot', 'prejudice', 'discrimination', 'stereotyp'],
        'insult': ['stupid', 'idiot', 'moron', 'dumb', 'loser', 'pathetic', 'ugly'],
        'profanity': ['fuck', 'shit', 'damn', 'ass', 'crap', 'bitch'],
        'threat': ['kill', 'hurt', 'destroy', 'attack', 'fight', 'beat', 'die', 'threat'],
        'identity_attack': ['retard', 'cripple', 'gay', 'faggot', 'queer', 'homo', 'tranny'],
        'sexual': ['porn', 'sex', 'masturbat', 'dick', 'cock', 'pussy'],
    }
    
    # General toxic markers (including patterns with *censoring*)
    general_patterns = [
        r'\bf+u+c*k+\b', r'\bs+h+[i1]+t+\b', r'\ba+s+s+h+o+l+e+\b', r'\bd+[i1]+c+k+\b',
        r'\bf\*+c*k+\b', r'\bs\*+[i1]+t+\b', r'\ba\*+s+\b'  # censored versions
    ]
    
    # Get terms for the specific category
    specific_terms = category_terms.get(category, [])
    
    # Check for category-specific terms
    for term in specific_terms:
        if term in text_lower:
            # Find the actual match in the original text for better display
            match = re.search(r'\b\w*' + re.escape(term) + r'\w*\b', text_lower)
            if match:
                found_terms.append(match.group(0))
    
    # Check for general toxic patterns
    for pattern in general_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            found_terms.append(match.group(0))
    
    # Get unique terms
    return list(set(found_terms))


def _has_directive_language(text: str) -> bool:
    """Check if text uses directive language or imperatives."""
    directive_patterns = [
        r'\byou should\b',
        r'\byou need to\b',
        r'\bgo\b.*\byourself\b',
        r'^[A-Z\s]*!',  # All caps followed by exclamation
        r'\b[a-zA-Z]+\s+yourself\b'
    ]
    
    # Check for multiple exclamation or question marks
    if re.search(r'[!?]{2,}', text):
        return True
    
    # Check for imperative start with verbs
    imperative_starts = [
        r'^\s*[Gg]o\b',
        r'^\s*[Ss]top\b',
        r'^\s*[Gg]et\b', 
        r'^\s*[Ss]hut\b',
        r'^\s*[Kk]ill\b',
        r'^\s*[Ff]uck\b',
        r'^\s*[Ll]eave\b'
    ]
    
    for pattern in directive_patterns + imperative_starts:
        if re.search(pattern, text, re.IGNORECASE):
            return True
            
    return False


def _has_explicit_content(text: str) -> bool:
    """Check if the text contains explicit language."""
    explicit_patterns = [
        r'\bfuck\b', r'\bfucking\b', r'\bfucked\b', r'\bfucker\b',
        r'\bshit\b', r'\bshitting\b', r'\bshitty\b',
        r'\basshole\b', r'\bass\b',
        r'\bbitch\b', r'\bbitching\b', r'\bbitchy\b',
        r'\bdick\b', r'\bdickie\b',
        r'\bpussy\b',
        r'\bnigger\b', r'\bnigga\b',
        # Patterns with asterisk censoring
        r'\bf\*+c*k+\b', r'\bf\*+k\b',
        r'\bs\*+[i1]+t+\b',
        r'\ba\*+s+\b',
        r'\bb\*+t+c+h+\b',
        r'\bd\*+c*k+\b',
        # Patterns with double asterisk censoring
        r'\ba\*\*hole\b',
        r'\bf\*\*k\b',
        r'\bs\*\*t\b'
    ]
    
    for pattern in explicit_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
            
    return False


def _has_implicit_content(text: str) -> bool:
    """Check if text contains implicit or coded language."""
    implicit_patterns = [
        r'\bsnowflake\b',
        r'\bthug\b',
        r'\burban\b',
        r'\bthose people\b',
        r'\byour kind\b',
        r'\bspecial person\b',
        r'\bwoke\b',
        r'\bbased\b'
    ]
    
    for pattern in implicit_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
            
    return False


def _display_confidence_explanation(explanation_obj: Dict[str, Any]) -> None:
    """
    Display confidence explanation in a user-friendly format.
    
    Args:
        explanation_obj: Explanation object from generate_confidence_explanation
    """
    # Get key elements from explanation
    explanation = explanation_obj.get('explanation', '')
    confidence_level = explanation_obj.get('confidence_level', '')
    confidence_factors = explanation_obj.get('confidence_factors', [])
    uncertainty_factors = explanation_obj.get('uncertainty_factors', [])
    suggestions = explanation_obj.get('improvement_suggestions', [])
    
    # Choose color based on confidence level
    if confidence_level == "high":
        color = "green"
        bold = True
    elif confidence_level == "moderate":
        color = "blue"
        bold = True
    elif confidence_level == "borderline":
        color = "yellow"
        bold = True
    else:  # low confidence
        color = "red"
        bold = False
    
    # Display header
    print("\n" + "=" * 60)
    print(colorize("CONFIDENCE EXPLANATION", "blue", bold=True))
    print("=" * 60)
    
    # Main explanation
    print(colorize(explanation, color, bold=bold))
    print()
    
    # Display confidence factors
    if confidence_factors:
        print(colorize("Confidence factors:", "blue"))
        for factor in confidence_factors:
            print(f" • {factor}")
        print()
    
    # Display uncertainty factors
    if uncertainty_factors:
        print(colorize("Uncertainty factors:", "yellow" if uncertainty_factors else "green"))
        for factor in uncertainty_factors:
            print(f" • {factor}")
        print()
    
    # Display suggestions
    if suggestions:
        print(colorize("Suggestions:", "green"))
        for suggestion in suggestions:
            print(f" • {suggestion}")
    
    print("=" * 60)


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
    mode_grp.add_argument("--batch", help="Process a file or directory in batch mode")
    mode_grp.add_argument("--stream", action="store_true", help="Process text from stdin in real-time streaming mode")
    mode_grp.add_argument("--evaluate", help="Run model evaluation on a validation dataset CSV file")
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

    # Optional Groq fallback -------------------------------------------------
    p.add_argument(
        "--allow-groq-fallback",
        action="store_true",
        help="Consult Groq for a second opinion when the local model is uncertain (gray-zone predictions)",
    )

    # Determine default from config for help text, but CLI will use config when flag absent
    try:
        from config_loader import load_config as _lc
        _cfg_default = _lc()
        _cfg_default_policy = _cfg_default.get("groq", {}).get("tie_policy", "prefer-groq")
    except Exception:
        _cfg_default_policy = "prefer-groq"

    p.add_argument(
        "--groq-tie-policy",
        choices=["prefer-groq", "prefer-local", "highest-confidence"],
        default=None,
        help=(
            "Tie-breaking rule when Groq and local model disagree. Choices: "
            "prefer-groq, prefer-local, highest-confidence. "
            f"If omitted, uses value from config (current default: {_cfg_default_policy})."
        ),
    )

    # Groq gray-zone bound overrides --------------------------------------
    p.add_argument(
        "--groq-lower-bound",
        type=float,
        help="Lower confidence bound that triggers Groq fallback (default 0.4)",
    )
    p.add_argument(
        "--groq-upper-bound",
        type=float,
        help="Upper confidence bound that triggers Groq fallback (default 0.6)",
    )
    
    # Confidence filtering -----------------------------------------------
    p.add_argument(
        "--confidence-filter",
        type=float,
        help="Only show results with confidence above this threshold (0.0-1.0)",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        help="Only show results with confidence above this minimum threshold (0.0-1.0)",
    )
    p.add_argument(
        "--max-confidence",
        type=float,
        help="Only show results with confidence below this maximum threshold (0.0-1.0)",
    )
    p.add_argument(
        "--confidence-explain",
        action="store_true",
        help="Provide natural language explanations for model confidence levels",
    )
    p.add_argument(
        "--confidence-sort",
        type=str,
        choices=["highest", "lowest"],
        help="Sort results by confidence level (highest or lowest first)",
    )
    
    # Evaluation options -------------------------------------------------
    p.add_argument(
        "--evaluation-output",
        help="Path to save evaluation results JSON file",
    )
    p.add_argument(
        "--no-headers",
        action="store_true",
        help="CSV validation dataset does not have headers",
    )
    p.add_argument(
        "--eval-threshold",
        type=float,
        default=0.5,
        help="Default threshold for all categories during evaluation (default: 0.5)",
    )
    p.add_argument(
        "--category-thresholds",
        help="JSON file with per-category thresholds for evaluation",
    )
    p.add_argument(
        "--optimize-thresholds",
        action="store_true",
        help="Optimize thresholds for best F1 score",
    )
    p.add_argument(
        "--min-threshold",
        type=float,
        default=0.1,
        help="Minimum threshold to test during optimization (default: 0.1)",
    )
    p.add_argument(
        "--max-threshold",
        type=float,
        default=0.9,
        help="Maximum threshold to test during optimization (default: 0.9)",
    )
    p.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        help="Step size for threshold testing during optimization (default: 0.05)",
    )
    p.add_argument(
        "--save-thresholds",
        action="store_true",
        help="Save optimal thresholds to configuration file",
    )
    p.add_argument(
        "--config-path",
        help="Path to configuration file for saving optimal thresholds",
    )
    
    # Visualization options ------------------------------------------------
    viz_group = p.add_argument_group("Visualization Options")
    viz_group.add_argument("--plot-dir", 
                         help="Directory where plot files will be saved")
    viz_group.add_argument("--plot-format", default="both",
                         choices=["png", "svg", "both"],
                         help="Format for plot files (png, svg, or both)")
    viz_group.add_argument("--plot-dpi", type=int, default=300,
                         help="DPI for raster image formats")
    viz_group.add_argument("--plot-pr-curves", action="store_true",
                         help="Generate precision-recall curves")
    viz_group.add_argument("--plot-threshold-sweep", action="store_true",
                         help="Generate threshold sweep plots")
    viz_group.add_argument("--plot-confusion-matrices", action="store_true",
                         help="Generate confusion matrix heatmaps")
    viz_group.add_argument("--plot-all", action="store_true",
                         help="Generate all available plot types")
    viz_group.add_argument("--selected-categories", 
                         help="Comma-separated list of categories to plot")
    
    # PDF report arguments
    viz_group.add_argument("--generate-pdf", action="store_true",
                         help="Generate a comprehensive PDF report")
    viz_group.add_argument("--pdf-path", 
                         help="Path for the PDF report (default: evaluation_report.pdf in plot-dir)")
    viz_group.add_argument("--report-title", default="Model Evaluation Report",
                         help="Custom title for the PDF report")
    viz_group.add_argument("--report-template",
                         help="Path to custom report template (optional)")
    
    # Inject threshold args after core flags to keep help tidy
    create_threshold_argument(p)

    # Additional mode_grp arguments
    mode_grp.add_argument(
        "--clear-groq-cache",
        action="store_true",
        help="Delete all cached Groq responses and exit",
    )

    mode_grp.add_argument(
        "--groq-cache-stats",
        action="store_true",
        help="Display statistics about the Groq API response cache and exit",
    )
    
    mode_grp.add_argument("--setup-config", action="store_true",
                        help="Launch the interactive configuration wizard")
    
    mode_grp.add_argument("--compare-models", action="store_true",
                        help="Compare multiple models on the same validation dataset")
    
    # Configuration wizard arguments (non-exclusive)
    wizard_group = p.add_argument_group("Configuration Wizard")
    wizard_group.add_argument("--config-output",
                            help="Path to save the generated config file (default: config.yaml)")
    wizard_group.add_argument("--wizard-defaults", 
                            choices=["high_precision", "high_recall", "balanced", "performance", "custom"],
                            help="Start with recommended defaults for specific use case profile")
    
    # Model comparison arguments
    compare_group = p.add_argument_group("Model Comparison")
    compare_group.add_argument("--model-paths", 
                             help="Comma-separated list of model paths to compare")
    compare_group.add_argument("--model-names", 
                             help="Optional comma-separated list of friendly names for the models")
    compare_group.add_argument("--config-paths", 
                             help="Optional comma-separated list of config paths for each model")
    compare_group.add_argument("--comparison-output", 
                             help="Directory to save comparison results and visualizations")
    compare_group.add_argument("--comparative-plots", action="store_true",
                             help="Generate comparative visualization plots across models")
    compare_group.add_argument("--significance-test", action="store_true",
                             help="Run statistical significance tests on model differences")
    compare_group.add_argument("--comparative-report", action="store_true",
                             help="Generate a PDF report comparing all models")
    compare_group.add_argument("--comparison-focus", choices=["precision", "recall", "f1"],
                             help="Focus comparison on specific metrics")

    # Performance monitoring arguments
    monitor_group = p.add_argument_group("Performance Monitoring")
    monitor_group.add_argument("--monitor", action="store_true",
                             help="Enable real-time monitoring dashboard during batch processing")
    monitor_group.add_argument("--monitor-interval", type=float, default=1.0,
                             help="Time interval in seconds between dashboard updates")
    monitor_group.add_argument("--monitor-metrics", 
                             help="Comma-separated list of metrics to display (default: all)")
    monitor_group.add_argument("--monitor-log", 
                             help="Save monitoring data to log file")
    monitor_group.add_argument("--monitor-port", type=int, default=8050,
                             help="Port to use for web-based monitoring dashboard")
    monitor_group.add_argument("--monitor-headless", action="store_true",
                             help="Run monitoring in headless mode (no UI, metrics logging only)")

    return p


# ---------------------------------------------------------------------------
# Single-sentence path --------------------------------------------------------
# ---------------------------------------------------------------------------

def _process_single(text: str, *, cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    threshold = args.threshold if args.threshold is not None else cfg["model"]["threshold"]
    model_name = args.model if args.model else cfg["model"]["name"]

    tie_policy_active = args.groq_tie_policy if args.groq_tie_policy is not None else cfg.get("groq", {}).get("tie_policy", "prefer-groq")

    res = predict_toxicity(
        texts=[text],
        thresholds={cat.name: threshold for cat in ToxicityCategory},
        model_name=model_name,
        show_progress=False,
        allow_groq_fallback=args.allow_groq_fallback,
        gray_min=args.groq_lower_bound,
        gray_max=args.groq_upper_bound,
        tie_policy=tie_policy_active,
    )[0]

    if args.probabilities:
        # Try to get raw probabilities from legacy helper for backwards-compat
        try:
            from model_loader import predict_proba  # type: ignore
            prob_map = predict_proba(text)[0]  # type: ignore[index]
        except Exception:
            prob_map = {}
        res["raw_probabilities"] = prob_map

    # Check confidence filtering for single text
    scores = {cat.name: data['score'] for cat, data in res['category_results'].items()}
    max_score = max(scores.values()) if scores else 0.0
    passes_filter, filter_reason = check_confidence_filter(max_score, args)
    
    if not passes_filter:
        if args.json:
            payload: Dict[str, Any] = {
                "text": text,
                "filtered": True,
                "filter_reason": filter_reason,
                "confidence": max_score,
                "timestamp": datetime.now().isoformat(),
            }
            # Add confidence explanation to JSON if requested
            if hasattr(args, 'confidence_explain') and args.confidence_explain:
                explanation_obj = generate_confidence_explanation(res, text)
                payload["confidence_explanation"] = explanation_obj
            _emit_json(payload, args.output)
        else:
            print(f"Result filtered: {filter_reason} (confidence: {max_score:.4f})")
            # Add confidence explanation for filtered results if requested
            if hasattr(args, 'confidence_explain') and args.confidence_explain:
                explanation_obj = generate_confidence_explanation(res, text)
                _display_confidence_explanation(explanation_obj)
        return res
    
    if args.json:
        # Build categories dict from category_results
        categories = {}
        probabilities = {}
        for cat, data in res.get('category_results', {}).items():
            cat_name = cat.name if hasattr(cat, 'name') else str(cat)
            categories[cat_name] = data.get('above_threshold', False)
            probabilities[cat_name] = data.get('score', 0.0)
        
        payload: Dict[str, Any] = {
            "text": text,
            "is_toxic": res["is_toxic"],
            "categories": categories,
            "probabilities": probabilities,
            "confidence": max_score,
            "timestamp": datetime.now().isoformat(),
        }
        if args.probabilities:
            payload["raw_probabilities"] = res.get("raw_probabilities", {})
        # Add confidence explanation to JSON if requested
        if hasattr(args, 'confidence_explain') and args.confidence_explain:
            explanation_obj = generate_confidence_explanation(res, text)
            payload["confidence_explanation"] = explanation_obj
        _emit_json(payload, args.output)
    else:
        _print_human_single(res, args)
    return res


def _print_human_single(res: Dict[str, Any], args: argparse.Namespace) -> None:
    verdict = "TOXIC" if res["is_toxic"] else "OK"
    print(f"Result: {verdict}")

    if res["is_toxic"]:
        # Extract category names from category_results where above_threshold is True
        detected = [
            cat.name for cat, data in res["category_results"].items() 
            if data.get("above_threshold", False)
        ]
        print("Detected categories:", ", ".join(detected))

    if args.probabilities:
        # Show scores from category_results
        for cat, data in res.get("category_results", {}).items():
            score = data.get("score", 0.0)
            print(f"  {cat.name}: {score:.4f}")
    
    # Add confidence explanation if requested
    if hasattr(args, 'confidence_explain') and args.confidence_explain:
        explanation_obj = generate_confidence_explanation(res, res.get('text', ''))
        _display_confidence_explanation(explanation_obj)


# ---------------------------------------------------------------------------
# JSON helper ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def _json_serialize(obj):
    """Custom JSON serializer to handle ToxicityCategory and other non-serializable objects."""
    if hasattr(obj, 'name'):
        return obj.name
    elif hasattr(obj, '__str__'):
        return str(obj)
    else:
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def _emit_json(obj: Any, path: Optional[str] = None) -> None:
    data = json.dumps(obj, indent=2, ensure_ascii=False, default=_json_serialize)
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
# Batch processing helpers ---------------------------------------------------
# ---------------------------------------------------------------------------

def display_batch_results(results: Dict[str, Any], args: argparse.Namespace) -> None:
    """Display comprehensive batch processing results with full probability distributions,
    Groq fallback tracking, and detailed toxicity metrics.
    
    Args:
        results: Dictionary containing batch processing results
        args: Command line arguments
    """

    if args.json:
        return  # Nothing to do – JSON already printed

    total_files = results.get("total_files", 0)
    toxic_files = results.get("toxic_files", 0)
    total_sentences = results.get("total_sentences", 0)
    toxic_sentences = results.get("toxic_sentences", 0)
    
    # Get confidence filtering statistics
    displayed_sentences = results.get("displayed_sentences", total_sentences)
    filtered_sentences = results.get("filtered_sentences", 0)
    below_range_sentences = results.get("below_range_sentences", 0)
    above_range_sentences = results.get("above_range_sentences", 0)
    
    # Get enhanced metrics
    toxicity_metrics = results.get("toxicity_metrics", {})
    avg_overall_score = toxicity_metrics.get("avg_overall_score", results.get("avg_overall_toxicity", 0.0))
    max_toxicity_score = toxicity_metrics.get("max_toxicity_score", 0.0)
    
    # Get Groq metrics
    groq_metrics = results.get("groq_metrics", {})
    total_usage_count = groq_metrics.get("total_usage_count", results.get("groq_fallback_count", 0))
    override_count = groq_metrics.get("override_count", 0)
    effectiveness_score = groq_metrics.get("effectiveness_score", 0.0)
    avg_confidence_improvement = groq_metrics.get("avg_confidence_improvement", 0.0)

    # Calculate percentages
    file_pct = (toxic_files / total_files * 100.0) if total_files else 0.0
    sent_pct = (toxic_sentences / total_sentences * 100.0) if total_sentences else 0.0
    
    # Determine toxicity level for color coding based on average overall score
    if avg_overall_score >= 0.7:
        toxicity_level = "high"
        color = "red"
    elif avg_overall_score >= 0.4:
        toxicity_level = "medium"
        color = "yellow"
    elif avg_overall_score >= 0.2:
        toxicity_level = "low"
        color = "blue"
    else:
        toxicity_level = "minimal"
        color = "green"

    # Display summary header
    print("\n" + "=" * 80)
    print(colorize("BATCH PROCESSING SUMMARY", "blue", bold=True))
    print("=" * 80)
    
    # Display confidence filter settings if any are active
    confidence_filter_settings = []
    if hasattr(args, 'confidence_filter') and args.confidence_filter is not None:
        try:
            confidence_value = float(args.confidence_filter)
            confidence_filter_settings.append(f"above {confidence_value:.4f}")
        except (ValueError, TypeError):
            confidence_filter_settings.append(f"above {args.confidence_filter}")
    
    if hasattr(args, 'min_confidence') and args.min_confidence is not None:
        try:
            min_value = float(args.min_confidence)
            confidence_filter_settings.append(f"above {min_value:.4f}")
        except (ValueError, TypeError):
            confidence_filter_settings.append(f"above {args.min_confidence}")
    
    if hasattr(args, 'max_confidence') and args.max_confidence is not None:
        try:
            max_value = float(args.max_confidence)
            confidence_filter_settings.append(f"below {max_value:.4f}")
        except (ValueError, TypeError):
            confidence_filter_settings.append(f"below {args.max_confidence}")
    
    if confidence_filter_settings:
        filter_desc = " and ".join(confidence_filter_settings)
        print(colorize(f"Confidence filter: showing results {filter_desc}", "blue", bold=True))

    # Display if confidence explanation is enabled
    if hasattr(args, 'confidence_explain') and args.confidence_explain:
        print(colorize("Confidence explanation: Enabled (providing natural language explanations)", "blue", bold=True))
    
    # Display if confidence sorting is enabled
    if hasattr(args, 'confidence_sort') and args.confidence_sort is not None:
        print(colorize(f"Confidence sorting: {args.confidence_sort} first", "blue", bold=True))
        print(f"Results will be sorted by confidence level ({args.confidence_sort} first)")

    # Display overall statistics
    print(f"Files processed: {total_files}")
    toxic_files_str = f"Toxic files: {toxic_files}/{total_files} ({file_pct:.1f}%)"
    print(colorize(toxic_files_str, color, bold=(toxicity_level in ["high", "medium"])))

    # Display sentence statistics with confidence filtering information
    print(f"\nTotal sentences: {total_sentences}")
    
    # Show confidence filtering statistics if applicable
    has_confidence_filter = (hasattr(args, 'confidence_filter') and args.confidence_filter is not None) or \
                          (hasattr(args, 'min_confidence') and args.min_confidence is not None) or \
                          (hasattr(args, 'max_confidence') and args.max_confidence is not None)
    
    if has_confidence_filter and filtered_sentences > 0:
        filtered_percent = (filtered_sentences / total_sentences) * 100
        print(f"Sentences displayed: {displayed_sentences} ({(100 - filtered_percent):.1f}%)")
        print(f"Sentences filtered by confidence: {filtered_sentences} ({filtered_percent:.1f}%)")
        
        # Show breakdown of range filtering if applicable
        if below_range_sentences > 0 or above_range_sentences > 0:
            if below_range_sentences > 0:
                below_percent = (below_range_sentences / total_sentences) * 100
                print(f"  - Below minimum confidence: {below_range_sentences} ({below_percent:.1f}%)")
            if above_range_sentences > 0:
                above_percent = (above_range_sentences / total_sentences) * 100
                print(f"  - Above maximum confidence: {above_range_sentences} ({above_percent:.1f}%)")
    
    # Calculate toxic percentage based on displayed sentences for consistency
    effective_sent_pct = (toxic_sentences / displayed_sentences * 100.0) if displayed_sentences else 0.0
    toxic_sentences_str = f"Toxic sentences: {toxic_sentences}/{displayed_sentences} ({effective_sent_pct:.1f}%)"
    print(colorize(toxic_sentences_str, color, bold=(toxicity_level in ["high", "medium"])))

    # Display overall toxicity metrics
    print("\nToxicity Metrics:")
    print(colorize(f"  Average toxicity score: {avg_overall_score:.4f} - {toxicity_level.upper()}", 
                  color, bold=(toxicity_level in ["high", "medium"])))
    print(f"  Maximum toxicity score: {max_toxicity_score:.4f}")

    # Display toxicity distribution if available
    toxicity_distribution = toxicity_metrics.get("toxicity_distribution", {})
    if toxicity_distribution and 'mean' in toxicity_distribution:
        print("\nToxicity Distribution:")
        print(f"  Mean: {toxicity_distribution.get('mean', 0.0):.4f}")
        print(f"  Median: {toxicity_distribution.get('median', 0.0):.4f}")
        
        if 'percentiles' in toxicity_distribution:
            percentiles = toxicity_distribution['percentiles']
            print(f"  75th percentile: {percentiles.get('75th', 0.0):.4f}")
            print(f"  90th percentile: {percentiles.get('90th', 0.0):.4f}")

    # Display per-category metrics if available
    per_category_metrics = toxicity_metrics.get("per_category_metrics", {})
    if per_category_metrics:
        print("\nCategory Metrics:")
        for category, metrics in sorted(
            per_category_metrics.items(), 
            key=lambda x: x[1].get('avg_score', 0.0),
            reverse=True
        ):
            category_avg = metrics.get('avg_score', 0.0)
            category_max = metrics.get('max_score', 0.0)
            
            category_color = "red" if category_avg >= 0.7 else "yellow" if category_avg >= 0.4 else "blue"
            print(colorize(f"  {category}: avg={category_avg:.4f}, max={category_max:.4f}", category_color))

    # Display Groq usage metrics
    if total_usage_count > 0:
        print("\nGroq API Usage:")
        print(f"  Total calls: {total_usage_count}")
        print(f"  Classification changes: {override_count} ({effectiveness_score*100:.1f}% effectiveness)")
        if avg_confidence_improvement > 0:
            print(f"  Average confidence improvement: {avg_confidence_improvement:.4f}")

    # Collect category distribution across all files
    all_categories = {}
    for file_path, file_result in results.get("file_results", {}).items():
        for category, count in file_result.get("category_counts", {}).items():
            all_categories[category] = all_categories.get(category, 0) + count

    if all_categories:
        print("\nCategory Distribution:")
        for category, count in sorted(all_categories.items(), key=lambda x: x[1], reverse=True):
            category_percent = (count / toxic_sentences * 100) if toxic_sentences > 0 else 0
            print(f"  - {category}: {count} ({category_percent:.1f}% of toxic sentences)")

    # If verbose, display detailed file information
    if getattr(args, "verbose", False):
        print("\n" + "=" * 80)
        print(colorize("FILE DETAILS", "blue", bold=True))
        print("=" * 80)
        
        # Sort files by appropriate criteria
        sorted_files = []
        for file_path, file_result in results.get("file_results", {}).items():
            toxicity_profile = file_result.get('toxicity_profile', {})
            toxicity_score = toxicity_profile.get('overall_score', file_result.get("overall_toxicity_score", 0.0))
            
            # Calculate max confidence for this file if confidence sorting is enabled
            max_confidence = 0.0
            if hasattr(args, 'confidence_sort') and args.confidence_sort is not None:
                sentences = file_result.get('sentences', [])
                for sentence in sentences:
                    if 'category_results' in sentence:
                        # Handle both category objects and string keys
                        scores = {}
                        for cat, data in sentence['category_results'].items():
                            cat_name = cat.name if hasattr(cat, 'name') else str(cat)
                            scores[cat_name] = data['score']
                    else:
                        scores = sentence.get('probabilities', sentence.get('scores', {}))
                    sentence_max_score = max(scores.values()) if scores else 0.0
                    if sentence_max_score > max_confidence:
                        max_confidence = sentence_max_score
            
            sorted_files.append((file_path, file_result, toxicity_score, max_confidence))
        
        # Sort according to settings
        if hasattr(args, 'confidence_sort') and args.confidence_sort is not None:
            # Sort by confidence
            sort_order = SortOrder(args.confidence_sort)
            sorted_files.sort(
                key=lambda x: x[3],  # max_confidence
                reverse=(sort_order == SortOrder.HIGHEST_FIRST)
            )
        else:
            # Default sort by toxicity score, highest first
            sorted_files.sort(key=lambda x: x[2], reverse=True)
        
        # Display details for each file
        for i, (file_path, file_result, toxicity_score, max_confidence) in enumerate(sorted_files):
            toxic_sentences = file_result.get("toxic_sentences", 0)
            total_sentences = file_result.get("total_sentences", 0)
            sentence_percent = (toxic_sentences / total_sentences * 100) if total_sentences > 0 else 0
            
            # Determine color based on toxicity score
            if toxicity_score >= 0.7:
                file_color = "red"
                bold = True
            elif toxicity_score >= 0.4:
                file_color = "yellow"
                bold = True
            elif toxicity_score >= 0.2:
                file_color = "blue"
                bold = False
            else:
                file_color = "green"
                bold = False
            
            # Display basic file info
            print(f"\n{i+1}. {file_path}")
            print(colorize(f"   Toxicity score: {toxicity_score:.4f}", file_color, bold=bold))
            
            # Display max confidence if sorting by confidence
            if hasattr(args, 'confidence_sort') and args.confidence_sort is not None:
                conf_color = "red" if max_confidence >= 0.7 else "yellow" if max_confidence >= 0.5 else "blue"
                conf_display = colorize(f"{max_confidence:.4f}", conf_color)
                print(f"   Max confidence: {conf_display}")
            
            print(f"   Sentences: {toxic_sentences}/{total_sentences} toxic ({sentence_percent:.1f}%)")
    
    print("\n" + "=" * 80)
    
    # If output directory was specified, display where results were saved
    if hasattr(args, 'output') and args.output:
        print(f"Detailed results saved to: {args.output}")
        print("  - batch_summary.json: Overall statistics")
        print("  - toxicity_report.json: Files categorized by toxicity level")
        print("  - category_distribution.json: Toxicity breakdown by category")
        print("  - groq_usage_report.json: Detailed Groq API usage statistics")
        print("  - *.result.json: Detailed results for each processed file")
        print("=" * 80)


def handle_stream_processing(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Process text input from stdin in real-time streaming mode.
    
    Args:
        args: Command line arguments containing model, threshold, and display options
        
    Returns:
        Dictionary containing streaming session statistics
    """
    print("\n=== Real-time Toxicity Analysis Streaming Mode ===")
    print("Type text and press Enter for immediate analysis.")
    print("Press Ctrl+D (or Ctrl+Z on Windows) to end the session.")
    
    # Display confidence filter if enabled
    confidence_filter_info = []
    if hasattr(args, 'confidence_filter') and args.confidence_filter is not None:
        try:
            confidence_value = float(args.confidence_filter)
            confidence_filter_info.append(f"above {confidence_value:.4f}")
        except (ValueError, TypeError):
            confidence_filter_info.append(f"above {args.confidence_filter}")
    
    if hasattr(args, 'min_confidence') and args.min_confidence is not None:
        try:
            min_value = float(args.min_confidence)
            confidence_filter_info.append(f"above {min_value:.4f}")
        except (ValueError, TypeError):
            confidence_filter_info.append(f"above {args.min_confidence}")
    
    if hasattr(args, 'max_confidence') and args.max_confidence is not None:
        try:
            max_value = float(args.max_confidence)
            confidence_filter_info.append(f"below {max_value:.4f}")
        except (ValueError, TypeError):
            confidence_filter_info.append(f"below {args.max_confidence}")
    
    if confidence_filter_info:
        filter_desc = " and ".join(confidence_filter_info)
        print(f"Confidence filter: showing results {filter_desc}")
    
    # Show if confidence explanation is enabled
    if hasattr(args, 'confidence_explain') and args.confidence_explain:
        print("Confidence explanation: Enabled (providing natural language explanations)")
    
    # Show if confidence sorting is enabled
    if hasattr(args, 'confidence_sort') and args.confidence_sort is not None:
        print(f"Confidence sorting: {args.confidence_sort} first")
        print("Note: In streaming mode, sort applies to summary display")
        
    print("=" * 50)
    
    # Load model and configuration
    cfg = load_config()
    model_name = args.model or cfg.get("model", {}).get("name", "unitary/toxic-bert")
    
    # Apply threshold configuration
    thresholds = cfg.get("thresholds", {})
    if args.threshold is not None:
        # Apply global threshold to all categories
        for cat in ToxicityCategory:
            thresholds[cat.name] = args.threshold
    
    # Apply per-category threshold overrides
    threshold_overrides = parse_threshold_args(args)
    if threshold_overrides:
        thresholds.update(threshold_overrides)
    
    # Initialize session statistics
    stats = {
        'total_lines': 0,
        'displayed_lines': 0,  # Track lines that pass confidence filter
        'filtered_lines': 0,   # Track lines filtered by confidence threshold
        'below_range_lines': 0,  # Track lines below min confidence
        'above_range_lines': 0,  # Track lines above max confidence
        'toxic_lines': 0,
        'categories': {},
        'groq_usage': {
            'total': 0,
            'overrides': 0
        },
        'session_start': datetime.now().isoformat(),
        'results': []
    }
    
    try:
        # Main streaming loop
        while True:
            try:
                # Prompt and get input
                if not args.json and not args.quiet:
                    prompt = colorize("> ", "blue", bold=True) if supports_color() and not getattr(args, "no_color", False) else "> "
                    line = input(prompt)
                else:
                    line = input()
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Process the line
                stats['total_lines'] += 1
                
                # Get toxicity prediction
                tie_policy_active = args.groq_tie_policy if args.groq_tie_policy is not None else cfg.get("groq", {}).get("tie_policy", "prefer-groq")
                
                result = predict_toxicity(
                    texts=[line],
                    thresholds=thresholds,
                    model_name=model_name,
                    show_progress=False,
                    allow_groq_fallback=args.allow_groq_fallback,
                    gray_min=args.groq_lower_bound,
                    gray_max=args.groq_upper_bound,
                    tie_policy=tie_policy_active,
                )[0]
                
                # Store result
                stats['results'].append(result)
                
                # Get maximum confidence score for filtering
                scores = {cat.name: data['score'] for cat, data in result['category_results'].items()}
                max_score = max(scores.values()) if scores else 0.0
                
                # Check if result passes confidence filter
                passes_filter, filter_reason = check_confidence_filter(max_score, args)
                
                if passes_filter:
                    display_result = True
                else:
                    display_result = False
                    stats['filtered_lines'] += 1
                    
                    # Track specific reason for filtering for range filters
                    if "below minimum" in filter_reason:
                        stats['below_range_lines'] += 1
                    elif "above maximum" in filter_reason:
                        stats['above_range_lines'] += 1
                
                # Update statistics for toxic content (regardless of filter)
                if result['is_toxic']:
                    stats['toxic_lines'] += 1
                    
                    # Update category counts
                    for category, data in result['category_results'].items():
                        if data['above_threshold']:
                            cat_name = category.name
                            stats['categories'][cat_name] = stats['categories'].get(cat_name, 0) + 1
                
                # Track Groq usage
                if result.get('groq_fallback_used', False):
                    stats['groq_usage']['total'] += 1
                    if result.get('groq_changed_classification', False):
                        stats['groq_usage']['overrides'] += 1
                
                # Display the result if it passes confidence filter
                if display_result:
                    stats['displayed_lines'] += 1
                    _display_stream_result(result, stats, args)
                else:
                    # Show minimal info for filtered results
                    if args.verbose:
                        filter_msg = f"Result filtered (confidence: {max_score:.4f}, {filter_reason})"
                        print(colorize(filter_msg, "yellow") if supports_color() and not getattr(args, "no_color", False) else filter_msg)
                        
                        # Add confidence explanation for filtered results if requested
                        if hasattr(args, 'confidence_explain') and args.confidence_explain:
                            explanation_obj = generate_confidence_explanation(result, line)
                            _display_confidence_explanation(explanation_obj)
                        
                        print("-" * 50)
                
            except EOFError:
                # End of input (Ctrl+D)
                break
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nStream processing interrupted.")
    
    # Display final statistics
    _display_stream_summary(stats, args)
    
    # Add end timestamp
    stats['session_end'] = datetime.now().isoformat()
    
    return stats


def _display_stream_result(result: Dict[str, Any], stats: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Display real-time result for a single stream input line.
    
    Args:
        result: Dictionary containing toxicity prediction result
        stats: Dictionary containing running session statistics
        args: Command line arguments that may affect display formatting
    """
    if args.json:
        # JSON output mode
        json_result = {
            'text': result['text'],
            'is_toxic': result['is_toxic'],
            'categories': {
                cat.name: {
                    "score": data["score"],
                    "above_threshold": data["above_threshold"],
                    "threshold": data["threshold"],
                }
                for cat, data in result['category_results'].items()
            },
            'scores': {cat.name: data['score'] for cat, data in result['category_results'].items()},
            'most_probable_category': result['most_probable_category'].name,
            'timestamp': datetime.now().isoformat()
        }
        if result.get('groq_fallback_used', False):
            json_result['groq_fallback_used'] = True
            if result.get('groq_changed_classification', False):
                json_result['groq_changed_classification'] = True
        
        # Add confidence explanation to JSON if requested
        if hasattr(args, 'confidence_explain') and args.confidence_explain:
            explanation_obj = generate_confidence_explanation(result, result.get('text', ''))
            json_result['confidence_explanation'] = explanation_obj
        
        print(json.dumps(json_result, default=_json_serialize))
        return
    
    # Determine toxicity status and color
    is_toxic = result.get('is_toxic', False)
    
    # Find max score and category
    max_score = 0.0
    max_category = None
    scores = {}
    for category, data in result.get('category_results', {}).items():
        scores[category.name] = data['score']
        if data['score'] > max_score:
            max_score = data['score']
            max_category = category.name
    
    # Determine color based on max score
    use_color = supports_color() and not getattr(args, "no_color", False)
    
    if max_score >= 0.7:
        color = "red"
        status = "HIGH TOXICITY"
        bold = True
    elif max_score >= 0.5:
        color = "red"
        status = "TOXIC"
        bold = True
    elif max_score >= 0.4:
        color = "yellow"
        status = "BORDERLINE"
        bold = True
    elif max_score >= 0.2:
        color = "blue"
        status = "LOW TOXICITY"
        bold = False
    else:
        color = "green"
        status = "SAFE"
        bold = False
    
    # Display the primary result
    if use_color:
        status_display = colorize(status, color, bold=bold)
    else:
        status_display = status
    
    # Add Groq indicator if used
    if result.get('groq_fallback_used', False):
        if result.get('groq_changed_classification', False):
            groq_indicator = colorize(" (reclassified by Groq)", "yellow", bold=True) if use_color else " (reclassified by Groq)"
        else:
            groq_indicator = " (via Groq)"
        status_display += groq_indicator
    
    print(f"Result: {status_display}")
    
    # Display confidence
    use_color = supports_color() and not getattr(args, "no_color", False)
    if use_color:
        confidence_display = colorize(f"Confidence: {max_score:.4f}", color, bold=(max_score >= 0.7))
    else:
        confidence_display = f"Confidence: {max_score:.4f}"
    print(f"{confidence_display} (category: {max_category})")
    
    # Show top 3 categories with scores
    if args.verbose or args.probabilities:
        print("Top categories:")
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for category, score in sorted_scores:
            if use_color:
                category_color = "red" if score >= 0.7 else "yellow" if score >= 0.5 else "blue"
                score_display = colorize(f'{score:.4f}', category_color)
            else:
                score_display = f'{score:.4f}'
            print(f"  - {category}: {score_display}")
    elif max_category and max_score >= 0.2:
        # Just show the top category for non-verbose mode
        if use_color:
            category_color = "red" if max_score >= 0.7 else "yellow" if max_score >= 0.5 else "blue"
            score_display = colorize(f'{max_score:.4f}', category_color)
        else:
            score_display = f'{max_score:.4f}'
        print(f"  Top: {max_category} ({score_display})")
    
    # Show running statistics
    if not args.quiet:
        total_displayed = stats.get('displayed_lines', stats.get('total_lines', 0))
        total_filtered = stats.get('filtered_lines', 0)
        below_range = stats.get('below_range_lines', 0)
        above_range = stats.get('above_range_lines', 0)
        toxic_displayed = stats['toxic_lines']
        
        # Calculate percentages based on displayed lines
        toxic_percent = (toxic_displayed / total_displayed) * 100 if total_displayed > 0 else 0
        
        stats_text = f"Running stats: {toxic_displayed}/{total_displayed} toxic lines ({toxic_percent:.1f}%)"
        if total_filtered > 0:
            filtered_percent = (total_filtered / stats['total_lines']) * 100
            stats_text += f", {total_filtered} filtered ({filtered_percent:.1f}%)"
            
            # Add breakdown of range filtering if applicable
            if below_range > 0 or above_range > 0:
                range_details = []
                if below_range > 0:
                    range_details.append(f"{below_range} below range")
                if above_range > 0:
                    range_details.append(f"{above_range} above range")
                stats_text += f" [{', '.join(range_details)}]"
        
        if use_color:
            if toxic_percent >= 50:
                stats_display = colorize(stats_text, "red")
            elif toxic_percent >= 20:
                stats_display = colorize(stats_text, "yellow")
            else:
                stats_display = colorize(stats_text, "green")
        else:
            stats_display = stats_text
            
        print(stats_display)
        
    # Add confidence explanation if requested
    if hasattr(args, 'confidence_explain') and args.confidence_explain:
        explanation_obj = generate_confidence_explanation(result, result.get('text', ''))
        _display_confidence_explanation(explanation_obj)
    
    print("-" * 50)


def _display_stream_summary(stats: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Display summary statistics at the end of a streaming session.
    
    Args:
        stats: Dictionary containing session statistics
        args: Command line arguments that may affect display formatting
    """
    total_lines = stats.get('total_lines', 0)
    displayed_lines = stats.get('displayed_lines', 0)
    filtered_lines = stats.get('filtered_lines', 0)
    below_range_lines = stats.get('below_range_lines', 0)
    above_range_lines = stats.get('above_range_lines', 0)
    toxic_lines = stats.get('toxic_lines', 0)
    
    print("\n" + "=" * 50)
    header = "STREAMING SESSION SUMMARY"
    use_color = supports_color() and not getattr(args, "no_color", False)
    
    if use_color:
        print(colorize(header, "blue", bold=True))
    else:
        print(header)
    print("=" * 50)
    
    # If no lines were processed
    if total_lines == 0:
        print("No text was analyzed during this session.")
        print("=" * 50)
        return
    
    # Calculate percentages - fallback to total_lines if displayed_lines is not set
    effective_displayed_lines = displayed_lines if displayed_lines > 0 else total_lines
    toxic_percent = (toxic_lines / effective_displayed_lines) * 100 if effective_displayed_lines > 0 else 0
    
    # Display basic stats
    print(f"Total lines processed: {total_lines}")
    
    # Display confidence filter stats if applicable
    if filtered_lines > 0:
        filtered_percent = (filtered_lines / total_lines) * 100
        print(f"Lines displayed: {displayed_lines} ({(100 - filtered_percent):.1f}%)")
        print(f"Lines filtered by confidence: {filtered_lines} ({filtered_percent:.1f}%)")
        
        # Show breakdown of range filtering if applicable
        if below_range_lines > 0 or above_range_lines > 0:
            if below_range_lines > 0:
                below_percent = (below_range_lines / total_lines) * 100
                print(f"  - Below minimum confidence: {below_range_lines} ({below_percent:.1f}%)")
            if above_range_lines > 0:
                above_percent = (above_range_lines / total_lines) * 100
                print(f"  - Above maximum confidence: {above_range_lines} ({above_percent:.1f}%)")
    
    # Display toxicity stats based on displayed lines
    toxic_lines_str = f"Toxic lines: {toxic_lines}/{effective_displayed_lines} ({toxic_percent:.1f}%)"
    
    if use_color:
        if toxic_percent >= 50:
            print(colorize(toxic_lines_str, "red", bold=True))
        elif toxic_percent >= 20:
            print(colorize(toxic_lines_str, "yellow", bold=True))
        elif toxic_percent > 0:
            print(colorize(toxic_lines_str, "blue"))
        else:
            print(colorize(toxic_lines_str, "green"))
    else:
        print(toxic_lines_str)
    
    # Display category distribution if any toxic lines were found
    if toxic_lines > 0 and stats.get('categories'):
        print("\nCategory distribution:")
        for category, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True):
            category_percent = (count / toxic_lines) * 100
            print(f"  - {category}: {count} ({category_percent:.1f}% of toxic lines)")
    
    # Display top entries sorted by confidence if requested
    if hasattr(args, 'confidence_sort') and args.confidence_sort is not None:
        results = stats.get('results', [])
        if results:
            # Filter to displayed results only (those that passed confidence filter)
            if filtered_lines > 0:
                displayed_results = []
                for res in results:
                    # Get max confidence from result
                    if 'category_results' in res:
                        scores = {cat.name: data['score'] for cat, data in res['category_results'].items()}
                    else:
                        scores = res.get('probabilities', res.get('scores', {}))
                    max_score = max(scores.values()) if scores else 0.0
                    passes_filter, _ = check_confidence_filter(max_score, args)
                    if passes_filter:
                        displayed_results.append(res)
            else:
                displayed_results = results
            
            # Sort results by confidence (max score)
            sort_order = SortOrder(args.confidence_sort)
            sorted_results = sorted(
                displayed_results,
                key=lambda x: max((x.get('probabilities', x.get('scores', {})) or {}).values() or [0]),
                reverse=(sort_order == SortOrder.HIGHEST_FIRST)
            )
            
            # Display top 5 sorted results
            display_limit = min(5, len(sorted_results))
            if display_limit > 0:
                print(f"\nTop {display_limit} entries by confidence ({args.confidence_sort} first):")
                
                for i, result in enumerate(sorted_results[:display_limit]):
                    # Get max score and category
                    scores = result.get('probabilities', result.get('scores', {}))
                    max_score = max(scores.values()) if scores else 0.0
                    max_category = max(scores.items(), key=lambda x: x[1])[0] if scores else None
                    
                    # Get truncated text
                    text = result.get('text', '')
                    if len(text) > 60:
                        text = text[:57] + "..."
                    
                    # Determine color based on score
                    if use_color:
                        if max_score >= 0.7:
                            score_color = "red"
                        elif max_score >= 0.5:
                            score_color = "yellow"
                        elif max_score >= 0.3:
                            score_color = "blue"
                        else:
                            score_color = "green"
                        score_display = colorize(f'{max_score:.4f}', score_color)
                    else:
                        score_display = f'{max_score:.4f}'
                    
                    print(f"{i+1}. {text}")
                    print(f"   Confidence: {score_display} ({max_category})")
                    classification = 'TOXIC' if result.get('is_toxic', result.get('toxic', False)) else 'SAFE'
                    print(f"   Classification: {classification}")
                    print()
    
    # Display Groq usage if any
    groq_usage = stats.get('groq_usage', {})
    groq_total = groq_usage.get('total', 0)
    groq_overrides = groq_usage.get('overrides', 0)
    
    if groq_total > 0:
        groq_percent = (groq_total / total_lines) * 100
        override_percent = (groq_overrides / groq_total) * 100 if groq_total > 0 else 0
        
        print(f"\nGroq API usage:")
        print(f"  - Used {groq_total} times ({groq_percent:.1f}% of lines)")
        print(f"  - Changed classification {groq_overrides} times ({override_percent:.1f}% effectiveness)")
    
    # Display confidence filter settings if used
    confidence_filter_settings = []
    if hasattr(args, 'confidence_filter') and args.confidence_filter is not None:
        try:
            confidence_value = float(args.confidence_filter)
            confidence_filter_settings.append(f"above {confidence_value:.4f}")
        except (ValueError, TypeError):
            confidence_filter_settings.append(f"above {args.confidence_filter}")
    
    if hasattr(args, 'min_confidence') and args.min_confidence is not None:
        try:
            min_value = float(args.min_confidence)
            confidence_filter_settings.append(f"above {min_value:.4f}")
        except (ValueError, TypeError):
            confidence_filter_settings.append(f"above {args.min_confidence}")
    
    if hasattr(args, 'max_confidence') and args.max_confidence is not None:
        try:
            max_value = float(args.max_confidence)
            confidence_filter_settings.append(f"below {max_value:.4f}")
        except (ValueError, TypeError):
            confidence_filter_settings.append(f"below {args.max_confidence}")
    
    if confidence_filter_settings:
        filter_desc = " and ".join(confidence_filter_settings)
        print(f"\nConfidence filter applied: {filter_desc}")
    
    # Output in JSON format if requested
    if args.json:
        json_summary = {
            'total_lines': total_lines,
            'displayed_lines': displayed_lines,
            'filtered_lines': filtered_lines,
            'below_range_lines': below_range_lines,
            'above_range_lines': above_range_lines,
            'toxic_lines': toxic_lines,
            'toxic_percent': round(toxic_percent, 2),
            'categories': stats.get('categories', {}),
            'groq_usage': groq_usage,
            'session_start': stats.get('session_start'),
            'session_end': stats.get('session_end')
        }
        
        # Add confidence filter settings to JSON
        confidence_filters = {}
        if hasattr(args, 'confidence_filter') and args.confidence_filter is not None:
            confidence_filters['threshold'] = args.confidence_filter
        if hasattr(args, 'min_confidence') and args.min_confidence is not None:
            confidence_filters['min_confidence'] = args.min_confidence
        if hasattr(args, 'max_confidence') and args.max_confidence is not None:
            confidence_filters['max_confidence'] = args.max_confidence
        
        if confidence_filters:
            json_summary['confidence_filters'] = confidence_filters
            
        # Add confidence sort setting to JSON
        if hasattr(args, 'confidence_sort') and args.confidence_sort is not None:
            json_summary['confidence_sort'] = args.confidence_sort
            
            # Add sorted results if sorting is enabled
            if results:
                # Filter to displayed results only
                if filtered_lines > 0:
                    displayed_results = []
                    for res in results:
                        # Get max confidence from result
                        if 'category_results' in res:
                            scores = {cat.name: data['score'] for cat, data in res['category_results'].items()}
                        else:
                            scores = res.get('probabilities', res.get('scores', {}))
                        max_score = max(scores.values()) if scores else 0.0
                        passes_filter, _ = check_confidence_filter(max_score, args)
                        if passes_filter:
                            displayed_results.append(res)
                else:
                    displayed_results = results
                
                # Sort results by confidence (max score)
                sort_order = SortOrder(args.confidence_sort)
                sorted_results = sorted(
                    displayed_results,
                    key=lambda x: max((x.get('probabilities', x.get('scores', {})) or {}).values() or [0]),
                    reverse=(sort_order == SortOrder.HIGHEST_FIRST)
                )
                
                # Add top 10 results to JSON
                json_summary['sorted_results'] = sorted_results[:10]
            
        print("\nJSON Summary:")
        print(json.dumps(json_summary, indent=2, default=_json_serialize))
    
    # Note about confidence explanation if enabled
    if hasattr(args, 'confidence_explain') and args.confidence_explain:
        print("\nConfidence explanation was enabled for this session")
    
    # Note about confidence sorting if enabled
    if hasattr(args, 'confidence_sort') and args.confidence_sort is not None:
        print(f"\nResults sorted by confidence: {args.confidence_sort} first")
    
    print("=" * 50)


def handle_batch_processing(args: argparse.Namespace) -> Dict[str, Any]:
    """Entry-point used by the CLI when ``--batch`` is supplied.  The helper
    validates *args*, delegates actual work to
    :pyfunc:`batch_processor.batch_process`, dumps JSON or human readable
    output, then returns the results so tests/integration code can inspect
    them programmatically.
    """

    # ------------------------------------------------------------------
    # Validate paths ----------------------------------------------------
    # ------------------------------------------------------------------
    input_path = Path(args.batch)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    output_path: Optional[Path] = None
    if getattr(args, "output", None):
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Delegate to processing layer -------------------------------------
    # ------------------------------------------------------------------
    model_obj = load_model(model_name=args.model or "unitary/toxic-bert")
    cfg = load_config()

    selected_categories = getattr(args, "categories", None)

    # Add confidence filter settings to config
    confidence_config = {}
    if hasattr(args, 'confidence_filter') and args.confidence_filter is not None:
        confidence_config['confidence_filter'] = args.confidence_filter
    if hasattr(args, 'min_confidence') and args.min_confidence is not None:
        confidence_config['min_confidence'] = args.min_confidence
    if hasattr(args, 'max_confidence') and args.max_confidence is not None:
        confidence_config['max_confidence'] = args.max_confidence
    
    if confidence_config:
        cfg['confidence_filtering'] = confidence_config

    # Add confidence explanation setting to config
    if hasattr(args, 'confidence_explain') and args.confidence_explain:
        cfg['confidence_explain'] = True

    # Add confidence sorting setting to config
    if hasattr(args, 'confidence_sort') and args.confidence_sort is not None:
        cfg['confidence_sort'] = args.confidence_sort

    # Initialize monitoring if requested
    monitoring_context = None
    if getattr(args, "monitor", False):
        try:
            from monitor import start_monitoring, stop_monitoring
            
            # Configure monitoring
            monitor_config = {
                "log_path": getattr(args, "monitor_log", None),
                "headless": getattr(args, "monitor_headless", False),
                "update_interval": getattr(args, "monitor_interval", 1.0),
            }
            
            if getattr(args, "monitor_metrics", None):
                monitor_config["metrics"] = [m.strip() for m in args.monitor_metrics.split(",")]
            
            # Start monitoring
            monitoring_context = start_monitoring(monitor_config)
            
        except ImportError:
            print("Warning: Monitoring requires 'rich' and 'psutil' packages. Install them with: pip install rich psutil")
            monitoring_context = None
        except Exception as e:
            print(f"Warning: Failed to start monitoring: {str(e)}")
            monitoring_context = None

    try:
        results = batch_process(
            input_path=input_path,
            output_path=output_path,
            model=model_obj,
            config=cfg,
            show_progress=not getattr(args, "json", False) and not getattr(args, "quiet", False),
            selected_categories=selected_categories,
            monitor=monitoring_context,
        )
    finally:
        # Stop monitoring if it was started
        if monitoring_context:
            try:
                summary = stop_monitoring(monitoring_context)
                
                # Display summary if not in quiet mode
                if not getattr(args, "quiet", False) and not getattr(args, "json", False):
                    print("\nMonitoring Summary:")
                    print("-" * 80)
                    print(f"Total processing time: {summary['elapsed_seconds']:.2f} seconds")
                    print(f"Total texts processed: {summary['processed_texts']}")
                    print(f"Average throughput: {summary['avg_throughput']:.2f} texts/second")
                    print(f"Average latency: {summary['avg_latency'] * 1000:.2f} ms")
                    print(f"API calls: {summary['api_calls']} ({summary['api_errors']} errors)")
                    print(f"Error rate: {summary['error_rate'] * 100:.2f}%")
                    print(f"Toxic content: {summary['toxic_texts']} ({summary['toxic_percentage']:.1f}%)")
                    
                    if getattr(args, "monitor_log", None):
                        print(f"Detailed monitoring log saved to: {args.monitor_log}")
                        
            except Exception as e:
                print(f"Warning: Error stopping monitoring: {str(e)}")

    # Timestamp for traceability ---------------------------------------
    results["timestamp"] = datetime.now().isoformat(timespec="seconds")

    # Emit results ------------------------------------------------------
    if getattr(args, "json", False):
        _emit_json(results, path=args.output if (output_path is None and getattr(args, "output", None)) else None)
    else:
        display_batch_results(results, args)

    return results


def handle_wizard(args):
    """Handle configuration wizard based on command line arguments."""
    try:
        from config_wizard import setup_wizard
        
        if not args.setup_config:
            return False
        
        try:
            config_path = setup_wizard(
                wizard_mode=args.wizard_defaults,
                output_path=args.config_output,
                non_interactive=getattr(args, 'non_interactive', False)
            )
            print(f"Configuration wizard completed successfully. Config saved to: {config_path}")
            return True
        except KeyboardInterrupt:
            print("\nConfiguration wizard cancelled by user.")
            return True
        except Exception as e:
            print(f"Error running configuration wizard: {str(e)}")
            import traceback
            traceback.print_exc()
            return True
    except ImportError:
        print("Error: Configuration wizard requires the 'inquirer' package.")
        print("Please install it using: pip install inquirer")
        return True


def handle_comparison(args):
    """Handle model comparison based on command line arguments."""
    from evaluator import load_validation_dataset
    from model_loader import get_model
    from config_loader import load_config
    from model_comparison import (
        compare_models, generate_comparative_plots, display_comparison_results,
        statistical_significance, generate_comparative_report
    )
    
    if not args.compare_models:
        return False
    
    # Check required arguments
    if not args.model_paths:
        print("Error: --model-paths is required for model comparison")
        return True
    
    if not args.evaluate:
        print("Error: --evaluate is required to specify validation dataset")
        return True
    
    # Load validation dataset
    print(f"Loading validation dataset from {args.evaluate}...")
    texts, labels = load_validation_dataset(args.evaluate, not args.no_headers)
    
    # Split model paths and names
    model_paths = [path.strip() for path in args.model_paths.split(",")]
    
    if args.model_names:
        model_names = [name.strip() for name in args.model_names.split(",")]
    else:
        # Use filenames as model names
        model_names = [Path(path).stem for path in model_paths]
    
    # Check if we have the same number of names as models
    if len(model_names) != len(model_paths):
        print(f"Warning: Number of model names ({len(model_names)}) doesn't match number of models ({len(model_paths)})")
        # Use default naming if mismatch
        model_names = [f"Model_{i+1}" for i in range(len(model_paths))]
    
    # Parse config paths if provided
    config_paths = []
    if args.config_paths:
        config_paths = [path.strip() for path in args.config_paths.split(",")]
        
        # Check if we have the same number of configs as models
        if len(config_paths) != len(model_paths):
            print(f"Warning: Number of config paths ({len(config_paths)}) doesn't match number of models ({len(model_paths)})")
            # Use None for missing configs
            if len(config_paths) < len(model_paths):
                config_paths.extend([None] * (len(model_paths) - len(config_paths)))
    else:
        # Use default config for all models
        config_paths = [None] * len(model_paths)
    
    # Set up output directory
    output_dir = args.comparison_output or "model_comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    models_dict = {}
    thresholds = {}
    
    for i, (model_path, model_name, config_path) in enumerate(zip(model_paths, model_names, config_paths)):
        print(f"Loading model {i+1}/{len(model_paths)}: {model_name}")
        
        try:
            # Load model and config
            model = get_model(model_name=model_path)
            config = load_config(config_path) if config_path else None
            
            # Add to models dictionary
            models_dict[model_name] = model
            
            # Extract thresholds from config if available
            if config and "thresholds" in config:
                thresholds[model_name] = config["thresholds"]
            
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            print("Skipping this model in the comparison")
    
    if not models_dict:
        print("Error: No models could be loaded for comparison")
        return True
    
    print(f"Successfully loaded {len(models_dict)} models for comparison")
    
    # Run comparison
    print("Comparing models on validation dataset...")
    comparison_results = compare_models(
        models_dict, texts, labels, thresholds=thresholds
    )
    
    # Run significance tests if requested
    if args.significance_test:
        print("Running statistical significance tests...")
        significance_results = statistical_significance(comparison_results)
        comparison_results["significance"] = significance_results
    
    # Display results
    display_comparison_results(comparison_results, detailed=True)
    
    # Generate comparative plots if requested
    if args.comparative_plots:
        print(f"Generating comparative plots in {output_dir}...")
        plot_files = generate_comparative_plots(
            comparison_results,
            output_dir,
            plot_types=["bar", "radar", "confusion"],
            dpi=getattr(args, 'plot_dpi', 300),
            fmt=getattr(args, 'plot_format', 'both')
        )
        print(f"Generated {len(plot_files)} comparative plots")
    
    # Generate comparative report if requested
    if args.comparative_report:
        print("Generating comparative PDF report...")
        report_path = Path(output_dir) / "model_comparison_report.pdf"
        report_file = generate_comparative_report(
            comparison_results,
            str(report_path),
            include_plots=True,
            title="Toxicity Classification Model Comparison"
        )
        print(f"Comparative report saved to: {report_file}")
    
    # Export comparison results to JSON
    results_path = Path(output_dir) / "comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    print(f"Comparison results saved to: {results_path}")
    return True


def handle_visualization(args, optimization_results, evaluation_results):
    """Handle visualization generation based on command line arguments."""
    from evaluator import (
        plot_precision_recall_curve, plot_threshold_sweep, 
        plot_confusion_matrices, generate_pdf_report
    )
    
    # Check if any plot type is requested
    plot_requested = (
        hasattr(args, 'plot_pr_curves') and args.plot_pr_curves or 
        hasattr(args, 'plot_threshold_sweep') and args.plot_threshold_sweep or 
        hasattr(args, 'plot_confusion_matrices') and args.plot_confusion_matrices or 
        hasattr(args, 'plot_all') and args.plot_all or
        hasattr(args, 'generate_pdf') and args.generate_pdf
    )
    
    if not plot_requested or not (hasattr(args, 'plot_dir') and args.plot_dir or hasattr(args, 'pdf_path') and args.pdf_path):
        return
    
    # Prepare plot directory if specified
    plot_dir = None
    if hasattr(args, 'plot_dir') and args.plot_dir:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare categories if specified
    selected_categories = None
    if hasattr(args, 'selected_categories') and args.selected_categories:
        selected_categories = [cat.strip() for cat in args.selected_categories.split(',')]
    
    # Get plot format and DPI
    plot_format = getattr(args, 'plot_format', 'both')
    plot_dpi = getattr(args, 'plot_dpi', 300)
    
    try:
        # Generate plots based on arguments (only if plot_dir is specified)
        if plot_dir:
            if (hasattr(args, 'plot_pr_curves') and args.plot_pr_curves) or (hasattr(args, 'plot_all') and args.plot_all):
                if optimization_results:
                    try:
                        auprc_scores = plot_precision_recall_curve(
                            optimization_results,
                            plot_dir,
                            categories=selected_categories,
                            dpi=plot_dpi,
                            fmt=plot_format
                        )
                        print(f"AUPRC Scores: {', '.join([f'{k}: {v:.3f}' for k, v in auprc_scores.items()])}")
                    except Exception as e:
                        print(f"Warning: Error generating precision-recall curves: {str(e)}")
                else:
                    print("Warning: Precision-recall curves require threshold optimization results.")
            
            if (hasattr(args, 'plot_threshold_sweep') and args.plot_threshold_sweep) or (hasattr(args, 'plot_all') and args.plot_all):
                if optimization_results:
                    try:
                        plot_files = plot_threshold_sweep(
                            optimization_results,
                            plot_dir,
                            categories=selected_categories,
                            dpi=plot_dpi,
                            fmt=plot_format
                        )
                    except Exception as e:
                        print(f"Warning: Error generating threshold sweep plots: {str(e)}")
                else:
                    print("Warning: Threshold sweep plots require threshold optimization results.")
            
            if (hasattr(args, 'plot_confusion_matrices') and args.plot_confusion_matrices) or (hasattr(args, 'plot_all') and args.plot_all):
                if evaluation_results:
                    try:
                        plot_files = plot_confusion_matrices(
                            evaluation_results,
                            plot_dir,
                            categories=selected_categories,
                            dpi=plot_dpi,
                            fmt=plot_format
                        )
                    except Exception as e:
                        print(f"Warning: Error generating confusion matrix plots: {str(e)}")
                else:
                    print("Warning: Confusion matrix plots require evaluation results.")
                
    except ImportError as e:
        print(f"Error: {e}")
        print("To use visualization features, install the required packages:")
        print("pip install matplotlib scikit-learn")
    
    # Generate PDF report if requested
    if hasattr(args, 'generate_pdf') and args.generate_pdf:
        if evaluation_results:
            try:
                # Determine PDF path
                pdf_path = getattr(args, 'pdf_path', None)
                if not pdf_path:
                    if hasattr(args, 'plot_dir') and args.plot_dir:
                        pdf_path = Path(args.plot_dir) / "evaluation_report.pdf"
                    else:
                        pdf_path = "evaluation_report.pdf"
                
                # Generate the report
                report_path = generate_pdf_report(
                    evaluation_results,
                    pdf_path,
                    optimization_results=optimization_results,
                    title=getattr(args, 'report_title', "Model Evaluation Report"),
                    template_path=getattr(args, 'report_template', None)
                )
                print(f"PDF report generated: {report_path}")
            except ImportError as e:
                print(f"Error: {e}")
                print("To use PDF report generation, install the required packages:")
                print("pip install reportlab matplotlib")
            except Exception as e:
                print(f"Error generating PDF report: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("Warning: PDF report generation requires evaluation results.")


def handle_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Handle model evaluation on a validation dataset.
    
    Args:
        args: Command line arguments containing evaluation parameters
        
    Returns:
        Dictionary containing evaluation results
    """
    from evaluator import (
        load_validation_dataset, 
        evaluate_model, 
        display_evaluation_results, 
        export_evaluation_results,
        optimize_thresholds,
        display_threshold_optimization,
        save_optimal_thresholds
    )
    
    print(f"Loading validation dataset from {args.evaluate}...")
    texts, labels = load_validation_dataset(args.evaluate, has_header=not args.no_headers)
    
    print(f"Dataset loaded: {len(texts)} samples with {len(labels)} categories")
    print(f"Categories: {', '.join(labels.keys())}")
    
    # Load model and configuration
    model_obj = load_model(model_name=args.model or "unitary/toxic-bert")
    cfg = load_config()
    
    # Check if threshold optimization is requested
    if hasattr(args, 'optimize_thresholds') and args.optimize_thresholds:
        print("Starting threshold optimization...")
        
        # Run threshold optimization
        optimization_results = optimize_thresholds(
            model=model_obj,
            texts=texts,
            labels=labels,
            threshold_range=(args.min_threshold, args.max_threshold),
            step_size=args.threshold_step,
            parallel=True  # Use parallel processing for better performance
        )
        
        # Display optimization results
        display_threshold_optimization(optimization_results)
        
        # Save optimal thresholds if requested
        if hasattr(args, 'save_thresholds') and args.save_thresholds:
            config_path = args.config_path or "toxicity_detector.yaml"
            if not Path(config_path).exists():
                # Create a basic config file if it doesn't exist
                basic_config = {
                    "model": {"name": args.model or "unitary/toxic-bert"},
                    "thresholds": optimization_results['optimal_thresholds']
                }
                
                with open(config_path, 'w') as f:
                    if config_path.endswith('.json'):
                        json.dump(basic_config, f, indent=2)
                    else:
                        import yaml
                        yaml.dump(basic_config, f, default_flow_style=False)
                
                print(f"Created new configuration file with optimal thresholds: {config_path}")
            else:
                save_optimal_thresholds(optimization_results['optimal_thresholds'], config_path)
                print(f"Saved optimal thresholds to configuration file: {config_path}")
        
        # Export optimization results if requested
        if args.evaluation_output:
            output_data = {
                "optimization_results": optimization_results,
                "optimal_thresholds": optimization_results['optimal_thresholds'],
                "default_evaluation": optimization_results['default_results'],
                "optimized_evaluation": optimization_results['optimized_results']
            }
            export_evaluation_results(output_data, args.evaluation_output)
        
        # Generate visualizations if requested
        handle_visualization(args, optimization_results, optimization_results['optimized_results'])
        
        return optimization_results
    
    else:
        # Standard evaluation without optimization
        
        # Load category-specific thresholds if specified
        thresholds = None
        if args.category_thresholds:
            print(f"Loading category thresholds from {args.category_thresholds}...")
            with open(args.category_thresholds, 'r') as f:
                if args.category_thresholds.endswith('.yaml') or args.category_thresholds.endswith('.yml'):
                    import yaml
                    config_data = yaml.safe_load(f)
                    # Extract thresholds from config structure
                    thresholds = config_data.get('thresholds', config_data)
                else:
                    thresholds = json.load(f)
        elif args.eval_threshold != 0.5:
            # Apply the same threshold to all categories
            thresholds = {category: args.eval_threshold for category in labels.keys()}
            print(f"Using threshold {args.eval_threshold} for all categories")
        else:
            # Use default thresholds from config or 0.5
            thresholds = cfg.get("thresholds", {})
            if not thresholds:
                thresholds = {category: 0.5 for category in labels.keys()}
            print("Using default thresholds")
        
        print(f"Evaluating model on {len(texts)} samples...")
        eval_results = evaluate_model(model_obj, texts, labels, thresholds=thresholds)
        
        # Display results
        display_evaluation_results(eval_results)
        
        # Export results if requested
        if args.evaluation_output:
            export_evaluation_results(eval_results, args.evaluation_output)
        
        # Generate visualizations if requested
        handle_visualization(args, None, eval_results)
        
        return eval_results


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

    # ------------------------------------------------------------------
    # Validate Groq bound values ---------------------------------------
    # ------------------------------------------------------------------
    if args.groq_lower_bound is not None and not (0.0 <= args.groq_lower_bound <= 1.0):
        parser.error("--groq-lower-bound must be between 0.0 and 1.0")

    if args.groq_upper_bound is not None and not (0.0 <= args.groq_upper_bound <= 1.0):
        parser.error("--groq-upper-bound must be between 0.0 and 1.0")

    if args.groq_lower_bound is not None and args.groq_upper_bound is not None:
        if args.groq_lower_bound >= args.groq_upper_bound:
            parser.error("--groq-lower-bound must be smaller than --groq-upper-bound")

    if args.confidence_filter is not None and not (0.0 <= args.confidence_filter <= 1.0):
        parser.error("--confidence-filter must be between 0.0 and 1.0")

    if args.min_confidence is not None and not (0.0 <= args.min_confidence <= 1.0):
        parser.error("--min-confidence must be between 0.0 and 1.0")
    
    if args.max_confidence is not None and not (0.0 <= args.max_confidence <= 1.0):
        parser.error("--max-confidence must be between 0.0 and 1.0")
    
    if (args.min_confidence is not None and args.max_confidence is not None and 
        args.min_confidence >= args.max_confidence):
        parser.error("--min-confidence must be less than --max-confidence")
    
    # Check for conflicting confidence options
    confidence_options = [args.confidence_filter, args.min_confidence, args.max_confidence]
    if args.confidence_filter is not None and (args.min_confidence is not None or args.max_confidence is not None):
        parser.error("--confidence-filter cannot be used with --min-confidence or --max-confidence")

    # ---------------------------------------------------------------------
    # Groq cache maintenance / info commands -----------------------------
    # ---------------------------------------------------------------------
    if args.clear_groq_cache:
        from groq_cache import GroqCache
        removed = GroqCache().clear()
        print(f"Cleared {removed} cached Groq responses.")
        return 0

    if args.groq_cache_stats:
        from groq_cache import GroqCache

        stats = GroqCache().stats()
        print("\nGroq API Response Cache Statistics")
        print("=" * 40)
        print(f"Cache location: {stats['dir']}")
        print(f"Total entries: {stats['entries']}")

        # Show size in human-friendly MB as well as bytes if available
        size_mb = stats.get("size_mb")
        size_bytes = stats.get("size_bytes")
        if size_mb is not None and size_bytes is not None:
            print(f"Total size: {size_mb:.2f} MB ({size_bytes} bytes)")

        if stats.get("oldest"):
            print(f"Oldest entry: {stats['oldest']}")
        if stats.get("newest"):
            print(f"Newest entry: {stats['newest']}")

        if stats['entries'] == 0:
            print("The cache is currently empty. It will populate as you use --allow-groq-fallback.")
        else:
            print("\nUse '--clear-groq-cache' to remove cached entries if they become stale.")

        # Nothing else to do
        return 0

    # Handle configuration wizard if requested (before other command checks)
    if args.setup_config:
        if handle_wizard(args):
            return 0

    # Handle model comparison if requested
    if args.compare_models:
        if handle_comparison(args):
            return 0

    if args.text:
        # Use _process_single to support confidence filtering
        res = _process_single(args.text, cfg=cfg, args=args)
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
        from config_loader import create_default_config
        path = create_default_config(Path("toxicity_detector.yaml"))
        print(f"Default configuration file written to {path}")
        return 0

    if args.batch:
        handle_batch_processing(args)
        return 0
    
    if args.stream:
        handle_stream_processing(args)
        return 0
    
    if args.evaluate:
        handle_evaluation(args)
        return 0

    # Default: show help if no command specified
    parser.print_help()
    return 1


if __name__ == "__main__":
    exit(main())