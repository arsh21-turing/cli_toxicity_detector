#!/usr/bin/env python3
"""
Multi-model comparison and analysis framework for toxicity classification.

This module provides functionality to compare multiple models side-by-side
on the same validation dataset, including detailed performance metrics,
visualizations, and statistical significance testing.
"""

import os
import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
import logging

# Import from evaluator module
from evaluator import evaluate_model

# Set up logging
logger = logging.getLogger(__name__)


def compare_models(
    models_dict: Dict[str, Any],
    texts: List[str],
    labels: Dict[str, List[int]],
    thresholds: Optional[Dict[str, Dict[str, float]]] = None,
    categories: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate multiple models on the same validation dataset.
    
    Args:
        models_dict: Dictionary mapping model names to model objects
        texts: List of text samples
        labels: Dictionary mapping category names to binary label lists (0/1)
        thresholds: Optional dictionary mapping model names to their category thresholds
        categories: Optional list of categories to evaluate (defaults to all in labels)
        
    Returns:
        Dictionary containing comparison results with:
            - per_model: Dictionary mapping model names to their evaluation results
            - comparative: Dictionary with head-to-head metrics and comparisons
            - winner: Dictionary detailing the best model for each metric
    """
    if not models_dict:
        raise ValueError("No models provided for comparison")
    
    if not texts or not labels:
        raise ValueError("No validation data provided")
    
    # Initialize results structure
    comparison_results = {
        "per_model": {},
        "comparative": {},
        "winner": {}
    }
    
    # If categories not specified, use all categories in the labels
    if categories is None:
        categories = list(labels.keys())
    
    # Initialize default thresholds if not provided
    if thresholds is None:
        thresholds = {}
        
    # Evaluate each model
    for model_name, model in models_dict.items():
        logger.info(f"Evaluating model: {model_name}")
        
        # Get thresholds for this model if available, or use default (0.5)
        model_thresholds = thresholds.get(model_name, {})
        
        # Evaluate the model
        evaluation_results = evaluate_model(
            model, texts, labels, 
            thresholds=model_thresholds, 
            categories=categories
        )
        
        # Store evaluation results for this model
        comparison_results["per_model"][model_name] = evaluation_results
    
    # Calculate comparative metrics
    comparison_results["comparative"] = _calculate_comparative_metrics(
        comparison_results["per_model"], categories
    )
    
    # Determine winners for each metric
    comparison_results["winner"] = _determine_winners(
        comparison_results["per_model"], categories
    )
    
    return comparison_results


def _calculate_comparative_metrics(
    per_model_results: Dict[str, Dict[str, Any]],
    categories: List[str]
) -> Dict[str, Any]:
    """
    Calculate comparative metrics across models.
    
    Args:
        per_model_results: Dictionary mapping model names to their evaluation results
        categories: List of categories to include in the comparison
        
    Returns:
        Dictionary with comparative metrics
    """
    # Get list of model names
    model_names = list(per_model_results.keys())
    
    if len(model_names) < 2:
        # No comparison needed if there's only one model
        return {"note": "Comparison requires at least two models"}
    
    comparative = {
        "overall": {},
        "per_category": {category: {} for category in categories},
        "relative_performance": {}
    }
    
    # Calculate relative performance for each pair of models
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i >= j:  # Skip self-comparison and avoid duplicates
                continue
                
            pair_key = f"{model1}_vs_{model2}"
            comparative["relative_performance"][pair_key] = {
                "overall": {},
                "per_category": {category: {} for category in categories}
            }
            
            # Compare overall metrics
            for metric in ["precision", "recall", "f1"]:
                model1_value = per_model_results[model1]["overall"].get(metric, 0.0)
                model2_value = per_model_results[model2]["overall"].get(metric, 0.0)
                
                # Calculate absolute and relative differences
                abs_diff = model1_value - model2_value
                rel_diff = (abs_diff / model2_value * 100) if model2_value != 0 else float('inf')
                
                comparative["relative_performance"][pair_key]["overall"][metric] = {
                    "absolute_difference": abs_diff,
                    "relative_difference_percent": rel_diff,
                    "better_model": model1 if abs_diff > 0 else (model2 if abs_diff < 0 else "tie")
                }
            
            # Compare per-category metrics
            for category in categories:
                for metric in ["precision", "recall", "f1"]:
                    if (category in per_model_results[model1]["per_category"] and
                        category in per_model_results[model2]["per_category"]):
                        model1_value = per_model_results[model1]["per_category"][category].get(metric, 0.0)
                        model2_value = per_model_results[model2]["per_category"][category].get(metric, 0.0)
                        
                        # Calculate absolute and relative differences
                        abs_diff = model1_value - model2_value
                        rel_diff = (abs_diff / model2_value * 100) if model2_value != 0 else float('inf')
                        
                        comparative["relative_performance"][pair_key]["per_category"][category][metric] = {
                            "absolute_difference": abs_diff,
                            "relative_difference_percent": rel_diff,
                            "better_model": model1 if abs_diff > 0 else (model2 if abs_diff < 0 else "tie")
                        }
    
    # Calculate model agreement statistics
    comparative["agreement"] = _calculate_model_agreement(per_model_results, categories)
    
    return comparative


def _calculate_model_agreement(
    per_model_results: Dict[str, Dict[str, Any]],
    categories: List[str]
) -> Dict[str, Any]:
    """
    Calculate agreement statistics between models.
    
    Args:
        per_model_results: Dictionary mapping model names to their evaluation results
        categories: List of categories to include in the calculation
        
    Returns:
        Dictionary with agreement statistics
    """
    model_names = list(per_model_results.keys())
    agreement = {}
    
    # Calculate agreement only if we have sentence-level predictions
    sample_model = model_names[0]
    for category in categories:
        if (category in per_model_results[sample_model]["per_category"] and
            "predictions" in per_model_results[sample_model]["per_category"][category]):
            
            # If we have predictions, calculate agreement
            agreement[category] = {}
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i >= j:  # Skip self-comparison and avoid duplicates
                        continue
                    
                    pair_key = f"{model1}_vs_{model2}"
                    
                    preds1 = per_model_results[model1]["per_category"][category]["predictions"]
                    preds2 = per_model_results[model2]["per_category"][category]["predictions"]
                    
                    # Calculate agreement rate
                    agreements = sum(1 for p1, p2 in zip(preds1, preds2) if p1 == p2)
                    total = len(preds1)
                    agreement_rate = agreements / total if total > 0 else 1.0
                    
                    agreement[category][pair_key] = {
                        "agreement_rate": agreement_rate,
                        "agreements": agreements,
                        "total": total
                    }
    
    return agreement


def _determine_winners(
    per_model_results: Dict[str, Dict[str, Any]],
    categories: List[str]
) -> Dict[str, Any]:
    """
    Determine the best performing model for each metric.
    
    Args:
        per_model_results: Dictionary mapping model names to their evaluation results
        categories: List of categories to include
        
    Returns:
        Dictionary with best model for each metric
    """
    model_names = list(per_model_results.keys())
    winners = {
        "overall": {},
        "per_category": {category: {} for category in categories}
    }
    
    # Find winners for overall metrics
    for metric in ["precision", "recall", "f1"]:
        best_model = ""
        best_value = -1.0
        
        for model in model_names:
            value = per_model_results[model]["overall"].get(metric, 0.0)
            if value > best_value:
                best_value = value
                best_model = model
        
        winners["overall"][metric] = {
            "model": best_model,
            "value": best_value
        }
    
    # Find winners for per-category metrics
    for category in categories:
        for metric in ["precision", "recall", "f1"]:
            best_model = ""
            best_value = -1.0
            
            for model in model_names:
                if category in per_model_results[model]["per_category"]:
                    value = per_model_results[model]["per_category"][category].get(metric, 0.0)
                    if value > best_value:
                        best_value = value
                        best_model = model
            
            winners["per_category"][category][metric] = {
                "model": best_model,
                "value": best_value
            }
    
    return winners


def display_comparison_results(
    comparison_results: Dict[str, Any],
    detailed: bool = False
) -> None:
    """
    Format and display model comparison results.
    
    Args:
        comparison_results: Results from compare_models function
        detailed: Whether to show detailed results for each category
    """
    try:
        from color_utils import colorize, COLORS
    except ImportError:
        # Fallback if color_utils is not available
        def colorize(text, color):
            return text
        
        COLORS = {
            'CYAN': '',
            'GREEN': '',
            'YELLOW': '',
            'RED': ''
        }
    
    # Make sure we have the colors available
    if 'CYAN' not in COLORS:
        COLORS.update({
            'CYAN': '',
            'GREEN': '',
            'YELLOW': '',
            'RED': ''
        })
    
    per_model_results = comparison_results.get("per_model", {})
    winners = comparison_results.get("winner", {})
    
    model_names = list(per_model_results.keys())
    
    if not model_names:
        print("No models to compare")
        return
    
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Display overall metrics
    print("\nOVERALL METRICS")
    print("-"*80)
    
    headers = ["Metric"] + model_names + ["Best Model"]
    headers_str = "| " + " | ".join(f"{h:<15}" for h in headers) + " |"
    
    print(headers_str)
    print("|" + "-"*len(headers_str) + "|")
    
    for metric in ["precision", "recall", "f1"]:
        row = [metric.capitalize()]
        best_model = winners.get("overall", {}).get(metric, {}).get("model", "")
        
        for model_name in model_names:
            value = per_model_results[model_name]["overall"].get(metric, 0.0)
            # Colorize if this model is the winner for this metric
            if model_name == best_model:
                row.append(colorize(f"{value:.4f}", COLORS['CYAN']))
            else:
                row.append(f"{value:.4f}")
        
        row.append(best_model)
        
        print("| " + " | ".join(f"{cell:<15}" for cell in row) + " |")
    
    # Display per-category metrics if detailed mode
    if detailed:
        # Get list of all categories
        categories = set()
        for model_name in model_names:
            categories.update(per_model_results[model_name]["per_category"].keys())
        
        # Sort categories for consistent display
        categories = sorted(categories)
        
        for category in categories:
            print(f"\nCATEGORY: {category}")
            print("-"*80)
            
            # Check if we have results for this category
            if category not in winners.get("per_category", {}):
                print(f"No comparison data available for category: {category}")
                continue
            
            # Print headers
            print(headers_str)
            print("|" + "-"*len(headers_str) + "|")
            
            for metric in ["precision", "recall", "f1"]:
                row = [metric.capitalize()]
                best_model = winners.get("per_category", {}).get(category, {}).get(metric, {}).get("model", "")
                
                for model_name in model_names:
                    value = 0.0
                    if (category in per_model_results[model_name]["per_category"] and
                        metric in per_model_results[model_name]["per_category"][category]):
                        value = per_model_results[model_name]["per_category"][category][metric]
                    
                    # Colorize if this model is the winner for this metric
                    if model_name == best_model:
                        row.append(colorize(f"{value:.4f}", COLORS['CYAN']))
                    else:
                        row.append(f"{value:.4f}")
                
                row.append(best_model)
                
                print("| " + " | ".join(f"{cell:<15}" for cell in row) + " |")
    
    # Display model agreement if available
    if "comparative" in comparison_results and "agreement" in comparison_results["comparative"]:
        agreement = comparison_results["comparative"]["agreement"]
        if agreement:
            print("\nMODEL AGREEMENT")
            print("-"*80)
            
            for category, category_agreement in agreement.items():
                if category_agreement:
                    print(f"\nCategory: {category}")
                    
                    for pair_key, stats in category_agreement.items():
                        agreement_rate = stats["agreement_rate"]
                        color = COLORS['GREEN'] if agreement_rate >= 0.8 else (COLORS['YELLOW'] if agreement_rate >= 0.5 else COLORS['RED'])
                        agreement_str = colorize(f"{agreement_rate:.2%}", color)
                        
                        print(f"  {pair_key}: {agreement_str} agreement ({stats['agreements']}/{stats['total']} predictions)")
    
    # Display significance test results if available
    if "significance" in comparison_results:
        print("\nSTATISTICAL SIGNIFICANCE")
        print("-"*80)
        
        significance = comparison_results["significance"]
        
        for test_name, test_results in significance.items():
            print(f"\n{test_name}:")
            
            for comparison, result in test_results.items():
                p_value = result.get("p_value", 1.0)
                significant = result.get("significant", False)
                
                p_value_str = f"p={p_value:.4f}"
                if significant:
                    p_value_str = colorize(p_value_str, COLORS['GREEN'])
                
                print(f"  {comparison}: {p_value_str}" + (" (significant)" if significant else ""))
    
    print("\n" + "="*80)


def statistical_significance(
    comparison_results: Dict[str, Any],
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Evaluate statistical significance of differences between models.
    
    Args:
        comparison_results: Results from compare_models function
        confidence_level: Confidence level for significance tests (default: 0.95)
        
    Returns:
        Dictionary with significance test results
    """
    from scipy import stats
    
    per_model_results = comparison_results.get("per_model", {})
    model_names = list(per_model_results.keys())
    
    if len(model_names) < 2:
        return {"error": "Need at least two models for significance testing"}
    
    # Initialize results
    significance_results = {
        "overall_f1": {},
        "per_category_f1": {}
    }
    
    # Get significance threshold
    alpha = 1.0 - confidence_level
    
    # Test overall F1 scores if we have prediction-level data
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i >= j:  # Skip self-comparison and duplicates
                continue
            
            pair_key = f"{model1}_vs_{model2}"
            
            # Check if we have raw predictions to calculate significance
            if all("predictions" in per_model_results[model].get("overall", {}) for model in [model1, model2]):
                preds1 = per_model_results[model1]["overall"]["predictions"]
                preds2 = per_model_results[model2]["overall"]["predictions"]
                
                # Perform McNemar's test for paired categorical data
                try:
                    contingency_table = [
                        [sum(1 for p1, p2 in zip(preds1, preds2) if p1 == 1 and p2 == 1),  # Both models predict positive
                         sum(1 for p1, p2 in zip(preds1, preds2) if p1 == 1 and p2 == 0)], # Model1 positive, Model2 negative
                        [sum(1 for p1, p2 in zip(preds1, preds2) if p1 == 0 and p2 == 1),  # Model1 negative, Model2 positive
                         sum(1 for p1, p2 in zip(preds1, preds2) if p1 == 0 and p2 == 0)]  # Both models predict negative
                    ]
                    
                    result = stats.fisher_exact(contingency_table)
                    p_value = result.pvalue
                    
                    significance_results["overall_f1"][pair_key] = {
                        "p_value": p_value,
                        "significant": p_value < alpha,
                        "test_method": "Fisher's exact test"
                    }
                except Exception as e:
                    logger.warning(f"Could not perform significance test for {pair_key}: {str(e)}")
                    continue
    
    # If we don't have raw predictions, just calculate overall F1 differences
    if not significance_results["overall_f1"]:
        # Calculate bootstrap confidence intervals for overall F1 differences
        # This is a simplified approach - in a real scenario you'd want to bootstrap
        # from the raw data, but here we're just using the F1 scores
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i >= j:  # Skip self-comparison and duplicates
                    continue
                
                pair_key = f"{model1}_vs_{model2}"
                
                model1_f1 = per_model_results[model1]["overall"].get("f1", 0.0)
                model2_f1 = per_model_results[model2]["overall"].get("f1", 0.0)
                
                # Calculate difference and rough estimate of significance
                # This is not a proper statistical test, just a heuristic
                diff = abs(model1_f1 - model2_f1)
                
                # A simple heuristic - differences > 0.05 are considered "significant"
                # In practice, you'd use bootstrap or other methods for proper testing
                significance_results["overall_f1"][pair_key] = {
                    "difference": diff,
                    "significant": diff > 0.05,
                    "test_method": "heuristic (difference > 0.05)",
                    "note": "Not a formal statistical test - just based on magnitude of difference"
                }
    
    # Test per-category F1 scores if available
    # This follows the same pattern as above, but for each category
    categories = set()
    for model in per_model_results.values():
        categories.update(model["per_category"].keys())
    
    for category in categories:
        significance_results["per_category_f1"][category] = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i >= j:  # Skip self-comparison and duplicates
                    continue
                
                pair_key = f"{model1}_vs_{model2}"
                
                # Check if we have data for this category for both models
                if (category in per_model_results[model1]["per_category"] and
                    category in per_model_results[model2]["per_category"]):
                    
                    model1_f1 = per_model_results[model1]["per_category"][category].get("f1", 0.0)
                    model2_f1 = per_model_results[model2]["per_category"][category].get("f1", 0.0)
                    
                    # Calculate difference and rough estimate of significance
                    diff = abs(model1_f1 - model2_f1)
                    
                    # Simple heuristic as above
                    significance_results["per_category_f1"][category][pair_key] = {
                        "difference": diff,
                        "significant": diff > 0.05,
                        "test_method": "heuristic (difference > 0.05)",
                        "note": "Not a formal statistical test - just based on magnitude of difference"
                    }
    
    return significance_results


def generate_comparative_plots(
    comparison_results: Dict[str, Any],
    output_path: str,
    plot_types: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    dpi: int = 300,
    fmt: str = 'both'
) -> List[str]:
    """
    Create comparative visualizations across models.
    
    Args:
        comparison_results: Results from compare_models function
        output_path: Directory path to save plot files
        plot_types: Optional list of plot types to generate 
                   (choices: 'bar', 'radar', 'confusion', 'all')
        categories: Optional list of categories to include (defaults to all)
        dpi: DPI for saved PNG images (default: 300)
        fmt: Output format: 'png', 'svg', or 'both' (default: 'both')
        
    Returns:
        List of paths to the generated plot files
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
        import numpy as np
    except ImportError:
        raise ImportError(
            "Required packages for plotting are not installed. "
            "Please install matplotlib using: pip install matplotlib"
        )
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get model names
    per_model_results = comparison_results.get("per_model", {})
    model_names = list(per_model_results.keys())
    
    if not model_names:
        logger.warning("No model results found for comparison plots")
        return []
    
    # Determine plot types to generate
    all_plot_types = ["bar", "radar"]
    if not plot_types:
        plot_types = all_plot_types
    elif "all" in plot_types:
        plot_types = all_plot_types
    
    # Determine categories to include
    if not categories:
        # Get categories from the first model
        sample_model = model_names[0]
        categories = list(per_model_results[sample_model]["per_category"].keys())
    
    # Determine output formats
    formats = []
    if fmt.lower() == 'png' or fmt.lower() == 'both':
        formats.append('png')
    if fmt.lower() == 'svg' or fmt.lower() == 'both':
        formats.append('svg')
    
    # Set up styles
    plt.style.use('default')
    
    # Initialize list of generated plot files
    plot_files = []
    
    # Generate bar charts
    if "bar" in plot_types:
        # Generate overall metrics bar chart
        bar_chart_path = _create_overall_bar_chart(
            comparison_results, output_path, model_names, formats, dpi
        )
        plot_files.extend(bar_chart_path)
        
        # Generate per-category bar charts
        for category in categories:
            category_bar_paths = _create_category_bar_chart(
                comparison_results, output_path, model_names, category, formats, dpi
            )
            plot_files.extend(category_bar_paths)
    
    # Generate radar charts
    if "radar" in plot_types:
        # Generate overall metrics radar chart
        radar_chart_path = _create_overall_radar_chart(
            comparison_results, output_path, model_names, formats, dpi
        )
        plot_files.extend(radar_chart_path)
        
        # Generate per-category radar charts
        category_radar_paths = _create_category_radar_chart(
            comparison_results, output_path, model_names, categories, formats, dpi
        )
        plot_files.extend(category_radar_paths)
    
    return plot_files


def _create_overall_bar_chart(
    comparison_results: Dict[str, Any],
    output_path: Path,
    model_names: List[str],
    formats: List[str],
    dpi: int
) -> List[str]:
    """Create bar chart comparing overall metrics across models."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    per_model_results = comparison_results["per_model"]
    winners = comparison_results["winner"]["overall"]
    
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(metrics))
    bar_width = 0.8 / len(model_names)
    
    plt.figure(figsize=(10, 6))
    
    for i, model_name in enumerate(model_names):
        # Extract values for this model
        values = []
        for metric in metrics:
            value = per_model_results[model_name]["overall"].get(metric, 0.0)
            values.append(value)
        
        # Plot bars for this model
        offset = (i - len(model_names)/2 + 0.5) * bar_width
        bars = plt.bar(x + offset, values, bar_width, label=model_name)
        
        # Add value labels on top of the bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Add winning model indicators
    for i, metric in enumerate(metrics):
        if metric in winners:
            winner_model = winners[metric]["model"]
            if winner_model:
                # Find the index of the winning model
                winner_idx = model_names.index(winner_model)
                offset = (winner_idx - len(model_names)/2 + 0.5) * bar_width
                
                # Add a star or indicator
                plt.plot(i + offset, winners[metric]["value"] + 0.03, 'v', color='red', markersize=8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Overall Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    
    # Save plot in all requested formats
    saved_paths = []
    for format in formats:
        file_path = output_path / f"comparison_overall_bar.{format}"
        plt.savefig(file_path, dpi=dpi if format == 'png' else None)
        saved_paths.append(str(file_path))
    
    plt.close()
    return saved_paths


def _create_category_bar_chart(
    comparison_results: Dict[str, Any],
    output_path: Path,
    model_names: List[str],
    category: str,
    formats: List[str],
    dpi: int
) -> List[str]:
    """Create bar chart comparing metrics for a specific category across models."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    per_model_results = comparison_results["per_model"]
    winners = comparison_results["winner"]["per_category"].get(category, {})
    
    # Check if we have data for this category
    has_data = False
    for model_name in model_names:
        if (category in per_model_results[model_name]["per_category"] and
            per_model_results[model_name]["per_category"][category]):
            has_data = True
            break
    
    if not has_data:
        return []
    
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(metrics))
    bar_width = 0.8 / len(model_names)
    
    plt.figure(figsize=(10, 6))
    
    for i, model_name in enumerate(model_names):
        # Extract values for this model
        values = []
        for metric in metrics:
            value = 0.0
            if (category in per_model_results[model_name]["per_category"] and
                metric in per_model_results[model_name]["per_category"][category]):
                value = per_model_results[model_name]["per_category"][category][metric]
            values.append(value)
        
        # Plot bars for this model
        offset = (i - len(model_names)/2 + 0.5) * bar_width
        bars = plt.bar(x + offset, values, bar_width, label=model_name)
        
        # Add value labels on top of the bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Add winning model indicators
    for i, metric in enumerate(metrics):
        if metric in winners:
            winner_model = winners[metric]["model"]
            if winner_model:
                # Find the index of the winning model
                winner_idx = model_names.index(winner_model)
                offset = (winner_idx - len(model_names)/2 + 0.5) * bar_width
                
                # Add a star or indicator
                plt.plot(i + offset, winners[metric]["value"] + 0.03, 'v', color='red', markersize=8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(f'Model Performance Comparison: {category}')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    
    # Save plot in all requested formats
    saved_paths = []
    for format in formats:
        file_path = output_path / f"comparison_{category}_bar.{format}"
        plt.savefig(file_path, dpi=dpi if format == 'png' else None)
        saved_paths.append(str(file_path))
    
    plt.close()
    return saved_paths


def _create_overall_radar_chart(
    comparison_results: Dict[str, Any],
    output_path: Path,
    model_names: List[str],
    formats: List[str],
    dpi: int
) -> List[str]:
    """Create radar chart comparing overall metrics across models."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    per_model_results = comparison_results["per_model"]
    
    # Define metrics and angles for the radar chart
    metrics = ["precision", "recall", "f1"]
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per metric and add labels
    plt.xticks(angles[:-1], metrics)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1)
    
    # Plot each model
    for i, model_name in enumerate(model_names):
        values = []
        for metric in metrics:
            values.append(per_model_results[model_name]["overall"].get(metric, 0.0))
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Overall Model Performance Comparison')
    
    # Save plot in all requested formats
    saved_paths = []
    for format in formats:
        file_path = output_path / f"comparison_overall_radar.{format}"
        plt.savefig(file_path, dpi=dpi if format == 'png' else None, bbox_inches='tight')
        saved_paths.append(str(file_path))
    
    plt.close()
    return saved_paths


def _create_category_radar_chart(
    comparison_results: Dict[str, Any],
    output_path: Path,
    model_names: List[str],
    categories: List[str],
    formats: List[str],
    dpi: int
) -> List[str]:
    """Create radar chart comparing F1 scores across categories."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    per_model_results = comparison_results["per_model"]
    
    # Check if we have enough categories for a radar chart (at least 3)
    if len(categories) < 3:
        logger.warning("Need at least 3 categories for radar chart, skipping")
        return []
    
    # Define angles for the radar chart
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per category and add labels
    plt.xticks(angles[:-1], categories)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1)
    
    # Plot each model's F1 scores
    for i, model_name in enumerate(model_names):
        values = []
        for category in categories:
            value = 0.0
            if (category in per_model_results[model_name]["per_category"] and
                "f1" in per_model_results[model_name]["per_category"][category]):
                value = per_model_results[model_name]["per_category"][category]["f1"]
            values.append(value)
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('F1 Score Comparison Across Categories')
    
    # Save plot in all requested formats
    saved_paths = []
    for format in formats:
        file_path = output_path / f"comparison_f1_by_category_radar.{format}"
        plt.savefig(file_path, dpi=dpi if format == 'png' else None, bbox_inches='tight')
        saved_paths.append(str(file_path))
    
    plt.close()
    return saved_paths


def generate_comparative_report(
    comparison_results: Dict[str, Any],
    output_path: str,
    include_plots: bool = True,
    title: str = "Model Comparison Report"
) -> str:
    """
    Create a comprehensive PDF report comparing multiple models.
    
    Args:
        comparison_results: Results from compare_models function
        output_path: Path to save the PDF report
        include_plots: Whether to include visualization plots in the report
        title: Title for the PDF report
        
    Returns:
        Path to the generated PDF report
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
            PageBreak, Image, ListFlowable, ListItem
        )
        import matplotlib.pyplot as plt
        import numpy as np
        import io
        import datetime
    except ImportError:
        raise ImportError(
            "Required packages for PDF report generation are not installed. "
            "Please install reportlab and matplotlib using: "
            "pip install reportlab matplotlib"
        )
    
    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If a PDF extension is not provided, add it
    if not output_path.name.lower().endswith('.pdf'):
        output_path = output_path.with_suffix('.pdf')
    
    # Create a PDF document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    # Initialize story with document elements
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading1_style = styles['Heading1']
    heading2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create custom styles
    table_title_style = ParagraphStyle(
        'TableTitle',
        parent=styles['Heading3'],
        spaceAfter=6
    )
    
    # Add title and date
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Add introduction
    story.append(Paragraph("Introduction", heading1_style))
    story.append(Paragraph(
        "This report presents a comparative analysis of multiple toxicity classification models "
        "evaluated on the same validation dataset. It includes head-to-head performance metrics, "
        "visualizations, and statistical significance analysis.",
        normal_style
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Get model information
    per_model_results = comparison_results.get("per_model", {})
    winners = comparison_results.get("winner", {})
    model_names = list(per_model_results.keys())
    
    # Handle empty case
    if not model_names:
        story.append(Paragraph("No models found for comparison", normal_style))
        doc.build(story)
        return str(output_path)
    
    # Add models section
    story.append(Paragraph("1. Models Overview", heading1_style))
    
    # Create table of models being compared
    model_data = [
        ["Model", "Type", "Version", "Categories"]
    ]
    
    for model_name in model_names:
        model_info = per_model_results[model_name].get("model_info", {})
        model_type = model_info.get("type", "Unknown")
        model_version = model_info.get("version", "Unknown")
        
        # Get categories for this model
        categories = list(per_model_results[model_name].get("per_category", {}).keys())
        categories_str = ", ".join(categories) if len(categories) <= 5 else (
            ", ".join(categories[:5]) + f", +{len(categories)-5} more"
        )
        
        model_data.append([
            model_name,
            model_type,
            model_version,
            categories_str
        ])
    
    model_table = Table(model_data)
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
    ]))
    
    story.append(Paragraph("Models Compared", table_title_style))
    story.append(model_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Add overall performance section
    story.append(Paragraph("2. Overall Performance Comparison", heading1_style))
    
    # Create table of overall metrics
    overall_data = [
        ["Metric"] + model_names + ["Best Model"]
    ]
    
    for metric in ["precision", "recall", "f1"]:
        row = [metric.capitalize()]
        best_model = winners.get("overall", {}).get(metric, {}).get("model", "")
        
        for model_name in model_names:
            value = per_model_results[model_name]["overall"].get(metric, 0.0)
            row.append(f"{value:.4f}")
        
        row.append(best_model)
        
        overall_data.append(row)
    
    overall_table = Table(overall_data)
    overall_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
    ]))
    
    story.append(Paragraph("Overall Metrics", table_title_style))
    story.append(overall_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Build the PDF
    doc.build(story)
    
    print(f"Comparative report generated: {output_path}")
    return str(output_path)