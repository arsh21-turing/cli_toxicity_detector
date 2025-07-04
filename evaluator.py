#!/usr/bin/env python3
"""
Evaluator module for toxicity classification model.

This module provides functionality to evaluate model performance 
on labeled validation datasets, calculating precision, recall, 
and F1 scores for each toxicity category. It also includes 
threshold optimization to find the best threshold values for 
each category to maximize the macro-F1 score.
"""

import csv
import json
import yaml
import numpy as np
import multiprocessing
import itertools
import time
import os
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from pathlib import Path
from tqdm import tqdm

from color_utils import colorize, supports_color


def _parallel_evaluate_threshold(args):
    """
    Worker function for parallel threshold evaluation.
    
    Args:
        args: Tuple of (category, threshold, y_true, scores)
        
    Returns:
        Tuple of (category, threshold, metrics)
    """
    category, threshold, y_true, scores = args
    y_pred = [1 if score >= threshold else 0 for score in scores]
    metrics = calculate_metrics(y_true, y_pred)
    return category, threshold, metrics


def load_validation_dataset(csv_path: str, has_header: bool = True) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Load labeled validation data from a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing validation data
        has_header: Whether the CSV file has a header row (default: True)
        
    Returns:
        Tuple containing:
            - List of text samples
            - Dictionary mapping category names to binary label lists (0/1)
            
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV file is empty or has invalid format
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {csv_path}")
        
    texts = []
    labels_dict = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        # Handle header if present
        if has_header:
            try:
                header = next(reader)
                # First column is the text, the rest are category labels
                category_names = header[1:]
                for category in category_names:
                    labels_dict[category] = []
            except StopIteration:
                raise ValueError("CSV file is empty or contains no data")
        
        # Process data rows
        for row in reader:
            if not row:
                continue
                
            if len(row) < 2:
                raise ValueError(f"Invalid row format, expected text and at least one label: {row}")
            
            text = row[0]
            row_labels = [int(label) for label in row[1:]]
            
            texts.append(text)
            
            # If this is the first data row and we don't have a header,
            # initialize the labels dictionary with generic category names
            if not labels_dict:
                category_names = [f"category_{i}" for i in range(len(row_labels))]
                for category in category_names:
                    labels_dict[category] = []
            
            # Add labels for each category
            for i, category in enumerate(labels_dict.keys()):
                if i < len(row_labels):
                    labels_dict[category].append(row_labels[i])
    
    if not texts:
        raise ValueError("No valid data found in the CSV file")
        
    # Validate that all label lists have the same length as texts
    for category, category_labels in labels_dict.items():
        if len(category_labels) != len(texts):
            raise ValueError(f"Inconsistent number of labels for category '{category}'")
    
    return texts, labels_dict


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Union[float, int]]:
    """
    Calculate precision, recall, and F1 score for binary classification.
    
    Args:
        y_true: List of true binary labels (0 or 1)
        y_pred: List of predicted binary labels (0 or 1)
        
    Returns:
        Dictionary containing precision, recall, F1, and support counts
    """
    # Count true positives, false positives, etc.
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": tp + fn,  # Total number of true positives in the dataset
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn
    }


def evaluate_model(
    model: Any, 
    texts: List[str], 
    labels: Dict[str, List[int]], 
    thresholds: Optional[Dict[str, float]] = None,
    categories: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate model performance on a validation dataset.
    
    Args:
        model: The toxicity classification model to evaluate
        texts: List of text samples
        labels: Dictionary mapping category names to binary label lists (0/1)
        thresholds: Optional dictionary mapping categories to threshold values
        categories: Optional list of categories to evaluate (defaults to all in labels)
        
    Returns:
        Dictionary containing evaluation results with:
            - overall: Dictionary with macro averages for precision, recall, F1
            - per_category: Dictionary mapping each category to individual metrics
            - confusion_matrices: Dictionary mapping each category to confusion matrix
    """
    from model_loader import predict_toxicity
    
    # If no specific categories are provided, evaluate all categories in the labels
    if categories is None:
        categories = list(labels.keys())
    else:
        # Ensure all requested categories exist in the labels
        for category in categories:
            if category not in labels:
                raise ValueError(f"Category '{category}' not found in validation labels")
    
    # Default thresholds to 0.5 if not provided
    if thresholds is None:
        thresholds = {category: 0.5 for category in categories}
    
    # Initialize results structures
    per_category_metrics = {}
    confusion_matrices = {}
    
    # Get model predictions for all texts
    print(f"Running predictions on {len(texts)} samples...")
    predictions = predict_toxicity(
        texts=texts,
        thresholds=thresholds,
        model_name=getattr(model, 'model_name', 'default'),
        show_progress=True
    )
    
    # Calculate metrics for each category
    for category in categories:
        y_true = labels[category]
        
        # Extract predicted scores and apply threshold for this category
        threshold = thresholds.get(category, 0.5)
        y_pred = []
        
        for prediction in predictions:
            # Extract score for this category from prediction results
            if 'category_results' in prediction:
                # Handle new format with category objects
                category_score = 0.0
                for cat_obj, data in prediction['category_results'].items():
                    if hasattr(cat_obj, 'name') and cat_obj.name == category:
                        category_score = data['score']
                        break
                    elif str(cat_obj) == category:
                        category_score = data['score']
                        break
            elif 'probabilities' in prediction and category in prediction['probabilities']:
                # Handle legacy format
                category_score = prediction['probabilities'][category]
            elif 'scores' in prediction and category in prediction['scores']:
                # Handle alternative format
                category_score = prediction['scores'][category]
            else:
                # Default handling if structure is different
                category_score = 0.0
            
            # Apply threshold
            y_pred.append(1 if category_score >= threshold else 0)
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred)
        metrics['threshold'] = threshold
        per_category_metrics[category] = metrics
        
        # Create confusion matrix for this category
        confusion_matrix = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_true, y_pred):
            confusion_matrix[t, p] += 1
        
        confusion_matrices[category] = confusion_matrix.tolist()
    
    # Calculate macro averages
    categories_with_support = [cat for cat in categories if per_category_metrics[cat]['support'] > 0]
    
    if categories_with_support:
        overall_precision = sum(per_category_metrics[cat]['precision'] for cat in categories_with_support) / len(categories_with_support)
        overall_recall = sum(per_category_metrics[cat]['recall'] for cat in categories_with_support) / len(categories_with_support)
        overall_f1 = sum(per_category_metrics[cat]['f1'] for cat in categories_with_support) / len(categories_with_support)
    else:
        overall_precision = 0.0
        overall_recall = 0.0
        overall_f1 = 0.0
    
    # Compile final results
    evaluation_results = {
        'overall': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'num_categories': len(categories),
            'num_samples': len(texts)
        },
        'per_category': per_category_metrics,
        'confusion_matrices': confusion_matrices
    }
    
    return evaluation_results


def optimize_thresholds(
    model: Any,
    texts: List[str],
    labels: Dict[str, List[int]],
    threshold_range: Tuple[float, float] = (0.1, 0.9),
    step_size: float = 0.05,
    categories: Optional[List[str]] = None,
    parallel: bool = True
) -> Dict[str, Any]:
    """
    Find optimal thresholds for each category to maximize F1 scores.
    
    Args:
        model: The toxicity classification model to evaluate
        texts: List of text samples
        labels: Dictionary mapping category names to binary label lists (0/1)
        threshold_range: Tuple of (min_threshold, max_threshold) to search
        step_size: Step size between threshold values to test
        categories: Optional list of categories to optimize (defaults to all in labels)
        parallel: Whether to use parallel processing for optimization
        
    Returns:
        Dictionary with:
            - optimal_thresholds: Dict mapping categories to their optimal thresholds
            - default_results: Evaluation results with default thresholds (0.5)
            - optimized_results: Evaluation results with optimal thresholds
            - improvement: The overall F1 score improvement
            - search_details: Detailed results of the threshold search per category
    """
    from model_loader import predict_toxicity
    
    # If no specific categories are provided, optimize all categories in the labels
    if categories is None:
        categories = list(labels.keys())
    else:
        # Ensure all requested categories exist in the labels
        for category in categories:
            if category not in labels:
                raise ValueError(f"Category '{category}' not found in validation labels")
    
    print(f"Optimizing thresholds for {len(categories)} categories...")
    print(f"Threshold range: {threshold_range}, Step size: {step_size}")
    
    # Process all texts at once to get predictions (only need to do this once)
    print("Processing texts for model predictions...")
    
    # Use a simple threshold dict for initial predictions - we'll extract raw scores
    initial_thresholds = {category: 0.5 for category in categories}
    predictions = predict_toxicity(
        texts=texts,
        thresholds=initial_thresholds,
        model_name=getattr(model, 'model_name', 'default'),
        show_progress=True
    )
    
    # Extract raw scores for each category
    category_scores = {}
    for category in categories:
        category_scores[category] = []
        for prediction in predictions:
            # Extract score for this category from prediction results
            if 'category_results' in prediction:
                # Handle new format with category objects
                category_score = 0.0
                for cat_obj, data in prediction['category_results'].items():
                    if hasattr(cat_obj, 'name') and cat_obj.name == category:
                        category_score = data['score']
                        break
                    elif str(cat_obj) == category:
                        category_score = data['score']
                        break
            elif 'probabilities' in prediction and category in prediction['probabilities']:
                # Handle legacy format
                category_score = prediction['probabilities'][category]
            elif 'scores' in prediction and category in prediction['scores']:
                # Handle alternative format
                category_score = prediction['scores'][category]
            else:
                # Default handling if structure is different
                category_score = 0.0
            
            category_scores[category].append(category_score)
    
    # Generate threshold values to test
    min_threshold, max_threshold = threshold_range
    threshold_values = list(np.arange(min_threshold, max_threshold + step_size/2, step_size))
    
    # Create a function to evaluate a specific threshold for a category
    def evaluate_threshold(category: str, threshold: float) -> Tuple[str, float, Dict[str, float]]:
        y_true = labels[category]
        scores = category_scores[category]
        y_pred = [1 if score >= threshold else 0 for score in scores]
        metrics = calculate_metrics(y_true, y_pred)
        return category, threshold, metrics
    
    # Create all category-threshold pairs to evaluate
    threshold_tasks = []
    for category in categories:
        for threshold in threshold_values:
            threshold_tasks.append((category, threshold))
    
    # Evaluate all thresholds for all categories
    search_results = {}
    
    if parallel and len(threshold_tasks) > 10:  # Only use parallel for larger tasks
        # Use parallel processing to speed up evaluation
        print(f"Evaluating {len(threshold_tasks)} thresholds with parallel processing...")
        
        # Prepare arguments for parallel processing
        parallel_args = [(category, threshold, labels[category], category_scores[category]) 
                        for category, threshold in threshold_tasks]
        
        # Create process pool and execute tasks
        with multiprocessing.Pool(processes=min(os.cpu_count(), 8)) as pool:
            results = list(tqdm(pool.imap(_parallel_evaluate_threshold, parallel_args), total=len(parallel_args)))
        
        # Process results
        for category, threshold, metrics in results:
            if category not in search_results:
                search_results[category] = {}
            search_results[category][threshold] = metrics
    else:
        # Use sequential processing
        print(f"Evaluating {len(threshold_tasks)} thresholds sequentially...")
        for category, threshold in tqdm(threshold_tasks):
            result = evaluate_threshold(category, threshold)
            category, threshold, metrics = result
            if category not in search_results:
                search_results[category] = {}
            search_results[category][threshold] = metrics
    
    # Find optimal threshold for each category based on F1 score
    optimal_thresholds = {}
    for category, thresholds_data in search_results.items():
        best_f1 = -1
        best_threshold = 0.5  # Default
        
        for threshold, metrics in thresholds_data.items():
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_threshold = threshold
        
        optimal_thresholds[category] = float(best_threshold)  # Ensure it's a plain Python float
    
    # Evaluate with default thresholds (0.5)
    default_thresholds = {category: 0.5 for category in categories}
    default_results = evaluate_model(model, texts, labels, thresholds=default_thresholds, categories=categories)
    
    # Evaluate with optimal thresholds
    optimized_results = evaluate_model(model, texts, labels, thresholds=optimal_thresholds, categories=categories)
    
    # Calculate improvement
    default_f1 = default_results['overall']['f1']
    optimized_f1 = optimized_results['overall']['f1']
    improvement = optimized_f1 - default_f1
    
    # Return complete results
    optimization_results = {
        'optimal_thresholds': optimal_thresholds,
        'default_results': default_results,
        'optimized_results': optimized_results,
        'improvement': improvement,
        'search_details': search_results
    }
    
    return optimization_results


def save_optimal_thresholds(optimal_thresholds: Dict[str, float], config_path: str) -> None:
    """
    Save optimal thresholds to a config file.
    
    Args:
        optimal_thresholds: Dictionary mapping categories to optimal threshold values
        config_path: Path to the config file to update
    
    Raises:
        FileNotFoundError: If the config file doesn't exist
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load existing config
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ('.yaml', '.yml'):
            config = yaml.safe_load(f) or {}
        else:
            config = json.load(f)
    
    # Update or create thresholds section
    if 'thresholds' not in config:
        config['thresholds'] = {}
    
    # Update thresholds with optimal values
    for category, threshold in optimal_thresholds.items():
        config['thresholds'][category] = float(threshold)  # Ensure it's a plain Python float
    
    # Write updated config back to file
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in ('.yaml', '.yml'):
            yaml.dump(config, f, default_flow_style=False)
        else:
            json.dump(config, f, indent=2)


def display_evaluation_results(evaluation_results: Dict[str, Any], show_confusion_matrices: bool = True) -> None:
    """
    Format and display evaluation results.
    
    Args:
        evaluation_results: Evaluation results from evaluate_model
        show_confusion_matrices: Whether to display confusion matrices for each category
    """
    overall = evaluation_results['overall']
    per_category = evaluation_results['per_category']
    
    print("\n" + "="*80)
    print(f"MODEL EVALUATION RESULTS")
    print("="*80)
    print(f"Total samples: {overall['num_samples']}")
    print(f"Number of categories: {overall['num_categories']}")
    print("\n" + "-"*80)
    
    # Overall metrics
    print(f"Overall (macro) metrics:")
    print(f"  Precision: {_format_metric_value(overall['precision'])}")
    print(f"  Recall:    {_format_metric_value(overall['recall'])}")
    print(f"  F1 Score:  {_format_metric_value(overall['f1'])}")
    print("\n" + "-"*80)
    
    # Per-category metrics
    print(f"Per-category metrics:")
    print(f"{'Category':<20} {'Support':<10} {'Precision':<15} {'Recall':<15} {'F1 Score':<15} {'Threshold':<10}")
    print("-"*80)
    
    categories = sorted(per_category.keys())
    for category in categories:
        metrics = per_category[category]
        print(
            f"{category:<20} "
            f"{metrics['support']:<10} "
            f"{_format_metric_value(metrics['precision']):<15} "
            f"{_format_metric_value(metrics['recall']):<15} "
            f"{_format_metric_value(metrics['f1']):<15} "
            f"{metrics['threshold']:<10.2f}"
        )
    
    # Display confusion matrices if requested
    if show_confusion_matrices:
        print("\n" + "-"*80)
        print("Confusion Matrices:")
        confusion_matrices = evaluation_results['confusion_matrices']
        
        for category in categories:
            matrix = confusion_matrices[category]
            print(f"\n{category}:")
            print(f"             Predicted")
            print(f"             Negative  Positive")
            print(f"Actual   Negative  {matrix[0][0]:<8}  {matrix[0][1]:<8}")
            print(f"         Positive  {matrix[1][0]:<8}  {matrix[1][1]:<8}")


def display_threshold_optimization(optimization_results: Dict[str, Any], show_all_thresholds: bool = False) -> None:
    """
    Display threshold optimization results.
    
    Args:
        optimization_results: Results from optimize_thresholds
        show_all_thresholds: Whether to show all tested thresholds and their metrics
    """
    optimal_thresholds = optimization_results['optimal_thresholds']
    default_results = optimization_results['default_results']
    optimized_results = optimization_results['optimized_results']
    improvement = optimization_results['improvement']
    search_details = optimization_results['search_details']
    
    print("\n" + "="*80)
    print(f"THRESHOLD OPTIMIZATION RESULTS")
    print("="*80)
    
    # Show overall improvement
    print(f"Overall F1 Score Improvement: {_format_improvement(improvement)}")
    print(f"Default F1 Score: {_format_metric_value(default_results['overall']['f1'])}")
    print(f"Optimized F1 Score: {_format_metric_value(optimized_results['overall']['f1'])}")
    print("\n" + "-"*80)
    
    # Show per-category results
    print(f"Per-category optimized thresholds:")
    print(f"{'Category':<20} {'Default F1':<15} {'Optimized F1':<15} {'Improvement':<15} {'Optimal Threshold':<20}")
    print("-"*80)
    
    categories = sorted(optimal_thresholds.keys())
    for category in categories:
        default_f1 = default_results['per_category'][category]['f1']
        optimized_f1 = optimized_results['per_category'][category]['f1']
        category_improvement = optimized_f1 - default_f1
        optimal_threshold = optimal_thresholds[category]
        
        print(
            f"{category:<20} "
            f"{_format_metric_value(default_f1):<15} "
            f"{_format_metric_value(optimized_f1):<15} "
            f"{_format_improvement(category_improvement):<15} "
            f"{optimal_threshold:<20.2f}"
        )
    
    # Show detailed threshold search results if requested
    if show_all_thresholds:
        print("\n" + "-"*80)
        print("Detailed threshold search results:")
        
        for category in categories:
            print(f"\n{category}:")
            print(f"{'Threshold':<10} {'Precision':<15} {'Recall':<15} {'F1 Score':<15}")
            print("-"*55)
            
            # Sort thresholds numerically
            thresholds = sorted(float(t) for t in search_details[category].keys())
            
            for threshold in thresholds:
                metrics = search_details[category][threshold]
                print(
                    f"{threshold:<10.2f} "
                    f"{_format_metric_value(metrics['precision']):<15} "
                    f"{_format_metric_value(metrics['recall']):<15} "
                    f"{_format_metric_value(metrics['f1']):<15}"
                )


def _format_metric_value(value: float) -> str:
    """Format a metric value with color coding based on value."""
    formatted = f"{value:.4f}"
    
    if not supports_color():
        return formatted
    
    if value >= 0.9:
        return colorize(formatted, "green")
    elif value >= 0.7:
        return colorize(formatted, "cyan")
    elif value >= 0.5:
        return colorize(formatted, "yellow")
    else:
        return colorize(formatted, "red")


def _format_improvement(value: float) -> str:
    """Format an improvement value with color coding based on value."""
    formatted = f"{value:+.4f}"  # Use + sign to indicate improvement
    
    if not supports_color():
        return formatted
    
    if value > 0.1:
        return colorize(formatted, "green")
    elif value > 0.05:
        return colorize(formatted, "cyan")
    elif value > 0:
        return colorize(formatted, "yellow")
    else:
        return colorize(formatted, "red")


def export_evaluation_results(evaluation_results: Dict[str, Any], output_path: str) -> None:
    """
    Export evaluation results to a JSON file.
    
    Args:
        evaluation_results: Evaluation results from evaluate_model
        output_path: Path to save the results JSON file
    """
    output_path = Path(output_path)
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"Evaluation results exported to: {output_path}")


def plot_precision_recall_curve(
    optimization_results: Dict[str, Any],
    output_path: str,
    categories: Optional[List[str]] = None,
    dpi: int = 300,
    fmt: str = 'both'
) -> Dict[str, float]:
    """
    Create precision-recall curves for each category.
    
    Args:
        optimization_results: Results from optimize_thresholds function
        output_path: Directory path to save plot files
        categories: Optional list of categories to plot (defaults to all)
        dpi: DPI for saved PNG images (default: 300)
        fmt: Output format: 'png', 'svg', or 'both' (default: 'both')
        
    Returns:
        Dictionary mapping category names to their AUPRC scores
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
        from sklearn.metrics import precision_recall_curve, auc
        import numpy as np
    except ImportError:
        raise ImportError(
            "Required packages for plotting are not installed. "
            "Please install matplotlib and scikit-learn using: "
            "pip install matplotlib scikit-learn"
        )
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine categories to plot
    search_details = optimization_results['search_details']
    optimal_thresholds = optimization_results['optimal_thresholds']
    
    if categories is None:
        categories = list(optimal_thresholds.keys())
    
    # Set up styles
    plt.style.use('default')
    colors = plt.cm.tab10.colors
    
    # Calculate AUPRC scores
    auprc_scores = {}
    
    # We need to simulate precision-recall curves from our threshold sweep data
    # For each category, create a precision-recall curve
    for i, category in enumerate(categories):
        # Extract thresholds and metrics for this category
        thresholds = []
        precision_values = []
        recall_values = []
        
        for threshold, metrics in sorted(search_details[category].items()):
            thresholds.append(float(threshold))
            precision_values.append(metrics['precision'])
            recall_values.append(metrics['recall'])
        
        # Sort by recall for proper PR curve
        sorted_data = sorted(zip(recall_values, precision_values))
        recall_sorted, precision_sorted = zip(*sorted_data) if sorted_data else ([], [])
        
        # Calculate AUPRC using trapezoidal rule
        if len(recall_sorted) > 1:
            auprc = np.trapezoid(precision_sorted, recall_sorted)
        else:
            auprc = 0.0
        auprc_scores[category] = auprc
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.plot(recall_sorted, precision_sorted, color=colors[i % len(colors)], lw=2, 
                label=f'AUPRC = {auprc:.3f}')
        
        # Find the point corresponding to optimal threshold
        optimal_threshold = optimal_thresholds[category]
        
        # Find the metrics for the optimal threshold
        optimal_metrics = search_details[category].get(optimal_threshold, {})
        if optimal_metrics:
            optimal_precision = optimal_metrics.get('precision', 0)
            optimal_recall = optimal_metrics.get('recall', 0)
            
            # Mark the optimal threshold point
            plt.scatter(
                optimal_recall, optimal_precision, 
                marker='o', color='red', s=100, zorder=5, 
                label=f'Optimal threshold: {optimal_threshold:.2f}'
            )
        
        # Format the plot
        plt.title(f'Precision-Recall Curve: {category}', fontsize=14)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='lower left', fontsize=10)
        plt.tight_layout()
        
        # Save in requested formats
        formats = []
        if fmt.lower() == 'png' or fmt.lower() == 'both':
            formats.append('png')
        if fmt.lower() == 'svg' or fmt.lower() == 'both':
            formats.append('svg')
            
        for format in formats:
            file_path = output_path / f"pr_curve_{category}.{format}"
            plt.savefig(file_path, dpi=dpi if format == 'png' else None, bbox_inches='tight')
        
        plt.close()
        
    # Create a combined plot with all categories
    plt.figure(figsize=(12, 10))
    
    for i, category in enumerate(categories):
        # Extract thresholds and metrics for this category
        thresholds = []
        precision_values = []
        recall_values = []
        
        for threshold, metrics in sorted(search_details[category].items()):
            thresholds.append(float(threshold))
            precision_values.append(metrics['precision'])
            recall_values.append(metrics['recall'])
        
        # Sort by recall for proper PR curve
        sorted_data = sorted(zip(recall_values, precision_values))
        recall_sorted, precision_sorted = zip(*sorted_data) if sorted_data else ([], [])
        
        # Plot the curve
        plt.plot(
            recall_sorted, precision_sorted, 
            color=colors[i % len(colors)], 
            lw=2, 
            label=f'{category} (AUPRC = {auprc_scores[category]:.3f})'
        )
    
    # Format the combined plot
    plt.title('Precision-Recall Curves: All Categories', fontsize=14)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    
    # Save the combined plot
    for format in formats:
        file_path = output_path / f"pr_curve_all_categories.{format}"
        plt.savefig(file_path, dpi=dpi if format == 'png' else None, bbox_inches='tight')
    
    plt.close()
    
    print(f"Precision-recall curves saved to {output_path}")
    return auprc_scores


def plot_threshold_sweep(
    optimization_results: Dict[str, Any],
    output_path: str,
    categories: Optional[List[str]] = None,
    dpi: int = 300,
    fmt: str = 'both'
) -> List[str]:
    """
    Create threshold sweep plots showing how metrics change with threshold.
    
    Args:
        optimization_results: Results from optimize_thresholds function
        output_path: Directory path to save plot files
        categories: Optional list of categories to plot (defaults to all)
        dpi: DPI for saved PNG images (default: 300)
        fmt: Output format: 'png', 'svg', or 'both' (default: 'both')
        
    Returns:
        List of paths to the generated plot files
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
    except ImportError:
        raise ImportError(
            "Required packages for plotting are not installed. "
            "Please install matplotlib using: pip install matplotlib"
        )
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine categories to plot
    search_details = optimization_results['search_details']
    optimal_thresholds = optimization_results['optimal_thresholds']
    
    if categories is None:
        categories = list(optimal_thresholds.keys())
    
    # Set up styles
    plt.style.use('default')
    
    # Determine output formats
    formats = []
    if fmt.lower() == 'png' or fmt.lower() == 'both':
        formats.append('png')
    if fmt.lower() == 'svg' or fmt.lower() == 'both':
        formats.append('svg')
    
    output_files = []
    
    # For each category, create a threshold sweep plot
    for category in categories:
        # Extract thresholds and metrics
        thresholds = []
        precision_values = []
        recall_values = []
        f1_values = []
        
        for threshold, metrics in sorted(search_details[category].items()):
            thresholds.append(float(threshold))
            precision_values.append(metrics['precision'])
            recall_values.append(metrics['recall'])
            f1_values.append(metrics['f1'])
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot metrics
        plt.plot(thresholds, precision_values, 'b-', lw=2, label='Precision')
        plt.plot(thresholds, recall_values, 'r-', lw=2, label='Recall')
        plt.plot(thresholds, f1_values, 'g-', lw=2, label='F1 Score')
        
        # Mark the optimal threshold with a vertical line
        optimal_threshold = optimal_thresholds[category]
        plt.axvline(
            x=optimal_threshold, 
            color='purple', 
            linestyle='--', 
            lw=1.5, 
            label=f'Optimal threshold: {optimal_threshold:.2f}'
        )
        
        # Find the F1 score at the optimal threshold
        optimal_metrics = search_details[category].get(optimal_threshold, {})
        optimal_f1 = optimal_metrics.get('f1', 0.0)
        
        # Add a marker at the optimal F1 point
        plt.plot(
            optimal_threshold, 
            optimal_f1, 
            'o', 
            color='purple', 
            markersize=8, 
            label=f'Optimal F1: {optimal_f1:.3f}'
        )
        
        # Format the plot
        plt.title(f'Threshold Sweep Analysis: {category}', fontsize=14)
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xlim([min(thresholds) - 0.05, max(thresholds) + 0.05])
        plt.ylim([-0.05, 1.05])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()
        
        # Save in requested formats
        for format in formats:
            file_path = output_path / f"threshold_sweep_{category}.{format}"
            plt.savefig(file_path, dpi=dpi if format == 'png' else None, bbox_inches='tight')
            output_files.append(str(file_path))
        
        plt.close()
    
    # Create a combined plot with F1 scores for all categories
    plt.figure(figsize=(12, 10))
    
    for i, category in enumerate(categories):
        # Extract thresholds and F1 values
        thresholds = []
        f1_values = []
        
        for threshold, metrics in sorted(search_details[category].items()):
            thresholds.append(float(threshold))
            f1_values.append(metrics['f1'])
        
        # Plot F1 curve for this category
        plt.plot(
            thresholds, 
            f1_values, 
            lw=2, 
            label=f'{category} F1'
        )
        
        # Mark optimal threshold with a vertical line (partially transparent)
        optimal_threshold = optimal_thresholds[category]
        plt.axvline(
            x=optimal_threshold, 
            color='purple', 
            linestyle='--', 
            lw=1, 
            alpha=0.3
        )
    
    # Format the combined plot
    plt.title('F1 Score Threshold Sweep: All Categories', fontsize=14)
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    # Save the combined plot
    for format in formats:
        file_path = output_path / f"threshold_sweep_all_categories.{format}"
        plt.savefig(file_path, dpi=dpi if format == 'png' else None, bbox_inches='tight')
        output_files.append(str(file_path))
    
    plt.close()
    
    print(f"Threshold sweep plots saved to {output_path}")
    return output_files


def plot_confusion_matrices(
    evaluation_results: Dict[str, Any],
    output_path: str,
    categories: Optional[List[str]] = None,
    dpi: int = 300,
    fmt: str = 'both'
) -> List[str]:
    """
    Create visual confusion matrix plots.
    
    Args:
        evaluation_results: Results from evaluate_model
        output_path: Directory path to save plot files
        categories: Optional list of categories to plot (defaults to all)
        dpi: DPI for saved PNG images (default: 300)
        fmt: Output format: 'png', 'svg', or 'both' (default: 'both')
        
    Returns:
        List of paths to the generated plot files
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError(
            "Required packages for plotting are not installed. "
            "Please install matplotlib using: pip install matplotlib"
        )
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine categories to plot
    confusion_matrices = evaluation_results['confusion_matrices']
    per_category = evaluation_results['per_category']
    
    if categories is None:
        categories = list(confusion_matrices.keys())
    
    # Set up styles
    plt.style.use('default')
    
    # Determine output formats
    formats = []
    if fmt.lower() == 'png' or fmt.lower() == 'both':
        formats.append('png')
    if fmt.lower() == 'svg' or fmt.lower() == 'both':
        formats.append('svg')
    
    output_files = []
    
    # For each category, create a confusion matrix visualization
    for category in categories:
        if category not in confusion_matrices:
            continue
        
        # Get the confusion matrix
        cm = np.array(confusion_matrices[category])
        
        # Create figure
        plt.figure(figsize=(9, 9))
        
        # Plot the confusion matrix
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix: {category}', fontsize=14)
        plt.colorbar()
        
        # Add labels
        classes = ['Negative', 'Positive']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontsize=12)
        plt.yticks(tick_marks, classes, fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        
        # Add count annotations
        thresh = cm.max() / 2.0
        total = cm.sum()
        
        for i, j in np.ndindex(cm.shape):
            count = cm[i, j]
            percentage = 100 * count / total if total > 0 else 0
            
            plt.text(
                j, i, f"{count}\n({percentage:.1f}%)",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12
            )
        
        # Get metrics for this category
        metrics = per_category[category]
        threshold = metrics['threshold']
        f1 = metrics['f1']
        precision = metrics['precision']
        recall = metrics['recall']
        
        # Add metrics as text
        plt.figtext(
            0.5, 0.01,
            f"Threshold: {threshold:.2f} | F1: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}",
            horizontalalignment="center",
            fontsize=12
        )
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save in requested formats
        for format in formats:
            file_path = output_path / f"confusion_matrix_{category}.{format}"
            plt.savefig(file_path, dpi=dpi if format == 'png' else None, bbox_inches='tight')
            output_files.append(str(file_path))
        
        plt.close()
    
    print(f"Confusion matrix plots saved to {output_path}")
    return output_files


def generate_pdf_report(
    evaluation_results: Dict[str, Any], 
    output_path: str,
    optimization_results: Optional[Dict[str, Any]] = None,
    title: str = "Model Evaluation Report",
    template_path: Optional[str] = None
) -> str:
    """
    Create a comprehensive PDF report with all evaluation results.
    
    Args:
        evaluation_results: Results from evaluate_model
        output_path: Path to save the PDF report file
        optimization_results: Optional results from optimize_thresholds
        title: Title for the PDF report (default: "Model Evaluation Report")
        template_path: Optional path to custom report template
        
    Returns:
        Path to the generated PDF file
        
    Raises:
        ImportError: If required packages are not installed
        ValueError: If evaluation_results is None or empty
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
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
        import numpy as np
        import io
        import datetime
    except ImportError:
        raise ImportError(
            "Required packages for PDF report generation are not installed. "
            "Please install reportlab matplotlib"
        )
    
    # Validate input
    if evaluation_results is None or not isinstance(evaluation_results, dict) or len(evaluation_results) == 0:
        raise ValueError("Evaluation results cannot be None or empty")
        
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
        "This report presents the evaluation results of the toxicity classification model "
        "on a validation dataset. It includes overall metrics, per-category performance, "
        "confusion matrices, and visualization of model behavior at different threshold values.",
        normal_style
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Add overall metrics section
    story.append(Paragraph("1. Overall Metrics", heading1_style))
    
    # Safely access overall metrics with defaults for missing values
    overall = evaluation_results.get('overall', {})
    if not overall:
        # Create default overall metrics if missing
        overall = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "num_categories": 0,
            "num_samples": 0
        }
    
    overall_data = [
        ["Metric", "Value"],
        ["Precision", f"{overall.get('precision', 0.0):.4f}"],
        ["Recall", f"{overall.get('recall', 0.0):.4f}"],
        ["F1 Score", f"{overall.get('f1', 0.0):.4f}"],
        ["Number of Categories", f"{overall.get('num_categories', len(evaluation_results.get('per_category', {})))}"],
        ["Number of Samples", f"{overall.get('num_samples', 0)}"]
    ]
    
    overall_table = Table(overall_data, colWidths=[2*inch, 2*inch])
    overall_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold')
    ]))
    
    story.append(Paragraph("Overall Model Performance", table_title_style))
    story.append(overall_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Add threshold optimization results if available
    section_num = 2
    if optimization_results and isinstance(optimization_results, dict) and 'optimal_thresholds' in optimization_results:
        story.append(Paragraph(f"{section_num}. Threshold Optimization Results", heading1_style))
        
        # Safely access optimization data with defaults
        improvement = optimization_results.get('improvement', 0.0)
        default_f1 = optimization_results.get('default_results', {}).get('overall', {}).get('f1', 0.0)
        optimized_f1 = optimization_results.get('optimized_results', {}).get('overall', {}).get('f1', 0.0)
        
        story.append(Paragraph(
            f"Threshold optimization improved the overall F1 score from {default_f1:.4f} to {optimized_f1:.4f}, "
            f"an improvement of {improvement:+.4f}" + 
            (f" ({improvement*100/default_f1:+.2f}%)." if default_f1 > 0 else "."),
            normal_style
        ))
        story.append(Spacer(1, 0.2*inch))
        
        # Create threshold optimization table
        optimal_thresholds = optimization_results.get('optimal_thresholds', {})
        opt_data = [
            ["Category", "Default F1", "Optimized F1", "Improvement", "Optimal Threshold"]
        ]
        
        if optimal_thresholds:
            categories = sorted(optimal_thresholds.keys())
            for category in categories:
                # Safely access category metrics with defaults
                default_results = optimization_results.get('default_results', {}).get('per_category', {})
                optimized_results = optimization_results.get('optimized_results', {}).get('per_category', {})
                
                default_f1 = default_results.get(category, {}).get('f1', 0.0)
                optimized_f1 = optimized_results.get(category, {}).get('f1', 0.0)
                category_improvement = optimized_f1 - default_f1
                
                opt_data.append([
                    category,
                    f"{default_f1:.4f}",
                    f"{optimized_f1:.4f}",
                    f"{category_improvement:+.4f}",
                    f"{optimal_thresholds.get(category, 0.5):.2f}"
                ])
        else:
            # Add a row indicating no optimization data available
            opt_data.append(["No optimization data", "-", "-", "-", "-"])
        
        opt_table = Table(opt_data, colWidths=[1.2*inch, 1.1*inch, 1.1*inch, 1.1*inch, 1.1*inch])
        opt_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
        ]))
        
        story.append(Paragraph("Threshold Optimization Summary", table_title_style))
        story.append(opt_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Add threshold sweep plots if we have the necessary data
        story.append(Paragraph("Threshold Sweep Analysis", heading2_style))
        
        search_details = optimization_results.get('search_details', {})
        if search_details and optimal_thresholds:
            # Create threshold sweep plots for the PDF
            categories = sorted(optimal_thresholds.keys())
            for category in categories:
                # Check if we have search details for this category
                if category not in search_details:
                    continue
                
                # Extract thresholds and metrics from search details
                cat_search_details = search_details.get(category, {})
                if not cat_search_details:
                    continue
                
                thresholds = []
                precision_values = []
                recall_values = []
                f1_values = []
                
                for threshold_str, metrics in sorted(cat_search_details.items()):
                    try:
                        threshold = float(threshold_str)
                        thresholds.append(threshold)
                        precision_values.append(metrics.get('precision', 0.0))
                        recall_values.append(metrics.get('recall', 0.0))
                        f1_values.append(metrics.get('f1', 0.0))
                    except (ValueError, TypeError):
                        # Skip invalid threshold values
                        continue
                
                if not thresholds:
                    continue
                
                # Create the plot
                plt.figure(figsize=(7, 5))
                plt.plot(thresholds, precision_values, 'b-', lw=2, label='Precision')
                plt.plot(thresholds, recall_values, 'r-', lw=2, label='Recall')
                plt.plot(thresholds, f1_values, 'g-', lw=2, label='F1 Score')
                
                # Mark optimal threshold if available
                optimal_threshold = optimal_thresholds.get(category)
                if optimal_threshold is not None:
                    plt.axvline(
                        x=optimal_threshold, 
                        color='purple', 
                        linestyle='--', 
                        lw=1.5, 
                        label=f'Optimal threshold: {optimal_threshold:.2f}'
                    )
                
                plt.title(f'Threshold Sweep: {category}')
                plt.xlabel('Threshold')
                plt.ylabel('Score')
                plt.xlim([min(thresholds), max(thresholds)])
                plt.ylim([0, 1.05])
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(loc='best', fontsize=8)
                plt.tight_layout()
                
                # Save to a BytesIO buffer
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150)
                img_buffer.seek(0)
                plt.close()
                
                # Add image to PDF
                img = Image(img_buffer, width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 0.1*inch))
        else:
            # Inform that there's not enough data for threshold plots
            story.append(Paragraph(
                "Insufficient data available for threshold sweep visualization.",
                normal_style
            ))
        
        # Add a page break before next section
        story.append(PageBreak())
        section_num += 1
    
    # Per-category metrics
    story.append(Paragraph(f"{section_num}. Per-Category Performance", heading1_style))
    
    per_category = evaluation_results.get('per_category', {})
    
    if per_category:
        categories = sorted(per_category.keys())
        
        category_data = [
            ["Category", "Support", "Precision", "Recall", "F1 Score", "Threshold"]
        ]
        
        for category in categories:
            metrics = per_category.get(category, {})
            category_data.append([
                category,
                str(metrics.get('support', 'N/A')),
                f"{metrics.get('precision', 0.0):.4f}",
                f"{metrics.get('recall', 0.0):.4f}",
                f"{metrics.get('f1', 0.0):.4f}",
                f"{metrics.get('threshold', 0.5):.2f}"
            ])
        
        category_table = Table(category_data)
        category_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
        ]))
        
        story.append(Paragraph("Per-Category Metrics", table_title_style))
        story.append(category_table)
        story.append(Spacer(1, 0.2*inch))
    else:
        # Inform that there are no per-category metrics
        story.append(Paragraph(
            "No per-category metrics available in the evaluation results.",
            normal_style
        ))
    
    # Confusion matrices
    story.append(Paragraph("Confusion Matrices", heading2_style))
    
    confusion_matrices = evaluation_results.get('confusion_matrices', {})
    
    if confusion_matrices and per_category:
        story.append(Paragraph(
            "The following confusion matrices show the model's prediction results for each category. "
            "Each cell shows the count and percentage of samples in that combination of actual and predicted values.",
            normal_style
        ))
        story.append(Spacer(1, 0.2*inch))
        
        # Create confusion matrix visualizations for each category that has data
        categories = sorted(per_category.keys())
        has_matrices = False
        
        for category in categories:
            if category not in confusion_matrices:
                continue
                
            cm = np.array(confusion_matrices[category])
            
            # Validate confusion matrix dimensions
            if cm.shape != (2, 2):
                # Skip invalid matrices
                continue
            
            has_matrices = True
            
            # Create confusion matrix plot
            plt.figure(figsize=(6, 5))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix: {category}')
            plt.colorbar()
            
            # Add labels
            classes = ['Negative', 'Positive']
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes)
            plt.yticks(tick_marks, classes)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            # Add count annotations
            thresh = cm.max() / 2.0 if cm.size > 0 and cm.max() > 0 else 0
            total = cm.sum() if cm.size > 0 else 1  # Avoid division by zero
            
            for i, j in np.ndindex(cm.shape):
                count = cm[i, j]
                percentage = 100 * count / total if total > 0 else 0
                
                plt.text(
                    j, i, f"{count}\n({percentage:.1f}%)",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10
                )
            
            plt.tight_layout()
            
            # Get metrics for this category
            metrics = per_category.get(category, {})
            threshold = metrics.get('threshold', 0.5)
            f1 = metrics.get('f1', 0.0)
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            
            # Add metrics as text
            plt.figtext(
                0.5, 0.01,
                f"Threshold: {threshold:.2f} | F1: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}",
                horizontalalignment="center",
                fontsize=8
            )
            
            # Save to a BytesIO buffer
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150)
            img_buffer.seek(0)
            plt.close()
            
            # Add image to PDF
            img = Image(img_buffer, width=6*inch, height=5*inch)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
            
        if not has_matrices:
            story.append(Paragraph(
                "No valid confusion matrices available for visualization.",
                normal_style
            ))
    else:
        story.append(Paragraph(
            "No confusion matrix data available in the evaluation results.",
            normal_style
        ))
    
    # If we have optimization results with search details, add precision-recall curves
    if (optimization_results and 
        isinstance(optimization_results, dict) and 
        'search_details' in optimization_results and
        optimization_results['search_details']):
        
        story.append(PageBreak())
        
        section_num += 1
        story.append(Paragraph(f"{section_num}. Precision-Recall Analysis", heading1_style))
        
        story.append(Paragraph(
            "This section shows precision-recall curves for each category, illustrating the trade-off "
            "between precision and recall at different threshold settings. The optimal threshold point "
            "is marked on each curve.",
            normal_style
        ))
        story.append(Spacer(1, 0.2*inch))
        
        # Safely access relevant data
        search_details = optimization_results.get('search_details', {})
        optimal_thresholds = optimization_results.get('optimal_thresholds', {})
        categories = sorted(optimal_thresholds.keys())
        
        try:
            from sklearn.metrics import precision_recall_curve, auc
            
            # Calculate AUPRC scores
            auprc_scores = {}
            auprc_data = [["Category", "AUPRC Score"]]
            has_pr_curves = False
            
            for category in categories:
                if category not in search_details:
                    continue
                
                # Extract true labels and scores
                y_true_flat = []
                y_score_flat = []
                
                try:
                    for threshold, metrics in sorted(search_details[category].items()):
                        # If metrics contains y_true and y_score directly, use those
                        if 'y_true' in metrics and 'y_score' in metrics:
                            y_true_flat.extend(metrics['y_true'])
                            y_score_flat.extend(metrics['y_score'])
                        else:
                            # Skip if we can't find the necessary data
                            continue
                except (TypeError, ValueError):
                    # Skip this category if there's an error extracting data
                    continue
                
                # If we have enough data and both positive and negative samples, calculate PR curve
                if len(y_true_flat) > 0 and len(y_score_flat) > 0 and len(set(y_true_flat)) > 1:
                    has_pr_curves = True
                    precision, recall, thresholds = precision_recall_curve(y_true_flat, y_score_flat)
                    
                    # Calculate Area Under PR Curve
                    auprc = auc(recall, precision)
                    auprc_scores[category] = auprc
                    
                    # Add to table
                    auprc_data.append([category, f"{auprc:.4f}"])
                    
                    # Create figure
                    plt.figure(figsize=(7, 5))
                    plt.plot(recall, precision, lw=2, label=f'AUPRC = {auprc:.3f}')
                    
                    # Find the point on the curve corresponding to optimal threshold
                    optimal_threshold = optimal_thresholds.get(category)
                    
                    # Find the closest threshold in our precision-recall curve thresholds
                    if optimal_threshold is not None and len(thresholds) > 0:
                        optimal_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - optimal_threshold))
                        
                        # Mark the optimal threshold point
                        if optimal_idx < len(precision) and optimal_idx < len(recall):
                            plt.scatter(
                                recall[optimal_idx], precision[optimal_idx], 
                                marker='o', color='red', s=100, zorder=5, 
                                label=f'Optimal threshold: {optimal_threshold:.2f}'
                            )
                    
                    # Format the plot
                    plt.title(f'Precision-Recall Curve: {category}')
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.xlim([0.0, 1.05])
                    plt.ylim([0.0, 1.05])
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.legend(loc='best', fontsize=8)
                    
                    # Save to a BytesIO buffer
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', dpi=150)
                    img_buffer.seek(0)
                    plt.close()
                    
                    # Add image to PDF
                    img = Image(img_buffer, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.2*inch))
            
            # Add AUPRC table if we have scores
            if auprc_scores:
                auprc_table = Table(auprc_data)
                auprc_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
                ]))
                
                story.append(Paragraph("Area Under Precision-Recall Curve (AUPRC) Scores", table_title_style))
                story.append(auprc_table)
                story.append(Spacer(1, 0.2*inch))
            
            if not has_pr_curves:
                story.append(Paragraph(
                    "Insufficient data available to generate precision-recall curves. "
                    "This may be due to missing y_true/y_score values or categories with only one class.",
                    normal_style
                ))
        except ImportError:
            story.append(Paragraph(
                "Could not generate precision-recall curves: sklearn is required but not installed. "
                "Please install scikit-learn to enable this functionality.",
                normal_style
            ))
    
    # Conclusions and recommendations
    story.append(PageBreak())
    
    section_num += 1
    story.append(Paragraph(f"{section_num}. Conclusions and Recommendations", heading1_style))
    
    # Generate insights based on the evaluation results
    conclusions = []
    
    # Overall performance
    overall_f1 = overall.get('f1', 0.0)
    if overall_f1 >= 0.8:
        conclusions.append(
            "The model demonstrates strong overall performance with a macro F1 score of "
            f"{overall_f1:.4f}, indicating good balance between precision and recall."
        )
    elif overall_f1 >= 0.6:
        conclusions.append(
            "The model shows moderate overall performance with a macro F1 score of "
            f"{overall_f1:.4f}. There is room for improvement in balancing precision and recall."
        )
    elif overall_f1 > 0:
        conclusions.append(
            "The model's overall performance is limited, with a macro F1 score of "
            f"{overall_f1:.4f}. Significant improvements are needed in both precision and recall."
        )
    else:
        conclusions.append(
            "No meaningful performance data is available for overall assessment."
        )
    
    # Category-specific insights if we have at least one category
    if per_category and len(per_category) > 0:
        # Filter out categories with missing F1 scores
        valid_categories = [(k, v) for k, v in per_category.items() if 'f1' in v]
        
        if valid_categories:
            best_category = max(valid_categories, key=lambda x: x[1]['f1'])
            worst_category = min(valid_categories, key=lambda x: x[1]['f1'])
            
            conclusions.append(
                f"The model performs best on the '{best_category[0]}' category with an F1 score of "
                f"{best_category[1]['f1']:.4f}, and worst on the '{worst_category[0]}' category with an F1 score of "
                f"{worst_category[1]['f1']:.4f}."
            )
        else:
            conclusions.append(
                "No valid F1 scores found for individual categories."
            )
    elif per_category and len(per_category) == 1:
        # Only one category
        category, metrics = next(iter(per_category.items()))
        if 'f1' in metrics:
            conclusions.append(
                f"The model was evaluated on a single category ('{category}') "
                f"with an F1 score of {metrics['f1']:.4f}."
            )
    else:
        conclusions.append(
            "No per-category metrics available for assessment."
        )
    
    # Threshold optimization insights
    if optimization_results and 'improvement' in optimization_results:
        improvement = optimization_results['improvement']
        if improvement > 0.1:
            conclusions.append(
                "Threshold optimization provides substantial improvement in model performance. "
                "It is strongly recommended to use the optimized thresholds in production."
            )
        elif improvement > 0.05:
            conclusions.append(
                "Threshold optimization offers moderate improvement in model performance. "
                "Using optimized thresholds is recommended for better results."
            )
        elif improvement > 0:
            conclusions.append(
                "Threshold optimization provides slight improvement in model performance. "
                "The optimized thresholds should be considered if precision-recall balance is critical."
            )
        else:
            conclusions.append(
                "Threshold optimization did not improve overall performance. "
                "The default threshold of 0.5 appears to be optimal for this model."
            )
    
    # If we have no meaningful conclusions, add a default one
    if not conclusions:
        conclusions.append(
            "Insufficient evaluation data available to draw meaningful conclusions."
        )
    
    # Add conclusions as a list
    story.append(Paragraph("Key Insights:", heading2_style))
    
    conclusion_items = [ListItem(Paragraph(text, normal_style)) for text in conclusions]
    story.append(ListFlowable(conclusion_items, bulletType='bullet'))
    story.append(Spacer(1, 0.2*inch))
    
    # Recommendations
    story.append(Paragraph("Recommendations:", heading2_style))
    
    recommendations = []
    
    # Threshold recommendations
    if optimization_results and 'improvement' in optimization_results and optimization_results['improvement'] > 0:
        recommendations.append(
            "Implement the optimized thresholds for each category to maximize the F1 scores "
            "and achieve the best balance between precision and recall."
        )
    
    # Category-specific recommendations
    if per_category:
        low_performing = [category for category, metrics in per_category.items() 
                         if 'f1' in metrics and metrics['f1'] < 0.6]
        if low_performing:
            categories_str = ", ".join(f"'{cat}'" for cat in low_performing)
            recommendations.append(
                f"Focus on improving model performance for the following categories: {categories_str}. "
                "Consider collecting more training data or adjusting model parameters for these categories."
            )
    
    # Missing data recommendations
    missing_support = [category for category, metrics in per_category.items() 
                      if 'support' not in metrics or metrics['support'] == 0]
    if missing_support:
        categories_str = ", ".join(f"'{cat}'" for cat in missing_support)
        recommendations.append(
            f"Collect validation data for the following categories with insufficient samples: {categories_str}. "
            "Accurate evaluation requires a representative set of positive and negative examples."
        )
    
    # If we have no meaningful recommendations, add a default one
    if not recommendations:
        recommendations.append(
            "Insufficient evaluation data to provide specific recommendations."
        )
    
    # Add recommendations as a list
    recommendation_items = [ListItem(Paragraph(text, normal_style)) for text in recommendations]
    story.append(ListFlowable(recommendation_items, bulletType='bullet'))
    
    # Add error handling note if there were data issues
    data_issues = False
    
    if not per_category:
        data_issues = True
    elif any('support' not in metrics for category, metrics in per_category.items()):
        data_issues = True
    elif not confusion_matrices:
        data_issues = True
    
    if data_issues:
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Note on Data Quality:", heading2_style))
        story.append(Paragraph(
            "This report was generated with incomplete evaluation data. Some metrics or visualizations "
            "may be missing or incomplete. For a more comprehensive analysis, please ensure that the "
            "evaluation data includes complete metrics for all categories, proper confusion matrices, "
            "and sufficient support counts.",
            normal_style
        ))
    
    # Build the PDF
    doc.build(story)
    
    print(f"PDF report generated: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    # Example code to demonstrate usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate toxicity classification model")
    parser.add_argument("--csv", required=True, help="Path to validation dataset CSV file")
    parser.add_argument("--model", required=True, help="Path to model file or directory")
    parser.add_argument("--output", help="Path to save evaluation results JSON file")
    parser.add_argument("--no-headers", action="store_true", help="CSV file does not have headers")
    
    # Evaluation arguments
    group = parser.add_argument_group("Evaluation Options")
    group.add_argument("--threshold", type=float, default=0.5, help="Default threshold for all categories")
    group.add_argument("--category-thresholds", help="JSON file with per-category thresholds")
    
    # Threshold optimization arguments
    group = parser.add_argument_group("Threshold Optimization")
    group.add_argument("--optimize", action="store_true", help="Optimize thresholds for best F1 score")
    group.add_argument("--min-threshold", type=float, default=0.1, help="Minimum threshold to test")
    group.add_argument("--max-threshold", type=float, default=0.9, help="Maximum threshold to test")
    group.add_argument("--threshold-step", type=float, default=0.05, help="Step size for threshold testing")
    group.add_argument("--save-thresholds", help="Path to save optimal thresholds config")
    
    args = parser.parse_args()
    
    # This section would typically load the model and run the evaluation
    print("Example usage: Implement your model loading and evaluation here") 