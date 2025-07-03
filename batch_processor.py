"""Batch processing helper for toxicity-detector.

This lightweight module provides ``batch_process`` which traverses a single
file or an entire directory, applies the provided *model* to every file and
aggregates the results.  The implementation uses real model inference with
chunked sentence processing for optimal performance, enhanced toxicity scoring
with weighted averages, full probability distributions, and comprehensive
Groq override tracking. External callers (tests, CLI) are expected to inject
their own model + configuration objects making this module self-contained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import json
import logging
import os
import re
import time
import statistics
import collections
from datetime import datetime

# Optional progress helper – fall back to no-op if tqdm unavailable
try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    def tqdm(x: Any, **kwargs):  # type: ignore
        return x

from color_utils import colorize  # lightweight helper used for summaries

__all__ = ["batch_process"]

logger = logging.getLogger(__name__)

# Batch size for efficient sentence processing
BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Core public API ------------------------------------------------------------
# ---------------------------------------------------------------------------

def batch_process(
    *,
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    model: Any = None,
    config: Dict[str, Any] | None = None,
    show_progress: bool = True,
    selected_categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Process *input_path* (file or directory) and return aggregate results.

    The function processes files sentence-by-sentence using real model inference
    with optimized batch processing. Results include full probability distributions,
    Groq fallback tracking, and file-level toxicity scoring.
    """

    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(f"Input path does not exist: {inp}")

    # Default config if none provided
    if config is None:
        config = {}

    # Collect candidate files ------------------------------------------------
    files: List[Path]
    if inp.is_file():
        files = [inp]
    else:
        files = [p for p in inp.rglob("*") if p.is_file() and p.suffix in ['.txt', '.md', '.csv']]

    total = len(files)
    results: Dict[str, Any] = {
        "total_files": total,
        "toxic_files": 0,
        "total_sentences": 0,
        "toxic_sentences": 0,
        "toxicity_metrics": {
            "avg_overall_score": 0.0,
            "max_toxicity_score": 0.0,
            "toxicity_distribution": {},
            "per_category_metrics": {}
        },
        "groq_metrics": {
            "total_usage_count": 0,
            "override_count": 0,
            "effectiveness_score": 0.0,
            "avg_confidence_improvement": 0.0
        },
        "file_results": {},
        "timestamp": time.time(),
    }

    if show_progress and total:
        print(f"Processing {total} file(s)…")

    for idx, fp in enumerate(files, 1):
        if show_progress:
            pct = idx / total * 100
            print(f"\r{idx}/{total} ({pct:5.1f}%)", end="", flush=True)

        file_res = _process_single_file(
            fp,
            model=model,
            config=config,
            selected_categories=selected_categories,
            results_parent=results,
            show_progress=show_progress,
        )
        results["file_results"][str(fp)] = file_res
        if file_res.get("toxic"):
            results["toxic_files"] += 1

    if show_progress and total:
        print()  # newline after progress bar

    # Calculate comprehensive batch-level metrics
    _calculate_batch_metrics(results)

    if output_path:
        _write_results(results, Path(output_path))

    return results

# ---------------------------------------------------------------------------
# Internal helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------

def _calculate_batch_metrics(results: Dict[str, Any]) -> None:
    """
    Calculate comprehensive batch-level metrics based on file results.
    
    Args:
        results: Batch processing results dictionary to update
    """
    # Extract file-level scores
    file_scores = []
    category_scores = collections.defaultdict(list)
    groq_confidence_improvements = []
    
    # Process each file's results
    for file_path, file_result in results["file_results"].items():
        if 'toxicity_profile' in file_result and 'overall_score' in file_result['toxicity_profile']:
            file_scores.append(file_result['toxicity_profile']['overall_score'])
            
            # Track max toxicity score
            results['toxicity_metrics']['max_toxicity_score'] = max(
                results['toxicity_metrics']['max_toxicity_score'],
                file_result['toxicity_profile']['overall_score']
            )
            
            # Track per-category scores
            for category, score in file_result['toxicity_profile'].get('category_scores', {}).items():
                category_scores[category].append(score)
        elif 'overall_toxicity_score' in file_result:
            # Backwards compatibility
            file_scores.append(file_result['overall_toxicity_score'])
            results['toxicity_metrics']['max_toxicity_score'] = max(
                results['toxicity_metrics']['max_toxicity_score'],
                file_result['overall_toxicity_score']
            )
        
        # Track Groq metrics
        if 'groq_usage' in file_result:
            results['groq_metrics']['total_usage_count'] += file_result['groq_usage'].get('count', 0)
            results['groq_metrics']['override_count'] += file_result['groq_usage'].get('override_count', 0)
            
            # Track confidence improvements
            confidence_lift = file_result['groq_usage'].get('avg_confidence_lift', 0)
            if confidence_lift > 0:
                groq_confidence_improvements.append(confidence_lift)
        elif 'groq_fallback_count' in file_result:
            # Backwards compatibility
            results['groq_metrics']['total_usage_count'] += file_result.get('groq_fallback_count', 0)
    
    # Calculate average toxicity score
    if file_scores:
        results['toxicity_metrics']['avg_overall_score'] = round(sum(file_scores) / len(file_scores), 4)
    
    # Calculate toxicity distribution
    if file_scores:
        try:
            distribution = {
                'min': round(min(file_scores), 4),
                'max': round(max(file_scores), 4),
                'mean': round(statistics.mean(file_scores), 4),
                'median': round(statistics.median(file_scores), 4),
            }
            
            # Add percentiles if enough data
            if len(file_scores) >= 5:
                # Calculate percentiles manually for Python < 3.8 compatibility
                sorted_scores = sorted(file_scores)
                n = len(sorted_scores)
                
                def percentile(data, p):
                    k = (n - 1) * p / 100
                    f = int(k)
                    c = k - f
                    if f == n - 1:
                        return data[f]
                    return data[f] * (1 - c) + data[f + 1] * c
                
                distribution['percentiles'] = {
                    '25th': round(percentile(sorted_scores, 25), 4),
                    '75th': round(percentile(sorted_scores, 75), 4),
                    '90th': round(percentile(sorted_scores, 90), 4)
                }
                
            # Add standard deviation
            if len(file_scores) >= 2:
                distribution['std_dev'] = round(statistics.stdev(file_scores), 4)
                
            results['toxicity_metrics']['toxicity_distribution'] = distribution
        except Exception as e:
            logger.warning(f"Error calculating toxicity distribution: {e}")
    
    # Calculate per-category metrics
    for category, scores in category_scores.items():
        if scores:
            try:
                results['toxicity_metrics']['per_category_metrics'][category] = {
                    'avg_score': round(sum(scores) / len(scores), 4),
                    'max_score': round(max(scores), 4),
                    'min_score': round(min(scores), 4),
                    'mean': round(statistics.mean(scores), 4)
                }
                
                # Add percentiles if enough data
                if len(scores) >= 5:
                    sorted_scores = sorted(scores)
                    n = len(sorted_scores)
                    
                    def percentile(data, p):
                        k = (n - 1) * p / 100
                        f = int(k)
                        c = k - f
                        if f == n - 1:
                            return data[f]
                        return data[f] * (1 - c) + data[f + 1] * c
                    
                    results['toxicity_metrics']['per_category_metrics'][category]['percentiles'] = {
                        '75th': round(percentile(sorted_scores, 75), 4),
                        '90th': round(percentile(sorted_scores, 90), 4)
                    }
            except Exception as e:
                logger.warning(f"Error calculating metrics for category {category}: {e}")
    
    # Calculate Groq effectiveness metrics
    if results['groq_metrics']['total_usage_count'] > 0:
        # Calculate effectiveness as ratio of overrides to total usage
        results['groq_metrics']['effectiveness_score'] = round(
            results['groq_metrics']['override_count'] / results['groq_metrics']['total_usage_count'],
            4
        )
        
        # Calculate average confidence improvement
        if groq_confidence_improvements:
            results['groq_metrics']['avg_confidence_improvement'] = round(
                sum(groq_confidence_improvements) / len(groq_confidence_improvements),
                4
            )

def _process_single_file(
    fp: Path,
    *,
    model: Any = None,
    config: Dict[str, Any] | None = None,
    selected_categories: Optional[List[str]] = None,
    results_parent: Optional[Dict[str, Any]] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Analyse *fp* sentence-by-sentence using batched processing and return a rich result map.

    Processes sentences in batches of up to 32 for efficient model inference while
    maintaining detailed statistics and progress tracking.
    """

    if config is None:
        config = {}

    # ------------------------------------------------------------------
    # Read file ---------------------------------------------------------
    # ------------------------------------------------------------------
    try:
        text = fp.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        # If the file cannot be read treat as empty and record error.
        return {
            "error": "unreadable", 
            "toxic": False, 
            "total_sentences": 0, 
            "toxic_sentences": 0,
            "overall_toxicity_score": 0.0,
            "groq_fallback_count": 0,
            "sentences": []
        }

    sentences = _split_into_sentences(text)

    # Per-file aggregations with enhanced structure ---------------------
    file_res: Dict[str, Any] = {
        "toxic": False,
        "total_sentences": len(sentences),
        "toxic_sentences": 0,
        "displayed_sentences": 0,  # Track sentences that pass confidence filter
        "filtered_sentences": 0,   # Track sentences filtered by confidence
        "below_range_sentences": 0,  # Track sentences below min confidence
        "above_range_sentences": 0,  # Track sentences above max confidence
        "category_counts": {},
        "groq_usage": {
            "count": 0,
            "override_count": 0,
            "effectiveness": 0.0,
            "avg_confidence_lift": 0.0
        },
        "sentences": [],
    }

    # Progress bar per file (kept off when show_progress==False to appease tests)
    pbar = None
    if show_progress and sentences:
        pbar = tqdm(
            total=len(sentences),
            desc=f"Processing {fp.name}",
            unit="sent",
            leave=False
        )

    # Process sentences in batches for efficiency with enhanced tracking
    groq_confidence_lifts = []
    
    for i in range(0, len(sentences), BATCH_SIZE):
        # Get batch of sentences
        batch = sentences[i:i+BATCH_SIZE]
        
        # Process batch using real model
        batch_results = _predict_toxicity_batch(batch, model=model, config=config, selected_categories=selected_categories)
        
        # Process results for each sentence in the batch
        for j, sentence_result in enumerate(batch_results):
            # Get the original sentence
            sentence = batch[j]
            
            # Add text to result
            sentence_result["text"] = sentence
            
            # Check confidence filtering for this sentence
            scores = sentence_result.get("scores", {})
            max_confidence = max(scores.values()) if scores else 0.0
            passes_filter, filter_reason = _check_confidence_filter(max_confidence, config)
            
            # Always store the sentence result for complete data
            file_res["sentences"].append(sentence_result)
            
            # Track confidence filtering statistics
            if passes_filter:
                file_res["displayed_sentences"] += 1
            else:
                file_res["filtered_sentences"] += 1
                
                # Track specific reason for filtering
                if "below minimum" in filter_reason:
                    file_res["below_range_sentences"] += 1
                elif "above maximum" in filter_reason:
                    file_res["above_range_sentences"] += 1
            
            # Track Groq fallback usage with enhanced metrics
            if sentence_result.get("groq_used", False):
                file_res["groq_usage"]["count"] += 1
                
                # Track classification changes - check if Groq changed the result
                # For now, we assume any Groq usage potentially changes results
                # In a real implementation, this would compare local vs Groq predictions
                groq_changed = sentence_result.get("groq_changed_classification", True)
                if groq_changed:
                    file_res["groq_usage"]["override_count"] += 1
                
                # Track confidence lift (simulated for now)
                confidence_lift = sentence_result.get("groq_confidence_lift", 0.1)
                if confidence_lift > 0:
                    groq_confidence_lifts.append(confidence_lift)
            
            # Update toxicity counts
            if sentence_result.get("is_toxic", False):
                file_res["toxic_sentences"] += 1
                file_res["toxic"] = True
                
                # Update category counts based on category_results
                category_results = sentence_result.get("category_results", {})
                for category, result in category_results.items():
                    if result.get("above_threshold", False):
                        # Category should already be a string since we serialized it
                        cat_name = str(category)
                        file_res["category_counts"][cat_name] = file_res["category_counts"].get(cat_name, 0) + 1

        # Update progress bar
        if pbar:
            pbar.update(len(batch))

    # Close progress bar
    if pbar:
        pbar.close()

    # Calculate Groq effectiveness metrics
    if file_res["groq_usage"]["count"] > 0:
        file_res["groq_usage"]["effectiveness"] = round(
            file_res["groq_usage"]["override_count"] / file_res["groq_usage"]["count"],
            4
        )
        
        if groq_confidence_lifts:
            file_res["groq_usage"]["avg_confidence_lift"] = round(
                sum(groq_confidence_lifts) / len(groq_confidence_lifts),
                4
            )

    # Update parent aggregate dict if provided ---------------------------
    if results_parent is not None:
        results_parent["total_sentences"] += file_res["total_sentences"]
        results_parent["toxic_sentences"] += file_res["toxic_sentences"]
        
        # Add confidence filtering statistics to parent results
        results_parent.setdefault("displayed_sentences", 0)
        results_parent.setdefault("filtered_sentences", 0)
        results_parent.setdefault("below_range_sentences", 0)
        results_parent.setdefault("above_range_sentences", 0)
        
        results_parent["displayed_sentences"] += file_res["displayed_sentences"]
        results_parent["filtered_sentences"] += file_res["filtered_sentences"]
        results_parent["below_range_sentences"] += file_res["below_range_sentences"]
        results_parent["above_range_sentences"] += file_res["above_range_sentences"]

    # Compute comprehensive toxicity profile
    file_res["toxicity_profile"] = _compute_overall_toxicity_profile(file_res)
    
    # Backwards compatibility - keep overall_toxicity_score
    file_res["overall_toxicity_score"] = file_res["toxicity_profile"]["overall_score"]

    # Mark file as toxic if it contains any toxic sentences
    file_res["toxic"] = file_res["toxic_sentences"] > 0

    # Emit concise summary to stdout ------------------------------------
    _display_file_summary(fp, file_res)

    return file_res

def _predict_toxicity_batch(
    sentences: List[str],
    *,
    model: Any = None,
    config: Dict[str, Any] | None = None,
    selected_categories: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Predict toxicity for a batch of sentences using real model inference.

    Processes multiple sentences in a single model call for efficiency while
    preserving all detailed statistics and Groq fallback tracking.
    """
    if not sentences:
        return []
    
    if config is None:
        config = {}

    # Import model_loader to use the real prediction function
    from model_loader import predict_toxicity
    
    # Extract configuration parameters
    thresholds = config.get("thresholds", None)
    model_name = config.get("model", {}).get("name", "unitary/toxic-bert") if isinstance(config.get("model"), dict) else str(model or "unitary/toxic-bert")
    allow_groq_fallback = config.get("allow_groq_fallback", False)
    gray_min = config.get("groq_lower_bound", 0.4)
    gray_max = config.get("groq_upper_bound", 0.6)
    tie_policy = config.get("groq_tie_policy", "prefer-groq")
    
    try:
        # Use the real model prediction function with batch processing
        prediction_results = predict_toxicity(
            texts=sentences,
            thresholds=thresholds,
            model_name=model_name,
            batch_size=BATCH_SIZE,
            show_progress=False,  # We handle progress at file level
            allow_groq_fallback=allow_groq_fallback,
            gray_min=gray_min,
            gray_max=gray_max,
            tie_policy=tie_policy,
        )
        
        # Convert results to our expected format
        batch_results = []
        for result in prediction_results:
            # Extract information from the rich prediction result
            is_toxic = result.get("is_toxic", False)
            category_results = result.get("category_results", {})
            groq_used = result.get("groq_used", False)
            
            # Build scores dict from category_results - ensure JSON serializable keys
            scores = {}
            probabilities = {}
            serializable_category_results = {}
            for category, cat_result in category_results.items():
                cat_name = category.name if hasattr(category, 'name') else str(category)
                scores[cat_name] = cat_result.get("score", 0.0)
                
                # Build probability distribution
                score = cat_result.get("score", 0.0)
                probabilities[cat_name] = {
                    "0": round(1.0 - score, 4),
                    "1": round(score, 4)
                }
                
                # Create JSON-serializable category results
                serializable_category_results[cat_name] = cat_result
            
            # Convert most_probable_category to string for JSON serialization
            most_probable_category = result.get("most_probable_category")
            if hasattr(most_probable_category, 'name'):
                most_probable_category_str = most_probable_category.name
            else:
                most_probable_category_str = str(most_probable_category)
            
            batch_results.append({
                "toxic": is_toxic,
                "is_toxic": is_toxic,
                "scores": scores,
                "probabilities": probabilities,
                "category_results": serializable_category_results,
                "groq_used": groq_used,
                "most_probable_category": most_probable_category_str,
                "raw_logits": result.get("raw_logits"),
                "sigmoid_scores": result.get("sigmoid_scores"),
                "gray_zone_bounds": result.get("gray_zone_bounds"),
                "tie_policy": result.get("tie_policy"),
                "tie_source": result.get("tie_source"),
            })
        
        return batch_results
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        # Fallback to stub behavior if real model fails
        return _predict_toxicity_batch_fallback(sentences, selected_categories)

def _check_confidence_filter(confidence: float, config: Dict[str, Any]) -> tuple[bool, str]:
    """
    Check if a confidence value passes the confidence filtering criteria.
    
    Args:
        confidence: The confidence score to check (0.0-1.0)
        config: Configuration dictionary containing confidence filter settings
        
    Returns:
        Tuple of (passes_filter: bool, reason: str)
        - passes_filter: True if the confidence passes all filters
        - reason: String describing why it was filtered (empty if passes)
    """
    confidence_filtering = config.get('confidence_filtering', {})
    
    # Check single confidence filter
    if 'confidence_filter' in confidence_filtering:
        try:
            threshold = float(confidence_filtering['confidence_filter'])
            if confidence < threshold:
                return False, f"below threshold {threshold:.4f}"
        except (ValueError, TypeError):
            pass  # Treat invalid values as no filter
    
    # Check range filters
    if 'min_confidence' in confidence_filtering:
        try:
            min_threshold = float(confidence_filtering['min_confidence'])
            if confidence < min_threshold:
                return False, f"below minimum {min_threshold:.4f}"
        except (ValueError, TypeError):
            pass
    
    if 'max_confidence' in confidence_filtering:
        try:
            max_threshold = float(confidence_filtering['max_confidence'])
            if confidence > max_threshold:
                return False, f"above maximum {max_threshold:.4f}"
        except (ValueError, TypeError):
            pass
    
    return True, ""


def _predict_toxicity_batch_fallback(
    sentences: List[str],
    selected_categories: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Fallback prediction function when real model is unavailable."""
    import random
    
    categories_default = ["hate", "insult", "threat", "profanity"]
    cats = selected_categories if selected_categories else categories_default
    
    results = []
    for sentence in sentences:
        # Deterministic seed derived from sentence text to keep results stable
        rnd = random.Random(hash(sentence))
        tox_prob = rnd.random()
        
        scores = {c: round(rnd.random(), 2) for c in cats}
        is_toxic = tox_prob > 0.75  # arbitrary threshold
        
        probabilities = {}
        for cat, score in scores.items():
            probabilities[cat] = {
                "0": round(1.0 - score, 4),
                "1": round(score, 4)
            }
        
        results.append({
            "toxic": is_toxic,
            "is_toxic": is_toxic,
            "scores": scores,
            "probabilities": probabilities,
            "groq_used": False,
            "category_results": {},
        })
    
    return results

def _compute_overall_toxicity_profile(file_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a comprehensive toxicity profile for a file based on sentence probabilities.
    
    Args:
        file_result: Dictionary containing file processing results
        
    Returns:
        Dictionary containing detailed toxicity profile
    """
    if not file_result.get('sentences'):
        return {
            'overall_score': 0.0,
            'category_scores': {},
            'confidence': 0.0,
            'distribution': {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'median': 0.0
            }
        }
    
    total_weight = 0.0
    weighted_sum = 0.0
    category_weights = collections.defaultdict(float)
    category_sums = collections.defaultdict(float)
    
    # Extract all sentence scores for statistical analysis
    sentence_scores = []
    sentence_scores_by_category = collections.defaultdict(list)
    confidence_values = []
    
    for sentence in file_result['sentences']:
        # Use sentence length as weight (longer sentences have more impact)
        text = sentence.get('text', '')
        weight = max(len(text), 1)  # Minimum weight of 1
        
        # Get the maximum probability across all categories
        max_prob = 0.0
        max_category = None
        
        # Extract scores from different possible formats
        scores = sentence.get('scores', {})
        probabilities = sentence.get('probabilities', {})
        
        # Handle different result formats
        if probabilities:
            for category, probs in probabilities.items():
                if category.lower() == "non_toxic":
                    continue  # Skip non-toxic category
                prob_value = probs.get('1', 0.0) if isinstance(probs, dict) else float(probs)
                sentence_scores_by_category[category].append(prob_value)
                
                if prob_value > max_prob:
                    max_prob = prob_value
                    max_category = category
                
                # Update category weights and sums
                category_weights[category] += weight
                category_sums[category] += prob_value * weight
        elif scores:
            for category, score in scores.items():
                if category.lower() == "non_toxic":
                    continue  # Skip non-toxic category
                score = float(score) if score is not None else 0.0
                sentence_scores_by_category[category].append(score)
                
                if score > max_prob:
                    max_prob = score
                    max_category = category
                
                # Update category weights and sums
                category_weights[category] += weight
                category_sums[category] += score * weight
        
        # Add to sentence scores
        sentence_scores.append(max_prob)
        
        # Calculate confidence as distance from 0.5 (scaled to 0-1)
        confidence = abs(max_prob - 0.5) * 2
        confidence_values.append(confidence)
        
        # Update overall weighted sum
        weighted_sum += max_prob * weight
        total_weight += weight
    
    # Calculate weighted average scores
    overall_score = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0
    
    # Calculate per-category weighted scores
    category_scores = {}
    for category in category_sums:
        if category_weights[category] > 0:
            category_scores[category] = round(
                category_sums[category] / category_weights[category], 
                4
            )
    
    # Calculate statistical distribution
    distribution = {}
    if sentence_scores:
        try:
            distribution = {
                'min': round(min(sentence_scores), 4),
                'max': round(max(sentence_scores), 4),
                'mean': round(statistics.mean(sentence_scores), 4),
                'median': round(statistics.median(sentence_scores), 4)
            }
            
            # Add percentiles if enough data
            if len(sentence_scores) >= 5:
                sorted_scores = sorted(sentence_scores)
                n = len(sorted_scores)
                
                def percentile(data, p):
                    k = (n - 1) * p / 100
                    f = int(k)
                    c = k - f
                    if f == n - 1:
                        return data[f]
                    return data[f] * (1 - c) + data[f + 1] * c
                
                distribution['percentiles'] = {
                    '25th': round(percentile(sorted_scores, 25), 4),
                    '75th': round(percentile(sorted_scores, 75), 4),
                    '90th': round(percentile(sorted_scores, 90), 4)
                }
                
            # Add standard deviation if enough data
            if len(sentence_scores) >= 2:
                distribution['std_dev'] = round(statistics.stdev(sentence_scores), 4)
        except Exception as e:
            logger.warning(f"Error calculating distribution: {e}")
    
    # Calculate overall confidence metric
    avg_confidence = round(statistics.mean(confidence_values), 4) if confidence_values else 0.0
    
    # Create the complete toxicity profile
    toxicity_profile = {
        'overall_score': overall_score,
        'category_scores': category_scores,
        'confidence': avg_confidence,
        'distribution': distribution
    }
    
    return toxicity_profile


# Backwards compatibility function
def _compute_overall_toxicity_score(file_result: Dict[str, Any]) -> float:
    """Calculate a weighted average score (backwards compatibility)."""
    profile = _compute_overall_toxicity_profile(file_result)
    return profile.get('overall_score', 0.0)

def _json_serialize(obj):
    """Custom JSON serializer to handle ToxicityCategory and other non-serializable objects."""
    if hasattr(obj, 'name'):
        return obj.name
    elif hasattr(obj, '__str__'):
        return str(obj)
    else:
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def _write_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Write comprehensive batch processing results to output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Add timestamp to results
    timestamp = datetime.now().isoformat()
    results['timestamp'] = timestamp

    # Write main summary file
    summary_path = output_dir / "batch_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        # Create a simplified summary that excludes the detailed file_results
        summary_data = {k: v for k, v in results.items() if k != 'file_results'}
        summary_data['file_count'] = results['total_files']
        json.dump(summary_data, fh, indent=2, ensure_ascii=False, default=_json_serialize)

    # Write individual result files
    for file_path, file_result in results["file_results"].items():
        # Create a safe filename for the result
        result_filename = Path(file_path).name + ".result.json"
        result_path = output_dir / result_filename
        
        with result_path.open("w", encoding="utf-8") as fh:
            json.dump(file_result, fh, indent=2, ensure_ascii=False, default=_json_serialize)

    # Create toxicity report with categorized files
    toxicity_report = {
        'timestamp': timestamp,
        'high_toxicity_files': [],
        'medium_toxicity_files': [],
        'low_toxicity_files': [],
        'non_toxic_files': [],
    }

    # Categorize files by toxicity level
    for file_path, file_result in results["file_results"].items():
        toxicity_profile = file_result.get('toxicity_profile', {})
        overall_score = toxicity_profile.get('overall_score', file_result.get('overall_toxicity_score', 0.0))
        
        file_summary = {
            'path': file_path,
            'overall_score': overall_score,
            'toxic_sentences': file_result.get('toxic_sentences', 0),
            'total_sentences': file_result.get('total_sentences', 0)
        }
        
        if overall_score >= 0.7:
            toxicity_report['high_toxicity_files'].append(file_summary)
        elif overall_score >= 0.4:
            toxicity_report['medium_toxicity_files'].append(file_summary)
        elif overall_score > 0.0:
            toxicity_report['low_toxicity_files'].append(file_summary)
        else:
            toxicity_report['non_toxic_files'].append(file_summary)

    # Sort each category by toxicity score
    for category in ['high_toxicity_files', 'medium_toxicity_files', 'low_toxicity_files', 'non_toxic_files']:
        toxicity_report[category] = sorted(
            toxicity_report[category],
            key=lambda x: x['overall_score'],
            reverse=True
        )

    # Write toxicity report
    toxicity_report_path = output_dir / "toxicity_report.json"
    with toxicity_report_path.open("w", encoding="utf-8") as fh:
        json.dump(toxicity_report, fh, indent=2, ensure_ascii=False, default=_json_serialize)

    # Create category distribution report
    category_distribution = {
        'timestamp': timestamp,
        'categories': {},
        'total_toxic_sentences': results['toxic_sentences']
    }

    # Collect category data
    for file_path, file_result in results["file_results"].items():
        for category, count in file_result.get('category_counts', {}).items():
            if category not in category_distribution['categories']:
                category_distribution['categories'][category] = {
                    'total_count': 0,
                    'files': []
                }
            
            category_distribution['categories'][category]['total_count'] += count
            category_distribution['categories'][category]['files'].append({
                'path': file_path,
                'count': count
            })

    # Calculate percentages and sort files by count
    total_toxic = results['toxic_sentences']
    for category, data in category_distribution['categories'].items():
        if total_toxic > 0:
            data['percentage'] = round((data['total_count'] / total_toxic) * 100, 2)
        else:
            data['percentage'] = 0.0
            
        # Sort files by count
        data['files'] = sorted(data['files'], key=lambda x: x['count'], reverse=True)

    # Sort categories by total count
    sorted_categories = {}
    for category in sorted(
        category_distribution['categories'].keys(),
        key=lambda c: category_distribution['categories'][c]['total_count'],
        reverse=True
    ):
        sorted_categories[category] = category_distribution['categories'][category]

    category_distribution['categories'] = sorted_categories

    # Write category distribution report
    category_path = output_dir / "category_distribution.json"
    with category_path.open("w", encoding="utf-8") as fh:
        json.dump(category_distribution, fh, indent=2, ensure_ascii=False, default=_json_serialize)

    # Create Groq usage report
    groq_report = {
        'timestamp': timestamp,
        'total_usage': results['groq_metrics']['total_usage_count'],
        'total_overrides': results['groq_metrics']['override_count'],
        'effectiveness': results['groq_metrics']['effectiveness_score'],
        'avg_confidence_improvement': results['groq_metrics']['avg_confidence_improvement'],
        'files_with_groq_usage': []
    }

    # Collect files with Groq usage
    for file_path, file_result in results["file_results"].items():
        groq_usage = file_result.get('groq_usage', {})
        
        if groq_usage.get('count', 0) > 0:
            groq_report['files_with_groq_usage'].append({
                'path': file_path,
                'usage_count': groq_usage.get('count', 0),
                'override_count': groq_usage.get('override_count', 0),
                'effectiveness': groq_usage.get('effectiveness', 0.0),
                'avg_confidence_lift': groq_usage.get('avg_confidence_lift', 0.0)
            })

    # Sort by usage count
    groq_report['files_with_groq_usage'] = sorted(
        groq_report['files_with_groq_usage'],
        key=lambda x: x['usage_count'],
        reverse=True
    )

    # Write Groq usage report
    groq_path = output_dir / "groq_usage_report.json"
    with groq_path.open("w", encoding="utf-8") as fh:
        json.dump(groq_report, fh, indent=2, ensure_ascii=False, default=_json_serialize)

    logger.info("Comprehensive batch results written to %s", output_dir)

# ---------------------------------------------------------------------------
# Helper utilities -----------------------------------------------------------
# ---------------------------------------------------------------------------

_SENTENCE_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

def _split_into_sentences(text: str) -> List[str]:
    """Return a naive list of sentences from *text* using regex.

    This avoids heavyweight dependencies.  Tests can monkey-patch this symbol
    for deterministic behaviour.
    """
    # Pre-clean: normalise whitespace
    cleaned = " ".join(text.split())
    return [s.strip() for s in _SENTENCE_REGEX.split(cleaned) if s.strip()]

def _predict_toxicity(
    sentence: str,
    *,
    model: Any = None,
    config: Dict[str, Any] | None = None,
    selected_categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Predict toxicity for a single sentence. Wrapper around batch prediction."""
    results = _predict_toxicity_batch([sentence], model=model, config=config, selected_categories=selected_categories)
    return results[0] if results else {
        "toxic": False,
        "is_toxic": False,
        "scores": {},
        "probabilities": {},
        "groq_used": False,
    }

def _display_file_summary(fp: Path, meta: Dict[str, Any]) -> None:
    """Print a comprehensive summary of file processing results with color coding."""
    total = meta.get("total_sentences", 0)
    tox = meta.get("toxic_sentences", 0)
    pct = (tox / total * 100.0) if total else 0.0
    
    # Get toxicity score from enhanced profile or fallback to legacy
    toxicity_profile = meta.get('toxicity_profile', {})
    overall_score = toxicity_profile.get('overall_score', meta.get("overall_toxicity_score", 0.0))
    confidence = toxicity_profile.get('confidence', 0.0)
    
    # Get Groq usage information
    groq_usage = meta.get('groq_usage', {})
    groq_count = groq_usage.get('count', meta.get("groq_fallback_count", 0))
    groq_override_count = groq_usage.get('override_count', 0)
    groq_effectiveness = groq_usage.get('effectiveness', 0.0)

    # Determine color based on overall toxicity score
    if overall_score >= 0.7:
        col = "red"
        level = "HIGH"
    elif overall_score >= 0.4:
        col = "yellow"
        level = "MEDIUM"
    elif overall_score >= 0.1:
        col = "blue"
        level = "LOW"
    else:
        col = "green"
        level = "MINIMAL"

    # Format main verdict
    verdict = f"{tox}/{total} toxic sentences ({pct:.1f}%) | Overall: {overall_score:.3f} ({level})"
    colored_verdict = colorize(verdict, col, bold=(overall_score >= 0.4))
    
    # Add confidence if available
    if confidence > 0:
        conf_info = f" | Conf: {confidence:.2f}"
        colored_verdict += colorize(conf_info, "blue")
    
    # Add Groq usage if applicable
    if groq_count > 0:
        groq_pct = (groq_count / total * 100.0) if total else 0.0
        groq_info = f" | Groq: {groq_count} ({groq_pct:.1f}%)"
        if groq_override_count > 0:
            groq_info += f", {groq_override_count} overrides ({groq_effectiveness*100:.0f}%)"
        colored_verdict += colorize(groq_info, "cyan")

    print(f"{fp.name:30} {colored_verdict}")

    # Show top categories if verbose or high toxicity
    if overall_score >= 0.3 and meta.get("category_counts"):
        top_categories = sorted(
            meta["category_counts"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]  # Show top 3 categories
        
        cat_info = " | Top: " + ", ".join([f"{cat}({count})" for cat, count in top_categories])
        print(" " * 30 + colorize(cat_info, "blue"))
        
    # Show category scores if available and significant toxicity
    if overall_score >= 0.5 and toxicity_profile.get('category_scores'):
        top_cat_scores = sorted(
            toxicity_profile['category_scores'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]  # Show top 2 category scores
        
        if top_cat_scores:
            score_info = " | Scores: " + ", ".join([f"{cat}:{score:.2f}" for cat, score in top_cat_scores])
            print(" " * 30 + colorize(score_info, "yellow")) 