import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, 
    roc_auc_score
)


class PerformanceTracker:
    """Tracks and analyzes model performance over time."""
    
    def __init__(self, reference_data: Optional[Dict] = None):
        """
        Initialize the performance tracker.
        
        Args:
            reference_data: Optional reference dataset with labeled examples
        """
        self.model_history = defaultdict(list)
        self.reference_data = reference_data or {}
        self.ground_truth = {}
        self._load_mock_data()  # Load some initial mock data
    
    def add_analysis(self, model_name: str, text: str, results: Dict, 
                    ground_truth: Optional[Dict] = None, execution_time: float = 0.0):
        """
        Add a model analysis result to the performance history.
        
        Args:
            model_name: Name of the model used (e.g., 'primary', 'groq')
            text: The text that was analyzed
            results: Analysis results from the model (category scores)
            ground_truth: Optional ground truth labels for evaluation
            execution_time: Time taken for analysis in seconds
        """
        # Create an entry with timestamps, results, and performance data
        timestamp = datetime.now()
        text_hash = hash(text) % 10000000  # Simple hash for text identification
        
        entry = {
            "timestamp": timestamp,
            "text_hash": text_hash,
            "text_snippet": text[:100] + "..." if len(text) > 100 else text,
            "results": results,
            "execution_time": execution_time
        }
        
        # Add ground truth if provided
        if ground_truth:
            entry["ground_truth"] = ground_truth
            # Also store in ground_truth dictionary for cross-model comparison
            self.ground_truth[text_hash] = ground_truth
            
        # Add to history
        self.model_history[model_name].append(entry)
    
    def get_performance_metrics(self, model_name: str, 
                              time_period: Optional[timedelta] = None) -> Dict:
        """
        Calculate performance metrics for a specific model.
        
        Args:
            model_name: Name of the model to evaluate
            time_period: Optional time period to restrict analysis (e.g., last 24 hours)
            
        Returns:
            Dictionary with accuracy, speed, and calibration metrics
        """
        if model_name not in self.model_history:
            return {"error": f"No data found for model {model_name}"}
        
        # Filter by time period if specified
        history = self.model_history[model_name]
        if time_period:
            cutoff_time = datetime.now() - time_period
            history = [entry for entry in history 
                      if entry["timestamp"] >= cutoff_time]
        
        if not history:
            return {"error": f"No data found for model {model_name} in specified time period"}
        
        # Speed metrics
        execution_times = [entry["execution_time"] for entry in history]
        avg_execution_time = np.mean(execution_times)
        median_execution_time = np.median(execution_times)
        p95_execution_time = np.percentile(execution_times, 95)
        
        # Count entries with ground truth
        entries_with_ground_truth = [
            entry for entry in history if "ground_truth" in entry
        ]
        
        # Accuracy metrics (only if ground truth is available)
        accuracy_metrics = {}
        if entries_with_ground_truth:
            # For each category, calculate binary classification metrics
            categories = set()
            for entry in entries_with_ground_truth:
                categories.update(entry["results"].keys())
                categories.update(entry["ground_truth"].keys())
                
            for category in categories:
                # Collect true and predicted values
                y_true = []
                y_pred = []
                y_scores = []
                
                for entry in entries_with_ground_truth:
                    if (category in entry["ground_truth"] and 
                        category in entry["results"]):
                        # Get ground truth (binarize to 0 or 1)
                        gt_value = entry["ground_truth"].get(category, 0)
                        gt_binary = 1 if gt_value >= 0.5 else 0
                        y_true.append(gt_binary)
                        
                        # Get predicted score (floats)
                        pred_score = entry["results"].get(category, 0.0)
                        y_scores.append(pred_score)
                        
                        # Binarize prediction using 0.5 threshold
                        y_pred.append(1 if pred_score >= 0.5 else 0)
                
                if y_true and sum(y_true) > 0:  # Only if we have positive examples
                    metrics = {
                        "accuracy": accuracy_score(y_true, y_pred),
                        "precision": precision_score(y_true, y_pred, zero_division=0),
                        "recall": recall_score(y_true, y_pred, zero_division=0),
                        "f1": f1_score(y_true, y_pred, zero_division=0),
                        "auc": roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0.5,
                        "samples": len(y_true),
                        "positive_rate": sum(y_true) / len(y_true)
                    }
                    accuracy_metrics[category] = metrics
        
        return {
            "model_name": model_name,
            "sample_count": len(history),
            "time_period": {
                "start": history[0]["timestamp"] if history else None,
                "end": history[-1]["timestamp"] if history else None,
            },
            "execution_time": {
                "mean": avg_execution_time,
                "median": median_execution_time,
                "p95": p95_execution_time,
            },
            "accuracy": accuracy_metrics if accuracy_metrics else None,
            "has_ground_truth": bool(entries_with_ground_truth)
        }
    
    def get_comparative_metrics(self, models: Optional[List[str]] = None,
                              time_period: Optional[timedelta] = None) -> Dict:
        """
        Get comparative performance metrics across models.
        
        Args:
            models: List of model names to compare (defaults to all models)
            time_period: Optional time period to restrict analysis
            
        Returns:
            Dictionary with comparative metrics
        """
        # Use all models if none specified
        if not models:
            models = list(self.model_history.keys())
        
        # Get metrics for each model
        model_metrics = {}
        for model in models:
            metrics = self.get_performance_metrics(model, time_period)
            if "error" not in metrics:
                model_metrics[model] = metrics
        
        # Calculate head-to-head metrics for texts that were analyzed by multiple models
        head_to_head = {}
        
        # Find text hashes that were analyzed by all models
        model_hashes = {}
        for model in model_metrics:
            hashes = set(entry["text_hash"] for entry in self.model_history[model])
            model_hashes[model] = hashes
        
        common_hashes = set.intersection(*[hashes for hashes in model_hashes.values()])
        
        # For each common text, compare results
        for text_hash in common_hashes:
            # Get entries for this text hash from each model
            entries = {}
            for model in models:
                # Find the entry for this text hash
                for entry in self.model_history[model]:
                    if entry["text_hash"] == text_hash:
                        entries[model] = entry
                        break
            
            # Skip if we don't have entries from all models
            if len(entries) < len(models):
                continue
            
            # Compare execution times
            execution_times = {model: entry["execution_time"] 
                              for model, entry in entries.items()}
            
            # Compare scores for each category
            categories = set()
            for model in entries:
                categories.update(entries[model]["results"].keys())
            
            category_scores = {}
            for category in categories:
                category_scores[category] = {
                    model: entries[model]["results"].get(category, 0.0)
                    for model in entries
                }
            
            # Add to head-to-head comparison
            head_to_head[text_hash] = {
                "text_snippet": next(iter(entries.values()))["text_snippet"],
                "execution_times": execution_times,
                "category_scores": category_scores
            }
            
            # Add ground truth if available
            if text_hash in self.ground_truth:
                head_to_head[text_hash]["ground_truth"] = self.ground_truth[text_hash]
        
        # Calculate summary stats
        summary = {
            "model_count": len(model_metrics),
            "common_samples": len(head_to_head),
            "speed_comparison": {},
            "agreement": {}
        }
        
        # Speed comparison
        if head_to_head:
            for model in models:
                times = [h["execution_times"][model] for h in head_to_head.values()]
                if times:
                    summary["speed_comparison"][model] = {
                        "mean": np.mean(times),
                        "median": np.median(times),
                        "p95": np.percentile(times, 95)
                    }
            
            # Calculate model agreement scores
            if len(models) >= 2:
                agreement_scores = []
                for h in head_to_head.values():
                    cat_agreement = []
                    
                    for category, scores in h["category_scores"].items():
                        model_scores = list(scores.values())
                        # Calculate variance of scores (lower = more agreement)
                        if len(model_scores) >= 2:
                            variance = np.var(model_scores)
                            # Convert variance to agreement score (1 - normalized variance)
                            # Variance is between 0 and 0.25 for binary classification
                            agreement = 1.0 - min(variance * 4, 1.0)
                            cat_agreement.append(agreement)
                    
                    if cat_agreement:
                        h_agreement = np.mean(cat_agreement)
                        agreement_scores.append(h_agreement)
                
                if agreement_scores:
                    summary["agreement"] = {
                        "mean": np.mean(agreement_scores),
                        "median": np.median(agreement_scores),
                        "min": min(agreement_scores),
                        "max": max(agreement_scores)
                    }
        
        return {
            "models": models,
            "metrics_per_model": model_metrics,
            "head_to_head": head_to_head,
            "summary": summary
        }
    
    def clear_history(self):
        """Reset the performance tracker history."""
        self.model_history = defaultdict(list)
        self.ground_truth = {}
    
    def _load_mock_data(self):
        """
        Load mock historical data for demonstration purposes.
        In a real implementation, this would be replaced with actual historical data.
        """
        # Generate some mock historical data for demonstration
        models = ["primary", "groq"]
        categories = ["identity_attack", "insult", "obscene", "severe_toxicity", 
                     "sexual_explicit", "threat", "toxicity"]
        
        # Sample texts
        sample_texts = [
            "This is a neutral message with no toxicity.",
            "You're an idiot and nobody likes you!",
            "I hope your family dies in a fire.",
            "This product is terrible and I want my money back.",
            "The service was excellent and the staff was friendly.",
            "I can't believe how stupid this policy is.",
            "Go jump off a bridge and die.",
            "This movie was absolutely amazing!",
            "The food was disgusting and made me sick.",
            "I hate everyone from that country."
        ]
        
        # Ground truth labels
        ground_truth_values = [
            {"identity_attack": 0, "insult": 0, "obscene": 0, "severe_toxicity": 0, 
             "sexual_explicit": 0, "threat": 0, "toxicity": 0},
            {"identity_attack": 0, "insult": 1, "obscene": 0, "severe_toxicity": 0, 
             "sexual_explicit": 0, "threat": 0, "toxicity": 1},
            {"identity_attack": 0, "insult": 0, "obscene": 0, "severe_toxicity": 1, 
             "sexual_explicit": 0, "threat": 1, "toxicity": 1},
            {"identity_attack": 0, "insult": 0, "obscene": 0, "severe_toxicity": 0, 
             "sexual_explicit": 0, "threat": 0, "toxicity": 0.5},
            {"identity_attack": 0, "insult": 0, "obscene": 0, "severe_toxicity": 0, 
             "sexual_explicit": 0, "threat": 0, "toxicity": 0},
            {"identity_attack": 0, "insult": 1, "obscene": 0, "severe_toxicity": 0, 
             "sexual_explicit": 0, "threat": 0, "toxicity": 0.5},
            {"identity_attack": 0, "insult": 0, "obscene": 0, "severe_toxicity": 1, 
             "sexual_explicit": 0, "threat": 1, "toxicity": 1},
            {"identity_attack": 0, "insult": 0, "obscene": 0, "severe_toxicity": 0, 
             "sexual_explicit": 0, "threat": 0, "toxicity": 0},
            {"identity_attack": 0, "insult": 0, "obscene": 0, "severe_toxicity": 0, 
             "sexual_explicit": 0, "threat": 0, "toxicity": 0.5},
            {"identity_attack": 1, "insult": 0, "obscene": 0, "severe_toxicity": 0, 
             "sexual_explicit": 0, "threat": 0, "toxicity": 1},
        ]
        
        # Create mock history over the past week
        now = datetime.now()
        base_execution_times = {"primary": 0.2, "groq": 0.4}
        
        for day_offset in range(7):
            # Create timestamp for this day
            timestamp = now - timedelta(days=day_offset, 
                                       hours=np.random.randint(0, 24))
            
            for i, (text, ground_truth) in enumerate(
                zip(sample_texts, ground_truth_values)):
                
                text_hash = hash(text) % 10000000  # Simple hash for text identification
                
                # Store ground truth
                self.ground_truth[text_hash] = ground_truth
                
                # Generate mock results for each model with slight variations
                for model in models:
                    # Mock execution time with some variance
                    execution_time = base_execution_times[model] * (0.8 + 0.4 * np.random.random())
                    
                    # Generate slightly different results for each model
                    results = {}
                    for category, true_value in ground_truth.items():
                        # Base score close to ground truth, with some noise
                        # Primary model is more accurate, Groq has more variance
                        if model == "primary":
                            noise = np.random.normal(0, 0.1)
                        else:
                            noise = np.random.normal(0, 0.2)
                            
                        # Calculate score, clipped to [0, 1]
                        score = max(0.0, min(1.0, true_value + noise))
                        results[category] = score
                    
                    # Create the entry
                    entry = {
                        "timestamp": timestamp,
                        "text_hash": text_hash,
                        "text_snippet": text[:100] + "..." if len(text) > 100 else text,
                        "results": results,
                        "ground_truth": ground_truth,
                        "execution_time": execution_time
                    }
                    
                    # Add to history
                    self.model_history[model].append(entry)
        
        # Sort entries by timestamp
        for model in self.model_history:
            self.model_history[model].sort(key=lambda x: x["timestamp"])


def display_performance_dashboard(performance_tracker: PerformanceTracker, 
                                active_tab: str = "Overview"):
    """
    Create Streamlit UI for performance monitoring.
    
    Args:
        performance_tracker: Instance of PerformanceTracker
        active_tab: Which tab to activate by default
    """
    # Get available models
    models = list(performance_tracker.model_history.keys())
    if not models:
        st.warning("No performance data available for any models.")
        
        # Add a note about how data is collected
        st.info(
            "Performance data is collected automatically when you analyze text. "
            "Try analyzing some text to see performance metrics."
        )
        return
    
    # Create tabs for different views
    tabs = ["Overview", "Trends", "Calibration", "Raw Data"]
    tab_icons = ["📊", "📈", "🎯", "📋"]
    
    # Find which tab should be active
    active_index = tabs.index(active_tab) if active_tab in tabs else 0
    
    # Top section with tabs and clear button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Get tab selection
        selected_tab = st.radio(
            "Dashboard View",
            [f"{icon} {tab}" for icon, tab in zip(tab_icons, tabs)],
            index=active_index,
            horizontal=True
        )
        
        selected_tab = selected_tab.split(" ", 1)[1]  # Remove icon
    
    with col2:
        # Add "Clear Performance Data" button
        if st.button("🗑 Clear Data", type="secondary"):
            # We need a unique key for the confirmation dialog
            st.session_state.confirm_clear = True
    
    # Show confirmation dialog if button was pressed
    if st.session_state.get('confirm_clear', False):
        st.warning("⚠️ Are you sure you want to clear all performance history? This cannot be undone.")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Yes, Clear", type="primary"):
                # Clear the data
                performance_tracker.clear_history()
                st.session_state.confirm_clear = False
                st.success("✅ Performance data cleared!")
                # Add rerun to refresh the page
                st.rerun()
        
        with col2:
            if st.button("Cancel", type="secondary"):
                st.session_state.confirm_clear = False
                st.info("Clearing cancelled.")
                # Add rerun to refresh the page
                st.rerun()
    
    # Time period and model selection in a more compact layout
    time_periods = {
        "24 Hours": timedelta(days=1),
        "3 Days": timedelta(days=3),
        "Week": timedelta(days=7),
        "Month": timedelta(days=30),
        "All": None
    }
    
    col1, col2 = st.columns(2)
    with col1:
        selected_period = st.selectbox(
            "Period", 
            list(time_periods.keys()),
            index=2  # Default to "Week"
        )
        time_period = time_periods[selected_period]
    
    with col2:
        # Model selection
        selected_models = st.multiselect(
            "Models", 
            models,
            default=models
        )
        
        if not selected_models:
            selected_models = models[:1]  # Default to first model if none selected
    
    # Content based on selected tab
    if selected_tab == "Overview":
        st.write("### Performance Summary")
        
        # Get comparative metrics
        comparative_metrics = performance_tracker.get_comparative_metrics(
            models=selected_models, time_period=time_period
        )
        
        # Display summary metrics in a more compact form
        st.write("#### Speed Comparison")
        speed_data = []
        if "summary" in comparative_metrics and "speed_comparison" in comparative_metrics["summary"]:
            for model, metrics in comparative_metrics["summary"]["speed_comparison"].items():
                speed_data.append({
                    "Model": model,
                    "Mean": f"{metrics['mean']:.3f}s",
                    "Median": f"{metrics['median']:.3f}s",
                    "P95": f"{metrics['p95']:.3f}s"
                })
        
        if speed_data:
            speed_df = pd.DataFrame(speed_data)
            st.dataframe(speed_df, hide_index=True, use_container_width=True)
        else:
            st.info("No speed data available for selected period.")
        
        # Display model agreement metrics
        if ("summary" in comparative_metrics and 
            "agreement" in comparative_metrics["summary"] and 
            comparative_metrics["summary"]["agreement"]):
                
            agreement = comparative_metrics["summary"]["agreement"]
            
            # Show just mean agreement in a metric
            st.metric(
                "Model Agreement Score", 
                f"{agreement['mean']:.3f}",
                help="1.0 means perfect agreement between models"
            )
        
        # Show a condensed version of accuracy metrics
        st.write("#### Accuracy Summary")
        
        accuracy_data = []
        for model in selected_models:
            metrics = performance_tracker.get_performance_metrics(model, time_period)
            if metrics and "accuracy" in metrics and metrics["accuracy"]:
                # Calculate average metrics across categories
                avg_acc = np.mean([m["accuracy"] for m in metrics["accuracy"].values()])
                avg_f1 = np.mean([m["f1"] for m in metrics["accuracy"].values()])
                
                accuracy_data.append({
                    "Model": model,
                    "Avg Accuracy": f"{avg_acc:.3f}",
                    "Avg F1": f"{avg_f1:.3f}",
                    "Categories": len(metrics["accuracy"])
                })
        
        if accuracy_data:
            acc_df = pd.DataFrame(accuracy_data)
            st.dataframe(acc_df, hide_index=True, use_container_width=True)
        else:
            st.info("No accuracy data available (requires ground truth labels).")
    
    elif selected_tab == "Trends":
        st.write("### Performance Trends")
        
        # Get recent data for trend analysis
        recent_data = []
        for model in selected_models:
            if model in performance_tracker.model_history:
                model_data = performance_tracker.model_history[model]
                for entry in model_data:
                    if time_period is None or (datetime.now() - entry['timestamp']) <= time_period:
                        recent_data.append({
                            'Model': model,
                            'Timestamp': entry['timestamp'],
                            'Execution Time': entry['execution_time'],
                            'Text Length': len(entry['text'])
                        })
        
        if recent_data:
            # Create trend chart
            trend_df = pd.DataFrame(recent_data)
            trend_df['Time'] = pd.to_datetime(trend_df['Timestamp'])
            
            # Plot execution time trends
            fig, ax = plt.subplots(figsize=(8, 4))
            for model in selected_models:
                model_data = trend_df[trend_df['Model'] == model]
                if not model_data.empty:
                    ax.scatter(model_data['Time'], model_data['Execution Time'], 
                             label=model, alpha=0.7)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Execution Time (seconds)')
            ax.set_title('Execution Time Trends')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No trend data available for selected period.")
    
    elif selected_tab == "Calibration":
        st.write("### Model Calibration")
        
        # Show confidence calibration if available
        calibration_data = []
        for model in selected_models:
            metrics = performance_tracker.get_performance_metrics(model, time_period)
            if metrics and "calibration" in metrics:
                for category, cal_data in metrics["calibration"].items():
                    calibration_data.append({
                        "Model": model,
                        "Category": category,
                        "ECE": f"{cal_data['ece']:.3f}",
                        "Reliability": f"{cal_data['reliability']:.3f}"
                    })
        
        if calibration_data:
            cal_df = pd.DataFrame(calibration_data)
            st.dataframe(cal_df, hide_index=True, use_container_width=True)
            
            st.info(
                "ECE (Expected Calibration Error): Lower is better. "
                "Reliability: Higher is better."
            )
        else:
            st.info("No calibration data available (requires ground truth labels).")
    
    elif selected_tab == "Raw Data":
        st.write("### Raw Performance Data")
        
        # Model selection for raw data
        selected_model_raw = st.selectbox(
            "Select Model", 
            selected_models,
            index=0
        )
        
        # Get data for selected model
        if selected_model_raw:
            history = performance_tracker.model_history[selected_model_raw]
            
            # Filter by time period
            if time_period:
                cutoff_time = datetime.now() - time_period
                history = [entry for entry in history if entry["timestamp"] >= cutoff_time]
            
            if history:
                # Create a dataframe for the raw data
                raw_data = []
                
                # Define all potential categories from history
                all_categories = set()
                for entry in history:
                    all_categories.update(entry["results"].keys())
                
                # Sort categories for consistent display
                sorted_categories = sorted(all_categories)
                
                # Build data rows
                for entry in history:
                    row = {
                        "Timestamp": entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                        "Text": entry["text_snippet"],
                        "Execution Time": f"{entry['execution_time']:.4f}s"
                    }
                    
                    # Add result scores for all categories
                    for category in sorted_categories:
                        score = entry["results"].get(category, None)
                        if score is not None:
                            row[f"{category}"] = score
                    
                    # Add ground truth if available
                    has_ground_truth = "ground_truth" in entry
                    if has_ground_truth:
                        row["Has Ground Truth"] = "✓"
                    else:
                        row["Has Ground Truth"] = "✗"
                        
                    raw_data.append(row)
                
                # Create and display the dataframe
                raw_df = pd.DataFrame(raw_data)
                
                # Apply color styling to score columns if color_utils available
                try:
                    from color_utils import apply_color_to_dataframe
                    
                    def multi_column_colors(df):
                        # Style each toxicity category column
                        for category in sorted_categories:
                            col_name = f"{category}"
                            if col_name in df.columns:
                                # Color each cell based on its value and category
                                df = apply_color_to_dataframe(
                                    df, score_column=col_name, category_column=None
                                )
                        return df
                    
                    # Apply styling
                    styled_df = multi_column_colors(raw_df)
                    st.dataframe(styled_df, height=400, use_container_width=True)
                    
                except ImportError:
                    # Fallback if color_utils is not available
                    st.dataframe(raw_df, height=400, use_container_width=True)
                
                # Add export option
                if st.button("Export Data as CSV"):
                    # Convert dataframe to CSV
                    csv = raw_df.to_csv(index=False)
                    
                    # Create a download button
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"{selected_model_raw}_performance_data.csv",
                        "text/csv",
                        key='download-csv'
                    )
                
            else:
                st.info(f"No data available for {selected_model_raw} in the selected time period.") 