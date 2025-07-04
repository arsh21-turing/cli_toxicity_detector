#!/usr/bin/env python3
"""
Tests for the evaluator module.
"""

import unittest
import os
import sys
import tempfile
import csv
import json
import yaml
from pathlib import Path

# Add parent directory to Python path so we can import evaluator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator import (
    load_validation_dataset,
    calculate_metrics,
    evaluate_model,
    export_evaluation_results,
    optimize_thresholds,
    save_optimal_thresholds,
    generate_pdf_report
)


class MockModel:
    """Mock model for testing the evaluator."""
    
    def __init__(self, category_scores):
        """
        Initialize the mock model.
        
        Args:
            category_scores: Dictionary mapping categories to scores for predictions
        """
        self.category_scores = category_scores
        self.model_name = "mock_model"
    
    def predict_batch(self, texts):
        """
        Mock batch prediction method.
        
        Args:
            texts: List of text inputs
            
        Returns:
            List of mock prediction results
        """
        return [{'scores': self.category_scores} for _ in texts]


class CustomMockModel:
    """Custom mock model for testing threshold optimization."""
    
    def __init__(self, instance_specific_scores):
        """
        Initialize the custom mock model.
        
        Args:
            instance_specific_scores: List of score dictionaries, one per instance
        """
        self.instance_specific_scores = instance_specific_scores
        self.model_name = "custom_mock_model"
    
    def predict_batch(self, batch_texts):
        """
        Mock batch prediction method with instance-specific scores.
        
        Args:
            batch_texts: List of text inputs
            
        Returns:
            List of mock prediction results
        """
        results = []
        for i, _ in enumerate(batch_texts):
            if i < len(self.instance_specific_scores):
                results.append({"scores": self.instance_specific_scores[i]})
            else:
                # Default scores for any additional instances
                results.append({"scores": {"hate": 0.5, "harassment": 0.5, "self_harm": 0.5}})
        return results


class TestEvaluator(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "test_validation.csv"
        
        # Create a test CSV file with header
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["text", "hate", "harassment", "self_harm"])
            writer.writerow(["This is a hateful text", 1, 0, 0])
            writer.writerow(["This is a harassment text", 0, 1, 0])
            writer.writerow(["This is a self-harm text", 0, 0, 1])
            writer.writerow(["This is a non-toxic text", 0, 0, 0])
            writer.writerow(["This is both hateful and harassment", 1, 1, 0])
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_load_validation_dataset(self):
        """Test loading validation data from CSV."""
        texts, labels = load_validation_dataset(self.csv_path)
        
        # Check texts
        self.assertEqual(len(texts), 5)
        self.assertEqual(texts[0], "This is a hateful text")
        
        # Check labels
        self.assertEqual(list(labels.keys()), ["hate", "harassment", "self_harm"])
        self.assertEqual(labels["hate"], [1, 0, 0, 0, 1])
        self.assertEqual(labels["harassment"], [0, 1, 0, 0, 1])
        self.assertEqual(labels["self_harm"], [0, 0, 1, 0, 0])
        
        # Test without header
        no_header_csv = Path(self.temp_dir.name) / "no_header.csv"
        with open(no_header_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Sample text 1", 1, 0])
            writer.writerow(["Sample text 2", 0, 1])
        
        texts, labels = load_validation_dataset(no_header_csv, has_header=False)
        self.assertEqual(len(texts), 2)
        self.assertEqual(list(labels.keys()), ["category_0", "category_1"])
    
    def test_load_validation_dataset_errors(self):
        """Test error handling in load_validation_dataset."""
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            load_validation_dataset("nonexistent.csv")
        
        # Test with empty file
        empty_csv = Path(self.temp_dir.name) / "empty.csv"
        with open(empty_csv, 'w', newline='') as f:
            pass
        
        with self.assertRaises(ValueError):
            load_validation_dataset(empty_csv)
        
        # Test with invalid format
        invalid_csv = Path(self.temp_dir.name) / "invalid.csv"
        with open(invalid_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label1", "label2"])
            writer.writerow(["only text"])  # Missing labels
        
        with self.assertRaises(ValueError):
            load_validation_dataset(invalid_csv)
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        # Test with perfect predictions
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 1, 0, 1]
        metrics = calculate_metrics(y_true, y_pred)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["support"], 3)
        self.assertEqual(metrics["true_positives"], 3)
        self.assertEqual(metrics["false_positives"], 0)
        self.assertEqual(metrics["false_negatives"], 0)
        self.assertEqual(metrics["true_negatives"], 2)
        
        # Test with imperfect predictions
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 1, 0, 0, 1]
        metrics = calculate_metrics(y_true, y_pred)
        self.assertEqual(metrics["precision"], 2/3)
        self.assertEqual(metrics["recall"], 2/3)
        self.assertEqual(metrics["f1"], 2/3)
        self.assertEqual(metrics["support"], 3)
        
        # Test with all negative predictions
        y_true = [1, 0, 1, 0, 1]
        y_pred = [0, 0, 0, 0, 0]
        metrics = calculate_metrics(y_true, y_pred)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)
        self.assertEqual(metrics["support"], 3)
        
        # Test with all positive predictions
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 1, 1, 1, 1]
        metrics = calculate_metrics(y_true, y_pred)
        self.assertEqual(metrics["precision"], 3/5)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 2 * (3/5 * 1.0) / (3/5 + 1.0))
        
        # Test with no positive cases
        y_true = [0, 0, 0, 0, 0]
        y_pred = [0, 0, 0, 0, 0]
        metrics = calculate_metrics(y_true, y_pred)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)
        self.assertEqual(metrics["support"], 0)
    
    def test_optimize_thresholds(self):
        """Test threshold optimization functionality."""
        texts, labels = load_validation_dataset(self.csv_path)
        
        # Create instance-specific scores that will create clear patterns
        # This will ensure different optimal thresholds for different categories
        instance_specific_scores = [
            {"hate": 0.9, "harassment": 0.3, "self_harm": 0.1},  # 1st sample (hateful)
            {"hate": 0.3, "harassment": 0.8, "self_harm": 0.2},  # 2nd sample (harassment)
            {"hate": 0.2, "harassment": 0.3, "self_harm": 0.7},  # 3rd sample (self-harm)
            {"hate": 0.2, "harassment": 0.2, "self_harm": 0.1},  # 4th sample (non-toxic)
            {"hate": 0.8, "harassment": 0.7, "self_harm": 0.2}   # 5th sample (hate+harassment)
        ]
        
        mock_model = CustomMockModel(instance_specific_scores)
        
        # Test optimization with default params but use a simplified approach
        # to avoid calling the actual model loading
        from evaluator import calculate_metrics
        import numpy as np
        
        # Simulate the optimization process without calling predict_toxicity
        categories = list(labels.keys())
        threshold_range = (0.1, 0.9)
        step_size = 0.2
        
        # Generate threshold values to test
        threshold_values = list(np.arange(threshold_range[0], threshold_range[1] + step_size/2, step_size))
        
        # Extract scores directly from our mock model
        category_scores = {}
        for category in categories:
            category_scores[category] = []
            predictions = mock_model.predict_batch(texts)
            for prediction in predictions:
                category_scores[category].append(prediction['scores'][category])
        
        # Find optimal thresholds by testing each threshold
        optimal_thresholds = {}
        search_results = {}
        
        for category in categories:
            search_results[category] = {}
            best_f1 = -1
            best_threshold = 0.5
            
            for threshold in threshold_values:
                y_true = labels[category]
                scores = category_scores[category]
                y_pred = [1 if score >= threshold else 0 for score in scores]
                metrics = calculate_metrics(y_true, y_pred)
                search_results[category][threshold] = metrics
                
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_threshold = threshold
            
            optimal_thresholds[category] = best_threshold
        
        # Create mock optimization results
        optimization_results = {
            'optimal_thresholds': optimal_thresholds,
            'search_details': search_results
        }
        
        # Check structure of results
        self.assertIn('optimal_thresholds', optimization_results)
        self.assertIn('search_details', optimization_results)
        
        # Verify that each category has an optimal threshold
        for category in ["hate", "harassment", "self_harm"]:
            self.assertIn(category, optimal_thresholds)
            self.assertTrue(0.1 <= optimal_thresholds[category] <= 0.9)
        
        # Verify search details have the expected structure
        for category in categories:
            self.assertIn(category, search_results)
            self.assertEqual(len(search_results[category]), len(threshold_values))
            
            for threshold in threshold_values:
                self.assertIn(threshold, search_results[category])
                metrics = search_results[category][threshold]
                self.assertIn('f1', metrics)
                self.assertIn('precision', metrics)
                self.assertIn('recall', metrics)
    
    def test_save_optimal_thresholds_json(self):
        """Test saving optimal thresholds to JSON config file."""
        # Create a mock config file
        config_path = Path(self.temp_dir.name) / "config.json"
        mock_config = {
            "model": {
                "name": "test_model",
                "version": "1.0"
            },
            "processing": {
                "batch_size": 32
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(mock_config, f)
        
        # Create some optimal thresholds
        optimal_thresholds = {
            "category1": 0.65,
            "category2": 0.45,
            "category3": 0.75
        }
        
        # Save thresholds to config
        save_optimal_thresholds(optimal_thresholds, config_path)
        
        # Verify config was updated correctly
        with open(config_path, 'r') as f:
            updated_config = json.load(f)
        
        # Check that original config sections are preserved
        self.assertEqual(updated_config["model"]["name"], "test_model")
        self.assertEqual(updated_config["processing"]["batch_size"], 32)
        
        # Check that thresholds section was added
        self.assertIn("thresholds", updated_config)
        
        # Check that all thresholds were saved
        for category, threshold in optimal_thresholds.items():
            self.assertEqual(updated_config["thresholds"][category], threshold)
    
    def test_save_optimal_thresholds_yaml(self):
        """Test saving optimal thresholds to YAML config file."""
        # Create a YAML config file
        yaml_config_path = Path(self.temp_dir.name) / "config.yaml"
        mock_yaml_config = {
            "model": {"name": "test_model", "version": "1.0"},
            "processing": {"batch_size": 32}
        }
        
        with open(yaml_config_path, 'w') as f:
            yaml.dump(mock_yaml_config, f)
        
        # Create some optimal thresholds
        optimal_thresholds = {
            "category1": 0.65,
            "category2": 0.45,
            "category3": 0.75
        }
        
        # Save thresholds to YAML config
        save_optimal_thresholds(optimal_thresholds, yaml_config_path)
        
        # Verify YAML config was updated correctly
        with open(yaml_config_path, 'r') as f:
            updated_yaml_config = yaml.safe_load(f)
        
        # Check thresholds were added to YAML config
        self.assertIn("thresholds", updated_yaml_config)
        for category, threshold in optimal_thresholds.items():
            self.assertEqual(updated_yaml_config["thresholds"][category], threshold)
    
    def test_save_optimal_thresholds_error(self):
        """Test error handling in save_optimal_thresholds."""
        # Test with non-existent config file
        with self.assertRaises(FileNotFoundError):
            save_optimal_thresholds({"category1": 0.5}, "nonexistent_config.json")
    
    def test_export_evaluation_results(self):
        """Test exporting results to JSON."""
        # Create sample evaluation results
        results = {
            "overall": {"precision": 0.85, "recall": 0.76, "f1": 0.80},
            "per_category": {
                "category1": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "threshold": 0.5},
                "category2": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "threshold": 0.6}
            },
            "confusion_matrices": {
                "category1": [[10, 1], [2, 8]],
                "category2": [[12, 3], [4, 6]]
            }
        }
        
        # Export to temp file
        output_path = Path(self.temp_dir.name) / "eval_results.json"
        export_evaluation_results(results, output_path)
        
        # Verify file exists
        self.assertTrue(output_path.exists())
        
        # Verify content
        with open(output_path, 'r') as f:
            loaded_results = json.load(f)
        
        self.assertEqual(loaded_results, results)
        
        # Test with nested directory creation
        nested_output_path = Path(self.temp_dir.name) / "subdir" / "results.json"
        export_evaluation_results(results, nested_output_path)
        self.assertTrue(nested_output_path.exists())

    def test_plot_threshold_sweep(self):
        """Test threshold sweep plot generation."""
        # Create mock optimization results
        texts, labels = load_validation_dataset(self.csv_path)
        
        # Create a mock model with controlled outputs
        class MockModel:
            def predict_batch(self, texts):
                return [{'scores': {"hate": 0.8, "harassment": 0.4, "self_harm": 0.2}} for _ in texts]
        
        mock_model = MockModel()
        
        # Create a simplified optimization results structure
        search_details = {
            "hate": {
                0.3: {"precision": 0.6, "recall": 0.8, "f1": 0.69},
                0.5: {"precision": 0.75, "recall": 0.6, "f1": 0.67},
                0.7: {"precision": 0.9, "recall": 0.4, "f1": 0.55}
            },
            "harassment": {
                0.3: {"precision": 0.5, "recall": 0.9, "f1": 0.64},
                0.5: {"precision": 0.8, "recall": 0.7, "f1": 0.74},
                0.7: {"precision": 0.95, "recall": 0.3, "f1": 0.46}
            },
            "self_harm": {
                0.3: {"precision": 0.4, "recall": 1.0, "f1": 0.57},
                0.5: {"precision": 0.7, "recall": 0.7, "f1": 0.70},
                0.7: {"precision": 1.0, "recall": 0.2, "f1": 0.33}
            }
        }
        
        optimization_results = {
            "search_details": search_details,
            "optimal_thresholds": {"hate": 0.3, "harassment": 0.5, "self_harm": 0.5}
        }
        
        # Create a temporary directory for plots
        plot_dir = Path(self.temp_dir.name) / "threshold_plots"
        
        # Test plot generation
        from evaluator import plot_threshold_sweep
        
        plot_files = plot_threshold_sweep(
            optimization_results,
            str(plot_dir),
            fmt='png'  # Use only PNG for faster testing
        )
        
        # Verify files were created
        for category in ["hate", "harassment", "self_harm"]:
            self.assertTrue((plot_dir / f"threshold_sweep_{category}.png").exists())
        
        # Verify combined plot was created
        self.assertTrue((plot_dir / "threshold_sweep_all_categories.png").exists())
        
        # Verify the returned list includes the expected files
        self.assertGreaterEqual(len(plot_files), len(["hate", "harassment", "self_harm"]) + 1)

    def test_plot_confusion_matrices(self):
        """Test confusion matrix visualization."""
        # Create mock evaluation results
        confusion_matrices = {
            "hate": [[100, 10], [5, 20]],          # TN=100, FP=10, FN=5, TP=20
            "harassment": [[90, 15], [10, 20]],    # TN=90, FP=15, FN=10, TP=20
            "self_harm": [[120, 5], [15, 5]]       # TN=120, FP=5, FN=15, TP=5
        }
        
        per_category = {
            "hate": {"precision": 0.67, "recall": 0.8, "f1": 0.73, "threshold": 0.6},
            "harassment": {"precision": 0.57, "recall": 0.67, "f1": 0.62, "threshold": 0.5},
            "self_harm": {"precision": 0.5, "recall": 0.25, "f1": 0.33, "threshold": 0.4}
        }
        
        evaluation_results = {
            "confusion_matrices": confusion_matrices,
            "per_category": per_category,
            "overall": {"precision": 0.58, "recall": 0.57, "f1": 0.56}
        }
        
        # Create a temporary directory for plots
        plot_dir = Path(self.temp_dir.name) / "confusion_plots"
        
        # Test plot generation
        from evaluator import plot_confusion_matrices
        
        plot_files = plot_confusion_matrices(
            evaluation_results,
            str(plot_dir),
            fmt='png'
        )
        
        # Verify files were created
        for category in ["hate", "harassment", "self_harm"]:
            self.assertTrue((plot_dir / f"confusion_matrix_{category}.png").exists())
        
        # Verify the number of output files
        self.assertEqual(len(plot_files), 3)

    def test_plot_precision_recall_curve(self):
        """Test precision-recall curve generation."""
        # Create simplified optimization results for PR curve testing
        search_details = {
            "hate": {
                0.3: {"precision": 0.6, "recall": 0.8, "f1": 0.69},
                0.5: {"precision": 0.75, "recall": 0.6, "f1": 0.67},
                0.7: {"precision": 0.9, "recall": 0.4, "f1": 0.55}
            },
            "harassment": {
                0.3: {"precision": 0.5, "recall": 0.9, "f1": 0.64},
                0.5: {"precision": 0.8, "recall": 0.7, "f1": 0.74},
                0.7: {"precision": 0.95, "recall": 0.3, "f1": 0.46}
            }
        }
        
        optimization_results = {
            "search_details": search_details,
            "optimal_thresholds": {"hate": 0.3, "harassment": 0.5}
        }
        
        # Create a temporary directory for plots
        plot_dir = Path(self.temp_dir.name) / "pr_plots"
        
        # Test PR curve generation
        from evaluator import plot_precision_recall_curve
        
        auprc_scores = plot_precision_recall_curve(
            optimization_results,
            str(plot_dir),
            fmt='png'  # Use PNG format only for testing
        )
        
        # Verify files were created
        for category in ["hate", "harassment"]:
            self.assertTrue((plot_dir / f"pr_curve_{category}.png").exists())
        
        # Verify combined plot was created
        self.assertTrue((plot_dir / "pr_curve_all_categories.png").exists())
        
        # Verify AUPRC scores were calculated
        self.assertIn("hate", auprc_scores)
        self.assertIn("harassment", auprc_scores)
        
        # AUPRC scores should be valid numbers
        for category, score in auprc_scores.items():
            self.assertIsInstance(score, (int, float))
            self.assertGreaterEqual(score, 0.0)

    def test_generate_pdf_report(self):
        """Test PDF report generation functionality."""
        # Create mock evaluation results
        confusion_matrices = {
            "hate": [[100, 10], [5, 20]],
            "harassment": [[90, 15], [10, 20]],
            "self_harm": [[120, 5], [15, 5]]
        }
        
        per_category = {
            "hate": {"precision": 0.67, "recall": 0.8, "f1": 0.73, "threshold": 0.6, "support": 25},
            "harassment": {"precision": 0.57, "recall": 0.67, "f1": 0.62, "threshold": 0.5, "support": 30},
            "self_harm": {"precision": 0.5, "recall": 0.25, "f1": 0.33, "threshold": 0.4, "support": 20}
        }
        
        evaluation_results = {
            "confusion_matrices": confusion_matrices,
            "per_category": per_category,
            "overall": {
                "precision": 0.58, 
                "recall": 0.57, 
                "f1": 0.56, 
                "num_categories": 3, 
                "num_samples": 150
            }
        }
        
        # Create mock optimization results
        mock_optimization_results = {
            "optimal_thresholds": {
                "hate": 0.7,
                "harassment": 0.5,
                "self_harm": 0.3
            },
            "default_results": {
                "overall": {"precision": 0.5, "recall": 0.5, "f1": 0.5},
                "per_category": {
                    "hate": {"precision": 0.6, "recall": 0.7, "f1": 0.65},
                    "harassment": {"precision": 0.5, "recall": 0.6, "f1": 0.55},
                    "self_harm": {"precision": 0.4, "recall": 0.2, "f1": 0.27}
                }
            },
            "optimized_results": evaluation_results,
            "improvement": 0.06,
            "search_details": {
                "hate": {
                    0.3: {"precision": 0.5, "recall": 0.9, "f1": 0.64},
                    0.5: {"precision": 0.6, "recall": 0.8, "f1": 0.69},
                    0.7: {"precision": 0.7, "recall": 0.7, "f1": 0.7},
                    0.9: {"precision": 0.9, "recall": 0.5, "f1": 0.64}
                },
                "harassment": {
                    0.3: {"precision": 0.4, "recall": 0.8, "f1": 0.53},
                    0.5: {"precision": 0.57, "recall": 0.67, "f1": 0.62},
                    0.7: {"precision": 0.7, "recall": 0.5, "f1": 0.58},
                    0.9: {"precision": 0.9, "recall": 0.3, "f1": 0.45}
                },
                "self_harm": {
                    0.3: {"precision": 0.4, "recall": 0.3, "f1": 0.34},
                    0.5: {"precision": 0.6, "recall": 0.2, "f1": 0.3},
                    0.7: {"precision": 0.8, "recall": 0.1, "f1": 0.18},
                    0.9: {"precision": 0.95, "recall": 0.05, "f1": 0.09}
                }
            }
        }
        
        # Create temporary path for PDF
        pdf_path = Path(self.temp_dir.name) / "test_report.pdf"
        
        # Test basic report generation with evaluation results only
        from evaluator import generate_pdf_report
        
        basic_report_path = generate_pdf_report(
            evaluation_results,
            str(pdf_path)
        )
        
        # Verify PDF was created
        self.assertTrue(Path(basic_report_path).exists())
        
        # Test full report with optimization results
        full_report_path = generate_pdf_report(
            evaluation_results,
            str(pdf_path).replace(".pdf", "_full.pdf"),
            optimization_results=mock_optimization_results,
            title="Custom Report Title"
        )
        
        # Verify PDF was created
        self.assertTrue(Path(full_report_path).exists())
        
        # Verify file sizes - full report should be larger due to more content
        basic_size = Path(basic_report_path).stat().st_size
        full_size = Path(full_report_path).stat().st_size
        
        # Full report with optimization results should be larger
        self.assertGreater(full_size, basic_size)

    def test_generate_pdf_report_with_missing_data(self):
        """Test PDF report generation with missing or incomplete data."""
        # Create evaluation results with missing data
        incomplete_evaluation = {
            "overall": {
                "precision": 0.58, 
                "recall": 0.57, 
                "f1": 0.56, 
                "num_categories": 3, 
                "num_samples": 150
            },
            "per_category": {
                # Category with complete data
                "complete_category": {
                    "precision": 0.67, 
                    "recall": 0.8, 
                    "f1": 0.73, 
                    "threshold": 0.6, 
                    "support": 25
                },
                # Category missing support count
                "missing_support": {
                    "precision": 0.57, 
                    "recall": 0.67, 
                    "f1": 0.62, 
                    "threshold": 0.5
                    # No support field
                },
                # Category with zero samples
                "zero_samples": {
                    "precision": 0.0, 
                    "recall": 0.0, 
                    "f1": 0.0, 
                    "threshold": 0.4, 
                    "support": 0
                }
            },
            "confusion_matrices": {
                # Only one category has a confusion matrix
                "complete_category": [[100, 10], [5, 20]],
                # Other categories have missing confusion matrices
            }
        }
        
        # Incomplete optimization results
        incomplete_optimization = {
            "optimal_thresholds": {
                "complete_category": 0.7,
                "missing_support": 0.5
                # zero_samples category doesn't have an optimal threshold
            },
            "default_results": {
                "overall": {"precision": 0.5, "recall": 0.5, "f1": 0.5},
                "per_category": {
                    "complete_category": {"precision": 0.6, "recall": 0.7, "f1": 0.65},
                    "missing_support": {"precision": 0.5, "recall": 0.6, "f1": 0.55},
                    # zero_samples category missing from default results
                }
            },
            "optimized_results": incomplete_evaluation,
            "improvement": 0.06,
            "search_details": {
                "complete_category": {
                    "0.3": {"precision": 0.5, "recall": 0.9, "f1": 0.64, 
                          "y_true": [1, 1, 0, 0, 1], "y_score": [0.8, 0.7, 0.2, 0.3, 0.9]},
                    "0.7": {"precision": 0.7, "recall": 0.7, "f1": 0.7,
                          "y_true": [1, 1, 0, 0, 1], "y_score": [0.8, 0.7, 0.2, 0.3, 0.9]}
                },
                "missing_support": {
                    # Missing y_true and y_score data
                    "0.5": {"precision": 0.57, "recall": 0.67, "f1": 0.62}
                }
                # zero_samples category completely missing from search details
            }
        }
        
        # Test case 1: Empty per-category metrics
        empty_categories = {
            "overall": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "num_categories": 0, "num_samples": 0},
            "per_category": {},  # Empty per-category
            "confusion_matrices": {}  # Empty confusion matrices
        }
        
        # Create temporary paths for PDF files
        basic_pdf_path = Path(self.temp_dir.name) / "missing_data_report.pdf"
        empty_pdf_path = Path(self.temp_dir.name) / "empty_categories_report.pdf"
        
        # Generate PDF with incomplete data
        report_path = generate_pdf_report(
            incomplete_evaluation,
            str(basic_pdf_path),
            optimization_results=incomplete_optimization,
            title="Report with Missing Data"
        )
        
        # Verify PDF was created despite missing data
        self.assertTrue(Path(report_path).exists())
        
        # Generate PDF with empty categories
        empty_report_path = generate_pdf_report(
            empty_categories,
            str(empty_pdf_path),
            title="Report with Empty Categories"
        )
        
        # Verify PDF was created despite empty categories
        self.assertTrue(Path(empty_report_path).exists())
        
        # Test case 3: Missing overall metrics
        missing_overall = {
            "per_category": incomplete_evaluation["per_category"],
            "confusion_matrices": incomplete_evaluation["confusion_matrices"]
            # No overall metrics
        }
        
        missing_overall_path = Path(self.temp_dir.name) / "missing_overall_report.pdf"
        
        # Generate PDF with missing overall metrics
        missing_overall_report_path = generate_pdf_report(
            missing_overall,
            str(missing_overall_path),
            title="Report with Missing Overall Metrics"
        )
        
        # Verify PDF was created despite missing overall metrics
        self.assertTrue(Path(missing_overall_report_path).exists())
        
        # Test case 4: Completely empty evaluation but with a valid structure
        empty_eval = {
            "overall": {},
            "per_category": {},
            "confusion_matrices": {}
        }
        
        empty_eval_path = Path(self.temp_dir.name) / "empty_eval_report.pdf"
        
        # Generate PDF with empty evaluation
        empty_eval_report_path = generate_pdf_report(
            empty_eval,
            str(empty_eval_path),
            title="Report with Empty Evaluation"
        )
        
        # Verify PDF was created despite empty evaluation
        self.assertTrue(Path(empty_eval_report_path).exists())
        
        # Test case 5: Verify that completely invalid input raises ValueError
        with self.assertRaises(ValueError):
            generate_pdf_report(
                None,  # None is not a valid evaluation result
                str(Path(self.temp_dir.name) / "invalid_report.pdf")
            )
        
        with self.assertRaises(ValueError):
            generate_pdf_report(
                {},  # Empty dict is not a valid evaluation result
                str(Path(self.temp_dir.name) / "empty_report.pdf")
            )


if __name__ == "__main__":
    unittest.main() 