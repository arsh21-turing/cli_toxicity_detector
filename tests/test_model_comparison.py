#!/usr/bin/env python3
"""
Tests for the model comparison module.
"""

import unittest
import os
import sys
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to Python path so we can import model_comparison
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_comparison import (
    compare_models,
    _calculate_comparative_metrics,
    _determine_winners,
    display_comparison_results,
    statistical_significance,
    generate_comparative_plots
)


class TestModelComparison(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create mock models
        self.model1 = Mock()
        self.model2 = Mock()
        self.models_dict = {
            "Model_A": self.model1,
            "Model_B": self.model2
        }
        
        # Create sample data
        self.texts = ["text1", "text2", "text3", "text4", "text5"]
        self.labels = {
            "hate": [1, 0, 1, 0, 1],
            "harassment": [0, 1, 0, 1, 0]
        }
        
        # Create mock evaluation results
        self.mock_eval_results_1 = {
            "overall": {
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.75
            },
            "per_category": {
                "hate": {
                    "precision": 0.85,
                    "recall": 0.75,
                    "f1": 0.8,
                    "predictions": [1, 0, 1, 0, 1]
                },
                "harassment": {
                    "precision": 0.75,
                    "recall": 0.65,
                    "f1": 0.7,
                    "predictions": [0, 1, 0, 1, 0]
                }
            },
            "confusion_matrices": {
                "hate": [[2, 1], [0, 2]],
                "harassment": [[2, 1], [1, 1]]
            }
        }
        
        self.mock_eval_results_2 = {
            "overall": {
                "precision": 0.75,
                "recall": 0.8,
                "f1": 0.77
            },
            "per_category": {
                "hate": {
                    "precision": 0.8,
                    "recall": 0.8,
                    "f1": 0.8,
                    "predictions": [1, 0, 1, 0, 1]
                },
                "harassment": {
                    "precision": 0.7,
                    "recall": 0.8,
                    "f1": 0.75,
                    "predictions": [0, 1, 0, 1, 0]
                }
            },
            "confusion_matrices": {
                "hate": [[2, 1], [0, 2]],
                "harassment": [[2, 0], [1, 2]]
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    @patch('model_comparison.evaluate_model')
    def test_compare_models_basic(self, mock_evaluate):
        """Test basic model comparison functionality."""
        # Mock the evaluate_model function to return our test results
        mock_evaluate.side_effect = [self.mock_eval_results_1, self.mock_eval_results_2]
        
        # Run comparison
        results = compare_models(
            self.models_dict, 
            self.texts, 
            self.labels
        )
        
        # Check structure
        self.assertIn("per_model", results)
        self.assertIn("comparative", results)
        self.assertIn("winner", results)
        
        # Check per-model results
        self.assertIn("Model_A", results["per_model"])
        self.assertIn("Model_B", results["per_model"])
        
        # Check that evaluate_model was called for each model
        self.assertEqual(mock_evaluate.call_count, 2)
    
    def test_calculate_comparative_metrics(self):
        """Test comparative metrics calculation."""
        per_model_results = {
            "Model_A": self.mock_eval_results_1,
            "Model_B": self.mock_eval_results_2
        }
        
        categories = ["hate", "harassment"]
        
        comparative = _calculate_comparative_metrics(per_model_results, categories)
        
        # Check structure
        self.assertIn("relative_performance", comparative)
        self.assertIn("agreement", comparative)
        
        # Check relative performance calculation
        self.assertIn("Model_A_vs_Model_B", comparative["relative_performance"])
        
        pair_comparison = comparative["relative_performance"]["Model_A_vs_Model_B"]
        
        # Check overall metrics comparison
        self.assertIn("overall", pair_comparison)
        for metric in ["precision", "recall", "f1"]:
            self.assertIn(metric, pair_comparison["overall"])
            self.assertIn("absolute_difference", pair_comparison["overall"][metric])
            self.assertIn("relative_difference_percent", pair_comparison["overall"][metric])
            self.assertIn("better_model", pair_comparison["overall"][metric])
    
    def test_determine_winners(self):
        """Test winner determination logic."""
        per_model_results = {
            "Model_A": self.mock_eval_results_1,
            "Model_B": self.mock_eval_results_2
        }
        
        categories = ["hate", "harassment"]
        
        winners = _determine_winners(per_model_results, categories)
        
        # Check structure
        self.assertIn("overall", winners)
        self.assertIn("per_category", winners)
        
        # Check overall winners
        for metric in ["precision", "recall", "f1"]:
            self.assertIn(metric, winners["overall"])
            self.assertIn("model", winners["overall"][metric])
            self.assertIn("value", winners["overall"][metric])
        
        # Check per-category winners
        for category in categories:
            self.assertIn(category, winners["per_category"])
            for metric in ["precision", "recall", "f1"]:
                self.assertIn(metric, winners["per_category"][category])
    
    def test_statistical_significance(self):
        """Test statistical significance calculation."""
        comparison_results = {
            "per_model": {
                "Model_A": self.mock_eval_results_1,
                "Model_B": self.mock_eval_results_2
            }
        }
        
        significance = statistical_significance(comparison_results)
        
        # Check structure
        self.assertIn("overall_f1", significance)
        self.assertIn("per_category_f1", significance)
        
        # Should have comparisons for each pair
        if "Model_A_vs_Model_B" in significance["overall_f1"]:
            pair_sig = significance["overall_f1"]["Model_A_vs_Model_B"]
            self.assertIn("significant", pair_sig)
            self.assertIn("test_method", pair_sig)
    
    def test_empty_models_dict(self):
        """Test error handling for empty models dictionary."""
        with self.assertRaises(ValueError):
            compare_models({}, self.texts, self.labels)
    
    def test_empty_validation_data(self):
        """Test error handling for empty validation data."""
        with self.assertRaises(ValueError):
            compare_models(self.models_dict, [], {})
    
    def test_single_model_comparison(self):
        """Test comparison with only one model."""
        single_model = {"Model_A": self.model1}
        
        with patch('model_comparison.evaluate_model') as mock_evaluate:
            mock_evaluate.return_value = self.mock_eval_results_1
            
            results = compare_models(single_model, self.texts, self.labels)
            
            # Should still work but comparative metrics should note insufficient models
            self.assertIn("per_model", results)
            self.assertIn("comparative", results)
            
            # Comparative should note that comparison requires at least two models
            if "note" in results["comparative"]:
                self.assertIn("Comparison requires at least two models", results["comparative"]["note"])
    
    @patch('builtins.print')
    def test_display_comparison_results(self, mock_print):
        """Test display of comparison results."""
        
        comparison_results = {
            "per_model": {
                "Model_A": self.mock_eval_results_1,
                "Model_B": self.mock_eval_results_2
            },
            "winner": {
                "overall": {
                    "precision": {"model": "Model_A", "value": 0.8},
                    "recall": {"model": "Model_B", "value": 0.8},
                    "f1": {"model": "Model_B", "value": 0.77}
                },
                "per_category": {
                    "hate": {
                        "precision": {"model": "Model_A", "value": 0.85},
                        "recall": {"model": "Model_B", "value": 0.8},
                        "f1": {"model": "Model_A", "value": 0.8}
                    }
                }
            }
        }
        
        # Test basic display
        display_comparison_results(comparison_results)
        
        # Should have called print multiple times
        self.assertTrue(mock_print.called)
        
        # Test detailed display
        display_comparison_results(comparison_results, detailed=True)
        
        # Should have called print even more times for detailed view
        self.assertTrue(mock_print.called)
    
    def test_thresholds_parameter(self):
        """Test comparison with custom thresholds for each model."""
        thresholds = {
            "Model_A": {"hate": 0.6, "harassment": 0.5},
            "Model_B": {"hate": 0.7, "harassment": 0.6}
        }
        
        with patch('model_comparison.evaluate_model') as mock_evaluate:
            mock_evaluate.side_effect = [self.mock_eval_results_1, self.mock_eval_results_2]
            
            results = compare_models(
                self.models_dict, 
                self.texts, 
                self.labels,
                thresholds=thresholds
            )
            
            # Check that evaluate_model was called with the correct thresholds
            calls = mock_evaluate.call_args_list
            
            # First call should have Model_A thresholds
            self.assertEqual(calls[0][1]['thresholds'], thresholds["Model_A"])
            
            # Second call should have Model_B thresholds
            self.assertEqual(calls[1][1]['thresholds'], thresholds["Model_B"])
    
    def test_categories_parameter(self):
        """Test comparison with specific categories."""
        categories = ["hate"]  # Only evaluate hate category
        
        with patch('model_comparison.evaluate_model') as mock_evaluate:
            mock_evaluate.side_effect = [self.mock_eval_results_1, self.mock_eval_results_2]
            
            results = compare_models(
                self.models_dict, 
                self.texts, 
                self.labels,
                categories=categories
            )
            
            # Check that evaluate_model was called with the correct categories
            calls = mock_evaluate.call_args_list
            
            for call in calls:
                self.assertEqual(call[1]['categories'], categories)


if __name__ == "__main__":
    unittest.main() 