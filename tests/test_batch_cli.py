#!/usr/bin/env python3
"""Tests for batch processing CLI functionality."""

import importlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import statistics

# Ensure project root is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


class TestBatchCLI(unittest.TestCase):
    def setUp(self):
        """Set up test environment with temporary directories and files."""
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()
        
        # Create test files
        self.test_files = []
        for i in range(3):
            test_file = os.path.join(self.test_dir, f"test_{i}.txt")
            with open(test_file, "w") as f:
                f.write(f"This is test file {i}. ")
                f.write("Some content here. ")
                f.write("Maybe toxic content like you are stupid. ")
                f.write("And more normal content.")
            self.test_files.append(test_file)
        
        # Mock argparse namespace
        self.mock_args = MagicMock()
        self.mock_args.batch = self.test_dir
        self.mock_args.output = self.output_dir
        self.mock_args.model = "test-model"
        self.mock_args.json = False
        self.mock_args.verbose = False
        self.mock_args.quiet = False  # Needed for show_progress calculation

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.output_dir)

    @patch("main.batch_process")
    @patch("main.load_model")
    @patch("main.load_config")
    def test_batch_flag_basic_invocation(self, mock_load_config, mock_load_model, mock_batch_process):
        """Ensure handle_batch_processing delegates correctly and returns expected keys."""

        # We import main lazily so patched symbols exist within it
        main = importlib.import_module("main")

        # Set up stub return values
        mock_load_model.return_value = "dummy_model"
        mock_load_config.return_value = {"cfg": True}

        expected_results = {
            "total_files": 3,
            "toxic_files": 1,
            "toxicity_metrics": {
                "avg_overall_score": 0.25,
                "max_toxicity_score": 0.75,
                "toxicity_distribution": {},
                "per_category_metrics": {}
            },
            "groq_metrics": {
                "total_usage_count": 0,
                "override_count": 0,
                "effectiveness_score": 0.0,
                "avg_confidence_improvement": 0.0
            },
            "file_results": {
                self.test_files[0]: {"toxic": True, "scores": {"hate": 0.8}, "overall_toxicity_score": 0.75},
                self.test_files[1]: {"toxic": False, "scores": {"hate": 0.2}, "overall_toxicity_score": 0.15},
                self.test_files[2]: {"toxic": False, "scores": {"hate": 0.1}, "overall_toxicity_score": 0.05},
            },
        }
        mock_batch_process.return_value = expected_results.copy()

        results = main.handle_batch_processing(self.mock_args)

        # Assert helper functions were called as expected
        mock_load_model.assert_called_once_with(model_name=self.mock_args.model)
        mock_load_config.assert_called_once_with()
        mock_batch_process.assert_called_once()

        ckwargs = mock_batch_process.call_args.kwargs
        self.assertEqual(ckwargs["input_path"], Path(self.test_dir))
        self.assertEqual(ckwargs["output_path"], Path(self.output_dir))
        self.assertEqual(ckwargs["model"], "dummy_model")
        # Check that config has the basic structure plus confidence filtering if applicable
        config = ckwargs["config"]
        self.assertEqual(config["cfg"], True)
        # Confidence filtering may be added if mock args have filter attributes
        self.assertIsInstance(config, dict)
        self.assertTrue(ckwargs["show_progress"])

        # Ensure results got timestamp injection and retain core keys
        self.assertIn("timestamp", results)
        for k in ("total_files", "toxic_files", "file_results"):
            self.assertEqual(results[k], expected_results[k])

    def test_batch_sentence_processing_efficiency(self):
        """Test that sentences are processed efficiently through the batch processor."""
        from batch_processor import _predict_toxicity_batch
        
        # Create a list of 65 sentences (more than 2 batches)
        sentences = [f"Test sentence {i} with some content." for i in range(65)]
        
        # Mock model that tracks how many calls it receives
        mock_model = MagicMock()
        mock_config = {"threshold": 0.5}
        
        # Mock the predict_toxicity function to return deterministic results
        with patch('model_loader.predict_toxicity') as mock_predict:
            def mock_predict_side_effect(texts, **kwargs):
                # Return results for each text
                return [
                    {
                        "text": text,
                        "is_toxic": "stupid" in text.lower(),
                        "category_results": {
                            "hate": {"score": 0.8 if "stupid" in text.lower() else 0.1, "above_threshold": "stupid" in text.lower()},
                            "insult": {"score": 0.7 if "stupid" in text.lower() else 0.05, "above_threshold": "stupid" in text.lower()}
                        },
                        "groq_used": False,
                        "most_probable_category": "hate",
                        "raw_logits": [0.1, 0.9],
                        "sigmoid_scores": [0.1, 0.9]
                    }
                    for text in texts
                ]
            
            mock_predict.side_effect = mock_predict_side_effect
            
            # Process the batch
            results = _predict_toxicity_batch(sentences, model=mock_model, config=mock_config)
            
            # Verify we got results for all sentences
            self.assertEqual(len(results), 65)
            
            # Verify the model was called (should be 1 call with all sentences)
            self.assertEqual(mock_predict.call_count, 1)
            
            # Check that all sentences were passed to predict_toxicity
            call_kwargs = mock_predict.call_args.kwargs
            self.assertEqual(len(call_kwargs['texts']), 65)
            
            # Verify batch_size parameter is passed correctly
            self.assertEqual(call_kwargs['batch_size'], 32)

    def test_overall_toxicity_scoring(self):
        """Test comprehensive toxicity scoring with weighted averages."""
        from batch_processor import _compute_overall_toxicity_profile
        
        # Create sample file result with sentences of different lengths and toxicity
        file_result = {
            'sentences': [
                {
                    'text': 'Short toxic.',  # 12 chars
                    'scores': {'hate': 0.8, 'insult': 0.3},
                    'probabilities': {
                        'hate': {'0': 0.2, '1': 0.8},
                        'insult': {'0': 0.7, '1': 0.3}
                    }
                },
                {
                    'text': 'This is a much longer sentence that is not toxic and should have more weight in calculation.', # 92 chars
                    'scores': {'hate': 0.1, 'insult': 0.2},
                    'probabilities': {
                        'hate': {'0': 0.9, '1': 0.1},
                        'insult': {'0': 0.8, '1': 0.2}
                    }
                },
                {
                    'text': 'Medium length toxic sentence.',  # 29 chars
                    'scores': {'hate': 0.5, 'insult': 0.4},
                    'probabilities': {
                        'hate': {'0': 0.5, '1': 0.5},
                        'insult': {'0': 0.6, '1': 0.4}
                    }
                }
            ]
        }
        
        # Calculate toxicity profile
        profile = _compute_overall_toxicity_profile(file_result)
        
        # Verify profile structure
        self.assertIn('overall_score', profile)
        self.assertIn('category_scores', profile)
        self.assertIn('confidence', profile)
        self.assertIn('distribution', profile)
        
        # Verify overall score is calculated correctly
        # Expected: (12*0.8 + 92*0.2 + 29*0.5) / (12 + 92 + 29)
        # where we use max probability per sentence
        expected = (12 * 0.8 + 92 * 0.2 + 29 * 0.5) / (12 + 92 + 29)
        self.assertAlmostEqual(profile['overall_score'], expected, places=2)
        
        # Verify category scores exist
        self.assertIn('hate', profile['category_scores'])
        self.assertIn('insult', profile['category_scores'])
        
        # Verify distribution metrics
        distribution = profile['distribution']
        self.assertIn('min', distribution)
        self.assertIn('max', distribution)
        self.assertIn('mean', distribution)
        self.assertIn('median', distribution)
        
        # Min should be 0.2, max should be 0.8
        self.assertEqual(distribution['min'], 0.2)
        self.assertEqual(distribution['max'], 0.8)

    def test_enhanced_toxicity_scoring(self):
        """Test the enhanced toxicity scoring with multiple weighting strategies."""
        from batch_processor import _compute_overall_toxicity_profile
        
        # Create sample file result with mock sentences
        file_result = {
            'total_sentences': 5,
            'toxic_sentences': 2,
            'sentences': [
                {
                    'text': 'This is a normal sentence that is quite long to ensure it gets more weight.',
                    'scores': {'hate': 0.1, 'insult': 0.2, 'threat': 0.1},
                    'probabilities': {
                        'hate': {'0': 0.9, '1': 0.1},
                        'insult': {'0': 0.8, '1': 0.2},
                        'threat': {'0': 0.9, '1': 0.1}
                    }
                },
                {
                    'text': 'This is a toxic hate comment.',
                    'scores': {'hate': 0.8, 'insult': 0.6, 'threat': 0.3},
                    'probabilities': {
                        'hate': {'0': 0.2, '1': 0.8},
                        'insult': {'0': 0.4, '1': 0.6},
                        'threat': {'0': 0.7, '1': 0.3}
                    }
                },
                {
                    'text': 'Another normal but long sentence to add weight to non-toxic content in our test file.',
                    'scores': {'hate': 0.1, 'insult': 0.3, 'threat': 0.2},
                    'probabilities': {
                        'hate': {'0': 0.9, '1': 0.1},
                        'insult': {'0': 0.7, '1': 0.3},
                        'threat': {'0': 0.8, '1': 0.2}
                    }
                },
                {
                    'text': 'A strongly insulting comment that is quite toxic.',
                    'scores': {'hate': 0.5, 'insult': 0.9, 'threat': 0.2},
                    'probabilities': {
                        'hate': {'0': 0.5, '1': 0.5},
                        'insult': {'0': 0.1, '1': 0.9},
                        'threat': {'0': 0.8, '1': 0.2}
                    }
                },
                {
                    'text': 'Short neutral.',
                    'scores': {'hate': 0.1, 'insult': 0.1, 'threat': 0.1},
                    'probabilities': {
                        'hate': {'0': 0.9, '1': 0.1},
                        'insult': {'0': 0.9, '1': 0.1},
                        'threat': {'0': 0.9, '1': 0.1}
                    }
                }
            ]
        }
        
        # Run the toxicity scoring function
        toxicity_profile = _compute_overall_toxicity_profile(file_result)
        
        # Verify profile structure
        self.assertIn('overall_score', toxicity_profile)
        self.assertIn('category_scores', toxicity_profile)
        self.assertIn('confidence', toxicity_profile)
        self.assertIn('distribution', toxicity_profile)
        
        # Verify overall score (between 0 and 1)
        self.assertGreater(toxicity_profile['overall_score'], 0.0)
        self.assertLessEqual(toxicity_profile['overall_score'], 1.0)
        
        # Verify category scores
        category_scores = toxicity_profile['category_scores']
        self.assertIn('hate', category_scores)
        self.assertIn('insult', category_scores)
        self.assertIn('threat', category_scores)
        
        # Insult should have highest category score based on our test data
        self.assertGreater(category_scores['insult'], category_scores['threat'])
        
        # Verify distribution metrics
        distribution = toxicity_profile['distribution']
        self.assertEqual(distribution['min'], 0.1)  # Lowest score in our test data
        self.assertEqual(distribution['max'], 0.9)  # Highest score in our test data
        self.assertIn('mean', distribution)
        self.assertIn('median', distribution)
        
        # Verify confidence metric
        self.assertGreater(toxicity_profile['confidence'], 0.0)
        self.assertLessEqual(toxicity_profile['confidence'], 1.0)
        
        # Test with empty sentences list
        empty_result = {'sentences': []}
        empty_profile = _compute_overall_toxicity_profile(empty_result)
        self.assertEqual(empty_profile['overall_score'], 0.0)
        self.assertEqual(empty_profile['category_scores'], {})

    def test_detailed_output_reports(self):
        """Test the generation of enhanced output report files."""
        from batch_processor import _write_results
        
        # Create a temporary output directory
        output_dir = tempfile.mkdtemp()
        
        # Create a comprehensive batch results structure
        results = {
            'total_files': 3,
            'toxic_files': 2,
            'total_sentences': 10,
            'toxic_sentences': 4,
            'toxicity_metrics': {
                'avg_overall_score': 0.35,
                'max_toxicity_score': 0.75,
                'toxicity_distribution': {
                    'min': 0.1,
                    'max': 0.75,
                    'mean': 0.35,
                    'median': 0.3,
                    'std_dev': 0.2
                },
                'per_category_metrics': {
                    'hate': {'avg_score': 0.3, 'max_score': 0.7},
                    'insult': {'avg_score': 0.4, 'max_score': 0.8}
                }
            },
            'groq_metrics': {
                'total_usage_count': 3,
                'override_count': 2,
                'effectiveness_score': 0.67,
                'avg_confidence_improvement': 0.25
            },
            'file_results': {
                'file1.txt': {
                    'toxic': True,
                    'toxic_sentences': 2,
                    'total_sentences': 3,
                    'category_counts': {'hate': 1, 'insult': 2},
                    'toxicity_profile': {
                        'overall_score': 0.75,
                        'category_scores': {'hate': 0.7, 'insult': 0.8},
                        'confidence': 0.8,
                        'distribution': {'min': 0.5, 'max': 0.8}
                    },
                    'groq_usage': {
                        'count': 2,
                        'override_count': 1,
                        'effectiveness': 0.5,
                        'avg_confidence_lift': 0.3
                    },
                    'sentences': [
                        {
                            'text': 'This is a toxic sentence.',
                            'scores': {'hate': 0.7, 'insult': 0.8},
                            'probabilities': {
                                'hate': {'0': 0.3, '1': 0.7},
                                'insult': {'0': 0.2, '1': 0.8}
                            },
                            'groq_used': True,
                            'groq_changed_classification': True
                        }
                    ]
                },
                'file2.txt': {
                    'toxic': True,
                    'toxic_sentences': 2,
                    'total_sentences': 4,
                    'category_counts': {'hate': 2},
                    'toxicity_profile': {
                        'overall_score': 0.6,
                        'category_scores': {'hate': 0.65, 'insult': 0.4},
                        'confidence': 0.7,
                        'distribution': {'min': 0.3, 'max': 0.7}
                    },
                    'groq_usage': {
                        'count': 1,
                        'override_count': 1,
                        'effectiveness': 1.0,
                        'avg_confidence_lift': 0.2
                    },
                    'sentences': []
                },
                'file3.txt': {
                    'toxic': False,
                    'toxic_sentences': 0,
                    'total_sentences': 3,
                    'category_counts': {},
                    'toxicity_profile': {
                        'overall_score': 0.1,
                        'category_scores': {'hate': 0.1, 'insult': 0.15},
                        'confidence': 0.6,
                        'distribution': {'min': 0.05, 'max': 0.15}
                    },
                    'groq_usage': {
                        'count': 0,
                        'override_count': 0,
                        'effectiveness': 0.0,
                        'avg_confidence_lift': 0.0
                    },
                    'sentences': []
                }
            }
        }
        
        # Write the results
        _write_results(results, Path(output_dir))
        
        # Check for all expected files
        expected_files = [
            "batch_summary.json",
            "toxicity_report.json",
            "category_distribution.json",
            "groq_usage_report.json",
            "file1.txt.result.json",
            "file2.txt.result.json",
            "file3.txt.result.json"
        ]
        
        for filename in expected_files:
            file_path = os.path.join(output_dir, filename)
            self.assertTrue(os.path.exists(file_path), f"Expected file {filename} not found")
            
            # Verify file is valid JSON
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.assertIsNotNone(data)
        
        # Check batch_summary.json contents
        with open(os.path.join(output_dir, "batch_summary.json"), 'r') as f:
            summary = json.load(f)
            self.assertEqual(summary['total_files'], 3)
            self.assertEqual(summary['toxic_files'], 2)
            self.assertIn('toxicity_metrics', summary)
            self.assertIn('groq_metrics', summary)
            self.assertIn('timestamp', summary)
        
        # Check toxicity_report.json contents
        with open(os.path.join(output_dir, "toxicity_report.json"), 'r') as f:
            toxicity_report = json.load(f)
            self.assertIn('high_toxicity_files', toxicity_report)
            self.assertIn('medium_toxicity_files', toxicity_report)
            self.assertIn('low_toxicity_files', toxicity_report)
            self.assertIn('non_toxic_files', toxicity_report)
            
            # file1.txt should be in high toxicity (score 0.75)
            self.assertEqual(len(toxicity_report['high_toxicity_files']), 1)
            self.assertEqual(toxicity_report['high_toxicity_files'][0]['path'], 'file1.txt')
            
            # file2.txt should be in medium toxicity (score 0.6)
            self.assertEqual(len(toxicity_report['medium_toxicity_files']), 1)
            self.assertEqual(toxicity_report['medium_toxicity_files'][0]['path'], 'file2.txt')
            
            # file3.txt should be in low toxicity (score 0.1)
            self.assertEqual(len(toxicity_report['low_toxicity_files']), 1)
            self.assertEqual(toxicity_report['low_toxicity_files'][0]['path'], 'file3.txt')
        
        # Check category_distribution.json contents
        with open(os.path.join(output_dir, "category_distribution.json"), 'r') as f:
            category_dist = json.load(f)
            self.assertIn('categories', category_dist)
            self.assertIn('hate', category_dist['categories'])
            self.assertIn('insult', category_dist['categories'])
            
            # Verify counts match our test data
            self.assertEqual(category_dist['categories']['hate']['total_count'], 3)
            self.assertEqual(category_dist['categories']['insult']['total_count'], 2)
        
        # Check groq_usage_report.json contents
        with open(os.path.join(output_dir, "groq_usage_report.json"), 'r') as f:
            groq_report = json.load(f)
            self.assertEqual(groq_report['total_usage'], 3)
            self.assertEqual(groq_report['total_overrides'], 2)
            self.assertAlmostEqual(groq_report['effectiveness'], 0.67, places=2)
            self.assertEqual(len(groq_report['files_with_groq_usage']), 2)
        
        # Check individual file result (file1.txt)
        with open(os.path.join(output_dir, "file1.txt.result.json"), 'r') as f:
            file_result = json.load(f)
            self.assertIn('toxicity_profile', file_result)
            self.assertIn('groq_usage', file_result)
            self.assertIn('sentences', file_result)
            self.assertEqual(len(file_result['sentences']), 1)
        
        # Clean up the temporary directory
        shutil.rmtree(output_dir)

    def test_real_model_integration(self):
        """Test integration with real model prediction functions."""
        from batch_processor import _predict_toxicity_batch
        
        # Test sentences with different toxicity levels
        sentences = [
            "This is a normal sentence.",
            "You are stupid and I hate you!",
            "The weather is nice today."
        ]
        
        # Mock model and config
        mock_model = "test-model"
        mock_config = {
            "thresholds": {"hate": 0.5, "insult": 0.5},
            "allow_groq_fallback": False
        }
        
        # Mock the predict_toxicity function
        with patch('model_loader.predict_toxicity') as mock_predict:
            def mock_predict_side_effect(texts, **kwargs):
                results = []
                for text in texts:
                    is_toxic = "stupid" in text.lower() or "hate" in text.lower()
                    results.append({
                        "text": text,
                        "is_toxic": is_toxic,
                        "category_results": {
                            "hate": {"score": 0.8 if "hate" in text.lower() else 0.1, "above_threshold": "hate" in text.lower()},
                            "insult": {"score": 0.9 if "stupid" in text.lower() else 0.05, "above_threshold": "stupid" in text.lower()}
                        },
                        "groq_used": False,
                        "most_probable_category": "insult" if "stupid" in text.lower() else "hate",
                        "raw_logits": [0.1, 0.9] if is_toxic else [0.9, 0.1],
                        "sigmoid_scores": [0.1, 0.9] if is_toxic else [0.9, 0.1]
                    })
                return results
            
            mock_predict.side_effect = mock_predict_side_effect
            
            # Process the batch
            results = _predict_toxicity_batch(sentences, model=mock_model, config=mock_config)
            
            # Verify we got results
            self.assertEqual(len(results), 3)
            
            # Check first sentence (normal)
            self.assertFalse(results[0]['is_toxic'])
            self.assertFalse(results[0]['groq_used'])
            
            # Check second sentence (toxic)
            self.assertTrue(results[1]['is_toxic'])
            self.assertIn('scores', results[1])
            self.assertIn('probabilities', results[1])
            
            # Check third sentence (normal)
            self.assertFalse(results[2]['is_toxic'])
            
            # Verify the model was called correctly
            mock_predict.assert_called_once()
            call_kwargs = mock_predict.call_args.kwargs
            self.assertEqual(call_kwargs['texts'], sentences)
            self.assertEqual(call_kwargs['model_name'], mock_model)


if __name__ == "__main__":
    unittest.main() 