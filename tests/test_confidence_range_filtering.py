#!/usr/bin/env python3
"""
Test for confidence range filtering (--min-confidence and --max-confidence) in the toxicity detection CLI.
Verifies that confidence filtering works across all processing modes.
"""
from __future__ import annotations

import json
import io
import unittest
from unittest.mock import patch, MagicMock, call
from contextlib import redirect_stdout
from pathlib import Path

# Import the functions we want to test
from main import check_confidence_filter, handle_stream_processing, _process_single
from categories import ToxicityCategory


class ConfidenceRangeFilteringTest(unittest.TestCase):
    """Test suite for confidence range filtering functionality."""
    
    def setUp(self):
        """Set up test case."""
        # Create mock args
        self.mock_args = MagicMock()
        self.mock_args.model = None
        self.mock_args.threshold = None
        self.mock_args.confidence_filter = None
        self.mock_args.min_confidence = None
        self.mock_args.max_confidence = None
        self.mock_args.allow_groq_fallback = False
        self.mock_args.groq_lower_bound = None
        self.mock_args.groq_upper_bound = None
        self.mock_args.groq_tie_policy = None
        self.mock_args.json = False
        self.mock_args.verbose = False
        self.mock_args.probabilities = False
        self.mock_args.quiet = False
        self.mock_args.no_color = False
        self.mock_args.confidence_explain = False
        self.mock_args.confidence_sort = None
    
    def test_check_confidence_filter_no_filter(self):
        """Test confidence filtering with no filters set."""
        passes, reason = check_confidence_filter(0.75, self.mock_args)
        self.assertTrue(passes)
        self.assertEqual(reason, "")
    
    def test_check_confidence_filter_legacy_filter(self):
        """Test confidence filtering with legacy --confidence-filter."""
        self.mock_args.confidence_filter = 0.6
        
        # Should pass above threshold
        passes, reason = check_confidence_filter(0.75, self.mock_args)
        self.assertTrue(passes)
        self.assertEqual(reason, "")
        
        # Should fail below threshold
        passes, reason = check_confidence_filter(0.5, self.mock_args)
        self.assertFalse(passes)
        self.assertIn("below threshold", reason)
        self.assertIn("0.6000", reason)
    
    def test_check_confidence_filter_min_only(self):
        """Test confidence filtering with only --min-confidence."""
        self.mock_args.min_confidence = 0.4
        
        # Should pass above minimum
        passes, reason = check_confidence_filter(0.75, self.mock_args)
        self.assertTrue(passes)
        self.assertEqual(reason, "")
        
        # Should fail below minimum
        passes, reason = check_confidence_filter(0.3, self.mock_args)
        self.assertFalse(passes)
        self.assertIn("below minimum", reason)
        self.assertIn("0.4000", reason)
    
    def test_check_confidence_filter_max_only(self):
        """Test confidence filtering with only --max-confidence."""
        self.mock_args.max_confidence = 0.8
        
        # Should pass below maximum
        passes, reason = check_confidence_filter(0.75, self.mock_args)
        self.assertTrue(passes)
        self.assertEqual(reason, "")
        
        # Should fail above maximum
        passes, reason = check_confidence_filter(0.9, self.mock_args)
        self.assertFalse(passes)
        self.assertIn("above maximum", reason)
        self.assertIn("0.8000", reason)
    
    def test_check_confidence_filter_range(self):
        """Test confidence filtering with both min and max set."""
        self.mock_args.min_confidence = 0.3
        self.mock_args.max_confidence = 0.7
        
        # Should pass within range
        passes, reason = check_confidence_filter(0.5, self.mock_args)
        self.assertTrue(passes)
        self.assertEqual(reason, "")
        
        # Should fail below minimum
        passes, reason = check_confidence_filter(0.2, self.mock_args)
        self.assertFalse(passes)
        self.assertIn("below minimum", reason)
        self.assertIn("0.3000", reason)
        
        # Should fail above maximum
        passes, reason = check_confidence_filter(0.8, self.mock_args)
        self.assertFalse(passes)
        self.assertIn("above maximum", reason)
        self.assertIn("0.7000", reason)
        
        # Edge cases: exactly at boundaries
        passes, reason = check_confidence_filter(0.3, self.mock_args)
        self.assertTrue(passes)  # At minimum should pass
        
        passes, reason = check_confidence_filter(0.7, self.mock_args)
        self.assertTrue(passes)  # At maximum should pass
    
    def test_check_confidence_filter_invalid_values(self):
        """Test confidence filtering with invalid values."""
        # Test with string values that can't be converted to float
        self.mock_args.min_confidence = "invalid"
        self.mock_args.max_confidence = "also_invalid"
        
        # Should pass (invalid values are ignored)
        passes, reason = check_confidence_filter(0.5, self.mock_args)
        self.assertTrue(passes)
        self.assertEqual(reason, "")
    
    @patch('main.predict_toxicity')
    @patch('main.load_config')
    def test_single_text_confidence_filtering(self, mock_load_config, mock_predict):
        """Test confidence filtering in single text processing mode."""
        # Setup mocks
        mock_load_config.return_value = {
            "model": {"name": "unitary/toxic-bert", "threshold": 0.5},
            "thresholds": {}
        }
        
        # Mock prediction result with medium confidence
        mock_result = {
            'text': 'Test text',
            'is_toxic': True,
            'most_probable_category': ToxicityCategory.INSULT,
            'categories': {'INSULT': True, 'HATE': False},  # Add for _print_human_single
            'probabilities': {'INSULT': 0.6, 'HATE': 0.4},  # Add for _print_human_single
            'category_results': {
                ToxicityCategory.INSULT: {'score': 0.6, 'above_threshold': True, 'threshold': 0.5},
                ToxicityCategory.HATE: {'score': 0.4, 'above_threshold': False, 'threshold': 0.5},
            }
        }
        mock_predict.return_value = [mock_result]
        
        # Test with range that excludes this confidence (0.6)
        self.mock_args.min_confidence = 0.7  # Above our score
        self.mock_args.max_confidence = 0.9
        
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            result = _process_single("Test text", cfg=mock_load_config.return_value, args=self.mock_args)
        
        output = captured_output.getvalue()
        
        # Should show filtered message
        self.assertIn("Result filtered", output)
        self.assertIn("below minimum", output)
        self.assertIn("0.7000", output)
        
        # Test with range that includes this confidence
        self.mock_args.min_confidence = 0.5  # Below our score
        self.mock_args.max_confidence = 0.8  # Above our score
        
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            result = _process_single("Test text", cfg=mock_load_config.return_value, args=self.mock_args)
        
        output = captured_output.getvalue()
        
        # Should not show filtered message
        self.assertNotIn("Result filtered", output)
    
    @patch('main.predict_toxicity')
    @patch('main.load_config')
    @patch('builtins.input')
    def test_stream_confidence_filtering_statistics(self, mock_input, mock_load_config, mock_predict):
        """Test that streaming mode tracks confidence filtering statistics correctly."""
        # Setup mocks
        mock_load_config.return_value = {
            "model": {"name": "unitary/toxic-bert", "threshold": 0.5},
            "thresholds": {}
        }
        
        # Mock input to simulate typed lines and EOF
        mock_input.side_effect = [
            "Low confidence text",    # Will have score 0.2 - below range
            "Medium confidence text", # Will have score 0.5 - in range
            "High confidence text",   # Will have score 0.9 - above range
            EOFError()
        ]
        
        # Mock predictions with different confidence levels
        low_confidence_result = {
            'text': 'Low confidence text',
            'is_toxic': False,
            'most_probable_category': ToxicityCategory.NON_TOXIC,
            'categories': {'INSULT': False},
            'probabilities': {'INSULT': 0.2},
            'category_results': {
                ToxicityCategory.INSULT: {'score': 0.2, 'above_threshold': False, 'threshold': 0.5},
            }
        }
        
        medium_confidence_result = {
            'text': 'Medium confidence text',
            'is_toxic': True,
            'most_probable_category': ToxicityCategory.INSULT,
            'categories': {'INSULT': True},
            'probabilities': {'INSULT': 0.5},
            'category_results': {
                ToxicityCategory.INSULT: {'score': 0.5, 'above_threshold': True, 'threshold': 0.5},
            }
        }
        
        high_confidence_result = {
            'text': 'High confidence text',
            'is_toxic': True,
            'most_probable_category': ToxicityCategory.INSULT,
            'categories': {'INSULT': True},
            'probabilities': {'INSULT': 0.9},
            'category_results': {
                ToxicityCategory.INSULT: {'score': 0.9, 'above_threshold': True, 'threshold': 0.5},
            }
        }
        
        mock_predict.side_effect = [
            [low_confidence_result],
            [medium_confidence_result],
            [high_confidence_result]
        ]
        
        # Set confidence range to filter out low and high confidence
        self.mock_args.min_confidence = 0.4
        self.mock_args.max_confidence = 0.6
        
        # Run function with captured stdout
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            stats = handle_stream_processing(self.mock_args)
        
        # Verify statistics
        self.assertEqual(stats['total_lines'], 3)
        self.assertEqual(stats['displayed_lines'], 1)    # Only medium confidence shown
        self.assertEqual(stats['filtered_lines'], 2)     # Low and high filtered
        self.assertEqual(stats['below_range_lines'], 1)  # Low confidence
        self.assertEqual(stats['above_range_lines'], 1)  # High confidence
        self.assertEqual(stats['toxic_lines'], 2)        # All toxic lines (regardless of filtering)
        
        # Check output contains filtering information
        output = captured_output.getvalue()
        self.assertIn("Confidence filter: showing results above 0.4000 and below 0.6000", output)
        self.assertIn("1 below range", output)
        self.assertIn("Above maximum confidence: 1", output)  # Check in summary section
    
    @patch('main.predict_toxicity')
    @patch('main.load_config')
    @patch('builtins.input')
    def test_stream_json_mode_with_filtering(self, mock_input, mock_load_config, mock_predict):
        """Test that JSON mode includes confidence filtering information."""
        # Setup mocks
        mock_load_config.return_value = {
            "model": {"name": "unitary/toxic-bert", "threshold": 0.5},
            "thresholds": {}
        }
        
        # Mock input
        mock_input.side_effect = [
            "Test text",
            EOFError()
        ]
        
        # Mock prediction
        mock_result = {
            'text': 'Test text',
            'is_toxic': False,
            'most_probable_category': ToxicityCategory.NON_TOXIC,
            'categories': {'INSULT': False},
            'probabilities': {'INSULT': 0.3},
            'category_results': {
                ToxicityCategory.INSULT: {'score': 0.3, 'above_threshold': False, 'threshold': 0.5},
            }
        }
        mock_predict.return_value = [mock_result]
        
        # Set confidence range and JSON mode
        self.mock_args.min_confidence = 0.4
        self.mock_args.max_confidence = 0.8
        self.mock_args.json = True
        
        # Run function with captured stdout
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            stats = handle_stream_processing(self.mock_args)
        
        # Verify JSON output contains confidence filter settings
        output = captured_output.getvalue()
        
        # Parse the final JSON summary - it's formatted with indentation
        # Find the JSON block between "JSON Summary:" and "="
        json_start = output.find("JSON Summary:")
        json_end = output.find("=" * 50, json_start + 1)
        
        if json_start != -1 and json_end != -1:
            json_text = output[json_start + len("JSON Summary:"):json_end].strip()
            try:
                json_summary = json.loads(json_text)
            except json.JSONDecodeError:
                json_summary = None
        else:
            json_summary = None
        
        self.assertIsNotNone(json_summary, f"Could not find JSON summary in output: {output}")
        self.assertIn('confidence_filters', json_summary)
        self.assertEqual(json_summary['confidence_filters']['min_confidence'], 0.4)
        self.assertEqual(json_summary['confidence_filters']['max_confidence'], 0.8)
        self.assertEqual(json_summary['below_range_lines'], 1)
        self.assertEqual(json_summary['above_range_lines'], 0)


if __name__ == '__main__':
    unittest.main() 