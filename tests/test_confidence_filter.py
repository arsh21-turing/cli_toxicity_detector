#!/usr/bin/env python3
"""
Test for the --confidence-filter flag in the toxicity detection CLI.
Verifies that confidence filtering works correctly across different modes.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
import io
from unittest.mock import patch, MagicMock
from contextlib import redirect_stdout
from pathlib import Path

# Import the functions we want to test
from main import handle_stream_processing, _display_stream_result, _display_stream_summary
from categories import ToxicityCategory


class TestConfidenceFilter(unittest.TestCase):
    """Test suite for confidence filtering functionality."""
    
    def setUp(self):
        """Set up test cases."""
        # Create mock args with confidence filter enabled
        self.mock_args = MagicMock()
        self.mock_args.model = None
        self.mock_args.threshold = None
        self.mock_args.confidence_filter = 0.7  # High confidence threshold
        self.mock_args.min_confidence = None    # Add min confidence support
        self.mock_args.max_confidence = None    # Add max confidence support
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
        
        # Create sample results with varying confidence levels
        self.high_confidence_result = {
            'text': 'This is a high confidence toxic comment',
            'is_toxic': True,
            'most_probable_category': ToxicityCategory.INSULT,
            'categories': {'INSULT': True, 'HATE': True, 'THREAT': True},
            'probabilities': {'INSULT': 0.85, 'HATE': 0.80, 'THREAT': 0.60},
            'category_results': {
                ToxicityCategory.INSULT: {'score': 0.85, 'above_threshold': True, 'threshold': 0.5},
                ToxicityCategory.HATE: {'score': 0.80, 'above_threshold': True, 'threshold': 0.5},
                ToxicityCategory.THREAT: {'score': 0.60, 'above_threshold': True, 'threshold': 0.5},
            }
        }
        
        self.medium_confidence_result = {
            'text': 'This is a medium confidence comment',
            'is_toxic': True,
            'most_probable_category': ToxicityCategory.HATE,
            'categories': {'INSULT': True, 'HATE': True, 'THREAT': False},
            'probabilities': {'INSULT': 0.65, 'HATE': 0.55, 'THREAT': 0.30},
            'category_results': {
                ToxicityCategory.INSULT: {'score': 0.65, 'above_threshold': True, 'threshold': 0.5},
                ToxicityCategory.HATE: {'score': 0.55, 'above_threshold': True, 'threshold': 0.5},
                ToxicityCategory.THREAT: {'score': 0.30, 'above_threshold': False, 'threshold': 0.5},
            }
        }
    
    def test_confidence_filter_high_confidence(self):
        """Test that high confidence results are displayed."""
        stats = {'total_lines': 1, 'displayed_lines': 1, 'filtered_lines': 0, 'toxic_lines': 1}
        
        # High confidence should pass the 0.7 threshold
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            _display_stream_result(self.high_confidence_result, stats, self.mock_args)
        
        output = captured_output.getvalue()
        
        # Should display the result
        self.assertIn("TOXIC", output)
        self.assertIn("Confidence: 0.8500", output)  # Max score from INSULT category
        self.assertIn("INSULT", output)
    
    def test_stream_summary_with_confidence_filter(self):
        """Test that stream summary includes confidence filter statistics."""
        stats = {
            'total_lines': 10,
            'displayed_lines': 6,  # 6 results passed confidence filter
            'filtered_lines': 4,   # 4 results were filtered
            'toxic_lines': 3,      # 3 toxic among displayed
            'categories': {'INSULT': 2, 'HATE': 1},
            'groq_usage': {'total': 0, 'overrides': 0},
            'session_start': '2023-06-01T12:00:00',
            'session_end': '2023-06-01T12:10:00'
        }
        
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            _display_stream_summary(stats, self.mock_args)
        
        output = captured_output.getvalue()
        
        # Check confidence filter information is displayed
        self.assertIn("STREAMING SESSION SUMMARY", output)
        self.assertIn("Total lines processed: 10", output)
        self.assertIn("Lines displayed: 6", output)
        self.assertIn("Lines filtered by confidence: 4", output)
        self.assertIn("Toxic lines: 3/6", output)  # Based on displayed lines
        self.assertIn("Confidence filter applied: above 0.7000", output)
    
    @patch('main.predict_toxicity')
    @patch('main.load_config')
    @patch('builtins.input')
    def test_stream_processing_with_confidence_filter(self, mock_input, mock_load_config, mock_predict):
        """Test the full streaming process with confidence filtering."""
        # Setup mocks
        mock_load_config.return_value = {
            "model": {"name": "unitary/toxic-bert", "threshold": 0.5},
            "thresholds": {}
        }
        
        # Mock input to simulate typed lines and EOF
        mock_input.side_effect = [
            "High confidence line",
            "Medium confidence line", 
            EOFError()
        ]
        
        # Mock predictions with varying confidence levels
        mock_predict.side_effect = [
            [self.high_confidence_result],
            [self.medium_confidence_result]
        ]
        
        # Run streaming with confidence filter
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            stats = handle_stream_processing(self.mock_args)
        
        # Verify statistics
        self.assertEqual(stats['total_lines'], 2)
        self.assertEqual(stats['displayed_lines'], 1)  # Only high confidence result
        self.assertEqual(stats['filtered_lines'], 1)   # Medium confidence filtered
        self.assertEqual(stats['toxic_lines'], 2)      # Two toxic results overall
        
        # Verify output contains confidence filter information
        output = captured_output.getvalue()
        self.assertIn("Confidence filter: showing results above 0.7000", output)


if __name__ == '__main__':
    unittest.main() 