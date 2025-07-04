#!/usr/bin/env python3
"""
Test for the --stream flag in the toxicity detection CLI.
Verifies that streaming mode processes text line-by-line with immediate feedback.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
import io
from unittest.mock import patch, MagicMock, call
from contextlib import redirect_stdout
from pathlib import Path

# Import the functions we want to test
from main import handle_stream_processing, _display_stream_result, _display_stream_summary
from categories import ToxicityCategory


class StreamProcessingTest(unittest.TestCase):
    """Test suite for streaming functionality."""
    
    def setUp(self):
        """Set up test case."""
        # Create mock args
        self.mock_args = MagicMock()
        self.mock_args.model = None
        self.mock_args.threshold = None
        self.mock_args.confidence_filter = None
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
        
        # Create sample results that match the actual predict_toxicity output
        self.sample_result_toxic = {
            'text': 'This is a toxic comment',
            'is_toxic': True,
            'most_probable_category': ToxicityCategory.INSULT,
            'category_results': {
                ToxicityCategory.INSULT: {'score': 0.8, 'above_threshold': True, 'threshold': 0.5},
                ToxicityCategory.HATE: {'score': 0.6, 'above_threshold': True, 'threshold': 0.5},
                ToxicityCategory.THREAT: {'score': 0.3, 'above_threshold': False, 'threshold': 0.5},
            }
        }
        
        self.sample_result_safe = {
            'text': 'This is a safe comment',
            'is_toxic': False,
            'most_probable_category': ToxicityCategory.NON_TOXIC,
            'category_results': {
                ToxicityCategory.INSULT: {'score': 0.2, 'above_threshold': False, 'threshold': 0.5},
                ToxicityCategory.HATE: {'score': 0.1, 'above_threshold': False, 'threshold': 0.5},
                ToxicityCategory.THREAT: {'score': 0.1, 'above_threshold': False, 'threshold': 0.5},
            }
        }
    
    def test_display_stream_result_toxic(self):
        """Test displaying a toxic result in stream mode."""
        stats = {'total_lines': 1, 'toxic_lines': 1, 'categories': {'INSULT': 1}}
        
        # Capture stdout for testing
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            _display_stream_result(self.sample_result_toxic, stats, self.mock_args)
        
        output = captured_output.getvalue()
        
        # Check key elements are in the output
        self.assertIn("TOXIC", output)  # Should include toxicity status
        self.assertIn("INSULT", output)   # Should mention top category
        self.assertIn("0.8", output)    # Should include score
        self.assertIn("Running stats: 1/1", output)  # Should show stats
    
    def test_display_stream_summary(self):
        """Test the display of a streaming session summary."""
        # Create sample stats
        stats = {
            'total_lines': 10,
            'toxic_lines': 3,
            'categories': {'INSULT': 2, 'TOXIC': 3, 'THREAT': 1},
            'groq_usage': {'total': 2, 'overrides': 1},
            'session_start': '2023-06-01T12:00:00',
            'session_end': '2023-06-01T12:10:00'
        }
        
        # Capture stdout for testing
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            _display_stream_summary(stats, self.mock_args)
        
        output = captured_output.getvalue()
        
        # Check key elements are in the output
        self.assertIn("STREAMING SESSION SUMMARY", output)
        self.assertIn("Total lines processed: 10", output)
        self.assertIn("Toxic lines: 3/10", output)
    
    @patch('main.predict_toxicity')
    @patch('main.load_config')
    @patch('builtins.input')
    def test_handle_stream_processing(self, mock_input, mock_load_config, mock_predict):
        """Test the full stream processing function."""
        # Setup mocks
        mock_load_config.return_value = {
            "model": {"name": "unitary/toxic-bert", "threshold": 0.5},
            "thresholds": {}
        }
        
        # Mock input to simulate typed lines and EOF
        mock_input.side_effect = [
            "This is a safe comment",
            "This is a toxic comment",
            EOFError()
        ]
        
        # Mock predictions
        mock_predict.side_effect = [
            [self.sample_result_safe],
            [self.sample_result_toxic]
        ]
        
        # Run function with captured stdout
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            stats = handle_stream_processing(self.mock_args)
        
        # Verify stats
        self.assertEqual(stats['total_lines'], 2)
        self.assertEqual(stats['toxic_lines'], 1)
        self.assertIn('results', stats)
        self.assertEqual(len(stats['results']), 2)


if __name__ == '__main__':
    unittest.main() 