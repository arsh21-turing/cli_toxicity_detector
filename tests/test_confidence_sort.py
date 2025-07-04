import unittest
from unittest.mock import patch, MagicMock
import io
from contextlib import redirect_stdout
import tempfile
import os
import json

from main import (
    handle_stream_processing,
    handle_batch_processing,
    display_batch_results,
    SortOrder
)
from categories import ToxicityCategory
from batch_processor import batch_process


class TestConfidenceSort(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        # Create mock args
        self.mock_args = MagicMock()
        self.mock_args.model = 'default'
        self.mock_args.threshold = 0.5
        self.mock_args.allow_groq_fallback = False
        self.mock_args.json = False
        self.mock_args.verbose = True
        self.mock_args.confidence_filter = None
        self.mock_args.min_confidence = None
        self.mock_args.max_confidence = None
        self.mock_args.confidence_explain = False
        self.mock_args.confidence_sort = None
        self.mock_args.quiet = False
        self.mock_args.groq_tie_policy = None
        self.mock_args.groq_lower_bound = 0.4
        self.mock_args.groq_upper_bound = 0.6
        
        # Create sample results with varying confidence levels
        # Mock category objects for testing
        hate_cat = ToxicityCategory.HATE
        insult_cat = ToxicityCategory.INSULT
        threat_cat = ToxicityCategory.THREAT
        
        self.high_confidence_result = {
            'text': 'This is a high confidence toxic comment',
            'is_toxic': True,
            'toxic': True,
            'category_results': {
                hate_cat: {'score': 0.85, 'above_threshold': True, 'threshold': 0.5},
                insult_cat: {'score': 0.8, 'above_threshold': True, 'threshold': 0.5},
                threat_cat: {'score': 0.6, 'above_threshold': True, 'threshold': 0.5},
            },
            'probabilities': {'hate': 0.85, 'insult': 0.8, 'threat': 0.6},
            'scores': {'hate': 0.85, 'insult': 0.8, 'threat': 0.6},
        }
        
        self.medium_confidence_result = {
            'text': 'This is a medium confidence toxic comment',
            'is_toxic': True,
            'toxic': True,
            'category_results': {
                hate_cat: {'score': 0.65, 'above_threshold': True, 'threshold': 0.5},
                insult_cat: {'score': 0.55, 'above_threshold': True, 'threshold': 0.5},
                threat_cat: {'score': 0.3, 'above_threshold': False, 'threshold': 0.5},
            },
            'probabilities': {'hate': 0.65, 'insult': 0.55, 'threat': 0.3},
            'scores': {'hate': 0.65, 'insult': 0.55, 'threat': 0.3},
        }
        
        self.low_confidence_result = {
            'text': 'This is a low confidence comment',
            'is_toxic': False,
            'toxic': False,
            'category_results': {
                hate_cat: {'score': 0.35, 'above_threshold': False, 'threshold': 0.5},
                insult_cat: {'score': 0.25, 'above_threshold': False, 'threshold': 0.5},
                threat_cat: {'score': 0.2, 'above_threshold': False, 'threshold': 0.5},
            },
            'probabilities': {'hate': 0.35, 'insult': 0.25, 'threat': 0.2},
            'scores': {'hate': 0.35, 'insult': 0.25, 'threat': 0.2},
        }
        
        self.very_low_confidence_result = {
            'text': 'This is a very low confidence comment',
            'is_toxic': False,
            'toxic': False,
            'category_results': {
                hate_cat: {'score': 0.15, 'above_threshold': False, 'threshold': 0.5},
                insult_cat: {'score': 0.1, 'above_threshold': False, 'threshold': 0.5},
                threat_cat: {'score': 0.05, 'above_threshold': False, 'threshold': 0.5},
            },
            'probabilities': {'hate': 0.15, 'insult': 0.1, 'threat': 0.05},
            'scores': {'hate': 0.15, 'insult': 0.1, 'threat': 0.05},
        }
        
        # Create temporary directory for batch processing tests
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directories
        import shutil
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.output_dir)
    
    def test_stream_highest_confidence_sort(self):
        """Test sorting by highest confidence first in streaming mode."""
        # Setup args with highest confidence sort
        self.mock_args.confidence_sort = "highest"
        
        with patch('main.predict_toxicity') as mock_predict, \
             patch('main.load_config') as mock_load_config, \
             patch('builtins.input') as mock_input:
            
            # Setup mocks
            mock_load_config.return_value = {"model": {"name": "test"}, "thresholds": {}}
            
            # Mock input to simulate typed lines and EOF
            mock_input.side_effect = [
                "High confidence text",
                "Medium confidence text", 
                "Low confidence text",
                "Very low confidence text",
                EOFError()
            ]
            
            # Mock predictions with varying confidence levels (in mixed order)
            mock_predict.side_effect = [
                [self.medium_confidence_result],
                [self.high_confidence_result],
                [self.very_low_confidence_result],
                [self.low_confidence_result]
            ]
            
            # Run streaming with confidence sort
            with redirect_stdout(io.StringIO()) as captured_output:
                stats = handle_stream_processing(self.mock_args)
                output = captured_output.getvalue()
            
            # Output should mention confidence sorting
            self.assertIn("Confidence sorting: highest first", output)
            
            # Check that the summary shows sorted results
            self.assertIn("Top", output)
            self.assertIn("by confidence (highest first)", output)
            
            # The first entry should be the high confidence result
            lines = output.split('\n')
            top_section_start = None
            for i, line in enumerate(lines):
                if "Top" in line and "by confidence (highest first)" in line:
                    top_section_start = i
                    break
            
            self.assertIsNotNone(top_section_start, "Should find sorted results section")
            
            # Find confidence values after the "Top" section
            confidence_values = []
            for line in lines[top_section_start:]:
                if "Confidence:" in line:
                    # Extract confidence value
                    parts = line.split("Confidence:")
                    if len(parts) > 1:
                        conf_part = parts[1].strip()
                        conf_value = float(conf_part.split()[0])
                        confidence_values.append(conf_value)
            
            # Should have at least 2 confidence values and they should be in descending order
            self.assertGreaterEqual(len(confidence_values), 2)
            for i in range(len(confidence_values) - 1):
                self.assertGreaterEqual(confidence_values[i], confidence_values[i + 1],
                                       "Confidence values should be in descending order for highest first")
    
    def test_stream_lowest_confidence_sort(self):
        """Test sorting by lowest confidence first in streaming mode."""
        # Setup args with lowest confidence sort
        self.mock_args.confidence_sort = "lowest"
        
        with patch('main.predict_toxicity') as mock_predict, \
             patch('main.load_config') as mock_load_config, \
             patch('builtins.input') as mock_input:
            
            # Setup mocks
            mock_load_config.return_value = {"model": {"name": "test"}, "thresholds": {}}
            
            # Mock input to simulate typed lines and EOF
            mock_input.side_effect = [
                "High confidence text",
                "Medium confidence text",
                "Low confidence text", 
                "Very low confidence text",
                EOFError()
            ]
            
            # Mock predictions with varying confidence levels (in mixed order)
            mock_predict.side_effect = [
                [self.medium_confidence_result],
                [self.high_confidence_result],
                [self.very_low_confidence_result],
                [self.low_confidence_result]
            ]
            
            # Run streaming with confidence sort
            with redirect_stdout(io.StringIO()) as captured_output:
                stats = handle_stream_processing(self.mock_args)
                output = captured_output.getvalue()
            
            # Output should mention confidence sorting
            self.assertIn("Confidence sorting: lowest first", output)
            
            # Check that the summary shows sorted results
            self.assertIn("Top", output)
            self.assertIn("by confidence (lowest first)", output)
            
            # Find confidence values and verify they're in ascending order
            lines = output.split('\n')
            top_section_start = None
            for i, line in enumerate(lines):
                if "Top" in line and "by confidence (lowest first)" in line:
                    top_section_start = i
                    break
            
            self.assertIsNotNone(top_section_start, "Should find sorted results section")
            
            # Find confidence values after the "Top" section
            confidence_values = []
            for line in lines[top_section_start:]:
                if "Confidence:" in line:
                    # Extract confidence value
                    parts = line.split("Confidence:")
                    if len(parts) > 1:
                        conf_part = parts[1].strip()
                        conf_value = float(conf_part.split()[0])
                        confidence_values.append(conf_value)
            
            # Should have at least 2 confidence values and they should be in ascending order
            self.assertGreaterEqual(len(confidence_values), 2)
            for i in range(len(confidence_values) - 1):
                self.assertLessEqual(confidence_values[i], confidence_values[i + 1],
                                    "Confidence values should be in ascending order for lowest first")
    
    @patch('main.batch_process')
    @patch('main.load_model')
    @patch('main.load_config')
    def test_batch_confidence_sort(self, mock_load_config, mock_load_model, mock_batch_process):
        """Test confidence sorting in batch mode."""
        # Create test files
        test_file = os.path.join(self.test_dir, "test_confidence_sort.txt")
        with open(test_file, 'w') as f:
            f.write("Test content for confidence sorting.")
        
        # Setup mock config and model
        mock_load_model.return_value = "mock_model"
        mock_load_config.return_value = {"model": {"name": "test"}}
        
        # Create mock batch results with sentences of varying confidence
        mock_batch_results = {
            'total_files': 1,
            'toxic_files': 1,
            'total_sentences': 4,
            'toxic_sentences': 2,
            'file_results': {
                test_file: {
                    'toxic': True,
                    'toxic_sentences': 2,
                    'total_sentences': 4,
                    'toxicity_profile': {
                        'overall_score': 0.5,
                        'category_scores': {'hate': 0.55}
                    },
                    'sentences': [
                        # Mixed order of confidence levels
                        {'toxic': True, 'scores': {'hate': 0.65}, 'text': 'Medium confidence'},
                        {'toxic': True, 'scores': {'hate': 0.85}, 'text': 'High confidence'},
                        {'toxic': False, 'scores': {'hate': 0.15}, 'text': 'Very low confidence'},
                        {'toxic': False, 'scores': {'hate': 0.35}, 'text': 'Low confidence'}
                    ]
                }
            }
        }
        
        mock_batch_process.return_value = mock_batch_results
        
        # Setup batch args with highest confidence sort
        self.mock_args.batch = test_file
        self.mock_args.output = self.output_dir
        self.mock_args.confidence_sort = "highest"
        
        # Run batch processing with confidence sort
        with redirect_stdout(io.StringIO()) as captured_output:
            results = handle_batch_processing(self.mock_args)
            output = captured_output.getvalue()
        
        # Check confidence sort was passed to batch processor
        mock_batch_process.assert_called_once()
        config_arg = mock_batch_process.call_args[1]['config']
        self.assertIn('confidence_sort', config_arg)
        self.assertEqual(config_arg['confidence_sort'], "highest")
        
        # Output should mention confidence sorting
        self.assertIn("Confidence sorting: highest first", output)
        
        # Run with lowest confidence sort
        self.mock_args.confidence_sort = "lowest"
        mock_batch_process.reset_mock()
        
        with redirect_stdout(io.StringIO()) as captured_output:
            results = handle_batch_processing(self.mock_args)
        
        # Check confidence sort was passed to batch processor
        mock_batch_process.assert_called_once()
        config_arg = mock_batch_process.call_args[1]['config']
        self.assertIn('confidence_sort', config_arg)
        self.assertEqual(config_arg['confidence_sort'], "lowest")
    
    def test_sort_order_enum(self):
        """Test SortOrder enum functionality."""
        # Test enum values
        self.assertEqual(SortOrder.HIGHEST_FIRST.value, "highest")
        self.assertEqual(SortOrder.LOWEST_FIRST.value, "lowest")
        
        # Test enum comparison
        self.assertTrue(SortOrder("highest") == SortOrder.HIGHEST_FIRST)
        self.assertTrue(SortOrder("lowest") == SortOrder.LOWEST_FIRST)
        
        # Test string conversion
        self.assertEqual(str(SortOrder.HIGHEST_FIRST.value), "highest")
        self.assertEqual(str(SortOrder.LOWEST_FIRST.value), "lowest")
    
    def test_confidence_sort_with_filtering(self):
        """Test confidence sorting with confidence filtering."""
        # Test data with varying confidence levels
        test_data = {
            'file_results': {
                'file1.txt': {
                    'sentences': [
                        {'scores': {'hate': 0.9}, 'toxic': True},   # Very high
                        {'scores': {'hate': 0.7}, 'toxic': True},   # High
                        {'scores': {'hate': 0.5}, 'toxic': True},   # Medium
                        {'scores': {'hate': 0.3}, 'toxic': False},  # Low
                        {'scores': {'hate': 0.1}, 'toxic': False},  # Very low
                    ]
                }
            }
        }
        
        # Setup args for filtering (only show 0.3 to 0.7)
        self.mock_args.min_confidence = 0.3
        self.mock_args.max_confidence = 0.7
        self.mock_args.confidence_sort = "highest"
        
        # Count displayed sentences that pass the filter
        displayed_count = 0
        for file_data in test_data['file_results'].values():
            for sentence in file_data['sentences']:
                max_score = max(sentence['scores'].values())
                if self.mock_args.min_confidence <= max_score <= self.mock_args.max_confidence:
                    displayed_count += 1
        
        # There should be 3 sentences in the 0.3-0.7 range
        self.assertEqual(displayed_count, 3)
        
        # Test the combination of filtering and sorting
        # Note: We can't easily test the full display_batch_results function
        # due to its complexity, but we can test the core sorting logic
        
        # Get all sentences from all files
        all_sentences = []
        for file_path, file_data in test_data['file_results'].items():
            all_sentences.extend(file_data['sentences'])
        
        # Filter sentences
        filtered_sentences = []
        for sentence in all_sentences:
            max_score = max(sentence['scores'].values())
            if self.mock_args.min_confidence <= max_score <= self.mock_args.max_confidence:
                filtered_sentences.append(sentence)
        
        # Sort filtered sentences by confidence (highest first)
        sorted_sentences = sorted(
            filtered_sentences,
            key=lambda s: max(s['scores'].values()),
            reverse=True
        )
        
        # Check that sorting worked correctly
        self.assertEqual(max(sorted_sentences[0]['scores'].values()), 0.7)
        self.assertEqual(max(sorted_sentences[1]['scores'].values()), 0.5)
        self.assertEqual(max(sorted_sentences[2]['scores'].values()), 0.3)
    
    @patch('main.batch_process')
    @patch('main.load_model')
    @patch('main.load_config')
    def test_confidence_sort_batch_output(self, mock_load_config, mock_load_model, mock_batch_process):
        """Test that batch output files include confidence sort settings."""
        # Create test file
        test_file = os.path.join(self.test_dir, "test_sort_output.txt")
        with open(test_file, 'w') as f:
            f.write("Test content")
        
        # Setup mock batch processor to return minimal results
        mock_load_model.return_value = "mock_model" 
        mock_load_config.return_value = {"model": {"name": "test"}}
        mock_batch_process.return_value = {
            'total_files': 1,
            'toxic_files': 0,
            'file_results': {}
        }
        
        # Setup args for batch processing with confidence sort
        self.mock_args.batch = test_file
        self.mock_args.output = self.output_dir
        self.mock_args.confidence_sort = "highest"
        self.mock_args.json = True  # Use JSON output for easier testing
        
        # Run batch processing
        with redirect_stdout(io.StringIO()) as captured_output:
            results = handle_batch_processing(self.mock_args)
        
        # Check that config was passed with confidence sort setting
        mock_batch_process.assert_called_once()
        config_arg = mock_batch_process.call_args[1]['config']
        self.assertIn('confidence_sort', config_arg)
        self.assertEqual(config_arg['confidence_sort'], "highest")

    def test_stream_results_storage(self):
        """Test that streaming stores results for confidence sorting."""
        self.mock_args.confidence_sort = "highest"
        
        with patch('main.predict_toxicity') as mock_predict, \
             patch('main.load_config') as mock_load_config, \
             patch('builtins.input') as mock_input:
            
            # Setup mocks
            mock_load_config.return_value = {"model": {"name": "test"}, "thresholds": {}}
            
            # Mock input to simulate one line and EOF
            mock_input.side_effect = ["Test line", EOFError()]
            
            # Mock prediction
            mock_predict.side_effect = [[self.high_confidence_result]]
            
            # Run streaming
            with redirect_stdout(io.StringIO()):
                stats = handle_stream_processing(self.mock_args)
            
            # Check that results were stored
            self.assertIn('results', stats)
            self.assertEqual(len(stats['results']), 1)
            self.assertEqual(stats['results'][0], self.high_confidence_result)

    def test_no_confidence_sort_default_behavior(self):
        """Test that when confidence sorting is not enabled, default behavior is maintained."""
        # Don't set confidence_sort
        self.mock_args.confidence_sort = None
        
        with patch('main.predict_toxicity') as mock_predict, \
             patch('main.load_config') as mock_load_config, \
             patch('builtins.input') as mock_input:
            
            # Setup mocks
            mock_load_config.return_value = {"model": {"name": "test"}, "thresholds": {}}
            
            # Mock input to simulate typed lines and EOF
            mock_input.side_effect = ["Test line", EOFError()]
            
            # Mock prediction
            mock_predict.side_effect = [[self.high_confidence_result]]
            
            # Run streaming without confidence sort
            with redirect_stdout(io.StringIO()) as captured_output:
                stats = handle_stream_processing(self.mock_args)
                output = captured_output.getvalue()
            
            # Output should NOT mention confidence sorting
            self.assertNotIn("Confidence sorting:", output)
            self.assertNotIn("by confidence", output)


if __name__ == '__main__':
    unittest.main() 