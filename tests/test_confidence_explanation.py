import unittest
from unittest.mock import patch, MagicMock
import io
from contextlib import redirect_stdout
import json

from main import (
    generate_confidence_explanation,
    _display_confidence_explanation,
    _find_toxic_terms,
    _has_directive_language,
    _has_explicit_content,
    _has_implicit_content
)


class TestConfidenceExplanation(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        # Create sample results with varying confidence levels
        self.high_confidence_result = {
            'text': 'This is a toxic comment with explicit hate language, you idiot!',
            'is_toxic': True,
            'category_results': {
                MockCategory('hate'): {'score': 0.85, 'above_threshold': True},
                MockCategory('insult'): {'score': 0.8, 'above_threshold': True},
                MockCategory('threat'): {'score': 0.6, 'above_threshold': True}
            }
        }
        
        self.medium_confidence_result = {
            'text': 'This comment is somewhat rude and disrespectful.',
            'is_toxic': True,
            'category_results': {
                MockCategory('hate'): {'score': 0.65, 'above_threshold': True},
                MockCategory('insult'): {'score': 0.55, 'above_threshold': True},
                MockCategory('threat'): {'score': 0.3, 'above_threshold': False}
            }
        }
        
        self.borderline_confidence_result = {
            'text': 'This might be interpreted negatively by some readers.',
            'is_toxic': True,
            'category_results': {
                MockCategory('hate'): {'score': 0.45, 'above_threshold': False},
                MockCategory('insult'): {'score': 0.52, 'above_threshold': True},
                MockCategory('threat'): {'score': 0.2, 'above_threshold': False}
            },
            'groq_fallback_used': True,
            'groq_changed_classification': True
        }
        
        self.low_confidence_result = {
            'text': 'This is a normal, neutral comment.',
            'is_toxic': False,
            'category_results': {
                MockCategory('hate'): {'score': 0.25, 'above_threshold': False},
                MockCategory('insult'): {'score': 0.15, 'above_threshold': False},
                MockCategory('threat'): {'score': 0.05, 'above_threshold': False}
            }
        }
    
    def test_generate_confidence_explanation(self):
        """Test generating confidence explanations for different confidence levels."""
        # Test high confidence explanation
        high_explanation = generate_confidence_explanation(
            self.high_confidence_result,
            self.high_confidence_result['text']
        )
        
        self.assertEqual(high_explanation['confidence_level'], "high")
        self.assertEqual(high_explanation['primary_category'], "hate")
        self.assertGreaterEqual(len(high_explanation['confidence_factors']), 1)
        self.assertIn("explanation", high_explanation)
        
        # Verify toxic term detection - should find "hate" term since hate has highest score
        # The text contains "hate language" so "hate" is found
        self.assertTrue(any("hate" in factor for factor in high_explanation['confidence_factors']))
        
        # Test medium confidence explanation
        medium_explanation = generate_confidence_explanation(
            self.medium_confidence_result,
            self.medium_confidence_result['text']
        )
        
        self.assertEqual(medium_explanation['confidence_level'], "moderate")
        self.assertEqual(medium_explanation['primary_category'], "hate")
        self.assertGreaterEqual(len(medium_explanation['confidence_factors']), 0)
        self.assertGreaterEqual(len(medium_explanation['uncertainty_factors']), 0)
        self.assertGreaterEqual(len(medium_explanation['improvement_suggestions']), 0)
        
        # Test borderline confidence explanation
        borderline_explanation = generate_confidence_explanation(
            self.borderline_confidence_result,
            self.borderline_confidence_result['text']
        )
        
        self.assertEqual(borderline_explanation['confidence_level'], "borderline")
        self.assertEqual(borderline_explanation['primary_category'], "insult")
        self.assertGreaterEqual(len(borderline_explanation['uncertainty_factors']), 1)
        self.assertGreaterEqual(len(borderline_explanation['improvement_suggestions']), 1)
        
        # Verify Groq mention
        self.assertIn("Groq", borderline_explanation['explanation'])
        
        # Test low confidence explanation
        low_explanation = generate_confidence_explanation(
            self.low_confidence_result,
            self.low_confidence_result['text']
        )
        
        self.assertEqual(low_explanation['confidence_level'], "low")
        self.assertEqual(low_explanation['primary_category'], "hate")
        self.assertGreaterEqual(len(low_explanation['uncertainty_factors']), 1)
    
    def test_display_confidence_explanation(self):
        """Test displaying confidence explanations."""
        # Generate an explanation
        explanation_obj = generate_confidence_explanation(
            self.medium_confidence_result,
            self.medium_confidence_result['text']
        )
        
        # Test display output
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            _display_confidence_explanation(explanation_obj)
        
        output = captured_output.getvalue()
        
        # Verify key sections are included
        self.assertIn("CONFIDENCE EXPLANATION", output)
        self.assertIn(explanation_obj['explanation'], output)
        
        # Check for factors and suggestions sections
        if explanation_obj['confidence_factors']:
            self.assertIn("Confidence factors:", output)
        if explanation_obj['uncertainty_factors']:
            self.assertIn("Uncertainty factors:", output)
        if explanation_obj['improvement_suggestions']:
            self.assertIn("Suggestions:", output)
    
    def test_toxic_term_detection(self):
        """Test the detection of toxic terms in text."""
        # Test hate category
        hate_text = "This contains racist and prejudiced language."
        hate_terms = _find_toxic_terms(hate_text, "hate")
        self.assertIn("racist", hate_terms)
        self.assertIn("prejudiced", hate_terms)
        
        # Test insult category
        insult_text = "You are an idiot and stupid."
        insult_terms = _find_toxic_terms(insult_text, "insult")
        self.assertIn("idiot", insult_terms)
        self.assertIn("stupid", insult_terms)
        
        # Test explicit pattern matching
        explicit_text = "What the f*ck is wrong with you?!"
        explicit_terms = _find_toxic_terms(explicit_text, "profanity")
        self.assertTrue(any('f*ck' in term for term in explicit_terms))
    
    def test_language_pattern_detection(self):
        """Test detection of different language patterns."""
        # Test directive language
        self.assertTrue(_has_directive_language("You should go away!"))
        self.assertTrue(_has_directive_language("STOP IT NOW!"))
        self.assertFalse(_has_directive_language("This is a normal sentence."))
        
        # Test explicit content
        self.assertTrue(_has_explicit_content("This contains fuck and shit."))
        self.assertTrue(_has_explicit_content("What an a**hole behavior."))
        self.assertFalse(_has_explicit_content("This is perfectly fine content."))
        
        # Test implicit content
        self.assertTrue(_has_implicit_content("Those people and their snowflake mentality."))
        self.assertTrue(_has_implicit_content("That's just how these urban thugs behave."))
        self.assertFalse(_has_implicit_content("This is normal text without coded language."))
    
    def test_explanation_adapts_to_content(self):
        """Test that explanations adapt to different content types."""
        # Test with explicit language
        explicit_result = {
            'text': 'This is fucking ridiculous!',
            'is_toxic': True,
            'category_results': {
                MockCategory('profanity'): {'score': 0.9, 'above_threshold': True},
                MockCategory('insult'): {'score': 0.7, 'above_threshold': True}
            }
        }
        
        explicit_explanation = generate_confidence_explanation(
            explicit_result, 
            explicit_result['text']
        )
        
        # Should mention explicit language as a confidence factor
        self.assertTrue(any("explicit language" in factor.lower() for factor in explicit_explanation['confidence_factors']))
        
        # Test with implicit/coded language
        implicit_result = {
            'text': 'These snowflakes and their urban friends are ruining everything.',
            'is_toxic': True,
            'category_results': {
                MockCategory('hate'): {'score': 0.65, 'above_threshold': True},
                MockCategory('insult'): {'score': 0.45, 'above_threshold': False}
            }
        }
        
        implicit_explanation = generate_confidence_explanation(
            implicit_result, 
            implicit_result['text']
        )
        
        # Should mention implicit/coded language as an uncertainty factor
        self.assertTrue(
            any("implicit" in factor.lower() or "coded" in factor.lower() 
                for factor in implicit_explanation['uncertainty_factors'])
        )
    
    def test_integration_with_json_output(self):
        """Test JSON output contains confidence explanation data."""
        # Generate explanation
        explanation_obj = generate_confidence_explanation(
            self.high_confidence_result,
            self.high_confidence_result['text']
        )
        
        # Verify JSON serialization works
        json_data = json.dumps(explanation_obj)
        parsed_data = json.loads(json_data)
        
        # Check all keys are preserved
        self.assertEqual(parsed_data['confidence_level'], explanation_obj['confidence_level'])
        self.assertEqual(parsed_data['primary_category'], explanation_obj['primary_category'])
        self.assertEqual(parsed_data['confidence_score'], explanation_obj['confidence_score'])
        self.assertEqual(len(parsed_data['confidence_factors']), len(explanation_obj['confidence_factors']))


class MockCategory:
    """Mock category object for testing."""
    def __init__(self, name):
        self.name = name
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"MockCategory('{self.name}')"


if __name__ == '__main__':
    unittest.main() 