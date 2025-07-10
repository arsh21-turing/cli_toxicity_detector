#!/usr/bin/env python3
"""
Tests for the configuration wizard module.
"""

import unittest
import os
import sys
import tempfile
import yaml
import json
from pathlib import Path
from unittest import mock
import numpy as np

# Add parent directory to Python path so we can import config_wizard
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_wizard import (
    setup_wizard,
    get_use_case_defaults,
    validate_config,
    save_config,
    USE_CASE_PROFILES
)


class TestConfigWizard(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_get_use_case_defaults(self):
        """Test getting defaults for different use cases."""
        # Test valid use cases
        for use_case in USE_CASE_PROFILES:
            defaults = get_use_case_defaults(use_case)
            self.assertIsInstance(defaults, dict)
            self.assertIn("thresholds", defaults)
            self.assertIn("model", defaults)
            self.assertIn("processing", defaults)
        
        # Test invalid use case
        with self.assertRaises(ValueError):
            get_use_case_defaults("nonexistent_profile")
    
    def test_validate_config(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = {
            "thresholds": {
                "hate": 0.5,
                "harassment": 0.6
            },
            "model": {
                "name": "toxicity-classifier", 
                "fallback_mode": "disabled"
            },
            "processing": {
                "batch_size": 32,
                "low_confidence_fallback": False
            }
        }
        is_valid, warnings = validate_config(valid_config)
        self.assertTrue(is_valid)
        self.assertEqual(len(warnings), 0)
        
        # Invalid configuration with missing sections
        invalid_config = {
            "model": {
                "fallback_mode": "disabled"
            }
        }
        is_valid, warnings = validate_config(invalid_config)
        self.assertFalse(is_valid)
        self.assertGreater(len(warnings), 0)
        
        # Configuration with warnings but still valid
        warning_config = {
            "thresholds": {
                "hate": 0.5,
                "harassment": 0.6
            },
            "model": {
                "name": "toxicity-classifier", 
                "fallback_mode": "aggressive"
            },
            "processing": {
                "batch_size": 32,
                "low_confidence_fallback": False
            }
        }
        is_valid, warnings = validate_config(warning_config)
        self.assertTrue(is_valid)
        self.assertGreater(len(warnings), 0)
    
    def test_save_config(self):
        """Test saving configuration to file."""
        config = {
            "thresholds": {
                "hate": 0.5,
                "harassment": 0.6
            },
            "model": {
                "name": "toxicity-classifier", 
                "fallback_mode": "disabled"
            }
        }
        
        # Test saving as YAML
        yaml_path = Path(self.temp_dir.name) / "test_config.yaml"
        saved_yaml_path = save_config(config, str(yaml_path))
        
        self.assertTrue(Path(saved_yaml_path).exists())
        
        # Verify YAML content
        with open(saved_yaml_path, 'r') as f:
            loaded_yaml = yaml.safe_load(f)
        self.assertEqual(loaded_yaml, config)
        
        # Test saving as JSON
        json_path = Path(self.temp_dir.name) / "test_config.json"
        saved_json_path = save_config(config, str(json_path))
        
        self.assertTrue(Path(saved_json_path).exists())
        
        # Verify JSON content
        with open(saved_json_path, 'r') as f:
            loaded_json = json.load(f)
        self.assertEqual(loaded_json, config)
    
    @mock.patch('inquirer.prompt')
    def test_setup_wizard_non_interactive(self, mock_prompt):
        """Test setup wizard in non-interactive mode."""
        # Non-interactive mode should not call inquirer.prompt
        output_path = Path(self.temp_dir.name) / "non_interactive_config.yaml"
        result = setup_wizard(
            wizard_mode="balanced", 
            output_path=str(output_path),
            non_interactive=True
        )
        
        self.assertEqual(result, str(output_path))
        self.assertTrue(Path(result).exists())
        self.assertEqual(mock_prompt.call_count, 0)
        
        # Verify saved config has expected structure from "balanced" profile
        with open(result, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        self.assertEqual(loaded_config["model"]["fallback_mode"], "balanced")
        self.assertIn("thresholds", loaded_config)
        
    @mock.patch('inquirer.prompt')
    def test_setup_wizard_interactive(self, mock_prompt):
        """Test setup wizard in interactive mode with mocked user input."""
        # Mock user selections
        mock_prompt.side_effect = [
            {"profile": "high_precision - Prioritize precision over recall (minimize false positives)"},
            {"model_name": "toxicity-classifier - Standard model with full category support", 
             "fallback_mode": "balanced - Use fallback for predictions with confidence 0.4-0.6"},
            {"batch_size": "64", "low_confidence_fallback": True, "confidence_threshold": "0.6"},
            {"threshold_mode": "set_all - Set one value for all categories"},
            {"all_threshold": "0.7"},
            # No validation issues or warnings
        ]
        
        output_path = Path(self.temp_dir.name) / "interactive_config.yaml"
        result = setup_wizard(output_path=str(output_path))
        
        self.assertEqual(result, str(output_path))
        self.assertTrue(Path(result).exists())
        
        # Verify config has expected structure from user selections
        with open(result, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        self.assertEqual(loaded_config["model"]["fallback_mode"], "balanced")
        self.assertEqual(loaded_config["processing"]["batch_size"], 64)
        self.assertEqual(loaded_config["processing"]["low_confidence_fallback"], True)
        
        # Check that all thresholds were set to 0.7
        for category, threshold in loaded_config["thresholds"].items():
            self.assertEqual(threshold, 0.7)


if __name__ == "__main__":
    unittest.main() 