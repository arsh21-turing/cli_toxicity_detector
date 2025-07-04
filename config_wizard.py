#!/usr/bin/env python3
"""
Configuration wizard for setting up optimal toxicity classification settings.

This module provides an interactive wizard to guide users through creating
optimized configuration files based on their specific use case and data
characteristics.
"""

import os
import sys
import yaml
import json
import inquirer
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union


# Define use case profiles with optimized defaults
USE_CASE_PROFILES = {
    "high_precision": {
        "description": "Prioritize precision over recall (minimize false positives)",
        "defaults": {
            "thresholds": {
                "hate": 0.8,
                "harassment": 0.75,
                "self_harm": 0.85,
                "sexual": 0.8,
                "violence": 0.75,
                "profanity": 0.7
            },
            "model": {
                "name": "toxicity-classifier",
                "version": "1.0.0",
                "fallback_mode": "high_confidence"
            },
            "processing": {
                "batch_size": 32,
                "confidence_threshold": 0.8,
                "low_confidence_fallback": True
            }
        }
    },
    "high_recall": {
        "description": "Prioritize recall over precision (minimize false negatives)",
        "defaults": {
            "thresholds": {
                "hate": 0.3,
                "harassment": 0.35,
                "self_harm": 0.25,
                "sexual": 0.3,
                "violence": 0.35,
                "profanity": 0.4
            },
            "model": {
                "name": "toxicity-classifier",
                "version": "1.0.0",
                "fallback_mode": "aggressive"
            },
            "processing": {
                "batch_size": 32,
                "confidence_threshold": 0.4,
                "low_confidence_fallback": True
            }
        }
    },
    "balanced": {
        "description": "Balance precision and recall (optimize for F1 score)",
        "defaults": {
            "thresholds": {
                "hate": 0.5,
                "harassment": 0.55,
                "self_harm": 0.6,
                "sexual": 0.55,
                "violence": 0.5,
                "profanity": 0.45
            },
            "model": {
                "name": "toxicity-classifier",
                "version": "1.0.0",
                "fallback_mode": "balanced"
            },
            "processing": {
                "batch_size": 32,
                "confidence_threshold": 0.6,
                "low_confidence_fallback": True
            }
        }
    },
    "performance": {
        "description": "Optimize for processing speed and resource efficiency",
        "defaults": {
            "thresholds": {
                "hate": 0.5,
                "harassment": 0.5,
                "self_harm": 0.5,
                "sexual": 0.5,
                "violence": 0.5,
                "profanity": 0.5
            },
            "model": {
                "name": "toxicity-classifier-lite",
                "version": "1.0.0",
                "fallback_mode": "disabled"
            },
            "processing": {
                "batch_size": 64,
                "confidence_threshold": 0.5,
                "low_confidence_fallback": False
            }
        }
    },
    "custom": {
        "description": "Start with default settings and customize",
        "defaults": {
            "thresholds": {
                "hate": 0.5,
                "harassment": 0.5,
                "self_harm": 0.5,
                "sexual": 0.5,
                "violence": 0.5,
                "profanity": 0.5
            },
            "model": {
                "name": "toxicity-classifier",
                "version": "1.0.0",
                "fallback_mode": "disabled"
            },
            "processing": {
                "batch_size": 32,
                "confidence_threshold": 0.5,
                "low_confidence_fallback": False
            }
        }
    }
}


def get_use_case_defaults(use_case: str) -> Dict[str, Any]:
    """
    Get recommended default configuration for a specific use case.
    
    Args:
        use_case: Name of the use case profile
        
    Returns:
        Dictionary with recommended configuration settings
        
    Raises:
        ValueError: If the use case is not recognized
    """
    if use_case not in USE_CASE_PROFILES:
        raise ValueError(f"Unknown use case: {use_case}. Valid options are: {', '.join(USE_CASE_PROFILES.keys())}")
    
    return USE_CASE_PROFILES[use_case]["defaults"].copy()


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration settings for consistency and potential issues.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Tuple containing:
            - Boolean indicating if the configuration is valid
            - List of warning messages for potential issues
    """
    warnings = []
    is_valid = True
    
    # Check for required sections
    required_sections = ["model", "thresholds"]
    for section in required_sections:
        if section not in config:
            warnings.append(f"Missing required section: {section}")
            is_valid = False
    
    # If validation can't continue without required sections, return early
    if not is_valid:
        return False, warnings
    
    # Check thresholds
    thresholds = config.get("thresholds", {})
    if not thresholds:
        warnings.append("No category thresholds defined")
        is_valid = False
    else:
        # Check threshold values
        for category, threshold in thresholds.items():
            if not isinstance(threshold, (int, float)):
                warnings.append(f"Invalid threshold value for {category}: {threshold}. Must be a number.")
                is_valid = False
            elif threshold < 0 or threshold > 1:
                warnings.append(f"Threshold for {category} is outside valid range [0, 1]: {threshold}")
                is_valid = False
    
    # Check model settings
    model = config.get("model", {})
    if "name" not in model:
        warnings.append("Model name is not specified")
        is_valid = False
    
    if model.get("fallback_mode") == "aggressive" and not config.get("processing", {}).get("low_confidence_fallback", False):
        warnings.append("Aggressive fallback mode is set but low_confidence_fallback is disabled")
    
    # Check processing settings
    processing = config.get("processing", {})
    batch_size = processing.get("batch_size")
    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            warnings.append(f"Invalid batch size: {batch_size}. Must be a positive integer.")
            is_valid = False
    
    # Performance warnings (not validation failures)
    if processing.get("low_confidence_fallback", False) and "fallback_mode" in model and model["fallback_mode"] != "disabled":
        if "confidence_threshold" in processing and processing["confidence_threshold"] < 0.4:
            warnings.append(f"Low confidence threshold ({processing['confidence_threshold']}) may result in excessive API calls")
    
    return is_valid, warnings


def save_config(config: Dict[str, Any], output_path: str) -> str:
    """
    Save configuration to a file in YAML or JSON format.
    
    Args:
        config: Configuration dictionary to save
        output_path: Path where to save the configuration file
        
    Returns:
        Path to the saved configuration file
        
    Raises:
        ValueError: If the file format is not supported
    """
    output_path = Path(output_path)
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine format based on extension
    if output_path.suffix.lower() in ('.yaml', '.yml'):
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    elif output_path.suffix.lower() == '.json':
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        # Default to YAML if extension is not recognized
        if not output_path.suffix:
            output_path = output_path.with_suffix('.yaml')
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}. Use .yaml, .yml, or .json.")
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return str(output_path)


def get_fallback_description(fallback_mode: str) -> str:
    """Get description for fallback mode."""
    descriptions = {
        "balanced": "Use fallback for predictions with confidence 0.4-0.6",
        "high_confidence": "Only use fallback for predictions <0.4 or >0.8",
        "aggressive": "Use fallback for all predictions <0.7",
        "disabled": "Never use fallback (fastest, but less accurate)"
    }
    return descriptions.get(fallback_mode, "Unknown fallback mode")


def setup_wizard(
    wizard_mode: Optional[str] = None,
    output_path: Optional[str] = None,
    non_interactive: bool = False
) -> str:
    """
    Run the interactive configuration wizard.
    
    Args:
        wizard_mode: Optional use case profile to use as defaults
        output_path: Path where to save the generated configuration file
        non_interactive: Whether to run in non-interactive mode with defaults
        
    Returns:
        Path to the created configuration file
    """
    # Set default output path if not provided
    if not output_path:
        output_path = "config.yaml"
    
    print("=" * 80)
    print("TOXICITY CLASSIFICATION CONFIGURATION WIZARD")
    print("=" * 80)
    print("This wizard will help you create an optimized configuration")
    print("file for your specific toxicity classification needs.")
    print()
    
    # Initialize config with defaults
    if wizard_mode and wizard_mode in USE_CASE_PROFILES:
        config = get_use_case_defaults(wizard_mode)
        print(f"Using '{wizard_mode}' profile as starting point.")
        print(f"Description: {USE_CASE_PROFILES[wizard_mode]['description']}")
    else:
        # Let user select a use case profile
        if not non_interactive:
            questions = [
                inquirer.List('profile',
                    message="Select a use case profile as starting point:",
                    choices=[f"{k} - {v['description']}" for k, v in USE_CASE_PROFILES.items()],
                )
            ]
            answers = inquirer.prompt(questions)
            selected_profile = answers['profile'].split(' - ')[0]
            config = get_use_case_defaults(selected_profile)
            print(f"Using '{selected_profile}' profile as starting point.")
        else:
            # Default to balanced profile in non-interactive mode
            config = get_use_case_defaults("balanced")
            print("Using 'balanced' profile as starting point (non-interactive mode).")
    
    print()
    
    # Interactive configuration customization
    if not non_interactive:
        # Model settings
        print("MODEL SETTINGS")
        print("-" * 80)
        
        model_questions = [
            inquirer.List('model_name',
                message="Select model type:",
                choices=[
                    "toxicity-classifier - Standard model with full category support",
                    "toxicity-classifier-lite - Faster but less accurate model"
                ],
                default="toxicity-classifier - Standard model with full category support" 
                    if config["model"]["name"] == "toxicity-classifier" else
                    "toxicity-classifier-lite - Faster but less accurate model"
            ),
            inquirer.List('fallback_mode',
                message="Select fallback behavior for low-confidence predictions:",
                choices=[
                    "balanced - Use fallback for predictions with confidence 0.4-0.6",
                    "high_confidence - Only use fallback for predictions <0.4 or >0.8",
                    "aggressive - Use fallback for all predictions <0.7",
                    "disabled - Never use fallback (fastest, but less accurate)"
                ],
                default=f"{config['model']['fallback_mode']} - {get_fallback_description(config['model']['fallback_mode'])}"
            )
        ]
        
        model_answers = inquirer.prompt(model_questions)
        config["model"]["name"] = model_answers["model_name"].split(" - ")[0]
        config["model"]["fallback_mode"] = model_answers["fallback_mode"].split(" - ")[0]
        
        # Processing settings
        print("\nPROCESSING SETTINGS")
        print("-" * 80)
        
        processing_questions = [
            inquirer.List('batch_size',
                message="Select batch size for processing:",
                choices=["16", "32", "64", "128"],
                default=str(config["processing"]["batch_size"])
            ),
            inquirer.Confirm('low_confidence_fallback',
                message="Enable fallback for low-confidence predictions?",
                default=config["processing"]["low_confidence_fallback"]
            )
        ]
        
        if config["model"]["fallback_mode"] != "disabled":
            processing_questions.append(
                inquirer.List('confidence_threshold',
                    message="Select confidence threshold for fallback:",
                    choices=["0.4", "0.5", "0.6", "0.7", "0.8"],
                    default=str(config["processing"]["confidence_threshold"])
                )
            )
        
        processing_answers = inquirer.prompt(processing_questions)
        config["processing"]["batch_size"] = int(processing_answers["batch_size"])
        config["processing"]["low_confidence_fallback"] = processing_answers["low_confidence_fallback"]
        
        if "confidence_threshold" in processing_answers:
            config["processing"]["confidence_threshold"] = float(processing_answers["confidence_threshold"])
        
        # Threshold settings
        print("\nTHRESHOLD SETTINGS")
        print("-" * 80)
        print("Set classification thresholds for each category.")
        print("Lower values increase sensitivity (more detections, more false positives).")
        print("Higher values increase specificity (fewer false positives, might miss some cases).")
        
        threshold_mode_question = [
            inquirer.List('threshold_mode',
                message="How would you like to set thresholds?",
                choices=[
                    "keep_defaults - Keep the default thresholds from selected profile",
                    "set_all - Set one value for all categories",
                    "customize - Set individual thresholds for each category"
                ]
            )
        ]
        
        threshold_mode = inquirer.prompt(threshold_mode_question)["threshold_mode"].split(" - ")[0]
        
        if threshold_mode == "set_all":
            all_threshold_question = [
                inquirer.List('all_threshold',
                    message="Select threshold value for all categories:",
                    choices=["0.3", "0.4", "0.5", "0.6", "0.7", "0.8"],
                    default="0.5"
                )
            ]
            
            all_threshold = float(inquirer.prompt(all_threshold_question)["all_threshold"])
            
            for category in config["thresholds"]:
                config["thresholds"][category] = all_threshold
        
        elif threshold_mode == "customize":
            for category in sorted(config["thresholds"].keys()):
                category_question = [
                    inquirer.List(f'threshold_{category}',
                        message=f"Threshold for '{category}':",
                        choices=["0.3", "0.4", "0.5", "0.6", "0.7", "0.8"],
                        default=str(config["thresholds"][category])
                    )
                ]
                
                category_threshold = float(inquirer.prompt(category_question)[f'threshold_{category}'])
                config["thresholds"][category] = category_threshold
    
    # Validate the config
    is_valid, warnings = validate_config(config)
    
    if not is_valid:
        print("\nERROR: The configuration has the following issues:")
        for warning in warnings:
            print(f"  - {warning}")
        
        if not non_interactive:
            fix_question = [
                inquirer.Confirm('fix',
                    message="Would you like to save anyway?",
                    default=False
                )
            ]
            
            fix = inquirer.prompt(fix_question)["fix"]
            if not fix:
                return setup_wizard(wizard_mode, output_path, non_interactive)
    
    elif warnings:
        print("\nWARNING: The configuration has potential issues:")
        for warning in warnings:
            print(f"  - {warning}")
        
        if not non_interactive:
            proceed_question = [
                inquirer.Confirm('proceed',
                    message="Do you want to proceed with this configuration?",
                    default=True
                )
            ]
            
            proceed = inquirer.prompt(proceed_question)["proceed"]
            if not proceed:
                return setup_wizard(wizard_mode, output_path, non_interactive)
    
    # Save the config
    try:
        config_path = save_config(config, output_path)
        print(f"\nConfiguration saved to: {config_path}")
        
        # Print config contents
        print("\nConfiguration Preview:")
        print("-" * 80)
        if Path(config_path).suffix.lower() in ('.yaml', '.yml'):
            print(yaml.dump(config, default_flow_style=False, sort_keys=False))
        else:
            print(json.dumps(config, indent=2))
        
        return config_path
    
    except Exception as e:
        print(f"Error saving configuration: {str(e)}")
        
        if not non_interactive:
            retry_question = [
                inquirer.Confirm('retry',
                    message="Would you like to retry with a different output path?",
                    default=True
                )
            ]
            
            retry = inquirer.prompt(retry_question)["retry"]
            if retry:
                new_path_question = [
                    inquirer.Text('new_path',
                        message="Enter new output path:",
                        default="config.yaml"
                    )
                ]
                new_path = inquirer.prompt(new_path_question)["new_path"]
                return setup_wizard(wizard_mode, new_path, non_interactive)
        
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Wizard for Toxicity Classification")
    parser.add_argument("--output", help="Path to save the generated config file")
    parser.add_argument("--use-case", choices=list(USE_CASE_PROFILES.keys()),
                      help="Use case profile to use as defaults")
    parser.add_argument("--non-interactive", action="store_true",
                      help="Run wizard in non-interactive mode with defaults")
    
    args = parser.parse_args()
    
    try:
        setup_wizard(args.use_case, args.output, args.non_interactive)
    except KeyboardInterrupt:
        print("\nWizard cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running wizard: {str(e)}")
        sys.exit(1) 