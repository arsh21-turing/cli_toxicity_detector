# color_utils.py
import colorsys
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Union

# Define category base colors
CATEGORY_COLORS = {
    "identity_attack": "#FF5733",  # Red-orange
    "insult": "#C70039",           # Deep red
    "obscene": "#900C3F",          # Burgundy
    "severe_toxicity": "#581845",  # Dark purple
    "sexual_explicit": "#E8249D",  # Pink
    "threat": "#FF0000",           # Bright red
    "toxicity": "#FF8D33",         # Orange
    "default": "#3498DB"           # Blue (for unknown categories)
}

# Define threshold levels for color intensity
THRESHOLDS = {
    "safe": 0.2,                  # Below this is considered safe
    "borderline": 0.5,             # Below this is borderline
    "toxic": 0.7,                  # Below this is toxic
    "highly_toxic": 1.0            # High toxicity
}

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color to RGB tuple.
    
    Args:
        hex_color: Hex color code (e.g., "#FF5733")
        
    Returns:
        Tuple of (R, G, B) values from 0-255
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    Convert RGB tuple to hex color.
    
    Args:
        rgb: Tuple of (R, G, B) values from 0-255
        
    Returns:
        Hex color code (e.g., "#FF5733")
    """
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def get_category_base_color(category_name: str) -> str:
    """
    Get the base color for a category.
    
    Args:
        category_name: Name of toxicity category
        
    Returns:
        Hex color code for the category
    """
    return CATEGORY_COLORS.get(category_name, CATEGORY_COLORS["default"])


def get_gradient_color(value: float, min_val: float = 0.0, max_val: float = 1.0, 
                      min_color: str = "#EAFAF1", max_color: str = "#FF0000") -> str:
    """
    Generate a color on a gradient between min_color and max_color.
    
    Args:
        value: Value to determine color position on gradient
        min_val: Minimum value of the range (default: 0.0)
        max_val: Maximum value of the range (default: 1.0)
        min_color: Color for minimum value (default: light green)
        max_color: Color for maximum value (default: red)
        
    Returns:
        Hex color code for the interpolated color
    """
    # Ensure value is within the range
    value = max(min_val, min(max_val, value))
    
    # Normalize value to 0-1
    normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
    
    # Convert hex to RGB
    rgb_min = hex_to_rgb(min_color)
    rgb_max = hex_to_rgb(max_color)
    
    # Interpolate RGB values
    rgb_result = tuple(
        int(rgb_min[i] + normalized * (rgb_max[i] - rgb_min[i]))
        for i in range(3)
    )
    
    # Convert back to hex
    return rgb_to_hex(rgb_result)


def get_confidence_adjustment(base_color: str, confidence: float) -> str:
    """
    Adjust a color based on confidence level by modifying saturation.
    
    Args:
        base_color: Base color as hex string
        confidence: Confidence level from 0.0-1.0
        
    Returns:
        Adjusted hex color
    """
    # Convert hex to RGB
    rgb = hex_to_rgb(base_color)
    
    # Convert RGB to HSV
    r, g, b = [x/255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    # Adjust saturation based on confidence
    # Lower confidence = lower saturation (more washed out)
    adjusted_s = s * (0.3 + 0.7 * confidence)
    
    # Convert back to RGB
    r, g, b = colorsys.hsv_to_rgb(h, adjusted_s, v)
    
    # Convert back to hex
    return "#{:02x}{:02x}{:02x}".format(
        int(r * 255), int(g * 255), int(b * 255)
    )


def get_toxicity_color(score: float, category: str = "toxicity", 
                      confidence: Optional[float] = None) -> str:
    """
    Get appropriate color for a toxicity score.
    
    Args:
        score: Toxicity score from 0.0-1.0
        category: Category name to determine base color
        confidence: Optional confidence level from 0.0-1.0
        
    Returns:
        Hex color code based on toxicity level
    """
    # Get base color for this category
    base_color = get_category_base_color(category)
    
    # For scores below the "safe" threshold, use green
    if score < THRESHOLDS["safe"]:
        color = "#2ECC71"  # Green for safe
    elif score < THRESHOLDS["borderline"]:
        # Gradient from green to yellow for borderline
        color = get_gradient_color(
            score, 
            THRESHOLDS["safe"], 
            THRESHOLDS["borderline"], 
            "#2ECC71",  # Green
            "#F1C40F"   # Yellow
        )
    elif score < THRESHOLDS["toxic"]:
        # Gradient from yellow to orange for toxic
        color = get_gradient_color(
            score, 
            THRESHOLDS["borderline"], 
            THRESHOLDS["toxic"], 
            "#F1C40F",  # Yellow
            base_color  # Category base color
        )
    else:
        # Use category color with intensity based on score
        intensity = (score - THRESHOLDS["toxic"]) / (THRESHOLDS["highly_toxic"] - THRESHOLDS["toxic"])
        
        # Make higher toxicity darker and more intense
        rgb = hex_to_rgb(base_color)
        darker_rgb = tuple(int(max(0, c * (1.0 - intensity * 0.4))) for c in rgb)
        color = rgb_to_hex(darker_rgb)
    
    # Apply confidence adjustment if provided
    if confidence is not None:
        color = get_confidence_adjustment(color, confidence)
    
    return color


def get_background_color_for_score(score: float, category: str = "toxicity") -> str:
    """
    Get a suitable background color for score display.
    
    Args:
        score: Toxicity score from 0.0-1.0
        category: Category name to determine base color
        
    Returns:
        Hex color code with transparency for background use
    """
    # Get base color
    color = get_toxicity_color(score, category)
    
    # Convert to RGB
    rgb = hex_to_rgb(color)
    
    # Make it lighter for background use
    lighter_rgb = tuple(int(min(255, c + 100)) for c in rgb)
    return rgb_to_hex(lighter_rgb)


def get_text_color_for_background(bg_color: str) -> str:
    """
    Get appropriate text color for a given background color.
    
    Args:
        bg_color: Background color as hex string
        
    Returns:
        Hex color code for text (black or white)
    """
    rgb = hex_to_rgb(bg_color)
    
    # Calculate luminance
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
    
    # Use black text on light backgrounds, white on dark backgrounds
    return "#000000" if luminance > 0.5 else "#FFFFFF"


def apply_color_to_dataframe(df: pd.DataFrame, 
                            score_column: str = "Score",
                            category_column: Optional[str] = None) -> pd.DataFrame.style:
    """
    Apply color styling to a DataFrame based on toxicity scores.
    
    Args:
        df: DataFrame to style
        score_column: Column name containing toxicity scores
        category_column: Optional column name containing category names
        
    Returns:
        Styled DataFrame
    """
    def color_scores(row):
        score = row[score_column]
        category = row.get(category_column, "toxicity") if category_column else "toxicity"
        color = get_toxicity_color(score, category)
        text_color = get_text_color_for_background(color)
        style = f"background-color: {color}; color: {text_color}"
        
        # Return a list-like structure for each column
        return [style] * len(row)
    
    return df.style.apply(color_scores, axis=1)


def get_html_badge_for_score(score: float, category: str = "toxicity", 
                           include_label: bool = True) -> str:
    """
    Generate an HTML badge for displaying toxicity scores.
    
    Args:
        score: Toxicity score from 0.0-1.0
        category: Category name
        include_label: Whether to include the category label
        
    Returns:
        HTML string for the badge
    """
    color = get_toxicity_color(score, category)
    text_color = get_text_color_for_background(color)
    
    label = f"{category}: " if include_label else ""
    percentage = f"{score:.1%}"
    
    return f"""
    <span style="
        background-color: {color}; 
        color: {text_color}; 
        padding: 2px 8px; 
        border-radius: 12px; 
        font-size: 0.8em; 
        font-weight: bold;
        display: inline-block;
        margin: 2px;
    ">
        {label}{percentage}
    </span>
    """


def get_text_with_highlight(text: str, scores: Dict[str, float]) -> str:
    """
    Generate HTML text with highlighted toxicity scores.
    
    Args:
        text: Original text
        scores: Dictionary of category scores
        
    Returns:
        HTML string with highlighted scores
    """
    result = text
    
    for category, score in scores.items():
        if score > 0.1:  # Only highlight significant scores
            badge = get_html_badge_for_score(score, category, include_label=True)
            result += f" {badge}"
    
    return result


def supports_color() -> bool:
    """
    Check if the terminal supports color output.
    
    Returns:
        True if color output is supported
    """
    import os
    import sys
    
    # Check if we're in a terminal
    if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
        return False
    
    # Check for color support
    term = os.environ.get('TERM', '')
    return term in ('xterm', 'xterm-256color', 'linux', 'screen', 'screen-256color')


def colorize(text: str, color: str, bold: bool = False) -> str:
    """
    Add ANSI color codes to text for terminal output.
    
    Args:
        text: Text to colorize
        color: Color name ('red', 'green', 'yellow', 'blue', etc.)
        bold: Whether to make text bold
        
    Returns:
        Colorized text string
    """
    if not supports_color():
        return text
    
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m',
        'bold': '\033[1m'
    }
    
    color_code = colors.get(color.lower(), '')
    bold_code = colors['bold'] if bold else ''
    reset_code = colors['reset']
    
    return f"{bold_code}{color_code}{text}{reset_code}"


def colorize_toxic(is_toxic: bool, enabled: bool = True) -> str:
    """
    Colorize toxicity status for terminal output.
    
    Args:
        is_toxic: Whether the content is toxic
        enabled: Whether color output is enabled
        
    Returns:
        Colorized text string
    """
    if not enabled:
        return "Toxic" if is_toxic else "Non-Toxic"
    
    if is_toxic:
        return colorize("Toxic", "red", bold=True)
    else:
        return colorize("Non-Toxic", "green", bold=True)


def colorize_percentage(percentage: float, enabled: bool = True) -> str:
    """
    Colorize percentage values for terminal output.
    
    Args:
        percentage: Percentage value (0-100)
        enabled: Whether color output is enabled
        
    Returns:
        Colorized text string
    """
    if not enabled:
        return f"{percentage:.1f}%"
    
    if percentage < 10:
        return colorize(f"{percentage:.1f}%", "green")
    elif percentage < 30:
        return colorize(f"{percentage:.1f}%", "yellow")
    else:
        return colorize(f"{percentage:.1f}%", "red", bold=True) 