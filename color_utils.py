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
    
    # Create a lighter version with transparency for background
    # Format as rgba for CSS
    alpha = min(0.9, score + 0.1)  # Higher score = more opaque
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha:.2f})"


def get_text_color_for_background(bg_color: str) -> str:
    """
    Determine appropriate text color (black/white) for a background color.
    
    Args:
        bg_color: Background color as hex
        
    Returns:
        "#FFFFFF" for white or "#000000" for black
    """
    # If bg_color is in rgba format, extract RGB values
    if bg_color.startswith("rgba("):
        parts = bg_color.strip(")").split("(")[1].split(",")
        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        # Convert hex to RGB
        r, g, b = hex_to_rgb(bg_color)
    
    # Calculate luminance
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    # Use white text for dark backgrounds, black for light backgrounds
    return "#FFFFFF" if luminance < 0.6 else "#000000"


def apply_color_to_dataframe(df: pd.DataFrame, 
                            score_column: str = "Score",
                            category_column: Optional[str] = None) -> pd.DataFrame.style:
    """
    Apply color styling to a pandas DataFrame based on score values.
    
    Args:
        df: DataFrame with scores
        score_column: Name of column containing scores
        category_column: Optional column with category names
        
    Returns:
        Styled DataFrame with conditional formatting
    """
    def color_scores(row):
        score = row[score_column]
        category = row[category_column] if category_column and category_column in row else "toxicity"
        
        bg_color = get_background_color_for_score(score, category)
        text_color = get_text_color_for_background(bg_color)
        
        return [f'background-color: {bg_color}; color: {text_color}' if col == score_column else '' 
                for col in row.index]
    
    return df.style.apply(color_scores, axis=1)


def get_html_badge_for_score(score: float, category: str = "toxicity", 
                           include_label: bool = True) -> str:
    """
    Generate HTML for a colored badge showing toxicity score.
    
    Args:
        score: Toxicity score from 0.0-1.0
        category: Category name for color coding
        include_label: Whether to include category name in badge
        
    Returns:
        HTML string for colored badge
    """
    bg_color = get_background_color_for_score(score, category)
    text_color = get_text_color_for_background(bg_color)
    label = f"{category}: " if include_label else ""
    
    # Define badge appearance
    badge_style = f"""
        display: inline-block;
        padding: 4px 8px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 0.9em;
        background-color: {bg_color};
        color: {text_color};
        margin: 2px;
    """
    
    return f'<span style="{badge_style}">{label}{score:.2f}</span>'


def get_text_with_highlight(text: str, scores: Dict[str, float]) -> str:
    """
    Generate HTML highlighting text based on toxicity scores.

    Args:
        text: Original text to highlight
        scores: Dictionary of category -> score mappings
        
    Returns:
        HTML with colored highlighting based on toxicity levels
    """
    # Get overall toxicity level
    overall_score = scores.get("toxicity", 0.0)

    # If below threshold, return original text
    if overall_score < THRESHOLDS["borderline"]:
        return text

    # Get highlight color
    bg_color = get_background_color_for_score(overall_score, "toxicity")
    text_color = get_text_color_for_background(bg_color)

    # Create highlighting style
    style = f'background-color: {bg_color}; color: {text_color}; padding: 2px; border-radius: 3px;'

    return f'<span style="{style}">{text}</span>' 