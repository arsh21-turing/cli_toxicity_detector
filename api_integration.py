import requests
import streamlit as st
import time
import json
from typing import Optional, Dict, Any, Tuple, List, Union

class GroqAPI:
    """
    Handles Groq LLM API integration for toxicity analysis.
    """
    
    # Default API URL and model
    API_URL = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "llama3-70b-8192"
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = DEFAULT_MODEL, enabled: bool = False):
        """
        Initialize the Groq API integration.
        
        Args:
            api_key: The Groq API key (optional, can be set later)
            model_name: The model to use (default: llama3-70b-8192)
            enabled: Whether the integration is enabled (default: False)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.enabled = enabled
    
    def set_api_key(self, api_key: str) -> bool:
        """
        Set and optionally validate the API key.
        
        Args:
            api_key: The API key to set
            
        Returns:
            bool: True if the key was set successfully
        """
        self.api_key = api_key.strip() if api_key else None
        return bool(self.api_key)
    
    def toggle_enabled(self, status: bool) -> None:
        """
        Enable or disable the Groq integration.
        
        Args:
            status: True to enable, False to disable
        """
        self.enabled = status
    
    def is_enabled(self) -> bool:
        """
        Check if the Groq integration is enabled.
        
        Returns:
            bool: True if enabled and API key is set
        """
        return self.enabled and bool(self.api_key)
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test the API connection with the current key.
        
        Returns:
            Tuple of (success_bool, message_str)
        """
        if not self.api_key:
            return False, "No API key provided. Please enter a valid Groq API key."
        
        # Simple models endpoint check - doesn't consume tokens
        url = f"{self.API_URL}/models"
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                # Also check if our model is available
                models = response.json().get("data", [])
                model_ids = [model.get("id", "") for model in models]
                
                if self.model_name in model_ids:
                    return True, f"Connection successful. Model '{self.model_name}' is available."
                else:
                    available_models = ", ".join(model_ids[:5])
                    return True, f"Connection successful, but model '{self.model_name}' not found. Available models include: {available_models}..."
            else:
                error_msg = response.json().get("error", {}).get("message", "Unknown error")
                return False, f"Connection failed: {error_msg} (Status code: {response.status_code})"
        
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for toxicity using the Groq LLM API.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing toxicity analysis results
            
        Raises:
            ValueError: If API is disabled or key is not set
            ConnectionError: If API request fails
        """
        if not self.is_enabled():
            raise ValueError("Groq API integration is disabled or no API key is set")
        
        if not text.strip():
            return {"error": "Empty text provided"}
        
        url = f"{self.API_URL}/chat/completions"
        
        prompt = """
        Analyze the following text for toxicity and harmful content. 
        Categorize the text using the following categories:
        - identity_attack: Comments that negatively target identity aspects
        - insult: Insulting or inflammatory content
        - obscene: Obscene or vulgar language
        - severe_toxicity: Extremely toxic language
        - sexual_explicit: Sexual content or references
        - threat: Content that threatens harm
        - toxicity: General toxic language
        
        For each category, provide a probability score between 0.0 and 1.0, where:
        - 0.0 means the text does not contain that type of toxicity
        - 1.0 means the text definitely contains that type of toxicity
        
        Analyze the following text:
        "{text}"
        
        Return your analysis as a valid JSON object with each category and its score.
        """
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a toxicity analysis assistant that evaluates text for harmful content."},
                {"role": "user", "content": prompt.format(text=text)}
            ],
            "temperature": 0.1,  # Low temperature for more consistent results
            "max_tokens": 500
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                response_json = response.json()
                
                # Extract the response content
                content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Extract the JSON part from the response
                try:
                    # Try to find JSON in the response - the LLM might include surrounding text
                    import re
                    json_match = re.search(r'({[\s\S]*})', content)
                    if json_match:
                        json_str = json_match.group(1)
                        analysis = json.loads(json_str)
                    else:
                        analysis = json.loads(content)  # Try parsing directly
                        
                    return analysis
                except json.JSONDecodeError:
                    return {
                        "error": "Failed to parse JSON response from Groq",
                        "raw_response": content
                    }
            else:
                error_msg = response.json().get("error", {}).get("message", "Unknown error")
                raise ConnectionError(f"API request failed: {error_msg} (Status code: {response.status_code})")
                
        except Exception as e:
            raise ConnectionError(f"API request error: {str(e)}")


def create_groq_config_ui(default_enabled: bool = False, default_comparison_mode: bool = False) -> GroqAPI:
    """
    Create a Groq API configuration UI section in the Streamlit sidebar.
    
    Args:
        default_enabled: Whether Groq integration should be enabled by default
        default_comparison_mode: Whether side-by-side comparison mode is enabled by default
        
    Returns:
        Configured GroqAPI instance
    """
    # Create the sidebar section
    st.sidebar.markdown("## Groq Integration")
    st.sidebar.markdown("Configure Groq API for fallback toxicity analysis.")
    
    # Initialize session state for Groq configuration if not already present
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = ""
    if 'groq_enabled' not in st.session_state:
        st.session_state.groq_enabled = False
    if 'groq_connection_status' not in st.session_state:
        st.session_state.groq_connection_status = ("", "")
    if 'groq_key_valid' not in st.session_state:
        st.session_state.groq_key_valid = False
    if 'groq_api' not in st.session_state:
        st.session_state.groq_api = GroqAPI(
            api_key=st.session_state.groq_api_key,
            enabled=st.session_state.groq_enabled
        )
    if 'side_by_side_mode' not in st.session_state:
        st.session_state.side_by_side_mode = default_comparison_mode
    
    # API key input
    api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.groq_api_key,
        help="Enter your Groq API key. It will be securely stored in the session."
    )
    
    # Update API key if changed
    if api_key != st.session_state.groq_api_key:
        st.session_state.groq_api_key = api_key
        st.session_state.groq_api.set_api_key(api_key)
        st.session_state.groq_connection_status = ("", "")  # Reset status when key changes
        st.session_state.groq_key_valid = False
        st.session_state.groq_enabled = False
        st.session_state.side_by_side_mode = False
    
    # Test connection button
    test_col1, test_col2 = st.sidebar.columns([1, 1])
    with test_col1:
        test_button = st.button(
            "Test Connection",
            help="Verify that your API key works with Groq"
        )
    
    # Display connection status
    status_color, status_message = st.session_state.groq_connection_status
    
    # Test connection if button is clicked
    if test_button:
        with st.sidebar.status("Testing Groq API connection..."):
            success, message = st.session_state.groq_api.test_connection()
            time.sleep(0.5)  # Brief delay to make the status visible
            if success:
                status_color = "green"
                st.session_state.groq_key_valid = True
            else:
                status_color = "red"
                st.session_state.groq_key_valid = False
                st.session_state.groq_enabled = False
                st.session_state.side_by_side_mode = False
            st.session_state.groq_connection_status = (status_color, message)
    
    # Display connection status if available
    if status_message:
        if status_color == "green":
            st.sidebar.success(status_message)
        else:
            st.sidebar.error(status_message)
    
    # Enable/disable toggle (only if key is valid)
    enabled = st.sidebar.toggle(
        "Enable Groq Integration",
        value=st.session_state.groq_enabled if st.session_state.groq_key_valid else False,
        disabled=not st.session_state.groq_key_valid,
        help="Enable or disable Groq API integration for fallback analysis (requires valid API key)"
    )
    if enabled != st.session_state.groq_enabled:
        st.session_state.groq_enabled = enabled
        st.session_state.groq_api.toggle_enabled(enabled)
    
    # Side-by-side comparison mode toggle (only active when Groq is enabled and key is valid)
    if st.session_state.groq_enabled and st.session_state.groq_key_valid:
        side_by_side = st.sidebar.toggle(
            "Side-by-Side Analysis Mode",
            value=st.session_state.side_by_side_mode,
            help="Run both primary model and Groq API simultaneously and compare results"
        )
        if side_by_side != st.session_state.side_by_side_mode:
            st.session_state.side_by_side_mode = side_by_side
    else:
        st.session_state.side_by_side_mode = False
        st.sidebar.toggle(
            "Side-by-Side Analysis Mode",
            value=False,
            disabled=True,
            help="Enable Groq integration with a valid API key to use side-by-side mode"
        )
    
    # Model selection
    available_models = ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
    selected_model = st.sidebar.selectbox(
        "Groq Model",
        options=available_models,
        index=0,
        help="Select which Groq LLM model to use for analysis"
    )
    if selected_model != st.session_state.groq_api.model_name:
        st.session_state.groq_api.model_name = selected_model
    
    # Provide info about usage and limits
    with st.sidebar.expander("Groq API Information"):
        st.write("""
        **About Groq Integration:**
        
        Groq provides fast LLM inference for toxicity analysis. 
        This integration uses Groq as a fallback for analyzing content when needed.
        
        **Usage Notes:**
        - Your API key is stored only in this session
        - API calls will consume tokens from your Groq account
        - The default model is Llama-3 70B
        
        **Side-by-Side Analysis Mode:**
        When enabled, both the primary model and Groq will analyze the same text,
        allowing you to compare results and identify potential discrepancies.
        """)
    
    # Return the configured API instance
    return st.session_state.groq_api


def compare_model_results(results_dict):
    """
    Compares results from multiple models to highlight differences.
    
    Args:
        results_dict: Dictionary mapping model names to their result objects
        
    Returns:
        Dictionary with comparison metrics and difference highlights
    """
    if len(results_dict) < 2:
        return {"error": "Need at least two models to compare"}
    
    comparison = {
        "models": list(results_dict.keys()),
        "differences": {},
        "agreement_score": 0.0,
        "largest_disagreement": {"category": None, "difference": 0.0}
    }
    
    # Get common categories across all models
    common_categories = set()
    for model_name, results in results_dict.items():
        categories = {k for k, v in results.items() if isinstance(v, (int, float))}
        if not common_categories:
            common_categories = categories
        else:
            common_categories &= categories
    
    if not common_categories:
        comparison["error"] = "No common categories found across models"
        return comparison
    
    # Calculate differences by category
    differences = {}
    max_diff = 0.0
    max_diff_category = None
    
    for category in common_categories:
        model_scores = [results.get(category, 0.0) for results in results_dict.values()]
        min_score = min(model_scores)
        max_score = max(model_scores)
        diff = max_score - min_score
        
        differences[category] = {
            "min": min_score,
            "max": max_score,
            "difference": diff,
            "scores": {model: results.get(category, 0.0) for model, results in results_dict.items()}
        }
        
        if diff > max_diff:
            max_diff = diff
            max_diff_category = category
    
    # Calculate overall agreement score (1.0 = perfect agreement, 0.0 = maximum disagreement)
    avg_diff = sum(d["difference"] for d in differences.values()) / len(differences) if differences else 0
    agreement_score = 1.0 - min(avg_diff, 1.0)  # Cap at 1.0
    
    comparison["differences"] = differences
    comparison["agreement_score"] = agreement_score
    comparison["largest_disagreement"] = {
        "category": max_diff_category,
        "difference": max_diff
    }
    
    return comparison


def display_side_by_side_comparison(primary_results, groq_results, comparison_metrics, threshold=0.5):
    """
    Displays side-by-side model comparison in Streamlit with color coding.
    
    Args:
        primary_results: Results from the primary model
        groq_results: Results from the Groq API
        comparison_metrics: Dictionary with comparison metrics from compare_model_results()
        threshold: Decision threshold for toxicity classification (default: 0.5)
    """
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Try to import color utilities
    try:
        from color_utils import (
            get_toxicity_color, 
            get_text_color_for_background,
            get_background_color_for_score,
            get_html_badge_for_score,
            THRESHOLDS
        )
        COLOR_UTILS_AVAILABLE = True
    except ImportError:
        COLOR_UTILS_AVAILABLE = False
    
    # Create a combined DataFrame for direct comparison
    common_categories = comparison_metrics["differences"].keys()
    
    data = []
    for category in common_categories:
        primary_score = primary_results.get(category, 0.0)
        groq_score = groq_results.get(category, 0.0)
        difference = abs(primary_score - groq_score)
        
        data.append({
            "Category": category,
            "Primary Model": primary_score,
            "Groq API": groq_score,
            "Absolute Difference": difference
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values("Absolute Difference", ascending=False)
    
    # Display agreement score
    agreement = comparison_metrics["agreement_score"]
    agreement_color = "green" if agreement >= 0.8 else "orange" if agreement >= 0.6 else "red"
    
    st.markdown(f"""
    ### Model Comparison Results
    <div style="margin: 10px 0;">
        <span style="font-weight: bold;">Agreement Score:</span>
        <span style="color: {agreement_color}; font-weight: bold; margin-left: 10px;">{agreement:.2f}</span>
        <span style="margin-left: 10px;">(1.0 = perfect agreement)</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Display largest disagreement
    largest_disagreement = comparison_metrics["largest_disagreement"]
    if largest_disagreement["category"] and largest_disagreement["difference"] > 0.1:
        st.markdown(f"""
        **Largest Disagreement:** {largest_disagreement["category"]} 
        (Difference: {largest_disagreement["difference"]:.2f})
        """)
    
    # Create side-by-side bar chart with color-coding
    st.subheader("Score Comparison")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(df))
    width = 0.35
    
    # Get colors for each bar if color utils available
    if COLOR_UTILS_AVAILABLE:
        primary_colors = [
            get_toxicity_color(score, category) 
            for score, category in zip(df["Primary Model"], df["Category"])
        ]
        
        groq_colors = [
            get_toxicity_color(score, category) 
            for score, category in zip(df["Groq API"], df["Category"])
        ]
    else:
        # Fallback colors
        primary_colors = ['royalblue'] * len(df)
        groq_colors = ['tomato'] * len(df)
    
    # Plot bars with gap between them
    primary_bars = ax.barh(x - width/2, df["Primary Model"], width, 
                         label="Primary Model", color=primary_colors)
    groq_bars = ax.barh(x + width/2, df["Groq API"], width, 
                       label="Groq API", color=groq_colors)
    
    # Add threshold line
    ax.axvline(x=threshold, color='black', linestyle='--', alpha=0.7)
    ax.text(threshold, -0.6, f'Threshold ({threshold})', color='black', alpha=0.7)
    
    # Add threshold lines for severity levels if color utils available
    if COLOR_UTILS_AVAILABLE:
        for threshold_name, threshold_value in THRESHOLDS.items():
            if threshold_value <= 1.0:  # Only show thresholds in range
                ax.axvline(x=threshold_value, color='gray', linestyle='--', alpha=0.4)
                ax.text(threshold_value, len(df), threshold_name, 
                       color='gray', alpha=0.7, fontsize=8, ha='center')
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.01
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', va='center', size=9)
    
    add_labels(primary_bars)
    add_labels(groq_bars)
    
    # Formatting
    ax.set_yticks(x)
    ax.set_yticklabels(df["Category"])
    ax.invert_yaxis()  # Categories listed from top to bottom
    ax.set_xlabel('Score (0-1)')
    ax.set_xlim(0, 1.1)
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display detailed comparison table with color coding if available
    if COLOR_UTILS_AVAILABLE:
        st.subheader("Detailed Comparison")
        
        # Custom styling function for the comparison table
        def color_differences(row):
            primary_score = row["Primary Model"]
            groq_score = row["Groq API"]
            diff = row["Absolute Difference"]
            category = row["Category"]
            
            styles = [''] * len(row)
            
            # Index for each column - depends on DataFrame column order
            primary_idx = row.index.get_loc("Primary Model")
            groq_idx = row.index.get_loc("Groq API")
            diff_idx = row.index.get_loc("Absolute Difference")
            
            # Color Primary Model score
            primary_bg = get_background_color_for_score(primary_score, category)
            primary_text = get_text_color_for_background(primary_bg)
            styles[primary_idx] = f'background-color: {primary_bg}; color: {primary_text}'
            
            # Color Groq API score
            groq_bg = get_background_color_for_score(groq_score, category)
            groq_text = get_text_color_for_background(groq_bg)
            styles[groq_idx] = f'background-color: {groq_bg}; color: {groq_text}'
            
            # Color difference cell based on magnitude
            if diff > 0.25:
                # Large difference - red
                styles[diff_idx] = 'background-color: rgba(255, 0, 0, 0.7); color: white; font-weight: bold'
            elif diff > 0.1:
                # Medium difference - orange
                styles[diff_idx] = 'background-color: rgba(255, 165, 0, 0.7); color: black; font-weight: bold'
            elif diff > 0.05:
                # Small difference - yellow
                styles[diff_idx] = 'background-color: rgba(255, 255, 0, 0.7); color: black'
                
            # Highlight threshold crossings (where one model is above threshold and other is below)
            primary_above = primary_score >= threshold
            groq_above = groq_score >= threshold
            if primary_above != groq_above:
                # Add a border to indicate threshold crossing disagreement
                styles[primary_idx] += '; border: 2px solid red'
                styles[groq_idx] += '; border: 2px solid red'
            
            return styles
        
        # Format and display the styled dataframe
        styled_df = df.style.apply(color_differences, axis=1).format({
            "Primary Model": "{:.3f}",
            "Groq API": "{:.3f}",
            "Absolute Difference": "{:.3f}"
        })
        
        st.dataframe(styled_df, height=400)
        
        # Display note about threshold crossings and legend
        st.markdown("""
        **Note:** 
        - Red borders indicate cases where models disagree on whether the content crosses the toxicity threshold.
        - Differences are highlighted based on magnitude: red (>0.25), orange (>0.1), yellow (>0.05).
        """)
        
        # Add color legend
        st.markdown("#### Color Legend")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                f'<div style="color: #2ECC71;">Safe (< {THRESHOLDS["safe"]})</div>',
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f'<div style="color: #F1C40F;">Borderline ({THRESHOLDS["safe"]} - {THRESHOLDS["borderline"]})</div>',
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f'<div style="color: #FF8D33;">Toxic ({THRESHOLDS["borderline"]} - {THRESHOLDS["toxic"]})</div>',
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                f'<div style="color: #C70039;">Highly Toxic (> {THRESHOLDS["toxic"]})</div>',
                unsafe_allow_html=True
            )
    else:
        # Fallback to simple table with threshold crossing highlighting
        st.subheader("Detailed Comparison")
        
        # Format the dataframe for display
        display_df = df.copy()
        
        # Highlight threshold crossings (cases where one model is above threshold and other is below)
        def highlight_threshold_crossings(row):
            primary_above = row["Primary Model"] >= threshold
            groq_above = row["Groq API"] >= threshold
            
            if primary_above != groq_above:
                return ['background-color: #ffdddd'] * len(row)
            else:
                return [''] * len(row)
        
        # Apply styling to the dataframe
        styled_df = display_df.style.apply(highlight_threshold_crossings, axis=1)
        styled_df = styled_df.format({
            "Primary Model": "{:.3f}",
            "Groq API": "{:.3f}",
            "Absolute Difference": "{:.3f}"
        })
        
        st.dataframe(styled_df, height=400)
        
        # Display note about threshold crossings
        st.markdown("""
        **Note:** Rows highlighted in red indicate cases where the models disagree on whether the content
        crosses the toxicity threshold. These represent potential false positives or false negatives.
        """) 