#!/usr/bin/env python3
"""
Streamlit Web UI for Toxicity Detector

A web interface that allows users to enter text and get real-time
toxicity analysis using the existing model with profile management
and file upload capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Dict, List, Optional, Union, Tuple
import json
import copy
import uuid
import time
import io
import csv
import os
import re
import logging
import threading
import queue
from datetime import datetime

# Import our project modules
from model_loader import analyze_text, get_model, DEFAULT_CATEGORIES
import re

# Try to import categories module
try:
    from categories import get_category_description, get_all_categories, get_toxic_categories
    from categories import ToxicityCategory
    CATEGORY_DESCRIPTIONS_AVAILABLE = True
except ImportError:
    CATEGORY_DESCRIPTIONS_AVAILABLE = False
    
# Try to import Groq functionality
try:
    from groq_cache import GroqCache
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Try to import file processor module
try:
    from file_processor import analyze_file, display_file_results
    FILE_PROCESSOR_AVAILABLE = True
except ImportError:
    FILE_PROCESSOR_AVAILABLE = False

# Try to import batch processor module
try:
    from batch_processor import batch_process
    BATCH_PROCESSOR_AVAILABLE = True
except ImportError:
    BATCH_PROCESSOR_AVAILABLE = False

# Try to import logger module
try:
    from logger import logger, setup_logger
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False
    logger = logging.getLogger("toxicity_detector")
    logger.setLevel(logging.INFO)

# Import configuration if available
try:
    from config_loader import load_config, save_config
    CONFIG = load_config()
    CONFIG_LOADER_AVAILABLE = True
except ImportError:
    CONFIG = {
        "model": {"threshold": 0.6},
        "categories": {cat: 0.6 for cat in DEFAULT_CATEGORIES},
        "groq": {"enabled": False, "api_key": ""},
        "batch": {"size": 150}
    }
    CONFIG_LOADER_AVAILABLE = False

# Set page title and config
st.set_page_config(
    page_title="Toxicity Detector",
    page_icon="🛡️",
    layout="wide"
)

# Function to load external CSS
def load_css(css_file):
    """
    Load external CSS file
    
    Args:
        css_file: Path to the CSS file
    """
    try:
        with open(css_file, 'r') as f:
            css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found: {css_file}")
        # Fallback to a minimal inline style if the file is not found
        st.markdown("""
        <style>
            .main-header { text-align: center; margin-bottom: 30px; }
        </style>
        """, unsafe_allow_html=True)

# Load external CSS
load_css('styles.css')

# Add CSS for the logging panel
st.markdown("""
<style>
.log-panel {
    height: 300px;
    overflow-y: auto;
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    font-family: monospace;
    padding: 10px;
    color: #212529;
    margin-top: 10px;
    animation: fadeIn 0.5s ease-out;
    transition: all 0.3s ease;
}
.log-panel:hover {
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}
.log-entry {
    padding: 3px 5px;
    margin-bottom: 3px;
    border-bottom: 1px solid #f1f1f1;
    transition: all 0.2s ease;
}
.log-entry:hover {
    background-color: rgba(0,0,0,0.05);
    transform: translateX(2px);
}
.log-timestamp {
    color: #6c757d;
    margin-right: 5px;
    font-weight: bold;
}
.log-level {
    font-weight: bold;
    padding: 1px 5px;
    border-radius: 3px;
    margin-right: 5px;
}
.log-level-DEBUG {
    background-color: #e9ecef;
    color: #6c757d;
}
.log-level-INFO {
    background-color: #cfe2ff;
    color: #0a58ca;
}
.log-level-WARNING {
    background-color: #fff3cd;
    color: #856404;
}
.log-level-ERROR {
    background-color: #f8d7da;
    color: #721c24;
}
.log-level-CRITICAL {
    background-color: #721c24;
    color: white;
}
.log-message {
    font-family: monospace;
    white-space: pre-wrap;
}
.log-controls {
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px;
    background-color: #f1f3f5;
    border-radius: 4px;
}
.auto-scroll-toggle {
    display: flex;
    align-items: center;
    gap: 5px;
}
.auto-scroll-indicator {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background-color: #28a745;
    display: inline-block;
    margin-right: 5px;
}
.auto-scroll-disable .auto-scroll-indicator {
    background-color: #dc3545;
}
.log-actions {
    display: flex;
    gap: 10px;
}
.log-filter {
    display: flex;
    align-items: center;
    gap: 10px;
}
.log-panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.log-panel-title {
    font-weight: bold;
    font-size: 16px;
    color: #343a40;
}
.log-search {
    flex: 1;
    margin: 0 10px;
}
</style>
""", unsafe_allow_html=True)

# Custom log handler for Streamlit UI
class StreamlitLogHandler(logging.Handler):
    """Custom logging handler that stores logs for display in Streamlit."""
    
    def __init__(self, max_entries=100):
        """Initialize the handler with a maximum number of entries to store."""
        super().__init__()
        self.log_queue = queue.Queue(maxsize=max_entries)
        self.max_entries = max_entries
        
        # Default formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.setFormatter(formatter)
    
    def emit(self, record):
        """Process a log record by formatting it and adding to the queue."""
        try:
            # Format the log message
            msg = self.format(record)
            
            # Add log level color formatting for UI display
            level_color = self._get_level_color(record.levelname)
            formatted_record = {
                'message': msg,
                'level': record.levelname,
                'levelno': record.levelno,
                'color': level_color,
                'timestamp': datetime.fromtimestamp(record.created).strftime('%H:%M:%S'),
                'time_ms': record.created * 1000  # For sorting, milliseconds
            }
            
            # Add to queue, removing oldest if necessary
            if self.log_queue.full():
                try:
                    self.log_queue.get_nowait()  # Remove oldest
                except queue.Empty:
                    pass  # Queue was emptied by another thread
            
            self.log_queue.put_nowait(formatted_record)
        except Exception:
            self.handleError(record)
    
    def get_logs(self, level=logging.NOTSET):
        """Get all logs at or above the specified level."""
        logs = list(self.log_queue.queue)
        if level > logging.NOTSET:
            logs = [log for log in logs if log['levelno'] >= level]
        return sorted(logs, key=lambda x: x['time_ms'])
    
    def clear(self):
        """Clear all logs."""
        with self.log_queue.mutex:
            self.log_queue.queue.clear()
    
    def _get_level_color(self, levelname):
        """Get appropriate color for log level."""
        colors = {
            'DEBUG': '#6c757d',     # Gray
            'INFO': '#0d6efd',      # Blue
            'WARNING': '#ffc107',   # Yellow
            'ERROR': '#dc3545',     # Red
            'CRITICAL': '#721c24'   # Dark red
        }
        return colors.get(levelname, '#6c757d')  # Default gray

# Initialize the log handler for streamlit UI
if 'log_handler' not in st.session_state:
    log_handler = StreamlitLogHandler(max_entries=100)
    logger.addHandler(log_handler)
    st.session_state.log_handler = log_handler
else:
    log_handler = st.session_state.log_handler

# Helper functions for enhanced visual feedback
def create_animated_progress_bar(percent, label=""):
    """Create an animated progress bar with CSS"""
    bar_html = f"""
    <div>
        <div style="font-size: 14px; margin-bottom: 5px;">{label}</div>
        <div class="animated-progress-bar">
            <div style="height: 100%; width: {percent}%; 
                 background-color: {get_progress_color(percent)}; 
                 transition: width 1s ease-out;"></div>
        </div>
        <div style="text-align: center; font-size: 12px; margin-top: 5px;">{percent:.1f}%</div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

def get_progress_color(percent):
    """Get appropriate color for progress percentage"""
    if percent < 30:
        return "#4CAF50"  # Green
    elif percent < 70:
        return "#FF9800"  # Orange
    else:
        return "#F44336"  # Red

def category_pill(category, count=None):
    """Create a hoverable category pill with animation"""
    display_name = get_category_display_name(category)
    count_display = f" ({count})" if count is not None else ""
    pill_html = f'<span class="category-pill {category}">{display_name}{count_display}</span>'
    return pill_html

def category_pills_row(categories, counts=None):
    """Create a row of category pills"""
    pills_html = []
    for i, cat in enumerate(categories):
        count = counts[i] if counts is not None and i < len(counts) else None
        pills_html.append(category_pill(cat, count))
    
    all_pills = "".join(pills_html)
    return f'<div style="margin: 10px 0;">{all_pills}</div>'

def status_message(message, icon="ℹ️"):
    """Display an animated status message"""
    st.markdown(f"""
    <div class="status-message">
        {icon} {message}
    </div>
    """, unsafe_allow_html=True)

def animated_result(is_toxic, category=None, confidence=None):
    """Create an animated result box"""
    result_class = "toxic" if is_toxic else "non-toxic"
    icon = "⚠️" if is_toxic else "✅"
    header = "Toxic Content Detected" if is_toxic else "Non-Toxic Content"
    
    category_html = ""
    if is_toxic and category:
        display_name = get_category_display_name(category)
        category_html = f'<div class="toxic-category">Category: {display_name}</div>'
    
    confidence_html = ""
    if confidence is not None:
        conf_value = confidence if is_toxic else 1 - confidence
        confidence_html = f'<div style="font-size: 14px; margin-top: 5px;">Confidence: {conf_value:.2f}</div>'
    
    st.markdown(f"""
    <div class="toxic-result {result_class}">
        <div class="toxic-header">{icon} {header}</div>
        {category_html}
        {confidence_html}
    </div>
    """, unsafe_allow_html=True)

def add_log_panel_js():
    """Add JavaScript for auto-scrolling the log panel"""
    js = """
    <script>
    // Function to handle auto-scrolling of the log panel
    function setupLogPanel() {
        const logPanel = document.querySelector('.log-panel');
        const toggleAutoScroll = document.querySelector('#toggleAutoScroll');
        
        if (logPanel && toggleAutoScroll) {
            let autoScroll = true;
            
            // Set auto-scroll based on checkbox state
            toggleAutoScroll.addEventListener('change', function() {
                autoScroll = this.checked;
                if (autoScroll) {
                    logPanel.scrollTop = logPanel.scrollHeight;
                }
            });
            
            // Auto-scroll when new logs are added if enabled
            const observer = new MutationObserver(function(mutations) {
                if (autoScroll) {
                    logPanel.scrollTop = logPanel.scrollHeight;
                }
            });
            
            observer.observe(logPanel, { childList: true, subtree: true });
            
            // Initial scroll to bottom
            logPanel.scrollTop = logPanel.scrollHeight;
        }
    }

    // Setup on load and whenever DOM changes (for Streamlit re-renders)
    document.addEventListener('DOMContentLoaded', setupLogPanel);
    const observer = new MutationObserver(function(mutations) {
        if (document.querySelector('.log-panel')) {
            setupLogPanel();
        }
    });

    observer.observe(document.body, { childList: true, subtree: true });
    </script>
    """
    st.components.v1.html(js, height=0)

def create_log_download_buttons(logs):
    """Create buttons to download logs in different formats

    Args:
        logs: List of log entries to export
    """
    if not logs:
        st.warning("No logs to download")
        return

    col1, col2 = st.columns(2)

    # Create a text version of the logs
    text_logs = "\n".join([f"{log['timestamp']} - {log['level']} - {log['message']}" for log in logs])

    # Create CSV log data
    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer)
    csv_writer.writerow(["Timestamp", "Level", "Message"])
    for log in logs:
        csv_writer.writerow([log['timestamp'], log['level'], log['message']])

    # Download as text file
    with col1:
        st.download_button(
            "📄 Download as Text",
            text_logs,
            file_name=f"toxicity_analysis_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

    # Download as CSV
    with col2:
        st.download_button(
            "📊 Download as CSV",
            csv_buffer.getvalue(),
            file_name=f"toxicity_analysis_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def export_logs_json(logs):
    """Export logs as a JSON file

    Args:
        logs: List of log entries to export

    Returns:
        JSON string of logs
    """
    # Convert to a simplified format for JSON export
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "total_entries": len(logs),
        "entries": [
            {
                "timestamp": log["timestamp"],
                "level": log["level"],
                "message": log["message"]
            }
            for log in logs
        ]
    }
    return json.dumps(export_data, indent=2)

def create_custom_text_export(logs, options):
    """Create a custom text export of logs based on options

    Args:
        logs: List of log entries
        options: Dictionary of export options

    Returns:
        Text string for export
    """
    lines = []
    # Add a header with generation time
    lines.append(f"# Toxicity Detector Logs - Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"# Total entries: {len(logs)}")
    lines.append("")
    
    # Add config information if requested
    if options.get("Include Current Config", False):
        lines.append("# Current Configuration:")
        filtered_config = {k: v for k, v in CONFIG.items() if k in ["model", "categories"]}
        for key, value in filtered_config.items():
            lines.append(f"# {key}: {json.dumps(value)}")
        lines.append("")
    
    # Format each log entry
    for log in logs:
        parts = []
        if options.get("Include Timestamps", True):
            parts.append(log["timestamp"])
        if options.get("Include Log Levels", True):
            parts.append(log["level"])
        parts.append(log["message"])
        
        lines.append(" - ".join(parts))
    
    return "\n".join(lines)

def create_custom_csv_export(logs, options):
    """Create a custom CSV export of logs based on options

    Args:
        logs: List of log entries
        options: Dictionary of export options

    Returns:
        CSV string for export
    """
    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer)
    
    # Create headers based on options
    headers = []
    if options.get("Include Timestamps", True):
        headers.append("Timestamp")
    if options.get("Include Log Levels", True):
        headers.append("Level")
    headers.append("Message")
    
    # Add config columns if requested
    if options.get("Include Current Config", False):
        headers.append("DefaultThreshold")
        # Add category thresholds
        for cat in DEFAULT_CATEGORIES:
            headers.append(f"Threshold_{cat}")
    
    csv_writer.writerow(headers)
    
    # Write each log entry
    for log in logs:
        row = []
        if options.get("Include Timestamps", True):
            row.append(log["timestamp"])
        if options.get("Include Log Levels", True):
            row.append(log["level"])
        row.append(log["message"])
        
        # Add config information if requested
        if options.get("Include Current Config", False):
            row.append(CONFIG["model"].get("threshold", 0.6))
            # Add category thresholds
            for cat in DEFAULT_CATEGORIES:
                row.append(CONFIG.get("categories", {}).get(cat, CONFIG["model"].get("threshold", 0.6)))
        
        csv_writer.writerow(row)
    
    return csv_buffer.getvalue()

def create_custom_json_export(logs, options):
    """Create a custom JSON export of logs based on options

    Args:
        logs: List of log entries
        options: Dictionary of export options

    Returns:
        JSON string for export
    """
    # Create the base export data
    export_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_entries": len(logs),
            "export_options": options
        },
        "entries": []
    }
    
    # Add config if requested
    if options.get("Include Current Config", False):
        filtered_config = {k: v for k, v in CONFIG.items() if k in ["model", "categories"]}
        export_data["config"] = filtered_config
    
    # Add each log entry based on options
    for log in logs:
        entry = {}
        if options.get("Include Timestamps", True):
            entry["timestamp"] = log["timestamp"]
        if options.get("Include Log Levels", True):
            entry["level"] = log["level"]
        entry["message"] = log["message"]
        
        export_data["entries"].append(entry)
    
    return json.dumps(export_data, indent=2)

def create_logging_panel():
    """Create the logging panel UI with download options"""
    # Only display logging panel if enabled in session state
    if not st.session_state.get("show_logging", False):
        return

    st.markdown("### Analysis Log Panel")

    # Log controls in 3 columns
    cols = st.columns([2, 3, 2])
    with cols[0]:
        level_options = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR
        }
        selected_level = st.selectbox(
            "Log Level",
            options=list(level_options.keys()),
            index=1  # Default to INFO
        )
        log_level = level_options[selected_level]

    with cols[1]:
        log_search = st.text_input("Search logs", placeholder="Filter logs by text...")

    with cols[2]:
        if st.button("Clear Logs"):
            log_handler.clear()
            st.success("Logs cleared")

    # Toggle for auto-scroll
    auto_scroll = st.checkbox("Auto-scroll to new logs", value=True, key="toggleAutoScroll")

    # Get filtered logs
    logs = log_handler.get_logs(level=log_level)

    # Apply search filter if provided
    if log_search:
        logs = [log for log in logs if log_search.lower() in log['message'].lower()]

    # Create download buttons for the logs
    create_log_download_buttons(logs)

    # Log statistics
    level_counts = {}
    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        level_counts[level] = len([log for log in log_handler.get_logs() if log['level'] == level])

    # Show small stats on log counts
    st.markdown(
        f"""
        <div style="font-size: 12px; color: #666; margin: 5px 0;">
            Total logs: {len(log_handler.get_logs())}, 
            Filtered: {len(logs)} | 
            DEBUG: {level_counts['DEBUG']}, 
            INFO: {level_counts['INFO']}, 
            WARNING: {level_counts['WARNING']}, 
            ERROR: {level_counts['ERROR']}, 
            CRITICAL: {level_counts['CRITICAL']}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Create log panel
    st.markdown('<div class="log-panel">', unsafe_allow_html=True)

    for log in logs:
        level_class = f"log-level-{log['level']}"
        st.markdown(
            f"""<div class="log-entry">
                <span class="log-timestamp">{log['timestamp']}</span>
                <span class="log-level {level_class}">{log['level']}</span>
                <span class="log-message">{log['message']}</span>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Add JavaScript for auto-scrolling
    add_log_panel_js()

    # Add JSON export option
    if logs:
        json_logs = export_logs_json(logs)
        st.download_button(
            "📋 Download as JSON",
            json_logs,
            file_name=f"toxicity_analysis_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    # Add export options section
    with st.expander("Advanced Export Options"):
        export_type = st.radio(
            "Export Format",
            ["Text", "CSV", "JSON"],
            horizontal=True
        )

        export_options = {
            "Include Timestamps": True,
            "Include Log Levels": True,
            "Include Current Config": st.session_state.get("show_logging", False)
        }

        export_columns = st.columns(len(export_options))
        for i, (option, default) in enumerate(export_options.items()):
            with export_columns[i]:
                export_options[option] = st.checkbox(option, value=default)

        if st.button("Generate Custom Export"):
            # Generate export based on selection
            if export_type == "Text":
                text_content = create_custom_text_export(logs, export_options)
                st.download_button(
                    "Download Custom Text Export",
                    text_content,
                    file_name=f"toxicity_logs_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            elif export_type == "CSV":
                csv_content = create_custom_csv_export(logs, export_options)
                st.download_button(
                    "Download Custom CSV Export",
                    csv_content,
                    file_name=f"toxicity_logs_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:  # JSON
                json_content = create_custom_json_export(logs, export_options)
                st.download_button(
                    "Download Custom JSON Export",
                    json_content,
                    file_name=f"toxicity_logs_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

def analyze_text_with_logging(analyzer, text, log_individual_categories=False):
    """
    Analyze text with the analyzer and log the results

    Args:
        analyzer: The toxicity analyzer function
        text: The text to analyze
        log_individual_categories: Whether to log individual category probabilities
        
    Returns:
        The analysis result
    """
    logger.info(f"Analyzing text: '{text[:50]}{'...' if len(text) > 50 else ''}'")

    start_time = time.time()

    try:
        # Call the analyzer
        result = analyzer(text)
        
        # Log the completion time
        end_time = time.time()
        duration = end_time - start_time
        
        # Log the result
        if result["is_toxic"]:
            logger.warning(f"TOXIC CONTENT DETECTED ({result['category']}) with confidence {result['confidence']:.4f} in {duration:.4f}s")
            
            # Log toxic categories
            if "toxic_categories" in result and len(result["toxic_categories"]) > 1:
                logger.warning(f"Multiple toxic categories: {', '.join(result['toxic_categories'])}")
        else:
            logger.info(f"Content classified as NON-TOXIC with confidence {1-result['confidence']:.4f} in {duration:.4f}s")
        
        # Optionally log individual category probabilities
        if log_individual_categories and "probabilities" in result:
            for cat, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
                threshold = CONFIG.get("categories", {}).get(cat, CONFIG["model"].get("threshold", 0.6))
                status = "OVER THRESHOLD" if prob > threshold else "under threshold"
                logger.debug(f"Category {cat}: {prob:.4f} ({status}, threshold={threshold:.2f})")
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        raise

# Create custom localStorage component
def create_localstorage_component():
    component_id = str(uuid.uuid4())

    # JavaScript code for localStorage interaction
    js_code = """
    <script>
    // Initialize localStorage handler
    function initLocalStorage() {
        // Get all profiles from localStorage
        function getProfiles() {
            const profilesStr = localStorage.getItem('toxicityProfiles');
            return profilesStr ? JSON.parse(profilesStr) : {};
        }
        
        // Save profiles to localStorage
        function saveProfiles(profiles) {
            localStorage.setItem('toxicityProfiles', JSON.stringify(profiles));
        }
        
        // Save a profile
        function saveProfile(profileName, profileData) {
            const profiles = getProfiles();
            profiles[profileName] = profileData;
            saveProfiles(profiles);
            return Object.keys(profiles);
        }
        
        // Delete a profile
        function deleteProfile(profileName) {
            const profiles = getProfiles();
            if (profiles[profileName]) {
                delete profiles[profileName];
                saveProfiles(profiles);
            }
            return Object.keys(profiles);
        }
        
        // Get a specific profile
        function getProfile(profileName) {
            const profiles = getProfiles();
            return profiles[profileName] || null;
        }
        
        // Load all profile names
        function loadProfileNames() {
            const profiles = getProfiles();
            return Object.keys(profiles);
        }
        
        // Set up event listeners for messages from Python
        window.addEventListener('message', function(event) {
            const data = event.data;
            
            if (data.type === 'saveProfile' && data.componentId === "${component_id}") {
                const profileNames = saveProfile(data.profileName, data.profileData);
                window.streamlitSendBack({
                    type: 'profileSaved',
                    profileNames: profileNames,
                    componentId: "${component_id}"
                });
            }
            else if (data.type === 'deleteProfile' && data.componentId === "${component_id}") {
                const profileNames = deleteProfile(data.profileName);
                window.streamlitSendBack({
                    type: 'profileDeleted',
                    profileNames: profileNames,
                    componentId: "${component_id}"
                });
            }
            else if (data.type === 'loadProfile' && data.componentId === "${component_id}") {
                const profileData = getProfile(data.profileName);
                window.streamlitSendBack({
                    type: 'profileLoaded',
                    profileData: profileData,
                    componentId: "${component_id}"
                });
            }
            else if (data.type === 'getProfileNames' && data.componentId === "${component_id}") {
                const profileNames = loadProfileNames();
                window.streamlitSendBack({
                    type: 'profileNamesLoaded',
                    profileNames: profileNames,
                    componentId: "${component_id}"
                });
            }
        });
        
        // Signal that component is ready
        window.streamlitSendBack({
            type: 'componentReady',
            componentId: "${component_id}"
        });
    }

    // Initialize when DOM content is loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initLocalStorage);
    } else {
        initLocalStorage();
    }
    </script>
    """

    # Render the component
    st.components.v1.html(js_code, height=0)

    return component_id

def get_category_display_name(category):
    """Format category name for display"""
    return category.replace('-', ' ').title()

def get_description(category):
    """Get the description for a category if available"""
    if CATEGORY_DESCRIPTIONS_AVAILABLE:
        return get_category_description(category)

    # Fallback descriptions if categories module is not available
    descriptions = {
        'insult': "Content that contains insulting language or personal attacks",
        'hate': "Content that expresses hatred or prejudice toward a particular group",
        'obscene': "Content that contains vulgar, explicit, or offensive language",
        'threat': "Content that contains threats of violence or harm",
        'sexual': "Content that contains explicit sexual references",
        'self-harm': "Content that references or encourages self-harm or suicide"
    }
    return descriptions.get(category, "No description available")

@st.cache_resource
def load_analyzer(config=None):
    """Load the toxicity analyzer (cached to avoid reloading)

    Args:
        config: Configuration to use for the analyzer
        
    Returns:
        The analyzer function
    """
    logger.info(f"Loading analyzer with config: {config.get('model', {}).get('threshold', 0.6) if config else 'default'}")
    
    # Create a wrapper function that uses analyze_text with config
    def analyzer_wrapper(text):
        # Extract threshold from config
        threshold = config.get("model", {}).get("threshold", 0.6) if config else 0.6
        
        # Get category-specific thresholds
        category_thresholds = config.get("categories", {}) if config else {}
        
        # Use category-specific thresholds if available, otherwise use default
        if category_thresholds:
            thresholds = {cat: category_thresholds.get(cat, threshold) for cat in DEFAULT_CATEGORIES}
        else:
            thresholds = threshold
        
        # Call analyze_text
        result = analyze_text(text, threshold=thresholds)
        
        # Transform the result to match expected format
        transformed_result = {
            "is_toxic": result["is_toxic"],
            "category": None,
            "confidence": 0.0,
            "probabilities": result["probabilities"],
            "toxic_categories": []
        }
        
        # Find the most toxic category
        if result["is_toxic"]:
            toxic_cats = [cat for cat, is_toxic in result["categories"].items() if is_toxic]
            if toxic_cats:
                # Find the category with highest probability
                max_prob = 0.0
                max_cat = toxic_cats[0]
                for cat in toxic_cats:
                    prob = result["probabilities"].get(cat, 0.0)
                    if prob > max_prob:
                        max_prob = prob
                        max_cat = cat
                
                transformed_result["category"] = max_cat
                transformed_result["confidence"] = max_prob
                transformed_result["toxic_categories"] = toxic_cats
        
        return transformed_result
    
    return analyzer_wrapper

def send_message_to_component(component_id, message_type, **kwargs):
    """Send a message to the JavaScript component"""
    message = {"type": message_type, "componentId": component_id, **kwargs}

    # Use Streamlit's JavaScript evaluation to send the message
    js_code = f"""
    window.postMessage({json.dumps(message)}, "*");
    """

    st.components.v1.html(f"""
    <script>
    {js_code}
    </script>
    """, height=0)

    # Small delay to allow time for message propagation
    time.sleep(0.1)

def save_profile(component_id, profile_name, profile_data):
    """Save a profile to localStorage"""
    send_message_to_component(
        component_id,
        "saveProfile",
        profileName=profile_name,
        profileData=profile_data
    )

def delete_profile(component_id, profile_name):
    """Delete a profile from localStorage"""
    send_message_to_component(
        component_id,
        "deleteProfile",
        profileName=profile_name
    )

def load_profile(component_id, profile_name):
    """Load a profile from localStorage"""
    send_message_to_component(
        component_id,
        "loadProfile",
        profileName=profile_name
    )

def get_profile_names(component_id):
    """Get all profile names from localStorage"""
    send_message_to_component(
        component_id,
        "getProfileNames"
    )

def create_profile_manager(component_id, updated_config):
    """Create profile management interface in the sidebar

    Args:
        component_id: ID of the localStorage component
        updated_config: Current configuration dictionary
        
    Returns:
        Updated configuration dictionary after profile operations
    """
    # Session state for managing profiles
    if "profile_names" not in st.session_state:
        st.session_state.profile_names = []
        get_profile_names(component_id)

    if "current_profile" not in st.session_state:
        st.session_state.current_profile = None

    if "profile_loaded" not in st.session_state:
        st.session_state.profile_loaded = None

    # Handle component messages
    if st.session_state.get("streamlit_message"):
        message = st.session_state.streamlit_message
        
        if message.get("componentId") == component_id:
            if message.get("type") == "componentReady":
                get_profile_names(component_id)
            
            elif message.get("type") == "profileNamesLoaded":
                st.session_state.profile_names = message.get("profileNames", [])
            
            elif message.get("type") == "profileSaved":
                st.session_state.profile_names = message.get("profileNames", [])
                st.sidebar.success(f"Profile '{st.session_state.current_profile}' saved!")
            
            elif message.get("type") == "profileDeleted":
                st.session_state.profile_names = message.get("profileNames", [])
                st.sidebar.info(f"Profile deleted")
                st.session_state.current_profile = None
            
            elif message.get("type") == "profileLoaded":
                profile_data = message.get("profileData")
                if profile_data:
                    # Update the configuration with profile data
                    updated_config = copy.deepcopy(profile_data)
                    st.session_state.profile_loaded = st.session_state.current_profile
                    st.sidebar.success(f"Profile '{st.session_state.current_profile}' loaded")
                    
                    # Force a rerun to update the UI with new config
                    st.experimental_rerun()

    st.sidebar.markdown("""
    <div class="sidebar-header">
    Profile Management
    </div>
    """, unsafe_allow_html=True)

    # Create a container for profile actions
    profile_col1, profile_col2 = st.sidebar.columns([3, 2])

    with profile_col1:
        # Profile selection dropdown
        selected_profile = st.selectbox(
            "Select Profile",
            options=[""] + st.session_state.profile_names,
            index=0 if st.session_state.current_profile is None else 
                  (st.session_state.profile_names.index(st.session_state.current_profile) + 1 
                   if st.session_state.current_profile in st.session_state.profile_names else 0),
            key="profile_selector"
        )
        
        if selected_profile != st.session_state.current_profile:
            st.session_state.current_profile = selected_profile if selected_profile else None

    with profile_col2:
        # Load button
        if st.session_state.current_profile:
            if st.button("Load Profile"):
                load_profile(component_id, st.session_state.current_profile)

    # New profile name input and save button
    new_profile_name = st.sidebar.text_input("New Profile Name", key="new_profile_name")

    save_col1, save_col2 = st.sidebar.columns([1, 1])

    with save_col1:
        # Create new profile or update existing
        if st.button("Save Profile"):
            if new_profile_name:
                # Create a profile data object with current settings
                profile_data = copy.deepcopy(updated_config)
                save_profile(component_id, new_profile_name, profile_data)
                st.session_state.current_profile = new_profile_name
            elif st.session_state.current_profile:
                # Update existing profile
                profile_data = copy.deepcopy(updated_config)
                save_profile(component_id, st.session_state.current_profile, profile_data)
            else:
                st.sidebar.warning("Please enter a profile name or select an existing profile")

    with save_col2:
        # Delete profile
        if st.session_state.current_profile and st.button("Delete Profile"):
            delete_profile(component_id, st.session_state.current_profile)

    # Show active profile
    if st.session_state.profile_loaded:
        st.sidebar.markdown(f"**Active Profile:** {st.session_state.profile_loaded}")

    st.sidebar.markdown("---")

    return updated_config

def create_sidebar_controls(component_id) -> Dict:
    """Create sidebar controls for adjusting thresholds and settings

    Args:
        component_id: ID of the localStorage component
        
    Returns:
        Dictionary containing the updated configuration
    """
    # Create a deep copy of the config to avoid modifying the original
    updated_config = copy.deepcopy(CONFIG)

    # Profile management section
    updated_config = create_profile_manager(component_id, updated_config)

    # Add settings header
    st.sidebar.markdown("""
    <div class="sidebar-header">
    Settings
    </div>
    """, unsafe_allow_html=True)

    # Add logging panel toggle
    show_logging = st.sidebar.checkbox(
        "Show Logging Panel",
        value=st.session_state.get("show_logging", False),
        help="Display real-time logging panel showing analysis details"
    )
    st.session_state.show_logging = show_logging

    # Log verbosity if logging panel is enabled
    if show_logging:
        log_detailed = st.sidebar.checkbox(
            "Detailed Category Logging",
            value=st.session_state.get("log_detailed", False),
            help="Log detailed information about each category's probability"
        )
        st.session_state.log_detailed = log_detailed

    # Add Groq fallback toggle
    if GROQ_AVAILABLE:
        groq_enabled = st.sidebar.checkbox(
            "Use Groq Fallback",
            value=updated_config.get("groq", {}).get("enabled", False),
            help="Enable Groq API as a fallback for toxicity analysis when the local model is uncertain"
        )
        
        # Update config
        if "groq" not in updated_config:
            updated_config["groq"] = {}
        updated_config["groq"]["enabled"] = groq_enabled
        
        # Show API key field if Groq is enabled
        if groq_enabled:
            groq_api_key = st.sidebar.text_input(
                "Groq API Key",
                value=updated_config.get("groq", {}).get("api_key", ""),
                type="password",
                help="Enter your Groq API key for fallback toxicity analysis"
            )
            updated_config["groq"]["api_key"] = groq_api_key

    # Add batch size control
    st.sidebar.markdown("""
    <div class="sidebar-header">
    Batch Processing
    </div>
    """, unsafe_allow_html=True)

    batch_size = st.sidebar.slider(
        "Batch Size",
        min_value=50,
        max_value=500,
        value=int(updated_config.get("batch", {}).get("size", 150)),
        step=50,
        help="Number of lines to process in each batch"
    )
    
    # Update batch settings in config
    if "batch" not in updated_config:
        updated_config["batch"] = {}
    updated_config["batch"]["size"] = batch_size

    # Add threshold controls
    st.sidebar.markdown("""
    <div class="sidebar-header">
    Threshold Settings
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("Adjust the sensitivity for each toxicity category.")

    # Default threshold control
    default_threshold = st.sidebar.slider(
        "Default Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(updated_config["model"].get("threshold", 0.6)),
        step=0.05,
        help="The default threshold for all categories"
    )
    updated_config["model"]["threshold"] = default_threshold

    # Add a separator
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Category-Specific Thresholds")

    # Individual category thresholds
    for category in DEFAULT_CATEGORIES:
        display_name = get_category_display_name(category)
        
        # Display a short description for the category
        st.sidebar.markdown(f"""
        <div class="slider-label">
        {display_name}
        </div>
        """, unsafe_allow_html=True)

        # Current threshold from config
        current_threshold = updated_config.get("categories", {}).get(category, default_threshold)
        
        # Slider for this category
        category_threshold = st.sidebar.slider(
            f"Threshold for {display_name}",
            min_value=0.0,
            max_value=1.0,
            value=float(current_threshold),
            step=0.05,
            key=f"threshold_{category}",
            label_visibility="collapsed",  # Hide the label since we have a markdown label above
            help=f"Adjust the sensitivity threshold for {display_name} category. Lower values are more sensitive."
        )
        
        # Update the config
        if "categories" not in updated_config:
            updated_config["categories"] = {}
        updated_config["categories"][category] = category_threshold

    # Add a save button if config_loader is available
    if CONFIG_LOADER_AVAILABLE:
        if st.sidebar.button("Save Settings as Default"):
            try:
                # Update config file
                save_config(updated_config)
                st.sidebar.success("Settings saved successfully!")
            except Exception as e:
                st.sidebar.error(f"Error saving settings: {str(e)}")

    # Add reset button
    if st.sidebar.button("Reset to App Defaults"):
        # This will trigger a page refresh, reloading the original config
        st.session_state.profile_loaded = None
        st.experimental_rerun()

    return updated_config

def read_text_file(uploaded_file) -> List[str]:
    """Read a text file and return the lines"""
    text_content = uploaded_file.getvalue().decode("utf-8")
    lines = text_content.split("\n")
    return [line.strip() for line in lines if line.strip()]

def read_csv_file(uploaded_file) -> Tuple[List[str], Optional[pd.DataFrame]]:
    """Read a CSV file and return the text column and original dataframe"""
    try:
        # Try to use pandas if available
        df = pd.read_csv(uploaded_file)

        # Try to identify a text/content column
        text_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if "text" in col_lower or "content" in col_lower or "comment" in col_lower:
                text_columns.append(col)
        
        if text_columns:
            # Use the first identified text column
            text_col = text_columns[0]
            return df[text_col].astype(str).tolist(), df
        else:
            # Use the first column as a fallback
            return df.iloc[:, 0].astype(str).tolist(), df
    except:
        # Fallback to manual CSV parsing
        csv_content = uploaded_file.getvalue().decode("utf-8")
        reader = csv.reader(io.StringIO(csv_content))
        
        # Skip the header
        headers = next(reader)
        
        # Try to identify a text/content column
        text_col_idx = 0
        for i, header in enumerate(headers):
            header_lower = header.lower()
            if "text" in header_lower or "content" in header_lower or "comment" in header_lower:
                text_col_idx = i
                break
        
        # Extract the text column
        rows = list(reader)
        texts = [row[text_col_idx] for row in rows if row and len(row) > text_col_idx]
        
        # Try to create a dataframe for convenience
        try:
            df = pd.DataFrame(rows, columns=headers)
            return texts, df
        except:
            return texts, None

def process_uploaded_file(uploaded_file, analyzer, config) -> Dict:
    """Process uploaded file and analyze for toxicity

    Args:
        uploaded_file: The uploaded file object
        analyzer: The toxicity analyzer function
        config: Configuration dictionary
        
    Returns:
        Dictionary with analysis results
    """
    start_time = time.time()
    file_extension = uploaded_file.name.split(".")[-1].lower()
    original_df = None

    # Read file based on extension
    if file_extension == "csv":
        lines, original_df = read_csv_file(uploaded_file)
    else:
        lines = read_text_file(uploaded_file)

    # Get file stats
    file_size_bytes = len(uploaded_file.getvalue())
    file_size_display = f"{file_size_bytes/1024:.1f} KB" if file_size_bytes < 1024*1024 else f"{file_size_bytes/(1024*1024):.2f} MB"

    # Initialize storage for results
    results = {
        "file_name": uploaded_file.name,
        "file_extension": file_extension,
        "file_size_bytes": file_size_bytes,
        "file_size_display": file_size_display,
        "timestamp": datetime.now().isoformat(),
        "start_time": start_time,
        "total_lines": len(lines),
        "toxic_lines": 0,
        "non_toxic_lines": 0,
        "category_counts": {cat: 0 for cat in DEFAULT_CATEGORIES},
        "confidence_metrics": {
            "min": 1.0,
            "max": 0.0,
            "avg": 0.0,
            "median": 0.0
        },
        "processing_metrics": {
            "start_time": start_time,
            "end_time": None,
            "duration_seconds": None,
            "lines_per_second": None,
            "batch_size": config.get("batch", {}).get("size", 150),
            "total_batches": None,
            "threshold": config["model"].get("threshold", 0.6)
        },
        "threshold_settings": {
            "default": config["model"].get("threshold", 0.6),
            "categories": {cat: config.get("categories", {}).get(cat, config["model"].get("threshold", 0.6)) for cat in DEFAULT_CATEGORIES}
        },
        "detailed_results": []
    }

    # Set up batch processing if available
    batch_size = config.get("batch", {}).get("size", 150)
    total_batches = (len(lines) + batch_size - 1) // batch_size
    results["processing_metrics"]["total_batches"] = total_batches

    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Storage for confidence values to calculate statistics
    confidence_values = []

    # Process file in batches
    for i in range(0, len(lines), batch_size):
        batch_lines = lines[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        status_text.text(f"Processing batch {batch_num}/{total_batches} ({len(batch_lines)} lines)...")
        
        # Process batch using simple loop processing
        batch_results = []
        for line in batch_lines:
            if line.strip():  # Skip empty lines
                analysis = analyzer(line)
                batch_results.append({
                    "text": line,
                    "analysis": analysis
                })
        
        # Aggregate results
        for result in batch_results:
            analysis = result["analysis"]
            confidence = analysis["confidence"]
            
            # Update confidence metrics
            confidence_values.append(confidence)
            results["confidence_metrics"]["min"] = min(results["confidence_metrics"]["min"], confidence)
            results["confidence_metrics"]["max"] = max(results["confidence_metrics"]["max"], confidence)
            
            # Store detailed result
            results["detailed_results"].append({
                "text": result["text"],
                "is_toxic": analysis["is_toxic"],
                "category": analysis["category"],
                "confidence": confidence,
                "probabilities": analysis["probabilities"] 
            })
            
            # Update summary statistics
            if analysis["is_toxic"]:
                results["toxic_lines"] += 1
                if analysis["category"] in results["category_counts"]:
                    results["category_counts"][analysis["category"]] += 1
            else:
                results["non_toxic_lines"] += 1
        
        # Update progress
        progress = min(1.0, (i + len(batch_lines)) / len(lines))
        progress_bar.progress(progress)

    # Complete progress bar
    progress_bar.progress(1.0)

    # Calculate final metrics
    end_time = time.time()
    duration = end_time - start_time

    # Update processing metrics
    results["processing_metrics"]["end_time"] = end_time
    results["processing_metrics"]["duration_seconds"] = duration
    results["processing_metrics"]["lines_per_second"] = len(lines) / duration if duration > 0 else 0

    # Update confidence metrics
    if confidence_values:
        results["confidence_metrics"]["avg"] = sum(confidence_values) / len(confidence_values)
        results["confidence_metrics"]["median"] = np.median(confidence_values)

    status_text.text(f"Processing complete! Analyzed {len(lines)} lines in {duration:.2f} seconds ({results['processing_metrics']['lines_per_second']:.1f} lines/sec)")

    # Store the original dataframe if available
    if original_df is not None:
        results["original_df"] = original_df

    # Create a dataframe with the detailed results
    results_df = pd.DataFrame(results["detailed_results"])
    results["results_df"] = results_df

    return results

def create_csv_export(results):
    """Create a CSV export of the analysis results

    Args:
        results: The analysis results dictionary
        
    Returns:
        StringIO object containing the CSV data
    """
    # Create the CSV data
    if "results_df" in results:
        # Use existing dataframe if available
        results_df = results["results_df"]
    else:
        # Create dataframe from detailed results
        results_df = pd.DataFrame(results["detailed_results"])

    # Check if we have original data
    if "original_df" in results and results["original_df"] is not None:
        # Try to merge with original data
        original_df = results["original_df"]
        
        # We can only merge if we can match rows
        if len(original_df) == len(results_df):
            # Create a combined dataframe
            combined_df = original_df.copy()
            
            # Add analysis results
            combined_df["is_toxic"] = results_df["is_toxic"]
            combined_df["category"] = results_df["category"]
            combined_df["confidence"] = results_df["confidence"]
            
            # Add individual probabilities
            for cat in DEFAULT_CATEGORIES:
                combined_df[f"prob_{cat}"] = results_df["probabilities"].apply(
                    lambda x: x.get(cat, 0.0) if isinstance(x, dict) else 0.0
                )
            
            # Use the combined dataframe for export
            export_df = combined_df
        else:
            # Just use the results dataframe
            export_df = results_df
    else:
        # Use the analysis results dataframe
        export_df = results_df

    # Expand probabilities column if it exists and isn't already expanded
    if "probabilities" in export_df.columns and not any(col.startswith("prob_") for col in export_df.columns):
        for cat in DEFAULT_CATEGORIES:
            export_df[f"prob_{cat}"] = export_df["probabilities"].apply(
                lambda x: x.get(cat, 0.0) if isinstance(x, dict) else 0.0
            )
        
        # Remove the original probabilities column
        export_df = export_df.drop(columns=["probabilities"])

    # Create a CSV string
    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)

    return csv_buffer

def get_confidence_level_counts(results):
    """Get counts of results by confidence level

    Args:
        results: The analysis results dictionary
        
    Returns:
        DataFrame with confidence level counts
    """
    # Define confidence levels
    confidence_levels = [
        {"name": "Very High (>0.9)", "min": 0.9, "max": 1.0, "color": "#d62728"},  # Red
        {"name": "High (0.8-0.9)", "min": 0.8, "max": 0.9, "color": "#e74c3c"},    # Light red
        {"name": "Medium-High (0.7-0.8)", "min": 0.7, "max": 0.8, "color": "#ff7f0e"},  # Orange
        {"name": "Medium (0.6-0.7)", "min": 0.6, "max": 0.7, "color": "#f39c12"},  # Light orange
        {"name": "Medium-Low (0.5-0.6)", "min": 0.5, "max": 0.6, "color": "#9467bd"},  # Purple
        {"name": "Low (0.4-0.5)", "min": 0.4, "max": 0.5, "color": "#8c564b"},     # Brown
        {"name": "Very Low (0.3-0.4)", "min": 0.3, "max": 0.4, "color": "#1f77b4"}, # Blue
        {"name": "Minimal (<0.3)", "min": 0.0, "max": 0.3, "color": "#2ca02c"}     # Green
    ]

    # Count occurrences in each level
    confidence_counts = []

    for level in confidence_levels:
        count = 0
        for result in results["detailed_results"]:
            confidence = result["confidence"] if result["is_toxic"] else 1.0 - result["confidence"]
            if level["min"] <= confidence < level["max"]:
                count += 1
        
        confidence_counts.append({
            "name": level["name"],
            "count": count,
            "color": level["color"]
        })

    # Create dataframe
    confidence_df = pd.DataFrame(confidence_counts)
    return confidence_df

def display_file_results_summary(results):
    """Display summary of file analysis results

    Args:
        results: Dictionary with analysis results
    """
    st.subheader(f"Results for {results['file_name']}")

    # Display detailed file metrics
    with st.expander("File & Processing Metrics", expanded=True):
        st.markdown("""
        <div class="metrics-container">
        """, unsafe_allow_html=True)

        # File metrics section
        st.markdown("""
        <div class="metrics-title">File Information</div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="metrics-grid">
        """, unsafe_allow_html=True)

        metrics_data = [
            ("File Name", results["file_name"]),
            ("File Type", f".{results['file_extension']}" if 'file_extension' in results else "Unknown"),
            ("File Size", results["file_size_display"]),
            ("Total Lines", f"{results['total_lines']:,}"),
            ("Processing Date", datetime.fromisoformat(results["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"))
        ]
        
        for label, value in metrics_data:
            st.markdown(
                f"""
                <div class="metric-item">
                    <span class="metric-label">{label}:</span>
                    <span class="metric-value">{value}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)
        
        # Processing metrics section
        st.markdown("""
        <div class="metrics-title">Processing Metrics</div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="metrics-grid">
        """, unsafe_allow_html=True)

        # Calculate or extract processing metrics
        duration = results["processing_metrics"].get("duration_seconds", 0)
        lines_per_sec = results["processing_metrics"].get("lines_per_second", 0)
        
        processing_metrics = [
            ("Processing Time", f"{duration:.2f} seconds"),
            ("Processing Rate", f"{lines_per_sec:.1f} lines/second"),
            ("Batch Size", f"{results['processing_metrics'].get('batch_size', 'N/A')}"),
            ("Total Batches", f"{results['processing_metrics'].get('total_batches', 'N/A')}"),
            ("Default Threshold", f"{results['threshold_settings'].get('default', 0.6):.2f}")
        ]
        
        for label, value in processing_metrics:
            st.markdown(
                f"""
                <div class="metric-item">
                    <span class="metric-label">{label}:</span>
                    <span class="metric-value">{value}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)
        
        # Confidence metrics section
        st.markdown("""
        <div class="metrics-title">Confidence Metrics</div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="metrics-grid">
        """, unsafe_allow_html=True)

        confidence_metrics = [
            ("Minimum Confidence", f"{results['confidence_metrics'].get('min', 0):.2f}"),
            ("Maximum Confidence", f"{results['confidence_metrics'].get('max', 0):.2f}"),
            ("Average Confidence", f"{results['confidence_metrics'].get('avg', 0):.2f}"),
            ("Median Confidence", f"{results['confidence_metrics'].get('median', 0):.2f}")
        ]
        
        for label, value in confidence_metrics:
            st.markdown(
                f"""
                <div class="metric-item">
                    <span class="metric-label">{label}:</span>
                    <span class="metric-value">{value}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Display summary statistics in cards
    st.markdown("""
    <div class="summary-stats">
    """, unsafe_allow_html=True)

    # Total lines card
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Total Lines</div>
        <div class="stat-value">{results['total_lines']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Toxic lines card
    toxic_percentage = results['toxic_lines'] / results['total_lines'] * 100 if results['total_lines'] > 0 else 0
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Toxic Content</div>
        <div class="stat-value">{results['toxic_lines']} ({toxic_percentage:.1f}%)</div>
    </div>
    """, unsafe_allow_html=True)

    # Non-toxic lines card
    non_toxic_percentage = results['non_toxic_lines'] / results['total_lines'] * 100 if results['total_lines'] > 0 else 0
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Safe Content</div>
        <div class="stat-value">{results['non_toxic_lines']} ({non_toxic_percentage:.1f}%)</div>
    </div>
    """, unsafe_allow_html=True)

    # Processing time card
    duration = results["processing_metrics"].get("duration_seconds", 0)
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Processing Time</div>
        <div class="stat-value">{duration:.2f}s</div>
        <div class="stat-label">({results['processing_metrics'].get('lines_per_second', 0):.1f} lines/sec)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Create download buttons
    col1, col2 = st.columns(2)

    with col1:
        # CSV Download
        csv_data = create_csv_export(results)
        csv_filename = f"toxicity_analysis_{results['file_name'].split('.')[0]}.csv"
        st.download_button(
            "📄 Download Report as CSV",
            csv_data.getvalue(),
            csv_filename,
            "text/csv",
            key="download_csv"
        )

    with col2:
        # JSON Download
        # Remove dataframes from the results before conversion to JSON
        json_results = {k: v for k, v in results.items() if k not in ["original_df", "results_df"]}
        results_json = json.dumps(json_results, indent=2)
        st.download_button(
            "📊 Download Results as JSON",
            results_json,
            f"toxicity_analysis_{results['file_name'].split('.')[0]}.json",
            "application/json",
            key="download_json"
        )

    # Display interactive visualizations
    st.subheader("Analysis Visualizations")

    # Create tabs for different visualizations
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Category Distribution", "Confidence Levels", "Time Series"])

    with viz_tab1:
        # Display category breakdown as interactive chart
        if results['toxic_lines'] > 0:
            # Prepare data for chart
            categories = []
            counts = []
            colors = []
            
            # Color palette for categories
            category_colors = {
                'insult': '#d62728',    # Red
                'hate': '#9467bd',      # Purple
                'obscene': '#ff7f0e',   # Orange
                'threat': '#e74c3c',    # Light Red
                'sexual': '#f39c12',    # Yellow
                'self-harm': '#8c564b'  # Brown
            }
            
            for cat, count in results['category_counts'].items():
                if count > 0:
                    categories.append(get_category_display_name(cat))
                    counts.append(count)
                    colors.append(category_colors.get(cat, '#1f77b4'))  # Default to blue
            
            # Sort by count (descending)
            sorted_indices = np.argsort(counts)[::-1]
            categories = [categories[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]
            colors = [colors[i] for i in sorted_indices]
            
            # Calculate percentages
            percentages = [count / results['toxic_lines'] * 100 for count in counts]
            
            # Create dataframe for chart
            df = pd.DataFrame({
                'Category': categories,
                'Count': counts,
                'Percentage': percentages,
                'Color': colors
            })
            
            # Create interactive bar chart
            cat_chart = alt.Chart(df).mark_bar().encode(
                y=alt.Y('Category:N', sort='-x', title=None),
                x=alt.X('Count:Q', title='Number of Occurrences'),
                color=alt.Color('Category:N', scale=alt.Scale(domain=categories, range=colors), legend=None),
                tooltip=['Category', 'Count', alt.Tooltip('Percentage', format='.1f')]
            ).properties(
                title="Category Distribution of Toxic Content",
                height=min(300, 50 * len(categories))  # Dynamic height based on number of categories
            ).interactive()
            
            # Create a pie chart for category distribution
            pie_data = df.copy()
            pie_data['angle'] = pie_data['Count'] / pie_data['Count'].sum() * 2 * 3.14159
            
            pie_chart = alt.Chart(pie_data).mark_arc().encode(
                theta=alt.Theta(field="angle", type="quantitative"),
                color=alt.Color('Category:N', scale=alt.Scale(domain=categories, range=colors)),
                tooltip=['Category', 'Count', alt.Tooltip('Percentage', format='.1f')]
            ).properties(
                title="Percentage Distribution by Category",
                width=300,
                height=300
            ).interactive()
            
            # Display the charts side by side
            col1, col2 = st.columns([2, 1])
            with col1:
                st.altair_chart(cat_chart, use_container_width=True)
            with col2:
                st.altair_chart(pie_chart, use_container_width=True)
            
            # Add descriptive text
            total_toxic = results['toxic_lines']
            most_common_cat = categories[0] if categories else "None"
            most_common_count = counts[0] if counts else 0
            most_common_pct = percentages[0] if percentages else 0
            
            st.markdown(f"""
            <div class="chart-container">
                <p><strong>Category Analysis Summary:</strong> Of the {total_toxic:,} toxic items detected, 
                <span style="color: {colors[0] if colors else '#000'}"><strong>{most_common_cat}</strong></span> 
                was the most prevalent category with {most_common_count:,} occurrences ({most_common_pct:.1f}% of toxic content).</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.info("No toxic content detected in the file.")

    with viz_tab2:
        # Create confidence level distribution chart
        confidence_df = get_confidence_level_counts(results)
        
        if not confidence_df.empty and confidence_df["count"].sum() > 0:
            confidence_chart = alt.Chart(confidence_df).mark_bar().encode(
                y=alt.Y('name:N', title=None, sort=None),  # Keep original order
                x=alt.X('count:Q', title='Number of Items'),
                color=alt.Color('name:N', scale=alt.Scale(domain=confidence_df['name'].tolist(), 
                                                        range=confidence_df['color'].tolist()),
                                legend=None),
                tooltip=['name', 'count']
            ).properties(
                title="Distribution by Confidence Level",
                height=300
            ).interactive()
            
            st.altair_chart(confidence_chart, use_container_width=True)
            
            # Create a histogram of confidence values
            confidence_values = []
            for result in results["detailed_results"]:
                confidence_values.append(result["confidence"] if result["is_toxic"] else 1.0 - result["confidence"])
            
            # Create dataframe for histogram
            confidence_hist_df = pd.DataFrame({
                'Confidence': confidence_values
            })
            
            # Create histogram
            hist_chart = alt.Chart(confidence_hist_df).mark_bar().encode(
                x=alt.X('Confidence:Q', bin=alt.Bin(maxbins=20), title='Confidence Score'),
                y=alt.Y('count()', title='Count'),
                tooltip=['count()', alt.Tooltip('Confidence:Q', bin=alt.Bin(maxbins=20))]
            ).properties(
                title="Histogram of Confidence Scores",
                height=250
            ).interactive()
            
            st.altair_chart(hist_chart, use_container_width=True)
            
            # Add descriptive text
            avg_confidence = results["confidence_metrics"].get("avg", 0)
            median_confidence = results["confidence_metrics"].get("median", 0)
            
            st.markdown(f"""
            <div class="chart-container">
                <p><strong>Confidence Analysis:</strong> The average confidence score is {avg_confidence:.2f}, 
                with a median of {median_confidence:.2f}. Higher confidence scores indicate stronger classification certainty.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No confidence data available.")

    with viz_tab3:
        # Only show if there are enough results for a meaningful time series
        if len(results["detailed_results"]) >= 10:
            # Add index and create time series data
            time_series_data = []
            window_size = max(1, min(100, int(len(results["detailed_results"]) / 20)))
            
            for i, result in enumerate(results["detailed_results"]):
                time_series_data.append({
                    "index": i,
                    "is_toxic": 1 if result["is_toxic"] else 0,
                    "confidence": result["confidence"],
                    "category": result["category"] if result["is_toxic"] else "non-toxic"
                })
            
            # Create rolling average
            ts_df = pd.DataFrame(time_series_data)
            ts_df["toxic_rolling"] = ts_df["is_toxic"].rolling(window=window_size, min_periods=1).mean()
            
            # Create time series chart
            line_chart = alt.Chart(ts_df).mark_line(color='red').encode(
                x=alt.X('index:Q', title='Document Position (line number)'),
                y=alt.Y('toxic_rolling:Q', title=f'Toxicity Rate (Rolling Avg, window={window_size})'),
                tooltip=['index:Q', alt.Tooltip('toxic_rolling:Q', format='.2f')]
            ).properties(
                title="Toxicity Trend Throughout Document",
                height=300
            ).interactive()
            
            # Add toxic points as scatter plot
            toxic_points = alt.Chart(ts_df[ts_df['is_toxic'] == 1]).mark_circle(size=60).encode(
                x='index:Q',
                y=alt.Y('is_toxic:Q', title='Toxicity'),
                color=alt.Color('category:N', scale=alt.Scale(scheme='category10')),
                tooltip=['index:Q', 'category:N', 'confidence:Q']
            )
            
            # Combine charts
            combined_chart = line_chart + toxic_points
            
            st.altair_chart(combined_chart, use_container_width=True)
            
            # Add descriptive text
            st.markdown(f"""
            <div class="chart-container">
                <p><strong>Trend Analysis:</strong> This chart shows how toxicity changes throughout the document, 
                with colored dots indicating toxic entries and their categories. The red line shows a rolling average 
                of toxicity rate with a window size of {window_size} entries.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add heatmap of toxicity distribution
            # Create a 2D grid representation for heatmap
            grid_size = min(50, max(10, int(np.sqrt(len(results["detailed_results"])))))
            heatmap_data = []
            
            for i, result in enumerate(results["detailed_results"]):
                row = i // grid_size
                col = i % grid_size
                heatmap_data.append({
                    "row": row,
                    "col": col,
                    "is_toxic": 1 if result["is_toxic"] else 0,
                    "confidence": result["confidence"] if result["is_toxic"] else 0,
                    "category": result["category"] if result["is_toxic"] else "non-toxic"
                })
            
            heatmap_df = pd.DataFrame(heatmap_data)
            
            heatmap = alt.Chart(heatmap_df).mark_rect().encode(
                x=alt.X('col:O', title=None, axis=alt.Axis(labels=False, ticks=False)),
                y=alt.Y('row:O', title=None, axis=alt.Axis(labels=False, ticks=False)),
                color=alt.Color('confidence:Q', 
                                scale=alt.Scale(domain=[0, 1], range=['#f7fbff', '#08306b']),
                                title='Toxicity Confidence'),
                tooltip=['row:O', 'col:O', 'confidence:Q', 'category:N']
            ).properties(
                title="Toxicity Distribution Heatmap",
                width=500,
                height=300
            ).interactive()
            
            st.altair_chart(heatmap, use_container_width=True)
            
            # Add descriptive text
            st.markdown(f"""
            <div class="chart-container">
                <p><strong>Distribution Analysis:</strong> This heatmap shows the distribution of toxic content 
                throughout the document. Darker cells indicate entries with higher toxicity confidence.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Not enough data for meaningful time series visualization. At least 10 entries are required.")

    # Create tabs for detailed results view
    result_tab1, result_tab2, result_tab3 = st.tabs(["Toxic Content", "All Content", "Category Search"])

    with result_tab1:
        # Show only toxic content
        toxic_results = [result for result in results['detailed_results'] if result['is_toxic']]
        if toxic_results:
            display_detailed_results(toxic_results)
        else:
            st.info("No toxic content detected in the file.")

    with result_tab2:
        # Show all content with pagination
        display_detailed_results(results['detailed_results'])

    with result_tab3:
        # Category search and filter
        search_col1, search_col2 = st.columns([3, 1])
        
        with search_col1:
            # Add a search box
            search_term = st.text_input("Search text content", key="search_text")
        
        with search_col2:
            # Add category filter dropdown
            category_options = ["All Categories"] + [get_category_display_name(cat) for cat in DEFAULT_CATEGORIES]
            selected_category = st.selectbox("Filter by Category", options=category_options, key="category_filter")
        
        # Filter results based on search and category
        filtered_results = []
        search_term_lower = search_term.lower()
        
        for result in results['detailed_results']:
            # Text search (case-insensitive)
            text_match = search_term == "" or search_term_lower in result['text'].lower()
            
            # Category filter
            if selected_category == "All Categories":
                category_match = True
            elif not result['is_toxic']:
                category_match = False
            else:
                category_match = get_category_display_name(result['category']) == selected_category
            
            # Add to filtered results if both match
            if text_match and category_match:
                filtered_results.append(result)
        
        # Display filtered results
        if filtered_results:
            st.write(f"Found {len(filtered_results)} matching results")
            display_detailed_results(filtered_results)
        else:
            st.info("No matching results found.")

def display_detailed_results(results, page_size=10):
    """Display paginated detailed results

    Args:
        results: List of detailed analysis results
        page_size: Number of items per page
    """
    # Initialize page number in session state
    if "page_number" not in st.session_state:
        st.session_state.page_number = 0

    # Calculate total pages
    total_pages = (len(results) + page_size - 1) // page_size

    # Create a filtered page of results
    start_idx = st.session_state.page_number * page_size
    end_idx = min(start_idx + page_size, len(results))
    page_results = results[start_idx:end_idx]

    # Display the current page of results
    for i, result in enumerate(page_results, start=start_idx + 1):
        with st.expander(f"#{i}: {result['text'][:50]}{'...' if len(result['text']) > 50 else ''}"):
            # Use animated result for each item
            animated_result(
                is_toxic=result['is_toxic'],
                category=result['category'] if result['is_toxic'] else None,
                confidence=result['confidence']
            )
            
            # Add category pills if toxic
            if result['is_toxic'] and result.get("toxic_categories"):
                st.markdown(
                    category_pills_row(result["toxic_categories"]),
                    unsafe_allow_html=True
                )
            
            # Display probabilities as a horizontal bar chart
            probs = result['probabilities']
            categories = list(probs.keys())
            probabilities = list(probs.values())
            
            # Sort by probability (descending)
            sorted_indices = np.argsort(probabilities)[::-1]
            categories = [categories[i] for i in sorted_indices]
            probabilities = [probabilities[i] for i in sorted_indices]
            
            df = pd.DataFrame({
                'Category': [get_category_display_name(c) for c in categories],
                'Probability': probabilities
            })
            
            # Create chart with enhanced styling
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('Probability:Q', scale=alt.Scale(domain=[0, 1])),
                y=alt.Y('Category:N', sort=None),
                color=alt.Color('Probability:Q', 
                               scale=alt.Scale(domain=[0, 0.4, 0.6, 1], 
                                              range=['#3498db', '#f39c12', '#e74c3c', '#c0392b'])),
                tooltip=['Category', alt.Tooltip('Probability', format='.3f')]
            ).properties(
                height=30 * len(categories)  # Height based on number of categories
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)

    # Pagination controls with enhanced styling
    if total_pages > 1:
        st.markdown('<div class="pagination-container">', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])
        
        with col1:
            if st.button("⏮ First", disabled=st.session_state.page_number <= 0, key="btn_first"):
                st.session_state.page_number = 0
                st.experimental_rerun()
        
        with col2:
            if st.button("◀ Prev", disabled=st.session_state.page_number <= 0, key="btn_prev"):
                st.session_state.page_number -= 1
                st.experimental_rerun()
        
        with col3:
            st.markdown(f'<div class="page-info">Page {st.session_state.page_number + 1} of {total_pages} ({len(results):,} items)</div>',
                       unsafe_allow_html=True)
        
        with col4:
            if st.button("Next ▶", disabled=st.session_state.page_number >= total_pages - 1, key="btn_next"):
                st.session_state.page_number += 1
                st.experimental_rerun()
                
        with col5:
            if st.button("Last ⏭", disabled=st.session_state.page_number >= total_pages - 1, key="btn_last"):
                st.session_state.page_number = total_pages - 1
                st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Page number input
        page_input = st.number_input(
            "Go to page", 
            min_value=1, 
            max_value=total_pages, 
            value=st.session_state.page_number + 1,
            step=1,
            key="page_input"
        )
        
        if st.button("Go", key="go_to_page"):
            st.session_state.page_number = page_input - 1
            st.experimental_rerun()

def file_upload_section(analyzer, config):
    """Create file upload section

    Args:
        analyzer: The analyzer function
        config: Current configuration
    """
    st.header("File Analysis")

    uploaded_file = st.file_uploader(
        "Upload file for toxicity analysis", 
        type=["txt", "csv"], 
        help="Upload a text or CSV file to analyze its content for toxicity"
    )

    # If file is uploaded, process it
    if uploaded_file:
        # Check session state to prevent reprocessing on rerun
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if "processed_file_key" not in st.session_state or st.session_state.processed_file_key != file_key:
            # Process the file
            with st.spinner("Analyzing file..."):
                results = process_uploaded_file(uploaded_file, analyzer, config)
                
                # Save results to session state
                st.session_state.file_results = results
                st.session_state.processed_file_key = file_key
        
        # Display results
        display_file_results_summary(st.session_state.file_results)
        
        # Option to download results as JSON
        results_json = json.dumps(st.session_state.file_results, indent=2)
        st.download_button(
            "Download Results as JSON",
            results_json,
            f"toxicity_analysis_{uploaded_file.name}.json",
            "application/json",
            key="download_results"
        )
    else:
        # Display upload area with guide
        st.markdown("""
        <div class="file-upload-container">
            <h3>Upload a File for Batch Analysis</h3>
            <p>Supported file formats:</p>
            <ul>
                <li><strong>TXT</strong>: Plain text files with one entry per line</li>
                <li><strong>CSV</strong>: CSV files with a text/content column</li>
            </ul>
            <p>The analysis will process the file line by line and provide comprehensive toxicity statistics
            with interactive visualizations and detailed reports.</p>
        </div>
        """, unsafe_allow_html=True)

    # Show the logging panel if enabled
    create_logging_panel()

def main():
    # Initialize the localStorage component
    component_id = create_localstorage_component()

    # Initialize Streamlit communication listener
    listener_code = """
    <script>
    // Listen for messages from the component
    window.addEventListener('message', function(event) {
        if (event.data && event.data.type && event.data.type.startsWith('profile')) {
            // Send the message to Streamlit
            const data = event.data;
            Streamlit.setComponentValue(data);
        }
    });
    </script>
    """

    # Create the listener component
    message = st.components.v1.html(listener_code, height=0)

    # Log application startup
    logger.info("Toxicity Detector Streamlit app started")

    # Page header
    st.markdown("""
    <h1 class="main-header">
    🛡️ Toxicity Detector
    </h1>
    """, unsafe_allow_html=True)

    # Create sidebar controls and get updated config
    updated_config = create_sidebar_controls(component_id)

    # Initialize or update the analyzer with the current config
    analyzer = load_analyzer(config=updated_config)

    # Create tabs for single text analysis and file upload
    single_tab, batch_tab = st.tabs(["Single Text Analysis", "Batch File Analysis"])

    with single_tab:
        # Text input for analysis
        text_input = st.text_area(
            "Enter text to analyze for toxicity:", 
            height=100, 
            key="text_input",
            placeholder="Type a sentence here to analyze it for toxicity..."
        )
        
        # Only analyze if there's text
        if text_input:
            # Analyze the text with logging
            result = analyze_text_with_logging(
                analyzer, 
                text_input,
                log_individual_categories=st.session_state.get("log_detailed", False)
            )
            
            # Display the results
            col1, col2 = st.columns([2, 3])
            
            with col1:
                # Use the enhanced animated result component
                animated_result(
                    is_toxic=result["is_toxic"],
                    category=result["category"],
                    confidence=result["confidence"]
                )
                
                # Show if Groq was used as fallback
                if result.get("source") == "groq":
                    status_message("⚡ Analysis provided by Groq AI")
            
            with col2:
                # Create a dataframe for the probability chart
                probs = result["probabilities"]
                categories = list(probs.keys())
                probabilities = list(probs.values())
                
                # Get thresholds for each category
                thresholds = []
                for cat in categories:
                    threshold = updated_config.get("categories", {}).get(cat, updated_config["model"].get("threshold", 0.6))
                    thresholds.append(threshold)
                
                # Sort by probability (descending)
                sorted_indices = np.argsort(probabilities)[::-1]
                categories = [categories[i] for i in sorted_indices]
                probabilities = [probabilities[i] for i in sorted_indices]
                thresholds = [thresholds[i] for i in sorted_indices]
                
                df = pd.DataFrame({
                    'Category': [get_category_display_name(c) for c in categories],
                    'Probability': probabilities,
                    'Threshold': thresholds
                })
                
                # Create the chart
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X('Probability:Q', scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y('Category:N', sort=None),
                    color=alt.Color('Probability:Q', 
                                    scale=alt.Scale(domain=[0, 0.4, 0.6, 1], 
                                                   range=['#3498db', '#f39c12', '#e74c3c', '#c0392b'])),
                    tooltip=['Category', alt.Tooltip('Probability', format='.3f'), alt.Tooltip('Threshold', format='.2f')]
                ).properties(
                    title='Toxicity Probabilities',
                    height=250
                ).interactive()
                
                # Add threshold markers for each category
                threshold_data = pd.DataFrame({
                    'Category': df['Category'],
                    'Threshold': df['Threshold']
                })
                
                threshold_marks = alt.Chart(threshold_data).mark_rule(
                    color='black',
                    strokeDash=[3, 3],
                    opacity=0.7
                ).encode(
                    x='Threshold:Q',
                    y='Category:N'
                )
                
                # Display the chart
                st.altair_chart(chart + threshold_marks, use_container_width=True)
                
                # Add a small legend explaining the threshold markers
                st.markdown("**Note:** Dashed lines indicate category-specific thresholds")
            
            # Category descriptions section
            if result["is_toxic"]:
                with st.expander("Category Description", expanded=True):
                    st.markdown(f"### {get_category_display_name(result['category'])}")
                    st.write(get_description(result["category"]))
                    
                    if "toxic_categories" in result and len(result["toxic_categories"]) > 1:
                        st.markdown("### Other Detected Categories")
                        for cat in result["toxic_categories"][1:]:  # Skip the first (main) category
                            st.markdown(f"**{get_category_display_name(cat)}**: {get_description(cat)}")
            
            # Technical details
            with st.expander("Technical Details", expanded=False):
                st.json(result)

    with batch_tab:
        # File upload section
        file_upload_section(analyzer, updated_config)

if __name__ == "__main__":
    main() 