import time
import psutil
import streamlit as st
from typing import Dict, Any, Optional
import datetime

class MetricsTracker:
    """
    Tracks processing metrics during batch operations.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics tracking to initial state."""
        self.start_time = time.time()
        self.processed_count = 0
        self.api_calls = 0
        self.memory_usage = 0
        self.error_count = 0
        self.update_memory_usage()

    def increment_processed(self, count: int = 1) -> int:
        """
        Increment the processed items counter.
        
        Args:
            count: Number of items to increment by (default: 1)
            
        Returns:
            Current processed count after incrementing
        """
        self.processed_count += count
        return self.processed_count

    def increment_api_calls(self, count: int = 1) -> int:
        """
        Increment the API calls counter.
        
        Args:
            count: Number of calls to increment by (default: 1)
            
        Returns:
            Current API call count after incrementing
        """
        self.api_calls += count
        return self.api_calls

    def increment_errors(self, count: int = 1) -> int:
        """
        Increment the error counter.
        
        Args:
            count: Number of errors to increment by (default: 1)
            
        Returns:
            Current error count after incrementing
        """
        self.error_count += count
        return self.error_count

    def update_memory_usage(self) -> float:
        """
        Update the current memory usage value.
        
        Returns:
            Current memory usage in MB
        """
        try:
            process = psutil.Process()
            self.memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except Exception:
            self.memory_usage = 0.0
        return self.memory_usage

    def get_throughput(self) -> float:
        """
        Calculate the current throughput (items per second).
        
        Returns:
            Items processed per second
        """
        elapsed_time = max(time.time() - self.start_time, 0.001)  # Avoid division by zero
        return self.processed_count / elapsed_time

    def get_error_rate(self) -> float:
        """
        Calculate the current error rate.
        
        Returns:
            Error rate as a percentage
        """
        total_processed = max(self.processed_count, 1)  # Avoid division by zero
        return (self.error_count / total_processed) * 100

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all current metrics as a dictionary.
        
        Returns:
            Dictionary with all metric values
        """
        self.update_memory_usage()
        elapsed_time = time.time() - self.start_time
        
        # Format elapsed time as HH:MM:SS
        elapsed_formatted = str(datetime.timedelta(seconds=int(elapsed_time)))
        
        return {
            "processed_count": self.processed_count,
            "api_calls": self.api_calls,
            "error_count": self.error_count,
            "memory_usage": self.memory_usage,
            "throughput": self.get_throughput(),
            "error_rate": self.get_error_rate(),
            "elapsed_time": elapsed_time,
            "elapsed_formatted": elapsed_formatted
        }


def create_metrics_dashboard(metrics_tracker: MetricsTracker, refresh_interval: float = 2):
    """
    Create a real-time metrics dashboard in the Streamlit sidebar.
    
    Args:
        metrics_tracker: Instance of MetricsTracker to display metrics from
        refresh_interval: Time between dashboard updates in seconds (default: 2)
        
    Returns:
        Dictionary of placeholder elements used for the dashboard
    """
    # Add CSS for better styling
    st.markdown("""
    <style>
    .metrics-dashboard {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .metric-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        border-bottom: 1px solid #e9ecef;
    }
    .metric-item:last-child {
        border-bottom: none;
    }
    .metric-label {
        font-weight: 600;
        color: #495057;
    }
    .metric-value {
        font-family: monospace;
        font-weight: 500;
    }
    .error-high { color: #dc3545; }
    .error-medium { color: #ffc107; }
    .error-low { color: #28a745; }
    </style>
    """, unsafe_allow_html=True)
    
    # Dashboard header
    st.sidebar.markdown("## 📊 Processing Metrics")
    st.sidebar.markdown(f"*Auto-refresh every {refresh_interval}s*")
    
    # Get current metrics
    metrics = metrics_tracker.get_metrics()
    
    # Create metrics display
    with st.sidebar.container():
        st.markdown('<div class="metrics-dashboard">', unsafe_allow_html=True)
        
        # Time elapsed
        st.markdown(f"""
        <div class="metric-item">
            <span class="metric-label">⏱️ Time Elapsed:</span>
            <span class="metric-value">{metrics['elapsed_formatted']}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Throughput
        st.markdown(f"""
        <div class="metric-item">
            <span class="metric-label">⚡ Throughput:</span>
            <span class="metric-value">{metrics['throughput']:.2f} items/sec</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Processed count
        st.markdown(f"""
        <div class="metric-item">
            <span class="metric-label">📊 Processed:</span>
            <span class="metric-value">{metrics['processed_count']:,}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # API calls
        st.markdown(f"""
        <div class="metric-item">
            <span class="metric-label">🔄 API Calls:</span>
            <span class="metric-value">{metrics['api_calls']:,}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Memory usage
        st.markdown(f"""
        <div class="metric-item">
            <span class="metric-label">💾 Memory:</span>
            <span class="metric-value">{metrics['memory_usage']:.1f} MB</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Error rate with color coding
        error_rate = metrics['error_rate']
        if error_rate > 5:
            error_class = "error-high"
            error_icon = "🔴"
        elif error_rate > 1:
            error_class = "error-medium"
            error_icon = "🟡"
        else:
            error_class = "error-low"
            error_icon = "🟢"
        
        st.markdown(f"""
        <div class="metric-item">
            <span class="metric-label">{error_icon} Error Rate:</span>
            <span class="metric-value {error_class}">{error_rate:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add refresh button
    if st.sidebar.button("🔄 Refresh Metrics Now", key="refresh_metrics"):
        st.rerun()
    
    # Add auto-refresh using session state
    if "last_metrics_update" not in st.session_state:
        st.session_state.last_metrics_update = time.time()
    
    # Check if it's time to auto-refresh
    current_time = time.time()
    if current_time - st.session_state.last_metrics_update >= refresh_interval:
        st.session_state.last_metrics_update = current_time
        # Use st.empty() to trigger a rerun without showing anything
        placeholder = st.empty()
        placeholder.markdown("")
    
    return metrics 