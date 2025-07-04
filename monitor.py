#!/usr/bin/env python3
"""
Real-time performance monitoring for batch processing.

This module provides a live dashboard to track metrics such as throughput,
memory usage, API call counts, confidence distribution, and error rates
during batch text processing.
"""

import os
import time
import threading
import logging
import json
import psutil
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# For terminal-based dashboard
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, BarColumn, TextColumn
from rich.text import Text

# Set up logging
logger = logging.getLogger(__name__)


class MonitoringContext:
    """Thread-safe context for monitoring metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the monitoring context.
        
        Args:
            config: Dictionary with monitoring configuration
        """
        self.config = config
        self.metrics = {
            "start_time": time.time(),
            "last_update_time": time.time(),
            "processed_texts": 0,
            "processed_bytes": 0,
            "processed_files": 0,
            "toxic_texts": 0,
            "error_count": 0,
            "api_calls": 0,
            "api_errors": 0,
            "confidence_buckets": {
                "very_low": 0,    # 0.0 - 0.2
                "low": 0,         # 0.2 - 0.4  
                "medium": 0,      # 0.4 - 0.6
                "high": 0,        # 0.6 - 0.8
                "very_high": 0    # 0.8 - 1.0
            },
            "category_counts": {},
            "memory_usage": 0,
            "cpu_usage": 0,
            "throughput": 0.0,
            "avg_latency": 0.0,
            "total_latency": 0.0,
            "error_rate": 0.0,
            "system": {}
        }
        self.lock = threading.RLock()
        self.running = True
        self.dashboard = None
        self.metrics_log = []
        self.log_path = config.get("log_path")
    
    def update(self, new_metrics: Dict[str, Any]) -> None:
        """
        Update metrics in a thread-safe manner.
        
        Args:
            new_metrics: Dictionary of metrics to update
        """
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.metrics["last_update_time"]
            
            # Update basic counters
            if "processed_texts" in new_metrics:
                self.metrics["processed_texts"] += new_metrics["processed_texts"]
                
                # Calculate throughput if enough time has passed
                if elapsed >= 1.0:
                    self.metrics["throughput"] = new_metrics["processed_texts"] / elapsed
            
            if "processed_bytes" in new_metrics:
                self.metrics["processed_bytes"] += new_metrics["processed_bytes"]
            
            if "processed_files" in new_metrics:
                self.metrics["processed_files"] += new_metrics["processed_files"]
            
            if "toxic_texts" in new_metrics:
                self.metrics["toxic_texts"] += new_metrics["toxic_texts"]
            
            if "error_count" in new_metrics:
                self.metrics["error_count"] += new_metrics["error_count"]
                
                # Update error rate
                total_texts = max(1, self.metrics["processed_texts"])
                self.metrics["error_rate"] = self.metrics["error_count"] / total_texts
            
            if "api_calls" in new_metrics:
                self.metrics["api_calls"] += new_metrics["api_calls"]
            
            if "api_errors" in new_metrics:
                self.metrics["api_errors"] += new_metrics["api_errors"]
            
            # Update confidence distribution - handle both single value and list of values
            if "confidence_value" in new_metrics:
                # Single confidence value
                confidence = new_metrics["confidence_value"]
                self._update_confidence_bucket(confidence)
            
            if "confidence_values" in new_metrics:
                # List of confidence values
                confidence_values = new_metrics["confidence_values"]
                for confidence in confidence_values:
                    self._update_confidence_bucket(confidence)
            
            # Update latency tracking
            if "latency" in new_metrics:
                latency = new_metrics["latency"]
                self.metrics["total_latency"] += latency
                total_texts = max(1, self.metrics["processed_texts"])
                self.metrics["avg_latency"] = self.metrics["total_latency"] / total_texts
            
            # Update category counts
            if "categories" in new_metrics:
                categories = new_metrics["categories"]
                for category in categories:
                    if category not in self.metrics["category_counts"]:
                        self.metrics["category_counts"][category] = 0
                    self.metrics["category_counts"][category] += 1
            
            # Update memory usage if provided
            if "memory_usage" in new_metrics:
                self.metrics["memory_usage"] = new_metrics["memory_usage"]
            
            # Update CPU usage if provided
            if "cpu_usage" in new_metrics:
                self.metrics["cpu_usage"] = new_metrics["cpu_usage"]
            
            # Update system metrics if provided
            if "system" in new_metrics:
                self.metrics["system"].update(new_metrics["system"])
            
            # Update last update time
            self.metrics["last_update_time"] = current_time
            
            # Log metrics if requested
            if self.log_path:
                self.metrics_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "metrics": self.metrics.copy()
                })
    
    def _update_confidence_bucket(self, confidence: float) -> None:
        """
        Update the appropriate confidence bucket based on the confidence value.
        
        Args:
            confidence: Confidence score (0.0 to 1.0)
        """
        if not isinstance(confidence, (int, float)):
            return
            
        if confidence < 0.2:
            self.metrics["confidence_buckets"]["very_low"] += 1
        elif confidence < 0.4:
            self.metrics["confidence_buckets"]["low"] += 1
        elif confidence < 0.6:
            self.metrics["confidence_buckets"]["medium"] += 1
        elif confidence < 0.8:
            self.metrics["confidence_buckets"]["high"] += 1
        else:
            self.metrics["confidence_buckets"]["very_high"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics snapshot.
        
        Returns:
            Dictionary with current metrics
        """
        with self.lock:
            return self.metrics.copy()
    
    def stop(self) -> Dict[str, Any]:
        """
        Stop monitoring and generate summary.
        
        Returns:
            Dictionary with summary metrics
        """
        with self.lock:
            self.running = False
            
            # Calculate final metrics
            end_time = time.time()
            total_elapsed = end_time - self.metrics["start_time"]
            
            summary = {
                "start_time": datetime.fromtimestamp(self.metrics["start_time"]).isoformat(),
                "end_time": datetime.fromtimestamp(end_time).isoformat(),
                "elapsed_seconds": total_elapsed,
                "processed_texts": self.metrics["processed_texts"],
                "processed_bytes": self.metrics["processed_bytes"],
                "processed_files": self.metrics["processed_files"],
                "toxic_texts": self.metrics["toxic_texts"],
                "toxic_percentage": (
                    self.metrics["toxic_texts"] / max(1, self.metrics["processed_texts"]) * 100
                ),
                "avg_throughput": (
                    self.metrics["processed_texts"] / max(1, total_elapsed)
                ),
                "avg_latency": self.metrics["avg_latency"],
                "error_count": self.metrics["error_count"],
                "error_rate": self.metrics["error_rate"],
                "api_calls": self.metrics["api_calls"],
                "api_errors": self.metrics["api_errors"],
                "confidence_distribution": self.metrics["confidence_buckets"],
                "category_distribution": self.metrics["category_counts"]
            }
            
            # Save metrics log if requested
            if self.log_path and self.metrics_log:
                try:
                    log_dir = Path(self.log_path).parent
                    log_dir.mkdir(parents=True, exist_ok=True)
                    
                    with open(self.log_path, 'w') as f:
                        json.dump({
                            "summary": summary,
                            "log": self.metrics_log
                        }, f, indent=2)
                    
                    logger.info(f"Monitoring log saved to: {self.log_path}")
                except Exception as e:
                    logger.error(f"Error saving monitoring log: {str(e)}")
            
            return summary


def get_system_metrics(process_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Get system and process metrics.
    
    Args:
        process_id: Optional process ID to monitor specifically
        
    Returns:
        Dictionary with system metrics
    """
    metrics = {
        "cpu": {
            "total": psutil.cpu_percent(interval=0.1),
            "cores": psutil.cpu_count()
        },
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent_used": psutil.virtual_memory().percent
        },
        "disk": {
            "read_bytes": psutil.disk_io_counters().read_bytes if psutil.disk_io_counters() else 0,
            "write_bytes": psutil.disk_io_counters().write_bytes if psutil.disk_io_counters() else 0
        }
    }
    
    # Add process-specific metrics if a PID is provided
    if process_id:
        try:
            process = psutil.Process(process_id)
            metrics["process"] = {
                "cpu_percent": process.cpu_percent(interval=0.1),
                "memory_percent": process.memory_percent(),
                "memory_bytes": process.memory_info().rss,
                "threads": len(process.threads()),
                "open_files": len(process.open_files())
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Error getting process metrics: {str(e)}")
            metrics["process"] = {"error": str(e)}
    
    return metrics


def create_dashboard(monitoring_context: MonitoringContext) -> Live:
    """
    Create the monitoring dashboard UI.
    
    Args:
        monitoring_context: The monitoring context
        
    Returns:
        Rich Live object for the dashboard
    """
    console = Console()
    
    def render_dashboard() -> Layout:
        """Render the dashboard layout with current metrics."""
        metrics = monitoring_context.get_metrics()
        
        # Create layout
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )
        
        # Set up the main section with columns
        layout["main"].split_row(
            Layout(name="stats", ratio=2),
            Layout(name="graphs", ratio=3),
        )
        
        # Split the graphs section
        layout["graphs"].split(
            Layout(name="throughput", ratio=1),
            Layout(name="distribution", ratio=1),
        )
        
        # Header content
        elapsed_time = time.time() - metrics["start_time"]
        header = Panel(
            Text(f"Toxicity Classification Monitor - Running for {format_time(elapsed_time)}", 
                 style="bold white", justify="center"),
            style="blue"
        )
        layout["header"].update(header)
        
        # Main stats table
        stats_table = Table(title="Processing Statistics", show_header=True, pad_edge=False)
        
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Processed Texts", str(metrics["processed_texts"]))
        stats_table.add_row("Toxic Content Found", f"{metrics['toxic_texts']} ({metrics['toxic_texts'] / max(1, metrics['processed_texts']) * 100:.1f}%)")
        stats_table.add_row("Files Processed", str(metrics["processed_files"]))
        stats_table.add_row("API Calls", f"{metrics['api_calls']} ({metrics['api_errors']} errors)")
        stats_table.add_row("Current Throughput", f"{metrics['throughput']:.2f} texts/sec")
        stats_table.add_row("Average Latency", f"{metrics['avg_latency'] * 1000:.2f} ms")
        stats_table.add_row("Error Rate", f"{metrics['error_rate'] * 100:.2f}%")
        
        if "process" in metrics["system"]:
            stats_table.add_row("Process Memory", f"{metrics['system']['process']['memory_bytes'] / (1024**2):.1f} MB ({metrics['system']['process']['memory_percent']:.1f}%)")
            stats_table.add_row("Process CPU", f"{metrics['system']['process']['cpu_percent']:.1f}%")
        
        layout["stats"].update(stats_table)
        
        # Throughput graph
        throughput_progress = Progress(
            TextColumn("[bold blue]Throughput:"),
            BarColumn(),
            TextColumn("[bold green]{task.fields[throughput]:.2f} texts/sec")
        )
        
        # Scale throughput to a reasonable bar size (assuming 100 is max)
        throughput_value = min(100, metrics["throughput"] * 10)
        task_id = throughput_progress.add_task("", total=100, throughput=metrics["throughput"], completed=throughput_value)
        
        # Memory and CPU usage
        memory_progress = Progress(
            TextColumn("[bold blue]Memory Usage:"),
            BarColumn(bar_width=None),
            TextColumn("[bold green]{task.percentage:.1f}%")
        )
        
        memory_value = metrics["system"].get("memory", {}).get("percent_used", 0)
        memory_task_id = memory_progress.add_task("", total=100, completed=memory_value)
        
        cpu_progress = Progress(
            TextColumn("[bold blue]CPU Usage:    "),
            BarColumn(bar_width=None),
            TextColumn("[bold green]{task.percentage:.1f}%")
        )
        
        cpu_value = metrics["system"].get("cpu", {}).get("total", 0)
        cpu_task_id = cpu_progress.add_task("", total=100, completed=cpu_value)
        
        # Combine progress bars
        layout["throughput"].update(Panel(
            throughput_progress,
            title="Performance Metrics",
            border_style="green",
        ))
        
        # Confidence distribution
        conf_buckets = metrics["confidence_buckets"]
        total_preds = sum(conf_buckets.values())
        
        conf_progress = Table(title="Confidence Distribution", show_header=True)
        conf_progress.add_column("Confidence", style="cyan")
        conf_progress.add_column("Count", style="green")
        conf_progress.add_column("Percentage", style="yellow")
        
        for name, count in conf_buckets.items():
            percentage = (count / max(1, total_preds)) * 100
            conf_progress.add_row(
                name.replace("_", " ").title(),
                str(count),
                f"{percentage:.1f}%"
            )
        
        # Category distribution
        category_counts = metrics["category_counts"]
        
        if category_counts:
            cat_table = Table(title="Category Distribution", show_header=True)
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Count", style="green")
            
            for name, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                cat_table.add_row(name, str(count))
            
            # Combine tables
            distribution_panel = Panel(
                Layout().split(
                    Layout(conf_progress, ratio=1),
                    Layout(cat_table, ratio=1)
                ),
                title="Distribution Analysis",
                border_style="yellow"
            )
        else:
            distribution_panel = Panel(
                conf_progress,
                title="Distribution Analysis",
                border_style="yellow"
            )
        
        layout["distribution"].update(distribution_panel)
        
        # Footer with system info
        system_info = f"CPU: {cpu_value:.1f}% | Memory: {memory_value:.1f}% | Disk I/O: R {format_bytes(metrics['system'].get('disk', {}).get('read_bytes', 0))}, W {format_bytes(metrics['system'].get('disk', {}).get('write_bytes', 0))}"
        footer = Panel(
            Text(system_info, justify="center"),
            style="blue"
        )
        layout["footer"].update(footer)
        
        return layout
    
    # Create a Live display
    live = Live(render_dashboard(), console=console, refresh_per_second=4, screen=True)
    monitoring_context.dashboard = live
    
    return live


def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.1f} PB"


def start_monitoring(
    config: Dict[str, Any] = None
) -> MonitoringContext:
    """
    Start the monitoring system.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Monitoring context for metric updates
    """
    if config is None:
        config = {}
    
    # Create monitoring context
    monitoring_context = MonitoringContext(config)
    
    # Start system metrics collection thread
    def collect_system_metrics():
        process_id = os.getpid()
        
        while monitoring_context.running:
            try:
                system_metrics = get_system_metrics(process_id)
                monitoring_context.update({"system": system_metrics})
            except Exception as e:
                logger.error(f"Error collecting system metrics: {str(e)}")
            
            time.sleep(1)  # Update once per second
    
    # Start the system metrics thread
    metrics_thread = threading.Thread(target=collect_system_metrics, daemon=True)
    metrics_thread.start()
    
    # Create and start dashboard unless in headless mode
    if not config.get("headless", False):
        dashboard = create_dashboard(monitoring_context)
        
        # Start dashboard in a separate thread
        def run_dashboard():
            with dashboard:
                while monitoring_context.running:
                    time.sleep(0.1)  # Small sleep to prevent high CPU usage
        
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
    
    return monitoring_context


def update_metrics(
    monitoring_context: MonitoringContext,
    new_metrics: Dict[str, Any]
) -> None:
    """
    Update monitoring metrics.
    
    Args:
        monitoring_context: The monitoring context from start_monitoring
        new_metrics: New metrics to add to the monitoring
    """
    monitoring_context.update(new_metrics)


def stop_monitoring(
    monitoring_context: MonitoringContext
) -> Dict[str, Any]:
    """
    Stop monitoring and get summary.
    
    Args:
        monitoring_context: The monitoring context
        
    Returns:
        Summary of monitoring metrics
    """
    return monitoring_context.stop()


def render_dashboard(
    monitoring_context: MonitoringContext
) -> None:
    """
    Force a dashboard render refresh.
    
    Args:
        monitoring_context: The monitoring context
    """
    if monitoring_context.dashboard:
        monitoring_context.dashboard.refresh()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor System Performance")
    parser.add_argument("--log", help="Path to save monitoring log")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no UI)")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Start monitoring
    config = {
        "log_path": args.log,
        "headless": args.headless
    }
    
    monitoring = start_monitoring(config)
    
    try:
        # Simulated processing loop
        for i in range(100):
            # Update metrics with simulated processing data
            update_metrics(monitoring, {
                "processed_texts": 10,
                "processed_bytes": 1024 * 5,
                "toxic_texts": 2,
                "latency": 0.05,
                "confidence_values": [np.random.random() for _ in range(10)],
                "categories": ["hate", "harassment"] if i % 5 == 0 else []
            })
            
            # Simulate processing time
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("Monitoring stopped by user")
    
    finally:
        # Stop monitoring and get summary
        summary = stop_monitoring(monitoring)
        print("\nMonitoring summary:")
        print(json.dumps(summary, indent=2)) 