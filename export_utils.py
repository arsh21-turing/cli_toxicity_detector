# export_utils.py
import os
import json
import csv
import shutil 
import tempfile
import zipfile
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
import streamlit as st

# Optional PDF report generation if dependencies are available
try:
    import fpdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Optional HTML report generation with Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def export_to_json(data: Dict, output_path: str) -> str:
    """
    Export data to a formatted JSON file.
    
    Args:
        data: Data structure to export
        output_path: Path to save the JSON file
        
    Returns:
        Path to the created JSON file
    """
    # Handle datetime serialization
    def json_serial(obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=json_serial)
    
    return output_path


def export_to_csv(dataframe: pd.DataFrame, output_path: str) -> str:
    """
    Export pandas DataFrame to CSV file.
    
    Args:
        dataframe: DataFrame to export
        output_path: Path to save the CSV file
        
    Returns:
        Path to the created CSV file
    """
    dataframe.to_csv(output_path, index=False)
    return output_path


def export_to_pdf(content: Dict, output_path: str) -> str:
    """
    Create a formatted PDF report.
    
    Args:
        content: Dictionary with report content sections
        output_path: Path to save the PDF file
        
    Returns:
        Path to the created PDF file
    """
    if not PDF_AVAILABLE:
        raise ImportError("fpdf package is required for PDF export")
    
    # Initialize PDF
    class PDF(fpdf.FPDF):
        def header(self):
            # Logo and title
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Toxicity Detection System - Comprehensive Report', 0, 1, 'C')
            self.ln(5)
            
            # Timestamp
            self.set_font('Arial', 'I', 10)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cell(0, 10, f'Generated: {timestamp}', 0, 1, 'C')
            self.ln(10)
        
        def footer(self):
            # Page numbers
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    # Create PDF instance
    pdf = PDF()
    pdf.add_page()
    
    # Add sections
    for section_title, section_content in content.items():
        # Section header
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, section_title, 0, 1, 'L')
        pdf.ln(2)
        
        # Section content
        pdf.set_font('Arial', '', 10)
        if isinstance(section_content, str):
            # Simple text content
            pdf.multi_cell(0, 5, section_content)
        elif isinstance(section_content, list):
            # List of items
            for item in section_content:
                pdf.cell(10, 5, '•', 0, 0)
                pdf.multi_cell(0, 5, str(item))
        elif isinstance(section_content, dict):
            # Key-value pairs
            for key, value in section_content.items():
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(40, 5, str(key), 0, 0)
                pdf.set_font('Arial', '', 10)
                pdf.multi_cell(0, 5, str(value))
        
        # Add some spacing between sections
        pdf.ln(5)
    
    # Save the PDF
    pdf.output(output_path)
    return output_path


def export_to_html(content: Dict, output_path: str, visualizations: List[str] = None) -> str:
    """
    Create an interactive HTML report.
    
    Args:
        content: Dictionary with report content sections
        output_path: Path to save the HTML file
        visualizations: List of paths to visualization files to embed
        
    Returns:
        Path to the created HTML file
    """
    # Format current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # CSS styles
    css = """
<style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
    .container { max-width: 1200px; margin: 0 auto; }
    .header { text-align: center; margin-bottom: 30px; }
    .section { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
    .section-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
    .item { margin-bottom: 10px; }
    .key { font-weight: bold; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
    th { background-color: #f2f2f2; }
    .viz-container { margin-top: 20px; text-align: center; }
</style>
"""

    # HTML header
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Toxicity Detection System - Comprehensive Report</title>
    {css}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Toxicity Detection System - Comprehensive Report</h1>
            <p>Generated: {timestamp}</p>
        </div>
"""

    # Add each section
    for section_title, section_content in content.items():
        html += f"""
        <div class="section">
            <div class="section-title">{section_title}</div>
        """
        
        if isinstance(section_content, str):
            # Simple text content
            html += f"<p>{section_content}</p>"
        elif isinstance(section_content, list):
            # List of items
            html += "<ul>"
            for item in section_content:
                html += f"<li>{item}</li>"
            html += "</ul>"
        elif isinstance(section_content, dict):
            # Key-value pairs as a table
            html += """
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            """
            for key, value in section_content.items():
                html += f"""
                <tr>
                    <td>{key}</td>
                    <td>{value}</td>
                </tr>
                """
            html += "</table>"

        html += "</div>"

    # Add visualizations if provided
    if visualizations:
        html += """
        <div class="section">
            <div class="section-title">Visualizations</div>
        """
        
        for viz_path in visualizations:
            if viz_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                # Get base64 encoding of image
                import base64
                with open(viz_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode()
                
                viz_name = os.path.basename(viz_path)
                html += f"""
                <div class="viz-container">
                    <h3>{viz_name}</h3>
                    <img src="data:image/png;base64,{encoded_string}" style="max-width: 100%;">
                </div>
                """
        
        html += "</div>"

    # Close HTML
    html += """
    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def create_report_visualizations(performance_tracker, output_dir: str) -> List[str]:
    """
    Generate visualization files for the report.

    Args:
        performance_tracker: Performance tracker instance
        output_dir: Directory to save visualizations
        
    Returns:
        List of paths to created visualization files
    """
    os.makedirs(output_dir, exist_ok=True)
    created_files = []

    # Get models
    models = list(performance_tracker.model_history.keys())
    if not models:
        return []

    # 1. Performance trends chart
    try:
        # Create a simple performance trends chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model in models:
            if model in performance_tracker.model_history:
                model_data = performance_tracker.model_history[model]
                if model_data:
                    # Get execution times
                    times = [entry['execution_time'] for entry in model_data]
                    timestamps = [entry['timestamp'] for entry in model_data]
                    
                    # Plot execution time trends
                    ax.plot(timestamps, times, label=model, marker='o', alpha=0.7)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title('Performance Trends')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        trend_path = os.path.join(output_dir, 'performance_trends.png')
        fig.savefig(trend_path, dpi=300, bbox_inches='tight')
        created_files.append(trend_path)
        plt.close(fig)
    except Exception as e:
        print(f"Error creating trends visualization: {e}")

    # 2. If there are current results, create a comparison chart
    if hasattr(st.session_state, 'last_analysis_results'):
        try:
            results = st.session_state.last_analysis_results
            if len(results) >= 2:
                # Create comparison bar chart
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Get common categories
                categories = set()
                for model, model_results in results.items():
                    categories.update(
                        k for k, v in model_results.items() 
                        if isinstance(v, (int, float))
                    )
                
                # Prepare data for plotting
                models_list = list(results.keys())
                x = list(range(len(categories)))
                width = 0.35 if len(models_list) <= 2 else 0.25
                
                # Plot each model's results
                for i, model in enumerate(models_list):
                    model_scores = [
                        results[model].get(cat, 0.0) 
                        for cat in sorted(categories)
                    ]
                    position = [pos + (i - len(models_list)/2 + 0.5) * width for pos in x]
                    ax.bar(position, model_scores, width, label=model)
                
                # Add labels and formatting
                ax.set_ylabel('Score')
                ax.set_title('Comparison of Model Scores by Category')
                ax.set_xticks(x)
                ax.set_xticklabels(sorted(categories), rotation=45, ha='right')
                ax.legend()
                
                # Add threshold line
                ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                
                # Save the chart
                comparison_path = os.path.join(output_dir, 'model_comparison.png')
                fig.savefig(comparison_path, dpi=300, bbox_inches='tight')
                created_files.append(comparison_path)
                plt.close(fig)
        except Exception as e:
            print(f"Error creating comparison visualization: {e}")

    return created_files


def generate_comprehensive_report(
    metrics_tracker,
    performance_tracker,
    monitoring_context,
    current_results: Optional[Dict] = None
) -> Tuple[str, bytes]:
    """
    Create a comprehensive report with all system data.

    Args:
        metrics_tracker: Instance of MetricsTracker
        performance_tracker: Instance of PerformanceTracker
        monitoring_context: Instance of MonitoringContext
        current_results: Optional dictionary with current analysis results
        
    Returns:
        Tuple of (filename, zip_bytes)
    """
    # Create temporary directory for report files
    temp_dir = tempfile.mkdtemp()
    try:
        # Timestamp for filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"toxicity_report_{timestamp}"
        
        # Create subdirectories
        metrics_dir = os.path.join(temp_dir, "metrics")
        performance_dir = os.path.join(temp_dir, "performance")
        system_dir = os.path.join(temp_dir, "system")
        results_dir = os.path.join(temp_dir, "results")
        viz_dir = os.path.join(temp_dir, "visualizations")
        
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(performance_dir, exist_ok=True)
        os.makedirs(system_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Export metrics data
        metrics_data = metrics_tracker.get_metrics()
        export_to_json(metrics_data, os.path.join(metrics_dir, "processing_metrics.json"))
        
        # 2. Export performance data
        models = list(performance_tracker.model_history.keys())
        
        # Overall performance metrics
        performance_overview = {}
        for model in models:
            metrics = performance_tracker.get_performance_metrics(model)
            performance_overview[model] = metrics
        
        export_to_json(performance_overview, 
                      os.path.join(performance_dir, "performance_metrics.json"))
        
        # Detailed performance history
        export_to_json(
            {"model_history": performance_tracker.model_history},
            os.path.join(performance_dir, "performance_history.json")
        )
        
        # Comparative metrics if multiple models
        if len(models) >= 2:
            comparative = performance_tracker.get_comparative_metrics()
            export_to_json(comparative, 
                          os.path.join(performance_dir, "comparative_metrics.json"))
        
        # 3. Export system health data
        system_metrics = monitoring_context.get_system_metrics()
        export_to_json(system_metrics, os.path.join(system_dir, "system_health.json"))
        
        # Create a readable summary of system health
        system_summary = {
            "CPU Usage": f"{system_metrics['cpu']['usage_percent']:.1f}%",
            "Memory Usage": f"{system_metrics['memory']['percent']:.1f}%",
            "Process Memory": f"{system_metrics['process']['memory_percent']:.1f}%",
            "Process CPU": f"{system_metrics['process']['cpu_percent']:.1f}%",
            "Elapsed Time": system_metrics['elapsed_formatted'],
            "Timestamp": system_metrics['timestamp'],
        }
        
        export_to_json(system_summary, os.path.join(system_dir, "system_summary.json"))
        
        # 4. Export current analysis results if available
        if current_results:
            export_to_json(current_results, os.path.join(results_dir, "current_analysis.json"))
            
            # Also save in session state for potential visualizations
            st.session_state.last_analysis_results = current_results
        
        # 5. Create visualizations
        viz_files = create_report_visualizations(performance_tracker, viz_dir)
        
        # 6. Generate CSV exports for tabular data
        # Create performance summary CSV
        performance_summary = []
        for model in models:
            metrics = performance_tracker.get_performance_metrics(model)
            if "accuracy" in metrics and metrics["accuracy"]:
                for category, cat_metrics in metrics["accuracy"].items():
                    performance_summary.append({
                        "Model": model,
                        "Category": category,
                        "Accuracy": cat_metrics['accuracy'],
                        "Precision": cat_metrics['precision'],
                        "Recall": cat_metrics['recall'],
                        "F1": cat_metrics['f1'],
                        "Samples": cat_metrics['samples']
                    })
        
        if performance_summary:
            perf_df = pd.DataFrame(performance_summary)
            export_to_csv(perf_df, os.path.join(performance_dir, "performance_summary.csv"))
        
        # 7. Create a PDF summary report
        if PDF_AVAILABLE:
            # Prepare content for PDF
            pdf_content = {
                "System Summary": system_summary,
                "Processing Metrics": {
                    "Processed Items": metrics_data['processed_count'],
                    "API Calls": metrics_data['api_calls'],
                    "Throughput": f"{metrics_data['throughput']:.2f} items/s",
                    "Memory Usage": f"{metrics_data['memory_usage']:.2f} MB",
                    "Elapsed Time": metrics_data['elapsed_formatted']
                },
                "Performance Overview": {
                    model: {
                        "Avg Execution Time": f"{metrics['execution_time']['mean']:.3f}s" 
                        if 'execution_time' in metrics else "N/A",
                        "Sample Count": metrics['sample_count'],
                        "Has Ground Truth": metrics['has_ground_truth']
                    }
                    for model, metrics in performance_overview.items()
                }
            }
            
            # Add current results if available
            if current_results:
                pdf_content["Current Analysis Results"] = {
                    model: ", ".join([
                        f"{k}: {v:.3f}" for k, v in results.items() 
                        if isinstance(v, (int, float))
                    ])
                    for model, results in current_results.items()
                }
            
            # Generate PDF
            export_to_pdf(pdf_content, os.path.join(temp_dir, f"{report_name}_summary.pdf"))
        
        # 8. Create an HTML report with interactive elements
        if PLOTLY_AVAILABLE:
            # Prepare content for HTML
            html_content = {
                "System Summary": system_summary,
                "Processing Metrics": {
                    "Processed Items": metrics_data['processed_count'],
                    "API Calls": metrics_data['api_calls'],
                    "Throughput": f"{metrics_data['throughput']:.2f} items/s",
                    "Memory Usage": f"{metrics_data['memory_usage']:.2f} MB",
                    "Elapsed Time": metrics_data['elapsed_formatted']
                },
                "Performance Overview": {
                    model: {
                        "Avg Execution Time": f"{metrics['execution_time']['mean']:.3f}s" 
                        if 'execution_time' in metrics else "N/A",
                        "Sample Count": metrics['sample_count'],
                        "Has Ground Truth": metrics['has_ground_truth']
                    }
                    for model, metrics in performance_overview.items()
                }
            }
            
            # Generate HTML
            export_to_html(
                html_content, 
                os.path.join(temp_dir, f"{report_name}_report.html"),
                visualizations=viz_files
            )
        
        # 9. Create zip file with all exports
        zip_path = os.path.join(tempfile.gettempdir(), f"{report_name}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
        
        # Read zip file as bytes for streaming
        with open(zip_path, 'rb') as f:
            zip_bytes = f.read()
        
        return f"{report_name}.zip", zip_bytes

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir) 