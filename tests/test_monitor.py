#!/usr/bin/env python3
"""
Tests for performance monitoring dashboard.
"""

import os
import sys
import time
import json
import pytest
import tempfile
from unittest import mock
from pathlib import Path

# Add parent directory to Python path so we can import monitor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor import (
    MonitoringContext, start_monitoring, update_metrics, 
    stop_monitoring, get_system_metrics
)


class TestMonitor:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_config = {
            "update_interval": 0.1,
            "headless": True,  # No UI for testing
        }
    
    def test_update_confidence_values_list(self):
        """Test that confidence_values list correctly increments confidence buckets."""
        # Initialize monitoring context
        context = MonitoringContext(self.test_config)
        
        # Initial bucket counts should be zero
        for bucket in context.metrics["confidence_buckets"].values():
            assert bucket == 0
        
        # Create a list of confidence values that will increment each bucket exactly once
        confidence_values = [
            0.1,    # very_low: < 0.2
            0.3,    # low: >= 0.2 and < 0.4
            0.5,    # medium: >= 0.4 and < 0.6
            0.7,    # high: >= 0.6 and < 0.8
            0.9     # very_high: >= 0.8
        ]
        
        # Update metrics with the confidence values list
        context.update({
            "confidence_values": confidence_values,
            "processed_texts": len(confidence_values)  # Also increment processed texts
        })
        
        # Verify each bucket was incremented exactly once
        assert context.metrics["confidence_buckets"]["very_low"] == 1
        assert context.metrics["confidence_buckets"]["low"] == 1
        assert context.metrics["confidence_buckets"]["medium"] == 1
        assert context.metrics["confidence_buckets"]["high"] == 1
        assert context.metrics["confidence_buckets"]["very_high"] == 1
        
        # Verify processed texts count was updated
        assert context.metrics["processed_texts"] == 5
    
    def test_update_confidence_values_boundary_conditions(self):
        """Test confidence bucketing with boundary values and edge cases."""
        context = MonitoringContext(self.test_config)
        
        # Test boundary conditions
        boundary_values = [
            0.0,    # very_low (lower boundary)
            0.199,  # very_low (just under next bucket)
            0.2,    # low (exact boundary)
            0.399,  # low (just under next bucket)
            0.4,    # medium (exact boundary)
            0.599,  # medium (just under next bucket)
            0.6,    # high (exact boundary)
            0.799,  # high (just under next bucket)
            0.8,    # very_high (exact boundary)
            1.0     # very_high (upper boundary)
        ]
        
        # Update with boundary values
        context.update({
            "confidence_values": boundary_values,
            "processed_texts": len(boundary_values)
        })
        
        # Very low should have 2 values: 0.0 and 0.199
        assert context.metrics["confidence_buckets"]["very_low"] == 2
        
        # Low should have 2 values: 0.2 and 0.399
        assert context.metrics["confidence_buckets"]["low"] == 2
        
        # Medium should have 2 values: 0.4 and 0.599
        assert context.metrics["confidence_buckets"]["medium"] == 2
        
        # High should have 2 values: 0.6 and 0.799
        assert context.metrics["confidence_buckets"]["high"] == 2
        
        # Very high should have 2 values: 0.8 and 1.0
        assert context.metrics["confidence_buckets"]["very_high"] == 2
    
    def test_update_confidence_values_mixed_with_single_value(self):
        """Test handling both confidence_values list and confidence_value in same update."""
        context = MonitoringContext(self.test_config)
        
        # Update with both a list and a single value
        context.update({
            "confidence_values": [0.1, 0.3, 0.5],  # should increment very_low, low, medium
            "confidence_value": 0.9,               # should increment very_high 
            "processed_texts": 4
        })
        
        # Verify buckets are correctly incremented
        assert context.metrics["confidence_buckets"]["very_low"] == 1
        assert context.metrics["confidence_buckets"]["low"] == 1
        assert context.metrics["confidence_buckets"]["medium"] == 1
        assert context.metrics["confidence_buckets"]["high"] == 0
        assert context.metrics["confidence_buckets"]["very_high"] == 1
    
    def test_update_confidence_values_cumulative(self):
        """Test that multiple updates cumulatively increment confidence buckets."""
        context = MonitoringContext(self.test_config)
        
        # First update
        context.update({
            "confidence_values": [0.1, 0.3],
            "processed_texts": 2
        })
        
        # Second update
        context.update({
            "confidence_values": [0.5, 0.7],
            "processed_texts": 2
        })
        
        # Third update
        context.update({
            "confidence_values": [0.9],
            "processed_texts": 1
        })
        
        # Verify cumulative counts
        assert context.metrics["confidence_buckets"]["very_low"] == 1
        assert context.metrics["confidence_buckets"]["low"] == 1
        assert context.metrics["confidence_buckets"]["medium"] == 1
        assert context.metrics["confidence_buckets"]["high"] == 1
        assert context.metrics["confidence_buckets"]["very_high"] == 1
        assert context.metrics["processed_texts"] == 5
    
    def test_update_confidence_values_invalid_inputs(self):
        """Test handling of invalid confidence values."""
        context = MonitoringContext(self.test_config)
        
        # Update with some invalid values mixed with valid ones
        context.update({
            "confidence_values": [
                0.5,          # valid
                "not a number", # invalid
                None,         # invalid
                1.5,          # outside range but numeric
                -0.2,         # outside range but numeric
                {},           # invalid
                0.7           # valid
            ],
            "processed_texts": 7
        })
        
        # Only valid values should be counted (and 1.5 should go in very_high, -0.2 in very_low)
        assert context.metrics["confidence_buckets"]["very_low"] == 1  # -0.2
        assert context.metrics["confidence_buckets"]["low"] == 0
        assert context.metrics["confidence_buckets"]["medium"] == 1    # 0.5
        assert context.metrics["confidence_buckets"]["high"] == 1      # 0.7
        assert context.metrics["confidence_buckets"]["very_high"] == 1 # 1.5
    
    def test_update_large_confidence_values_list(self):
        """Test handling of a large list of confidence values."""
        context = MonitoringContext(self.test_config)
        
        # Generate 1000 confidence values across the range
        import random
        random.seed(42)  # for reproducibility
        large_list = [random.random() for _ in range(1000)]
        
        # Count expected values per bucket
        expected_counts = {
            "very_low": 0,
            "low": 0,
            "medium": 0,
            "high": 0,
            "very_high": 0
        }
        
        for val in large_list:
            if val < 0.2:
                expected_counts["very_low"] += 1
            elif val < 0.4:
                expected_counts["low"] += 1
            elif val < 0.6:
                expected_counts["medium"] += 1
            elif val < 0.8:
                expected_counts["high"] += 1
            else:
                expected_counts["very_high"] += 1
        
        # Update with the large list
        context.update({
            "confidence_values": large_list,
            "processed_texts": len(large_list)
        })
        
        # Verify all buckets match expected counts
        for bucket, expected in expected_counts.items():
            assert context.metrics["confidence_buckets"][bucket] == expected, f"Bucket {bucket} should have {expected} items"
    
    def test_start_monitoring_basic_flow(self):
        """Test basic start_monitoring functionality."""
        # Mock dashboard creation to avoid UI elements in tests
        with mock.patch('monitor.create_dashboard', return_value=mock.MagicMock()):
            monitoring = start_monitoring(self.test_config)
            assert monitoring is not None
            assert isinstance(monitoring, MonitoringContext)
            
            # Verify initial metrics structure
            metrics = monitoring.get_metrics()
            assert "processed_texts" in metrics
            assert "confidence_buckets" in metrics
            assert "start_time" in metrics
            
            # Clean up
            stop_monitoring(monitoring)
    
    def test_get_system_metrics(self):
        """Test retrieval of system metrics."""
        metrics = get_system_metrics()
        
        # Verify basic structure
        assert "cpu" in metrics
        assert "memory" in metrics
        assert "disk" in metrics
        
        # Verify specific fields
        assert "total" in metrics["cpu"]
        assert "percent_used" in metrics["memory"]
        
        # Test with specific process monitoring
        process_metrics = get_system_metrics(os.getpid())
        assert "process" in process_metrics
    
    def test_update_and_stop_monitoring(self):
        """Test update_metrics and stop_monitoring functions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "monitor.log")
            
            # Configure monitoring with log
            config = self.test_config.copy()
            config["log_path"] = log_path
            
            # Start monitoring
            monitoring = MonitoringContext(config)
            
            # Update with test data
            update_metrics(monitoring, {
                "processed_texts": 100,
                "toxic_texts": 20,
                "confidence_values": [0.1, 0.3, 0.5, 0.7, 0.9]
            })
            
            # Get summary and stop
            summary = stop_monitoring(monitoring)
            
            # Verify summary metrics
            assert summary["processed_texts"] == 100
            assert summary["toxic_texts"] == 20
            assert summary["toxic_percentage"] == 20.0  # 20/100 * 100
            
            # Verify log file was created
            assert os.path.exists(log_path)
            
            # Verify log content
            with open(log_path, 'r') as f:
                log_data = json.load(f)
                assert "summary" in log_data
                assert "log" in log_data
                assert len(log_data["log"]) > 0
    
    def test_monitoring_context_thread_safety(self):
        """Test that MonitoringContext is thread-safe."""
        import threading
        import time
        
        context = MonitoringContext(self.test_config)
        errors = []
        
        def update_worker(worker_id):
            try:
                for i in range(10):
                    context.update({
                        "confidence_values": [0.1, 0.5, 0.9],
                        "processed_texts": 3
                    })
                    time.sleep(0.01)  # Small delay to increase chance of race conditions
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")
        
        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=update_worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        
        # Verify final counts are correct
        # 5 workers * 10 iterations * 3 confidence values = 150 total
        # Each iteration adds 1 very_low, 1 medium, 1 very_high
        assert context.metrics["processed_texts"] == 150
        assert context.metrics["confidence_buckets"]["very_low"] == 50
        assert context.metrics["confidence_buckets"]["medium"] == 50
        assert context.metrics["confidence_buckets"]["very_high"] == 50
    
    def test_empty_confidence_values_list(self):
        """Test handling of empty confidence values list."""
        context = MonitoringContext(self.test_config)
        
        # Update with empty list
        context.update({
            "confidence_values": [],
            "processed_texts": 0
        })
        
        # All buckets should remain at zero
        for bucket in context.metrics["confidence_buckets"].values():
            assert bucket == 0
    
    def test_confidence_values_with_zero_length_list(self):
        """Test that zero-length confidence_values list doesn't crash."""
        context = MonitoringContext(self.test_config)
        
        # This should not raise an exception
        context.update({
            "confidence_values": [],
            "processed_texts": 5  # processed_texts can be non-zero even with empty confidence list
        })
        
        assert context.metrics["processed_texts"] == 5
        assert all(count == 0 for count in context.metrics["confidence_buckets"].values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 