"""
Tests for cost tracker.
"""

from mlops.tracking.cost_tracker import CostConfig, CostTracker


def test_cost_tracker_defaults():
    """Test CostTracker initializes with defaults."""
    tracker = CostTracker()
    assert tracker.mlflow_uri == "http://localhost:5000"
    assert tracker.config.gpu_hour_cost == 0.90


def test_cost_config_custom():
    """Test CostConfig with custom values."""
    config = CostConfig(gpu_hour_cost=1.50, cpu_hour_cost=0.10)
    tracker = CostTracker(mlflow_uri="http://custom:5000", config=config)
    assert tracker.config.gpu_hour_cost == 1.50
    assert tracker.mlflow_uri == "http://custom:5000"
