"""
Cost Tracking for MLOps pipeline.

Migrated from neutron/tracking/cost_tracker.py.
"""

from dataclasses import dataclass


@dataclass
class CostConfig:
    """Configuration for cost tracking."""

    gpu_hour_cost: float = 0.90
    cpu_hour_cost: float = 0.05
    storage_gb_month_cost: float = 0.02


class CostTracker:
    """Tracks costs of ML experiments via MLflow."""

    def __init__(self, mlflow_uri: str = "http://localhost:5000", config: CostConfig | None = None):
        self.mlflow_uri = mlflow_uri
        self.config = config or CostConfig()
