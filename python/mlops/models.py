"""
Core data models for MLOps pipeline.

Migrated from neutron/core/models.py.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SearchStrategy(Enum):
    RANDOM = "random"
    GRID = "grid"
    BAYESIAN = "bayesian"
    SEMANTIC = "semantic"


@dataclass
class HyperparameterSpace:
    """Defines the search space for hyperparameter optimization."""

    learning_rate: tuple[float, float] = (1e-5, 1e-3)
    batch_size: list[int] = field(default_factory=lambda: [8, 16, 32])
    num_epochs: list[int] = field(default_factory=lambda: [1, 2, 3])
    weight_decay: tuple[float, float] = (0.0, 0.1)
    warmup_steps: list[int] = field(default_factory=lambda: [0, 100, 500])

    def sample(self) -> dict[str, Any]:
        """Sample a random configuration from the space."""
        return {
            "learning_rate": random.uniform(*self.learning_rate),
            "batch_size": random.choice(self.batch_size),
            "num_epochs": random.choice(self.num_epochs),
            "weight_decay": random.uniform(*self.weight_decay),
            "warmup_steps": random.choice(self.warmup_steps),
        }


@dataclass
class PipelineConfig:
    """Configuration for a training pipeline run."""

    experiment_id: str
    dataset_path: str
    hyperparameter_space: HyperparameterSpace
    search_strategy: SearchStrategy = SearchStrategy.RANDOM
    max_trials: int = 10
    parallel_trials: int = 2
    model_name: str = "distilbert-base-uncased"


@dataclass
class TrainingConfig:
    """Configuration for a single training run."""

    run_id: str
    experiment_id: str
    model_name: str
    dataset_path: str
    learning_rate: float
    batch_size: int
    num_epochs: int
    weight_decay: float = 0.01
    warmup_steps: int = 0


@dataclass
class OptimizationState:
    """Tracks the state of hyperparameter optimization."""

    current_strategy: SearchStrategy
    trials_completed: int
    best_accuracy: float
    best_config: dict[str, Any] = field(default_factory=dict)
