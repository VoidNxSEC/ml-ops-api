"""
Hyperparameter Optimizer for MLOps pipeline.

Migrated from neutron/optimization/optimizer.py.
"""

import itertools
from dataclasses import dataclass

from mlops.models import (
    HyperparameterSpace,
    OptimizationState,
    SearchStrategy,
    TrainingConfig,
)


class HyperparameterOptimizer:
    """Suggests hyperparameter configurations based on search strategy."""

    def __init__(
        self,
        hyperparameter_space: HyperparameterSpace,
        strategy: SearchStrategy = SearchStrategy.RANDOM,
    ):
        self.hyperparameter_space = hyperparameter_space
        self.strategy = strategy

    def suggest_configs(
        self, num: int, state: OptimizationState, experiment_id: str
    ) -> list[TrainingConfig]:
        """Generate hyperparameter configurations."""
        if self.strategy == SearchStrategy.RANDOM:
            return self._random_search(num, experiment_id)
        elif self.strategy == SearchStrategy.GRID:
            return self._grid_search(num, experiment_id, state.trials_completed)
        else:
            return self._random_search(num, experiment_id)

    def _random_search(self, num: int, experiment_id: str) -> list[TrainingConfig]:
        configs = []
        for i in range(num):
            sample = self.hyperparameter_space.sample()
            configs.append(
                TrainingConfig(
                    run_id=f"{experiment_id}-random-{i}",
                    experiment_id=experiment_id,
                    model_name="distilbert-base-uncased",
                    dataset_path="imdb",
                    learning_rate=sample["learning_rate"],
                    batch_size=sample["batch_size"],
                    num_epochs=sample["num_epochs"],
                    weight_decay=sample["weight_decay"],
                    warmup_steps=sample["warmup_steps"],
                )
            )
        return configs

    def _grid_search(
        self, num: int, experiment_id: str, offset: int = 0
    ) -> list[TrainingConfig]:
        space = self.hyperparameter_space
        grid = list(
            itertools.product(
                [space.learning_rate[0], space.learning_rate[1]],
                space.batch_size,
                space.num_epochs,
            )
        )
        configs = []
        for i, (lr, bs, ep) in enumerate(grid[offset : offset + num]):
            configs.append(
                TrainingConfig(
                    run_id=f"{experiment_id}-grid-{offset + i}",
                    experiment_id=experiment_id,
                    model_name="distilbert-base-uncased",
                    dataset_path="imdb",
                    learning_rate=lr,
                    batch_size=bs,
                    num_epochs=ep,
                )
            )
        return configs
