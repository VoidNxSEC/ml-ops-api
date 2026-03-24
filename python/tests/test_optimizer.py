"""
Tests for hyperparameter optimizer.

Migrated from neotron/tests/test_optimizer.py.
"""

from mlops.models import OptimizationState, SearchStrategy
from mlops.optimization.optimizer import HyperparameterOptimizer


def test_optimizer_initialization(sample_hyperparameter_space):
    """Test optimizer initialization"""
    optimizer = HyperparameterOptimizer(
        hyperparameter_space=sample_hyperparameter_space, strategy=SearchStrategy.RANDOM
    )
    assert optimizer.strategy == SearchStrategy.RANDOM
    assert optimizer.hyperparameter_space == sample_hyperparameter_space


def test_random_search_suggestions(sample_hyperparameter_space):
    """Test random search strategy"""
    optimizer = HyperparameterOptimizer(
        hyperparameter_space=sample_hyperparameter_space, strategy=SearchStrategy.RANDOM
    )

    state = OptimizationState(
        current_strategy=SearchStrategy.RANDOM, trials_completed=0, best_accuracy=0.0
    )

    configs = optimizer.suggest_configs(num=3, state=state, experiment_id="test-exp")

    assert len(configs) == 3
    for config in configs:
        assert config.experiment_id == "test-exp"
        assert config.learning_rate > 0


def test_grid_search_suggestions(sample_hyperparameter_space):
    """Test grid search strategy"""
    optimizer = HyperparameterOptimizer(
        hyperparameter_space=sample_hyperparameter_space, strategy=SearchStrategy.GRID
    )

    state = OptimizationState(
        current_strategy=SearchStrategy.GRID, trials_completed=0, best_accuracy=0.0
    )

    configs = optimizer.suggest_configs(num=5, state=state, experiment_id="test-exp")

    assert len(configs) <= 5
