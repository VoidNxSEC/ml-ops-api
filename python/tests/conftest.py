"""
Pytest configuration and fixtures for mlops tests.
"""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def mlflow_uri():
    """MLflow tracking URI for tests"""
    return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


@pytest.fixture
def sample_hyperparameter_space():
    """Sample hyperparameter space for testing"""
    from mlops.models import HyperparameterSpace

    return HyperparameterSpace(
        learning_rate=(1e-5, 1e-3),
        batch_size=[8, 16, 32],
        num_epochs=[1, 2, 3],
        weight_decay=(0.0, 0.1),
        warmup_steps=[0, 100, 500],
    )


@pytest.fixture
def sample_pipeline_config(sample_hyperparameter_space):
    """Sample pipeline configuration for testing"""
    from mlops.models import PipelineConfig, SearchStrategy

    return PipelineConfig(
        experiment_id="test-experiment",
        dataset_path="imdb",
        hyperparameter_space=sample_hyperparameter_space,
        search_strategy=SearchStrategy.RANDOM,
        max_trials=5,
        parallel_trials=2,
        model_name="distilbert-base-uncased",
    )
