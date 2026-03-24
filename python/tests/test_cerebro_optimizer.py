"""
Tests for Cerebro Semantic Optimizer.

Migrated from neotron/tests/plugins/test_cerebro_optimizer.py.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from mlops.models import (
    HyperparameterSpace,
    OptimizationState,
    SearchStrategy,
)
from mlops.llm.providers import LLMResponse
from mlops.plugins.cerebro_optimizer import CerebroOptimizer


class TestCerebroOptimizer(unittest.TestCase):
    def setUp(self):
        self.space = HyperparameterSpace()
        self.optimizer = CerebroOptimizer(self.space, config_path="tests/test_config.yaml")

        # Mock LLM Client
        self.optimizer.llm_client = MagicMock()

    def test_initialization(self):
        self.assertEqual(self.optimizer.strategy, SearchStrategy.SEMANTIC)
        self.assertIsNotNone(self.optimizer.llm_client)

    @patch("mlops.plugins.cerebro_optimizer.mlflow")
    def test_suggest_configs_success(self, mock_mlflow):
        mock_response_json = {
            "reasoning": "Test reasoning",
            "suggestions": [
                {
                    "learning_rate": 2e-5,
                    "batch_size": 16,
                    "num_epochs": 3,
                    "weight_decay": 0.01,
                    "warmup_steps": 100
                }
            ]
        }

        async def mock_generate(*args, **kwargs):
            return LLMResponse(
                content=json.dumps(mock_response_json),
                finish_reason="stop",
                total_tokens=100
            )

        self.optimizer.llm_client.generate.side_effect = mock_generate

        state = OptimizationState(
            current_strategy=SearchStrategy.SEMANTIC,
            trials_completed=0,
            best_accuracy=0.0
        )

        configs = self.optimizer.suggest_configs(1, state, "exp-001")

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].learning_rate, 2e-5)
        self.assertEqual(configs[0].batch_size, 16)
        self.optimizer.llm_client.generate.assert_called_once()

    def test_suggest_configs_json_parsing_error(self):
        async def mock_generate_error(*args, **kwargs):
            return LLMResponse(
                content="Invalid JSON",
                finish_reason="stop",
                total_tokens=10
            )

        self.optimizer.llm_client.generate.side_effect = mock_generate_error

        state = OptimizationState(
            current_strategy=SearchStrategy.SEMANTIC,
            trials_completed=0,
            best_accuracy=0.0
        )

        configs = self.optimizer.suggest_configs(2, state, "exp-002")

        self.assertEqual(len(configs), 2)
        self.assertGreaterEqual(configs[0].learning_rate, self.space.learning_rate[0])


if __name__ == "__main__":
    unittest.main()
