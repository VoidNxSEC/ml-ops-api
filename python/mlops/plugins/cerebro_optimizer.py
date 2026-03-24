"""
Cerebro Semantic Optimizer (PHANTOM Integration)

Uses LLMs to suggest hyperparameter configurations based on
semantic understanding of the search space and dataset characteristics.

Migrated from neutron/plugins/cerebro_optimizer.py.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml
try:
    import mlflow
except ImportError:
    mlflow = None

from mlops.llm.client import LLMClient
from mlops.models import (
    HyperparameterSpace,
    OptimizationState,
    SearchStrategy,
    TrainingConfig,
)
from mlops.optimization.optimizer import HyperparameterOptimizer

logger = logging.getLogger("mlops.plugins.cerebro")


class CerebroOptimizer(HyperparameterOptimizer):
    """
    Semantic Hyperparameter Optimizer using LLM reasoning.

    Integrated with PHANTOM (Cerebro) for intelligent search
    and MLflow for reasoning tracking.
    """

    def __init__(
        self,
        hyperparameter_space: HyperparameterSpace,
        config_path: str = "config/integrations.yaml",
    ):
        super().__init__(hyperparameter_space, strategy=SearchStrategy.SEMANTIC)
        self.llm_client = LLMClient()
        self.config = self._load_config(config_path)

        # Optimizer settings
        self.model = self.config.get("model", "claude-3-5-sonnet-20241022")
        self.temperature = self.config.get("temperature", 0.7)
        self.system_prompt = self.config.get("system_prompt", "")

    def _load_config(self, path: str) -> dict[str, Any]:
        """Load Cerebro configuration from YAML."""
        try:
            config_file = Path(path)
            if not config_file.exists():
                logger.warning(f"Config file {path} not found, using defaults")
                return {}

            with open(config_file, "r") as f:
                data = yaml.safe_load(f)

            return data.get("cerebro", {}).get("optimizer", {})
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            return {}

    async def suggest_configs_async(
        self, num: int, state: OptimizationState, experiment_id: str
    ) -> list[TrainingConfig]:
        """
        Generate hyperparameter configurations using LLM reasoning.

        Args:
            num: Number of configurations to suggest.
            state: Current optimization state (history).
            experiment_id: Experiment identifier.

        Returns:
            List of suggested TrainingConfig objects.
        """
        prompt = self._construct_prompt(num, state)

        try:
            logger.info(f"Requesting {num} suggestions from Cerebro ({self.model})...")

            if mlflow and mlflow.active_run():
                mlflow.log_param("cerebro_model", self.model)
                mlflow.log_text(prompt, "cerebro_prompt.txt")

            response = await self.llm_client.generate(
                prompt=prompt,
                system=self.system_prompt,
                temperature=self.temperature,
                max_tokens=4096
            )

            suggestions_data = self._parse_response(response.content)

            if mlflow and mlflow.active_run():
                mlflow.log_text(response.content, "cerebro_response.txt")
                if "reasoning" in suggestions_data:
                    mlflow.log_text(suggestions_data["reasoning"], "cerebro_reasoning.md")

            configs = []
            valid_suggestions = suggestions_data.get("suggestions", [])

            for i, suggestion in enumerate(valid_suggestions[:num]):
                config = TrainingConfig(
                    run_id=f"{experiment_id}-cerebro-{state.trials_completed + i}",
                    experiment_id=experiment_id,
                    model_name="distilbert-base-uncased",
                    dataset_path="imdb",
                    learning_rate=float(suggestion.get("learning_rate", 1e-5)),
                    batch_size=int(suggestion.get("batch_size", 16)),
                    num_epochs=int(suggestion.get("num_epochs", 3)),
                    weight_decay=float(suggestion.get("weight_decay", 0.01)),
                    warmup_steps=int(suggestion.get("warmup_steps", 0)),
                )
                configs.append(config)

            if len(configs) < num:
                logger.warning(f"Cerebro returned {len(configs)} configs, requested {num}. Filling with random.")
                random_configs = self._random_search(num - len(configs), experiment_id)
                configs.extend(random_configs)

            return configs

        except Exception as e:
            logger.error(f"Cerebro optimization failed: {e}")
            if mlflow and mlflow.active_run():
                mlflow.log_text(str(e), "cerebro_error.txt")

            return self._random_search(num, experiment_id)

    def suggest_configs(
        self, num: int, state: OptimizationState, experiment_id: str
    ) -> list[TrainingConfig]:
        """Synchronous wrapper for suggest_configs_async."""
        import asyncio
        return asyncio.run(self.suggest_configs_async(num, state, experiment_id))

    def _construct_prompt(self, num: int, state: OptimizationState) -> str:
        """Construct the prompt for the LLM."""
        space = self.hyperparameter_space

        history_str = "No prior trials."
        if state.trials_completed > 0:
            history_str = f"Completed {state.trials_completed} trials. Best accuracy: {state.best_accuracy:.4f}."
            if state.best_config:
                history_str += f"\nBest config so far: {json.dumps(state.best_config)}"

        return f"""
        Task: Suggest {num} hyperparameter configurations for a DistilBERT text classification model on the IMDB dataset.

        Search Space:
        - learning_rate: {space.learning_rate} (float range)
        - batch_size: {space.batch_size} (options)
        - num_epochs: {space.num_epochs} (options)
        - weight_decay: {space.weight_decay} (float range)
        - warmup_steps: {space.warmup_steps} (options)

        Optimization History:
        {history_str}

        Requirements:
        1. Suggest {num} distinct configurations.
        2. Explore promising regions based on the history.
        3. Provide a brief reasoning for your strategy.
        4. Output strictly valid JSON.

        Output Format (JSON):
        {{
            "reasoning": "Explanation of your search strategy...",
            "suggestions": [
                {{
                    "learning_rate": <float>,
                    "batch_size": <int>,
                    "num_epochs": <int>,
                    "weight_decay": <float>,
                    "warmup_steps": <int>
                }},
                ...
            ]
        }}
        """

    def _parse_response(self, text: str) -> dict[str, Any]:
        """Extract and parse JSON from LLM response."""
        text = text.strip()

        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        elif text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {text}")
            return {}
