"""
Ensemble Reasoning Module

Multi-model voting and aggregation strategies for improved accuracy and robustness.
Phase 1 MVP: Majority voting, confidence weighting, and basic debate.

Migrated from neutron/reasoning/ensemble_reasoning.py.
"""

import asyncio
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderResponse:
    """Single provider's response"""

    provider: str
    answer: str
    reasoning: str | None = None
    confidence: float = 0.0
    latency_ms: float = 0.0
    tokens_used: int = 0
    error: str | None = None


@dataclass
class EnsembleResult:
    """Aggregated result from ensemble"""

    answer: str
    confidence: float
    provider_votes: dict[str, str]
    reasoning: str
    strategy_used: str
    individual_responses: list[ProviderResponse] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_tokens_used: int = 0


class EnsembleReasoner:
    """
    Multi-provider ensemble reasoning system.

    Strategies:
    - majority: Simple majority vote
    - confidence_weighted: Weight by confidence scores
    - unanimous: Require all providers to agree
    - best_of_n: Select highest confidence response
    """

    def __init__(self, providers: list[str]):
        self.providers = providers
        self.history = []

    async def reason(
        self,
        task: str,
        strategy: str = "majority",
        temperature: float = 0.7,
        max_tokens: int = 512,
        timeout: int = 30,
    ) -> EnsembleResult:
        """
        Execute task across all providers and aggregate results.

        Args:
            task: The question or task to solve
            strategy: Voting strategy ("majority", "confidence_weighted", "unanimous", "best_of_n")
            temperature: LLM temperature
            max_tokens: Maximum tokens per response
            timeout: Timeout per provider in seconds

        Returns:
            EnsembleResult with aggregated answer
        """
        start_time = time.time()

        responses = await asyncio.gather(
            *[
                self._execute_on_provider(provider, task, temperature, max_tokens, timeout)
                for provider in self.providers
            ],
            return_exceptions=True,
        )

        valid_responses = []
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                error_resp = ProviderResponse(
                    provider=self.providers[i], answer="", error=str(resp), confidence=0.0
                )
                valid_responses.append(error_resp)
            else:
                valid_responses.append(resp)

        successful_responses = [r for r in valid_responses if r.error is None]

        if not successful_responses:
            raise RuntimeError("All providers failed")

        if strategy == "majority":
            result = self._majority_vote(successful_responses)
        elif strategy == "confidence_weighted":
            result = self._confidence_weighted(successful_responses)
        elif strategy == "unanimous":
            result = self._unanimous(successful_responses)
        elif strategy == "best_of_n":
            result = self._best_of_n(successful_responses)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        result.individual_responses = valid_responses
        result.total_latency_ms = (time.time() - start_time) * 1000
        result.total_tokens_used = sum(r.tokens_used for r in valid_responses)
        result.strategy_used = strategy

        self.history.append({"task": task, "result": result, "timestamp": time.time()})

        return result

    async def _execute_on_provider(
        self, provider: str, task: str, temperature: float, max_tokens: int, timeout: int
    ) -> ProviderResponse:
        """Execute task on a single provider"""
        from .dspy_adapter import DSPyProviderAdapter

        start_time = time.time()

        try:
            adapter = DSPyProviderAdapter(provider)

            response = await asyncio.wait_for(
                asyncio.to_thread(
                    adapter.basic_request, task, temperature=temperature, max_tokens=max_tokens
                ),
                timeout=timeout,
            )

            latency = (time.time() - start_time) * 1000
            confidence = self._estimate_confidence(response)

            return ProviderResponse(
                provider=provider,
                answer=response.strip(),
                confidence=confidence,
                latency_ms=latency,
                tokens_used=len(response.split()),
            )

        except TimeoutError:
            return ProviderResponse(
                provider=provider,
                answer="",
                error=f"Timeout after {timeout}s",
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return ProviderResponse(
                provider=provider,
                answer="",
                error=str(e),
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _estimate_confidence(self, response: str) -> float:
        """Estimate confidence score based on response characteristics."""
        response_lower = response.lower()

        confident_words = ["definitely", "certainly", "clearly", "obviously", "always"]
        uncertain_words = ["maybe", "possibly", "might", "could be", "perhaps", "unsure"]

        confidence = 0.5

        for word in confident_words:
            if word in response_lower:
                confidence += 0.1

        for word in uncertain_words:
            if word in response_lower:
                confidence -= 0.1

        word_count = len(response.split())
        if 20 < word_count < 200:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _majority_vote(self, responses: list[ProviderResponse]) -> EnsembleResult:
        """Simple majority voting"""
        answers = [r.answer for r in responses]
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)[0]

        winning_answer = most_common[0]
        vote_count = most_common[1]
        total_votes = len(answers)

        confidence = vote_count / total_votes

        reasoning = f"Majority vote: {vote_count}/{total_votes} providers agreed.\n"
        reasoning += f"Votes: {dict(answer_counts)}"

        return EnsembleResult(
            answer=winning_answer,
            confidence=confidence,
            provider_votes={r.provider: r.answer for r in responses},
            reasoning=reasoning,
            strategy_used="majority",
        )

    def _confidence_weighted(self, responses: list[ProviderResponse]) -> EnsembleResult:
        """Confidence-weighted voting"""
        answer_groups: dict[str, list[ProviderResponse]] = {}
        for r in responses:
            if r.answer not in answer_groups:
                answer_groups[r.answer] = []
            answer_groups[r.answer].append(r)

        weighted_scores = {}
        for answer, resps in answer_groups.items():
            weighted_scores[answer] = sum(r.confidence for r in resps)

        winning_answer = max(weighted_scores, key=weighted_scores.get)
        total_confidence = sum(weighted_scores.values())
        normalized_confidence = (
            weighted_scores[winning_answer] / total_confidence if total_confidence > 0 else 0
        )

        reasoning = "Confidence-weighted vote:\n"
        for answer, score in weighted_scores.items():
            reasoning += f"  '{answer[:50]}...': {score:.2f}\n"

        return EnsembleResult(
            answer=winning_answer,
            confidence=normalized_confidence,
            provider_votes={r.provider: r.answer for r in responses},
            reasoning=reasoning,
            strategy_used="confidence_weighted",
        )

    def _unanimous(self, responses: list[ProviderResponse]) -> EnsembleResult:
        """Require unanimous agreement"""
        answers = [r.answer for r in responses]
        unique_answers = set(answers)

        if len(unique_answers) == 1:
            return EnsembleResult(
                answer=answers[0],
                confidence=1.0,
                provider_votes={r.provider: r.answer for r in responses},
                reasoning=f"Unanimous agreement across all {len(responses)} providers",
                strategy_used="unanimous",
            )
        else:
            return EnsembleResult(
                answer="[NO CONSENSUS]",
                confidence=0.0,
                provider_votes={r.provider: r.answer for r in responses},
                reasoning=f"No unanimous agreement. Got {len(unique_answers)} different answers: {unique_answers}",
                strategy_used="unanimous",
            )

    def _best_of_n(self, responses: list[ProviderResponse]) -> EnsembleResult:
        """Select response with highest confidence"""
        best_response = max(responses, key=lambda r: r.confidence)

        return EnsembleResult(
            answer=best_response.answer,
            confidence=best_response.confidence,
            provider_votes={r.provider: r.answer for r in responses},
            reasoning=f"Highest confidence response from {best_response.provider} ({best_response.confidence:.2f})",
            strategy_used="best_of_n",
        )

    def get_history(self) -> list[dict[str, Any]]:
        """Get reasoning history"""
        return self.history

    def get_stats(self) -> dict[str, Any]:
        """Get statistics across all history"""
        if not self.history:
            return {"total_queries": 0}

        total_queries = len(self.history)
        avg_confidence = sum(h["result"].confidence for h in self.history) / total_queries
        avg_latency = sum(h["result"].total_latency_ms for h in self.history) / total_queries

        provider_success_rates = {}
        for h in self.history:
            for resp in h["result"].individual_responses:
                if resp.provider not in provider_success_rates:
                    provider_success_rates[resp.provider] = {"success": 0, "total": 0}

                provider_success_rates[resp.provider]["total"] += 1
                if resp.error is None:
                    provider_success_rates[resp.provider]["success"] += 1

        return {
            "total_queries": total_queries,
            "average_confidence": avg_confidence,
            "average_latency_ms": avg_latency,
            "provider_success_rates": {
                p: stats["success"] / stats["total"] if stats["total"] > 0 else 0
                for p, stats in provider_success_rates.items()
            },
        }
