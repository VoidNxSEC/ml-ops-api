"""
Tests for ensemble reasoning module.

Migrated from neotron/tests/reasoning/.
"""

import pytest

from mlops.reasoning.ensemble_reasoning import (
    EnsembleReasoner,
    EnsembleResult,
    ProviderResponse,
)


def test_provider_response_defaults():
    """Test ProviderResponse dataclass defaults."""
    resp = ProviderResponse(provider="test", answer="hello")
    assert resp.confidence == 0.0
    assert resp.error is None
    assert resp.tokens_used == 0


def test_ensemble_result_defaults():
    """Test EnsembleResult dataclass defaults."""
    result = EnsembleResult(
        answer="test",
        confidence=0.8,
        provider_votes={"p1": "test"},
        reasoning="test reason",
        strategy_used="majority",
    )
    assert result.individual_responses == []
    assert result.total_latency_ms == 0.0


def test_majority_vote_unanimous():
    """Test majority vote when all providers agree."""
    reasoner = EnsembleReasoner(providers=["p1", "p2", "p3"])

    responses = [
        ProviderResponse(provider="p1", answer="yes", confidence=0.9),
        ProviderResponse(provider="p2", answer="yes", confidence=0.8),
        ProviderResponse(provider="p3", answer="yes", confidence=0.7),
    ]

    result = reasoner._majority_vote(responses)
    assert result.answer == "yes"
    assert result.confidence == 1.0


def test_majority_vote_split():
    """Test majority vote with split responses."""
    reasoner = EnsembleReasoner(providers=["p1", "p2", "p3"])

    responses = [
        ProviderResponse(provider="p1", answer="yes", confidence=0.9),
        ProviderResponse(provider="p2", answer="yes", confidence=0.8),
        ProviderResponse(provider="p3", answer="no", confidence=0.7),
    ]

    result = reasoner._majority_vote(responses)
    assert result.answer == "yes"
    assert result.confidence == pytest.approx(2 / 3)


def test_best_of_n():
    """Test best_of_n selects highest confidence."""
    reasoner = EnsembleReasoner(providers=["p1", "p2"])

    responses = [
        ProviderResponse(provider="p1", answer="answer_a", confidence=0.6),
        ProviderResponse(provider="p2", answer="answer_b", confidence=0.9),
    ]

    result = reasoner._best_of_n(responses)
    assert result.answer == "answer_b"
    assert result.confidence == 0.9


def test_estimate_confidence_baseline():
    """Test confidence estimation baseline."""
    reasoner = EnsembleReasoner(providers=[])
    confidence = reasoner._estimate_confidence("Some neutral response")
    assert 0.0 <= confidence <= 1.0


def test_get_stats_empty():
    """Test stats with no history."""
    reasoner = EnsembleReasoner(providers=["p1"])
    stats = reasoner.get_stats()
    assert stats == {"total_queries": 0}
