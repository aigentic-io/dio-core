"""Tests for per-token cost model."""

import pytest

from aigentic.core.provider import Provider
from aigentic.core.fde import (
    ComplexityLevel,
    FederatedDecisionEngine,
    RoutingContext,
)


class TestProviderCost:
    """Tests for Provider cost calculations."""

    def test_estimated_cost_calculation(self):
        """Verify estimated_cost = input * rate_in + output * rate_out."""
        provider = Provider(
            name="test",
            type="cloud",
            cost_per_input_token=0.005,
            cost_per_output_token=0.02,
        )
        cost = provider.estimated_cost(100, 200)
        expected = 100 * 0.005 + 200 * 0.02
        assert cost == pytest.approx(expected)

    def test_cost_property_backward_compat(self):
        """Verify provider.cost returns default estimate (500 input + 500 output)."""
        provider = Provider(
            name="test",
            type="cloud",
            cost_per_input_token=0.005,
            cost_per_output_token=0.02,
        )
        expected = 500 * 0.005 + 500 * 0.02
        assert provider.cost == pytest.approx(expected)

    def test_free_provider_cost(self):
        """Verify zero costs for free providers."""
        provider = Provider(
            name="free",
            type="local",
            cost_per_input_token=0.0,
            cost_per_output_token=0.0,
        )
        assert provider.estimated_cost(1000, 2000) == 0.0
        assert provider.cost == 0.0


class TestFDECostScoring:
    """Tests for FDE cost scoring with per-token costs."""

    def test_fde_cost_scoring_with_tokens(self):
        """Verify FDE uses estimated cost from per-token rates."""
        fde = FederatedDecisionEngine()

        cheap = Provider(
            name="cheap",
            type="local",
            cost_per_input_token=0.000001,
            cost_per_output_token=0.000002,
        )
        expensive = Provider(
            name="expensive",
            type="cloud",
            cost_per_input_token=0.00005,
            cost_per_output_token=0.00015,
        )

        context = RoutingContext(
            prompt="What is Python?",
            complexity=ComplexityLevel.SIMPLE,
            estimated_input_tokens=100,
            estimated_output_tokens=100,
        )

        cheap_score = fde.score_provider(cheap, context)
        expensive_score = fde.score_provider(expensive, context)

        # Cheap provider should have a higher cost_score
        assert cheap_score.cost_score > expensive_score.cost_score

        # Both should have estimated_cost populated
        assert cheap_score.estimated_cost == pytest.approx(
            cheap.estimated_cost(100, 100)
        )
        assert expensive_score.estimated_cost == pytest.approx(
            expensive.estimated_cost(100, 100)
        )

    def test_fde_free_provider_perfect_cost_score(self):
        """Free providers should get a perfect cost score."""
        fde = FederatedDecisionEngine()

        free = Provider(
            name="free",
            type="local",
            cost_per_input_token=0.0,
            cost_per_output_token=0.0,
        )

        context = RoutingContext(
            prompt="test",
            complexity=ComplexityLevel.SIMPLE,
            estimated_input_tokens=100,
            estimated_output_tokens=100,
        )

        score = fde.score_provider(free, context)
        assert score.cost_score == 100.0
        assert score.estimated_cost == 0.0

    def test_estimate_output_tokens(self):
        """Verify output token estimation multipliers."""
        assert FederatedDecisionEngine._estimate_output_tokens(
            100, ComplexityLevel.SIMPLE
        ) == 100
        assert FederatedDecisionEngine._estimate_output_tokens(
            100, ComplexityLevel.MODERATE
        ) == 200
        assert FederatedDecisionEngine._estimate_output_tokens(
            100, ComplexityLevel.COMPLEX
        ) == 300


class TestFDECapabilityScoring:
    """Tests for FDE capability scoring with per-provider capability."""

    def test_capability_differentiates_cloud_providers(self):
        """Two cloud providers with different capability on complex query: higher wins."""
        fde = FederatedDecisionEngine()

        strong = Provider(
            name="strong-cloud",
            type="cloud",
            cost_per_input_token=0.0004,
            cost_per_output_token=0.0008,
            capability=0.9,
        )
        weak = Provider(
            name="weak-cloud",
            type="cloud",
            cost_per_input_token=0.0004,
            cost_per_output_token=0.0008,
            capability=0.6,
        )

        context = RoutingContext(
            prompt="Explain distributed consensus algorithms",
            complexity=ComplexityLevel.COMPLEX,
            estimated_input_tokens=100,
            estimated_output_tokens=300,
        )

        strong_score = fde.score_provider(strong, context)
        weak_score = fde.score_provider(weak, context)

        # Higher capability should yield higher capability_score
        assert strong_score.capability_score > weak_score.capability_score
        # And since costs are identical, higher capability should win overall
        assert strong_score.score > weak_score.score

    def test_capability_default_backward_compat(self):
        """Provider without explicit capability defaults to 1.0."""
        provider = Provider(name="default", type="cloud")
        assert provider.capability == 1.0

    def test_capability_scales_base_score(self):
        """Capability multiplies the base score."""
        fde = FederatedDecisionEngine()

        full_cap = Provider(name="full", type="cloud", capability=1.0)
        half_cap = Provider(name="half", type="cloud", capability=0.5)

        context = RoutingContext(
            prompt="Complex task",
            complexity=ComplexityLevel.COMPLEX,
            estimated_input_tokens=50,
            estimated_output_tokens=150,
        )

        full_score = fde.score_provider(full_cap, context)
        half_score = fde.score_provider(half_cap, context)

        # COMPLEX + cloud base = 100.0, so scores should be 100.0 and 50.0
        assert full_score.capability_score == pytest.approx(100.0)
        assert half_score.capability_score == pytest.approx(50.0)
