"""Tests for routing functionality."""

import pytest

from aigentic.core.provider import Provider
from aigentic.core.router import Policy, Request, Router


class TestRouter:
    """Test suite for Router class."""

    def test_add_provider(self):
        """Test adding providers to the router."""
        router = Router()
        provider = Provider(name="test-provider", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0)

        router.add_provider(provider)

        assert "test-provider" in router.providers
        assert router.providers["test-provider"] == provider

    def test_add_policy(self):
        """Test adding policies to the router."""
        router = Router()

        def test_rule(request: Request) -> str:
            return "TEST"

        policy = Policy(rule=test_rule, enforcement="strict")
        router.add_policy(policy)

        assert len(router.policies) == 1
        assert router.policies[0].rule == test_rule

    def test_classification_mapping(self):
        """Test setting classification mappings."""
        router = Router()
        router.set_classification_mapping("RESTRICTED", "local-provider")

        assert router.classification_map["RESTRICTED"] == "local-provider"

    def test_route_with_policy(self):
        """Test routing with a policy."""
        router = Router()

        # Add providers
        local = Provider(name="local", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0)
        cloud = Provider(name="cloud", type="cloud", cost_per_input_token=0.01, cost_per_output_token=0.03)
        router.add_provider(local)
        router.add_provider(cloud)

        # Add policy
        def privacy_rule(request: Request) -> str:
            if "secret" in request.prompt:
                return "RESTRICTED"
            return "PUBLIC"

        policy = Policy(rule=privacy_rule, enforcement="strict")
        router.add_policy(policy)

        # Set classification mapping
        router.set_classification_mapping("RESTRICTED", "local")
        router.set_classification_mapping("PUBLIC", "cloud")

        # Test routing
        assert router.route("This is a secret") == "local"
        assert router.route("This is public") == "cloud"

    def test_route_no_providers(self):
        """Test routing when no providers are available."""
        router = Router()
        assert router.route("test prompt") is None

    def test_route_no_policy_match(self):
        """Test routing when no policy matches."""
        router = Router()

        # Add providers
        provider = Provider(name="default", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0)
        router.add_provider(provider)

        # Route should return first available provider
        assert router.route("test prompt") == "default"

    def test_multiple_policies(self):
        """Test routing with multiple policies."""
        router = Router()

        local = Provider(name="local", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0)
        router.add_provider(local)

        def policy1(request: Request) -> str:
            return "CLASS1"

        def policy2(request: Request) -> str:
            return "CLASS2"

        router.add_policy(Policy(rule=policy1, enforcement="strict"))
        router.add_policy(Policy(rule=policy2, enforcement="advisory"))

        router.set_classification_mapping("CLASS1", "local")

        # First strict policy should take precedence
        assert router.route("test") == "local"

    def test_request_metadata(self):
        """Test Request object with metadata."""
        request = Request(prompt="test", metadata={"key": "value"})

        assert request.prompt == "test"
        assert request.metadata["key"] == "value"

    def test_policy_metadata(self):
        """Test Policy object with metadata."""
        def test_rule(request: Request) -> str:
            return "TEST"

        policy = Policy(
            rule=test_rule,
            enforcement="strict",
            metadata={"description": "Test policy"}
        )

        assert policy.metadata["description"] == "Test policy"

    def test_automatic_classification_routing(self):
        """Test automatic routing for RESTRICTED/PUBLIC classifications."""
        router = Router()

        # Add cloud and local providers
        cloud = Provider(name="cloud-provider", type="cloud", cost_per_input_token=0.01, cost_per_output_token=0.03)
        local = Provider(name="local-provider", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0)
        router.add_provider(cloud)
        router.add_provider(local)

        # Policy that returns RESTRICTED or PUBLIC
        def privacy_policy(request: Request) -> str:
            if "secret" in request.prompt:
                return "RESTRICTED"
            return "PUBLIC"

        router.add_policy(Policy(rule=privacy_policy, enforcement="strict"))

        # RESTRICTED should automatically route to local
        assert router.route("This is secret") == "local-provider"

        # PUBLIC should automatically route to cloud
        assert router.route("This is public") == "cloud-provider"

    def test_automatic_classification_private(self):
        """Test automatic routing for PRIVATE classification."""
        router = Router()

        cloud = Provider(name="cloud", type="cloud", cost_per_input_token=0.01, cost_per_output_token=0.03)
        local = Provider(name="local", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0)
        router.add_provider(cloud)
        router.add_provider(local)

        def policy(request: Request) -> str:
            return "PRIVATE"

        router.add_policy(Policy(rule=policy, enforcement="strict"))

        # PRIVATE should route to local (same as RESTRICTED)
        assert router.route("test") == "local"

    def test_explicit_mapping_overrides_automatic(self):
        """Test that explicit mappings override automatic behavior."""
        router = Router()

        cloud = Provider(name="cloud", type="cloud", cost_per_input_token=0.01, cost_per_output_token=0.03)
        local = Provider(name="local", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0)
        router.add_provider(cloud)
        router.add_provider(local)

        def policy(request: Request) -> str:
            return "RESTRICTED"

        router.add_policy(Policy(rule=policy, enforcement="strict"))

        # Normally RESTRICTED → local, but we explicitly map to cloud
        router.set_classification_mapping("RESTRICTED", "cloud")

        # Explicit mapping should take precedence
        assert router.route("test") == "cloud"
