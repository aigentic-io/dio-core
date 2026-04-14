"""Tests for routing functionality."""

import warnings

from aigentic.core import DIO, Provider
from aigentic.core.provider import MockProvider
from aigentic.core.router import Policy, Request, Router


class TestRouter:
    """Test suite for Router class."""

    def test_add_provider(self):
        """Test adding providers to the router."""
        router = Router()
        provider = Provider(name="test-provider", type="local", cost_per_million_input_token=0.0, cost_per_million_output_token=0.0)

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
        local = Provider(name="local", type="local", cost_per_million_input_token=0.0, cost_per_million_output_token=0.0)
        cloud = Provider(name="cloud", type="cloud", cost_per_million_input_token=100.0, cost_per_million_output_token=300.0)
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
        provider = Provider(name="default", type="local", cost_per_million_input_token=0.0, cost_per_million_output_token=0.0)
        router.add_provider(provider)

        # Route should return first available provider
        assert router.route("test prompt") == "default"

    def test_multiple_policies(self):
        """Test routing with multiple policies."""
        router = Router()

        local = Provider(name="local", type="local", cost_per_million_input_token=0.0, cost_per_million_output_token=0.0)
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
        cloud = Provider(name="cloud-provider", type="cloud", cost_per_million_input_token=100.0, cost_per_million_output_token=300.0)
        local = Provider(name="local-provider", type="local", cost_per_million_input_token=0.0, cost_per_million_output_token=0.0)
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

        cloud = Provider(name="cloud", type="cloud", cost_per_million_input_token=100.0, cost_per_million_output_token=300.0)
        local = Provider(name="local", type="local", cost_per_million_input_token=0.0, cost_per_million_output_token=0.0)
        router.add_provider(cloud)
        router.add_provider(local)

        def policy(request: Request) -> str:
            return "PRIVATE"

        router.add_policy(Policy(rule=policy, enforcement="strict"))

        # PRIVATE should route to local (same as RESTRICTED)
        assert router.route("test") == "local"

    # -------------------------------------------------------------------------
    # Model-level routing tests
    # -------------------------------------------------------------------------

    def test_provider_model_field(self):
        """Provider accepts a first-class model field."""
        p = Provider(name="openai-mini", type="cloud", cost_per_million_input_token=100.0, cost_per_million_output_token=300.0, model="gpt-4o-mini")
        assert p.model == "gpt-4o-mini"

    def test_provider_model_defaults_none(self):
        """Existing Provider construction without model field is unaffected."""
        p = Provider(name="cloud", type="cloud", cost_per_million_input_token=100.0, cost_per_million_output_token=300.0)
        assert p.model is None

    def test_provider_model_alongside_metadata(self):
        """model field and metadata coexist independently."""
        p = Provider(
            name="openai-mini",
            type="cloud",
            cost_per_million_input_token=100.0,
            cost_per_million_output_token=300.0,
            model="gpt-4o-mini",
            metadata={"vendor": "openai"},
        )
        assert p.model == "gpt-4o-mini"
        assert p.metadata["vendor"] == "openai"

    def test_two_cloud_providers_different_models_are_independent_arms(self):
        """Two providers sharing the same type but different models are distinct routing arms."""
        router = Router()
        cheap = Provider(name="openai-mini", type="cloud", cost_per_million_input_token=100.0, cost_per_million_output_token=300.0, model="gpt-4o-mini")
        expensive = Provider(name="openai-full", type="cloud", cost_per_million_input_token=200.0, cost_per_million_output_token=500.0, model="gpt-4o")
        router.add_provider(cheap)
        router.add_provider(expensive)

        assert len(router.providers) == 2
        assert router.providers["openai-mini"].model == "gpt-4o-mini"
        assert router.providers["openai-full"].model == "gpt-4o"

    def test_fde_routes_simple_query_to_cheaper_model(self):
        """FDE prefers the cheaper cloud model for simple queries when both are cloud type."""
        from aigentic.core.fde import FederatedDecisionEngine

        # capability pinned equal so this test isolates cost routing behavior
        cheap = Provider(name="mini", type="cloud", cost_per_million_input_token=100.0, cost_per_million_output_token=300.0, model="gpt-4o-mini", capability=0.5)
        expensive = Provider(name="full", type="cloud", cost_per_million_input_token=200.0, cost_per_million_output_token=500.0, model="gpt-4o", capability=0.5)
        fde = FederatedDecisionEngine()

        selected, score = fde.route({"mini": cheap, "full": expensive}, "What is Python?")
        assert selected == "mini"

    def test_fde_differentiates_models_by_cost_score(self):
        """FDE assigns a higher cost score to the cheaper model."""
        from aigentic.core.fde import ComplexityLevel, FederatedDecisionEngine, RoutingContext

        # capability pinned equal so this test isolates cost scoring behavior
        cheap = Provider(name="mini", type="cloud", cost_per_million_input_token=100.0, cost_per_million_output_token=300.0, model="gpt-4o-mini", capability=0.5)
        expensive = Provider(name="full", type="cloud", cost_per_million_input_token=200.0, cost_per_million_output_token=500.0, model="gpt-4o", capability=0.5)
        fde = FederatedDecisionEngine()
        ctx = RoutingContext(prompt="hi", complexity=ComplexityLevel.SIMPLE, estimated_input_tokens=100, estimated_output_tokens=100)

        cheap_score = fde.score_provider(cheap, ctx)
        expensive_score = fde.score_provider(expensive, ctx)
        assert cheap_score.cost_score > expensive_score.cost_score

    def test_explicit_mapping_overrides_automatic(self):
        """Test that explicit mappings override automatic behavior."""
        router = Router()

        cloud = Provider(name="cloud", type="cloud", cost_per_million_input_token=100.0, cost_per_million_output_token=300.0)
        local = Provider(name="local", type="local", cost_per_million_input_token=0.0, cost_per_million_output_token=0.0)
        router.add_provider(cloud)
        router.add_provider(local)

        def policy(request: Request) -> str:
            return "RESTRICTED"

        router.add_policy(Policy(rule=policy, enforcement="strict"))

        # Normally RESTRICTED → local, but we explicitly map to cloud
        router.set_classification_mapping("RESTRICTED", "cloud")

        # Explicit mapping should take precedence
        assert router.route("test") == "cloud"


class TestDIOMessages:
    """Tests for DIO.route() messages parameter (OpenAI format)."""

    def _setup(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            local = Provider(
                name="local", type="local",
                cost_per_million_input_token=0.0,
                cost_per_million_output_token=0.0,
                capability=0.5,
            )
        dio = DIO(use_fde=True)
        dio.add_provider(local)
        return dio

    def test_messages_passed_to_adapter(self):
        """Verify messages list is forwarded to the adapter unchanged."""
        received = {}

        class CapturingMock(MockProvider):
            def generate(self, messages, **kwargs):
                received["messages"] = messages
                return ("ok", {"input_tokens": 5, "output_tokens": 10})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            local = Provider(name="local", type="local", capability=0.5)
        dio = DIO(use_fde=True)
        dio.add_provider(local, adapter=CapturingMock(local))

        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is Python?"},
        ]
        dio.route("What is Python?", messages=msgs)
        assert received["messages"] == msgs

    def test_prompt_only_wraps_as_user_message(self):
        """When messages=None, prompt is wrapped as a single user message."""
        received = {}

        class CapturingMock(MockProvider):
            def generate(self, messages, **kwargs):
                received["messages"] = messages
                return ("ok", {"input_tokens": 5, "output_tokens": 10})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            local = Provider(name="local", type="local", capability=0.5)
        dio = DIO(use_fde=True)
        dio.add_provider(local, adapter=CapturingMock(local))

        dio.route("What is Python?")
        assert received["messages"] == [{"role": "user", "content": "What is Python?"}]

    def test_response_includes_usage(self):
        """Verify usage dict from adapter is stored in Response.usage."""
        dio = self._setup()
        r = dio.route("What is Python?")
        assert r.usage is not None
        assert "input_tokens" in r.usage
        assert "output_tokens" in r.usage


class TestFDEFallback:
    """FDE automatic fallback when a provider's adapter throws an exception."""

    class _Failing(MockProvider):
        def generate(self, prompt, **kwargs):
            raise ConnectionError("simulated API failure")

    def _setup(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cloud = Provider(
                name="cloud", type="cloud",
                cost_per_million_input_token=3.0,
                cost_per_million_output_token=15.0,
                capability=0.9,
            )
            local = Provider(
                name="local", type="local",
                cost_per_million_input_token=0.0,
                cost_per_million_output_token=0.0,
                capability=0.5,
            )
        dio = DIO(
            use_fde=True,
            fde_weights={"privacy": 0.40, "cost": 0.20, "capability": 0.30, "latency": 0.10},
        )
        dio.add_provider(cloud, adapter=self._Failing(cloud))
        dio.add_provider(local)
        return dio

    def test_fde_falls_back_when_top_provider_fails(self):
        """FDE skips a failing provider and uses the next eligible one."""
        # Complex query → cloud scores highest, but it fails → local is used
        r = self._setup().route(
            "Explain the CAP theorem in distributed systems and compare "
            "Cassandra vs Spanner vs CockroachDB consistency models."
        )
        assert r.provider == "local"
        assert r.was_fallback is True
        assert "skipped_providers" in r.metadata
        assert "cloud" in r.metadata["skipped_providers"]

    def test_fde_fallback_includes_failure_reason(self):
        r = self._setup().route(
            "Explain the CAP theorem in distributed systems and compare "
            "Cassandra vs Spanner vs CockroachDB consistency models."
        )
        assert "fallback_reason" in r.metadata
        assert "simulated API failure" in r.metadata["fallback_reason"]

    def test_fde_raises_when_all_providers_fail(self):
        """When every eligible provider fails, ValueError is raised."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cloud = Provider(
                name="cloud-fail", type="cloud",
                cost_per_million_input_token=3.0,
                cost_per_million_output_token=15.0,
                capability=0.9,
            )
            local = Provider(
                name="local-fail", type="local",
                cost_per_million_input_token=0.0,
                cost_per_million_output_token=0.0,
                capability=0.5,
            )
        dio = DIO(use_fde=True)
        dio.add_provider(cloud, adapter=self._Failing(cloud))
        dio.add_provider(local, adapter=self._Failing(local))

        import pytest
        with pytest.raises(ValueError, match="All 2 eligible provider"):
            dio.route("What is Python?")
