"""Smoke tests mirroring the routing decisions in each example script.

Each test class mirrors the provider setup and routing scenarios from one
example, asserting that the correct provider is chosen for each prompt.
All values (capability, cost) are set explicitly — no registry sync or
live API keys are required.

FDE SCORING STABILITY NOTE
---------------------------
FDE routing is deterministic given fixed providers, weights, and prompts:
the same inputs always produce the same winner. Tests are only included
where the winning margin is large enough (>2 points on a 0-100 scale)
to be robust against minor floating-point drift.

The Gemini model-tier scenario (gemini-2.0-flash-lite vs gemini-2.0-flash,
capability 0.79 vs 0.80) is intentionally excluded — the scoring margin is
<0.1 points and the result is sensitive to prompt token count. It works
correctly in practice (see examples/cloud_models.py output) but is not
suitable for a deterministic assertion.

AIgentic Premium DIO uses more sophisticated multi-dimensional scoring
algorithms (dynamic ELO, specialization profiles, real-time latency
measurements). Routing outcomes may differ from these tests even with
identical provider configs when premium scoring is enabled.
"""

import warnings

from aigentic.core import DIO, Provider
from aigentic.core.pii_detector import has_pii
from aigentic.core.provider import MockProvider

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_provider(**kwargs) -> Provider:
    """Create a Provider suppressing auto-load UserWarnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Provider(**kwargs)


# ── Quickstart ────────────────────────────────────────────────────────────────

class TestQuickstartSmoke:
    """Mirrors examples/quickstart.py routing scenarios."""

    def _setup(self) -> DIO:
        cloud = _make_provider(
            name="bedrock", type="cloud",
            cost_per_million_input_token=3.0,
            cost_per_million_output_token=15.0,
            capability=0.9,
        )
        local = _make_provider(
            name="vllm-secure", type="local",
            cost_per_million_input_token=0.5,
            cost_per_million_output_token=2.0,
            capability=0.5,
        )
        dio = DIO()
        dio.add_provider(cloud)
        dio.add_provider(local)

        def privacy_rule(request):
            return "RESTRICTED" if has_pii(request.prompt) else "PUBLIC"

        dio.add_policy(rule=privacy_rule, enforcement="strict")
        return dio

    def test_ssn_routes_to_local(self):
        assert self._setup().route("My SSN is 123-45-6789").provider == "vllm-secure"

    def test_email_routes_to_local(self):
        assert self._setup().route("Please contact me at user@example.com").provider == "vllm-secure"

    def test_clean_prompt_routes_to_cloud(self):
        assert self._setup().route("What is Python?").provider == "bedrock"

    def test_failover_on_connection_error(self):
        class _Failing(MockProvider):
            def generate(self, prompt, **kwargs):
                raise ConnectionError("simulated outage")

        primary = _make_provider(
            name="bedrock-primary", type="cloud",
            cost_per_million_input_token=3.0,
            cost_per_million_output_token=15.0,
        )
        backup = _make_provider(
            name="bedrock-backup", type="cloud",
            cost_per_million_input_token=2.5,
            cost_per_million_output_token=12.5,
        )
        dio = DIO()
        dio.add_provider(primary, adapter=_Failing(primary))
        dio.add_provider(backup)
        dio.set_fallback(primary, backup, trigger=ConnectionError)

        r = dio.route("Explain quantum computing")
        assert r.provider == "bedrock-backup"
        assert r.was_fallback is True


# ── Hybrid models ─────────────────────────────────────────────────────────────

class TestHybridModelSmoke:
    """Mirrors examples/hybrid_models.py routing scenarios.

    FDE weights: privacy=0.40, cost=0.20, capability=0.35, latency=0.05.
    Scoring margins: privacy hard-constraint (∞), simple ~12 pts, complex ~9 pts.
    """

    def _setup(self) -> DIO:
        dio = DIO(
            use_fde=True,
            fde_weights={"privacy": 0.40, "cost": 0.20, "capability": 0.35, "latency": 0.05},
            privacy_providers=["ollama-llama3"],
        )
        dio.add_provider(_make_provider(
            name="openai-gpt4o", type="cloud",
            cost_per_million_input_token=2.50,
            cost_per_million_output_token=10.00,
            capability=0.71,
        ))
        dio.add_provider(_make_provider(
            name="gemini-2.0-flash", type="cloud",
            cost_per_million_input_token=0.10,
            cost_per_million_output_token=0.40,
            capability=0.80,
        ))
        dio.add_provider(_make_provider(
            name="claude-3-5-haiku", type="cloud",
            cost_per_million_input_token=0.80,
            cost_per_million_output_token=4.00,
            capability=0.63,
        ))
        dio.add_provider(_make_provider(
            name="ollama-llama3", type="local",
            cost_per_million_input_token=0.0,
            cost_per_million_output_token=0.0,
            capability=0.47,
        ))
        return dio

    def test_email_pii_forced_local(self):
        r = self._setup().route("My email is john.doe@example.com. How can I reset my password?")
        assert r.provider == "ollama-llama3"

    def test_ssn_pii_forced_local(self):
        r = self._setup().route("My SSN is 123-45-6789. Can you verify my identity?")
        assert r.provider == "ollama-llama3"

    def test_simple_query_routes_to_cheapest(self):
        # Free local wins on cost with sufficient capability for simple queries
        r = self._setup().route("What is Python?")
        assert r.provider == "ollama-llama3"

    def test_complex_query_routes_to_capable_cloud(self):
        # gemini-2.0-flash has highest capability among cloud providers
        r = self._setup().route(
            "Explain the CAP theorem in distributed systems and provide "
            "real-world examples of systems that prioritize consistency vs availability."
        )
        assert r.provider == "gemini-2.0-flash"


# ── Cloud model tiers (OpenAI) ────────────────────────────────────────────────

class TestCloudModelTierSmoke:
    """Mirrors examples/cloud_models.py Demo B (OpenAI) routing scenarios.

    FDE weights: privacy=0.40, cost=0.15, capability=0.40, latency=0.05.
    Scoring margins: simple ~7.5 pts, complex ~2 pts.

    Demo A (Gemini) is excluded — the capability gap between gemini-2.0-flash-lite
    (0.79) and gemini-2.0-flash (0.80) is 0.01, leaving a scoring margin <0.1 pts
    that is too sensitive to prompt token count for a deterministic assertion.
    AIgentic Premium DIO's capability scoring resolves this with task-specific
    profiles that widen the effective differentiation between model tiers.
    """

    def _setup(self) -> DIO:
        dio = DIO(
            use_fde=True,
            fde_weights={"privacy": 0.40, "cost": 0.15, "capability": 0.40, "latency": 0.05},
        )
        dio.add_provider(_make_provider(
            name="openai-gpt4o-mini", type="cloud",
            cost_per_million_input_token=0.15,
            cost_per_million_output_token=0.60,
            capability=0.57,
        ))
        dio.add_provider(_make_provider(
            name="openai-gpt4o", type="cloud",
            cost_per_million_input_token=2.50,
            cost_per_million_output_token=10.00,
            capability=0.71,
        ))
        return dio

    def test_simple_query_routes_to_mini(self):
        assert self._setup().route("What is Python?").provider == "openai-gpt4o-mini"

    def test_complex_query_routes_to_full(self):
        r = self._setup().route(
            "Explain the CAP theorem in distributed systems and compare "
            "Cassandra vs Spanner vs CockroachDB consistency models."
        )
        assert r.provider == "openai-gpt4o"


# ── Private models (Ollama) ───────────────────────────────────────────────────

class TestPrivateModelSmoke:
    """Mirrors examples/private_models.py routing scenarios.

    FDE weights: privacy=0.40, cost=0.10, capability=0.30, latency=0.20.
    Both models run on the same host — cost is near-zero for both, so
    capability (Goldilocks) and latency drive the split.
    Scoring margins: simple ~6 pts, complex ~4 pts.
    """

    def _setup(self) -> DIO:
        dio = DIO(
            use_fde=True,
            fde_weights={"privacy": 0.40, "cost": 0.10, "capability": 0.30, "latency": 0.20},
        )
        dio.add_provider(_make_provider(
            name="llama3-latest", type="local",
            cost_per_million_input_token=0.01,
            cost_per_million_output_token=0.01,
            capability=0.47,
        ))
        dio.add_provider(_make_provider(
            name="gpt-oss-20b", type="local",
            cost_per_million_input_token=0.02,
            cost_per_million_output_token=0.02,
            capability=0.74,
        ))
        return dio

    def test_simple_query_routes_to_lightweight(self):
        # Goldilocks: llama3 (cap=0.47) scores higher than gpt-oss-20b (0.74)
        # for simple queries — smaller model is sufficient, lower latency wins
        assert self._setup().route("What is Python?").provider == "llama3-latest"

    def test_complex_query_routes_to_heavyweight(self):
        # gpt-oss-20b wins on capability score for complex reasoning
        r = self._setup().route(
            "Explain the CAP theorem in distributed systems and compare "
            "Cassandra vs Spanner vs CockroachDB consistency models."
        )
        assert r.provider == "gpt-oss-20b"
