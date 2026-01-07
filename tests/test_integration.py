"""Integration tests for DIO framework."""

import pytest

from aigentic.core import DIO, Provider
from aigentic.core.pii_detector import has_pii
from aigentic.core.router import Request


class TestDIOIntegration:
    """End-to-end integration tests matching workshop flow."""

    def test_basic_setup(self):
        """Test basic DIO setup from workshop slides (Slide 10)."""
        # Setup from slides
        cloud = Provider(name="bedrock", type="cloud", cost_per_input_token=0.01, cost_per_output_token=0.03)
        local = Provider(name="vllm-secure", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0)

        dio = DIO()
        dio.add_provider(cloud)
        dio.add_provider(local)

        # Verify providers are registered
        assert "bedrock" in dio.router.providers
        assert "vllm-secure" in dio.router.providers

    def test_privacy_policy_routing(self):
        """Test privacy policy routing from workshop slides (Slide 11)."""
        # Setup
        cloud = Provider(name="bedrock", type="cloud", cost_per_input_token=0.01, cost_per_output_token=0.03)
        local = Provider(name="vllm-secure", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0)

        dio = DIO()
        dio.add_provider(cloud)
        dio.add_provider(local)

        # Privacy rule from slides
        def privacy_rule(request):
            if has_pii(request.prompt):
                return "RESTRICTED"
            return "PUBLIC"

        dio.add_policy(rule=privacy_rule, enforcement="strict")

        # Set classification mappings
        dio.set_classification_mapping("RESTRICTED", local)
        dio.set_classification_mapping("PUBLIC", cloud)

        # Test with PII - should route to local
        response = dio.route("My SSN is 123-45-6789")
        assert response.provider == "vllm-secure"
        assert not response.was_fallback

        # Test without PII - should route to cloud
        response = dio.route("What is the weather today?")
        assert response.provider == "bedrock"

    def test_fallback_configuration(self):
        """Test fallback configuration from workshop slides (Slide 13)."""
        cloud = Provider(name="bedrock", type="cloud", cost_per_input_token=0.01, cost_per_output_token=0.03)
        local = Provider(name="vllm-secure", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0)

        dio = DIO()
        dio.add_provider(cloud)
        dio.add_provider(local)

        # Set fallback
        dio.set_fallback(primary=cloud, fallback=local, trigger=TimeoutError)

        # Verify fallback is configured
        assert dio.fallback_config is not None
        assert dio.fallback_config["primary"] == "bedrock"
        assert dio.fallback_config["fallback"] == "vllm-secure"

    def test_pii_detection_examples(self):
        """Test PII detection with various examples."""
        # SSN
        assert has_pii("My SSN is 123-45-6789")

        # Email
        assert has_pii("Contact me at user@example.com")

        # Credit card
        assert has_pii("My card is 1234-5678-9012-3456")

        # Phone
        assert has_pii("Call me at 555-123-4567")

        # No PII
        assert not has_pii("What is the weather today?")

    def test_response_metadata(self):
        """Test that responses include proper metadata."""
        cloud = Provider(name="bedrock", type="cloud", cost_per_input_token=0.01, cost_per_output_token=0.03)

        dio = DIO()
        dio.add_provider(cloud)

        response = dio.route("Test prompt")

        assert response.content is not None
        assert response.provider == "bedrock"
        assert response.was_fallback is False
        assert "classification" in response.metadata

    def test_multiple_pii_types(self):
        """Test routing with multiple PII types in one request."""
        cloud = Provider(name="cloud", type="cloud", cost_per_input_token=0.01, cost_per_output_token=0.03)
        local = Provider(name="local", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0)

        dio = DIO()
        dio.add_provider(cloud)
        dio.add_provider(local)

        def privacy_rule(request):
            if has_pii(request.prompt):
                return "RESTRICTED"
            return "PUBLIC"

        dio.add_policy(rule=privacy_rule, enforcement="strict")
        dio.set_classification_mapping("RESTRICTED", local)
        dio.set_classification_mapping("PUBLIC", cloud)

        # Request with multiple PII types
        response = dio.route(
            "My SSN is 123-45-6789 and email is user@example.com"
        )
        assert response.provider == "local"

    def test_workshop_quickstart_flow(self):
        """Test the complete quickstart flow for workshop attendees."""
        # This mirrors the 10-line getting started example
        from aigentic.core import DIO, Provider
        from aigentic.core.pii_detector import has_pii

        # Setup providers
        cloud = Provider(name="bedrock", type="cloud", cost_per_input_token=0.01, cost_per_output_token=0.03)
        local = Provider(name="vllm-secure", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0)

        # Initialize DIO
        dio = DIO()
        dio.add_provider(cloud)
        dio.add_provider(local)

        # Add privacy policy
        def privacy_rule(request):
            return "RESTRICTED" if has_pii(request.prompt) else "PUBLIC"

        dio.add_policy(rule=privacy_rule, enforcement="strict")
        dio.set_classification_mapping("RESTRICTED", local)
        dio.set_classification_mapping("PUBLIC", cloud)

        # Test routing
        sensitive = dio.route("My SSN is 123-45-6789")
        assert sensitive.provider == "vllm-secure"

        normal = dio.route("Hello world")
        assert normal.provider == "bedrock"
