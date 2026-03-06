"""Workshop Setup Helper - Automatically detects and configures available providers.

This script makes setup easy for workshop attendees - it auto-detects what's available
and sets up providers accordingly.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aigentic.core import DIO, Provider


def check_api_availability():
    """Check which APIs are available."""
    status = {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "google": bool(os.getenv("GOOGLE_API_KEY")),
        "claude": bool(os.getenv("ANTHROPIC_API_KEY")),
        "ollama": False,  # TODO: Check if Ollama is running
    }
    return status


def setup_dio_auto():
    """
    Automatically set up DIO with best available providers.

    Priority:
    1. Real cloud provider (OpenAI, Claude, or Gemini) if API key found
    2. Mock providers if no API keys

    Returns:
        DIO: Configured DIO instance
    """
    print("=" * 70)
    print("DIO Workshop - Automatic Provider Setup")
    print("=" * 70)

    # Check what's available
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    # Define local provider
    local = Provider(name="local-llm", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0, capability=0.5)

    dio = DIO()

    # Setup cloud provider
    print("\n🌐 Cloud Provider:")
    if openai_key:
        try:
            from aigentic.providers.openai import OpenAIProvider
            cloud = Provider(name="cloud-llm", type="cloud", cost_per_input_token=0.000005, cost_per_output_token=0.00002, capability=0.9,
                             metadata={"model": "gpt-4o-mini"})
            dio.add_provider(
                cloud,
                adapter=OpenAIProvider(cloud, api_key=openai_key)
            )
            print("  ✅ Using OpenAI GPT-4o-mini (REAL API)")
        except Exception as e:
            print(f"  ⚠️  OpenAI setup failed: {e}")
            print("  ℹ️  Falling back to mock provider")
            cloud = Provider(name="cloud-llm", type="cloud", cost_per_input_token=0.000005, cost_per_output_token=0.00002, capability=0.9)
            dio.add_provider(cloud)
    elif anthropic_key:
        try:
            from aigentic.providers.claude import ClaudeProvider
            cloud = Provider(name="cloud-llm", type="cloud", cost_per_input_token=0.000005, cost_per_output_token=0.00002, capability=0.9,
                             metadata={"model": "claude-3-5-haiku-20241022"})
            dio.add_provider(
                cloud,
                adapter=ClaudeProvider(cloud, api_key=anthropic_key)
            )
            print("  ✅ Using Claude 3.5 Haiku (REAL API)")
        except Exception as e:
            print(f"  ⚠️  Claude setup failed: {e}")
            print("  ℹ️  Falling back to mock provider")
            cloud = Provider(name="cloud-llm", type="cloud", cost_per_input_token=0.000005, cost_per_output_token=0.00002, capability=0.9)
            dio.add_provider(cloud)
    elif google_key:
        try:
            from aigentic.providers.gemini import GeminiProvider
            cloud = Provider(name="cloud-llm", type="cloud", cost_per_input_token=0.000005, cost_per_output_token=0.00002, capability=0.9,
                             metadata={"model": "gemini-2.5-flash"})
            dio.add_provider(
                cloud,
                adapter=GeminiProvider(cloud, api_key=google_key)
            )
            print("  ✅ Using Google Gemini 2.5 Flash (REAL API)")
        except Exception as e:
            print(f"  ⚠️  Gemini setup failed: {e}")
            print("  ℹ️  Falling back to mock provider")
            cloud = Provider(name="cloud-llm", type="cloud", cost_per_input_token=0.000005, cost_per_output_token=0.00002, capability=0.9)
            dio.add_provider(cloud)
    else:
        cloud = Provider(name="cloud-llm", type="cloud", cost_per_input_token=0.000005, cost_per_output_token=0.00002, capability=0.9)
        dio.add_provider(cloud)
        print("  ℹ️  Using mock cloud provider")
        print("  💡 Tip: Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY for real responses")

    # Setup local provider (always mock for workshop simplicity)
    print("\n🏠 Local Provider:")
    dio.add_provider(local)
    print("  ℹ️  Using mock local provider")
    print("  💡 Advanced: Install Ollama for real local inference")

    print("\n" + "=" * 70)
    print("Setup Complete! Ready for workshop.")
    print("=" * 70)

    return dio


def setup_dio_manual(cloud_provider="mock", local_provider="mock"):
    """
    Manually set up DIO with specific providers.

    Args:
        cloud_provider: "openai", "claude", "gemini", or "mock"
        local_provider: "ollama" or "mock"

    Returns:
        DIO: Configured DIO instance
    """
    local = Provider(name="local-llm", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0, capability=0.5)

    dio = DIO()

    # Setup cloud
    if cloud_provider == "openai":
        from aigentic.providers.openai import OpenAIProvider
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        cloud = Provider(name="cloud-llm", type="cloud", cost_per_input_token=0.000005, cost_per_output_token=0.00002, capability=0.9,
                         metadata={"model": "gpt-4o-mini"})
        dio.add_provider(cloud, adapter=OpenAIProvider(cloud, api_key=openai_key))
        print("✅ OpenAI configured")

    elif cloud_provider == "claude":
        from aigentic.providers.claude import ClaudeProvider
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        cloud = Provider(name="cloud-llm", type="cloud", cost_per_input_token=0.000005, cost_per_output_token=0.00002, capability=0.9,
                         metadata={"model": "claude-3-5-haiku-20241022"})
        dio.add_provider(cloud, adapter=ClaudeProvider(cloud, api_key=anthropic_key))
        print("✅ Claude configured")

    elif cloud_provider == "gemini":
        from aigentic.providers.gemini import GeminiProvider
        google_key = os.getenv("GOOGLE_API_KEY")
        if not google_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        cloud = Provider(name="cloud-llm", type="cloud", cost_per_input_token=0.000005, cost_per_output_token=0.00002, capability=0.9,
                         metadata={"model": "gemini-2.5-flash"})
        dio.add_provider(cloud, adapter=GeminiProvider(cloud, api_key=google_key))
        print("✅ Gemini configured")

    else:  # mock
        cloud = Provider(name="cloud-llm", type="cloud", cost_per_input_token=0.000005, cost_per_output_token=0.00002, capability=0.9)
        dio.add_provider(cloud)
        print("ℹ️  Mock cloud provider configured")

    # Setup local
    if local_provider == "ollama":
        from aigentic.providers.ollama import OllamaProvider
        local = Provider(name="local-llm", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0,
                         metadata={"model": "llama3.2:3b"})
        dio.add_provider(local, adapter=OllamaProvider(local))
        print("✅ Ollama configured")

    else:  # mock
        dio.add_provider(local)
        print("ℹ️  Mock local provider configured")

    return dio


def demo_setup():
    """Run a demo to verify setup works."""
    from aigentic.core.pii_detector import has_pii

    print("\n" + "=" * 70)
    print("Running Setup Demo")
    print("=" * 70)

    dio = setup_dio_auto()

    # Add privacy policy (from workshop slides)
    def privacy_rule(request):
        return "RESTRICTED" if has_pii(request.prompt) else "PUBLIC"

    dio.add_policy(rule=privacy_rule, enforcement="strict")
    print("\n🔒 Privacy policy added (PII → local, Public → cloud)")

    # Test routing
    print("\n🧪 Test 1: Public query (should use cloud)")
    response = dio.route("What is Python?")
    print(f"  Prompt: What is Python?")
    print(f"  Provider: {response.provider}")
    print(f"  Response preview: {response.content[:100]}...")

    print("\n🧪 Test 2: PII query (should use local)")
    response = dio.route("My SSN is 123-45-6789")
    print(f"  Prompt: My SSN is 123-45-6789")
    print(f"  Provider: {response.provider}")
    print(f"  Response preview: {response.content[:100]}...")

    print("\n" + "=" * 70)
    print("✅ Setup verified! You're ready for the workshop.")
    print("=" * 70)

    return dio


if __name__ == "__main__":
    # Run the demo
    dio = demo_setup()

    print("\n📝 Usage in your code:")
    print("""
    from workshop_setup import setup_dio_auto

    # Easy setup
    dio = setup_dio_auto()

    # Now use DIO as shown in workshop
    response = dio.route("Your prompt here")
    print(response.content)
    """)
