"""Demo: multi-provider FDE routing — cloud + private models together.

Registers providers from multiple vendors alongside a local Ollama instance.
FDE routes based on privacy (PII → local), cost, capability, and latency.

  openai-gpt4o        cloud   capability from registry  frontier reasoning
  gemini-2.0-flash    cloud   capability from registry  cost-effective cloud
  claude-3-5-haiku    cloud   capability from registry  fast, accurate
  ollama-llama3       local   capability from registry  private — PII goes here

Setup:
1. Get API keys (choose one or more):
   - OpenAI: https://platform.openai.com/api-keys
   - Google AI: https://makersuite.google.com/app/apikey
   - Anthropic Claude: https://console.anthropic.com/

2. Install dependencies:
   pip install openai google-genai anthropic python-dotenv

3. Add your API keys to .env file:
   OPENAI_API_KEY=your-key-here
   ANTHROPIC_API_KEY=your-key-here
   GOOGLE_API_KEY=your-key-here
   OLLAMA_BASE_URL=http://localhost:11434   # optional: remote Ollama (e.g. Tailscale homelab)

    python3 examples/hybrid_models.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for development mode
sys.path.insert(0, str(Path(__file__).parent.parent))

from aigentic.core import DIO, Provider
from aigentic.providers.openai import OpenAIProvider
from aigentic.providers.gemini import GeminiProvider
from aigentic.providers.claude import ClaudeProvider
from aigentic.providers.webhost import WebhostProvider


def demo_multi_provider(openai_key, google_key, claude_key, ollama_url=None):
    """Demo: multi-provider FDE routing with real API keys."""
    # ==========================================================================
    # STEP 2: Configure Providers — model= is now a first-class field
    # ==========================================================================

    openai_provider = Provider(
        name="openai-gpt4o",
        type="cloud",
        cost_per_input_token=0.0000025,     # $2.50/M tokens
        cost_per_output_token=0.00001,      # $10/M tokens
        model="gpt-4o",
        metadata={"vendor": "openai"},
    )

    gemini_provider = Provider(
        name="gemini-2.0-flash",
        type="cloud",
        cost_per_input_token=0.0000001,     # $0.10/M tokens
        cost_per_output_token=0.0000004,    # $0.40/M tokens
        model="gemini-2.0-flash",
        metadata={"vendor": "google"},
    )

    claude_provider = Provider(
        name="claude-3-5-haiku",
        type="cloud",
        cost_per_input_token=0.0000008,     # $0.80/M tokens
        cost_per_output_token=0.000004,     # $4/M tokens
        model="claude-3-5-haiku-20241022",
        metadata={"vendor": "anthropic"},
    )

    local_provider = Provider(
        name="ollama-llama3",
        type="local",
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        model="llama3:latest",
        metadata={"vendor": "ollama"},
    )

    print("=" * 80)
    print("DEMO: MULTI-PROVIDER FDE (privacy + cost + capability)")
    print(f"  {openai_provider.name:20}  cloud  capability={openai_provider.capability:.2f}  frontier reasoning")
    print(f"  {gemini_provider.name:20}  cloud  capability={gemini_provider.capability:.2f}  cost-effective cloud")
    print(f"  {claude_provider.name:20}  cloud  capability={claude_provider.capability:.2f}  fast, accurate")
    print(f"  {local_provider.name:20}  local  capability={local_provider.capability:.2f}  private — PII goes here")
    print("=" * 80)
    print()

    # ==========================================================================
    # WORKSHOP EXERCISE 1: Initialize DIO with FDE (COMPLETED)
    # ==========================================================================

    dio = DIO(
        use_fde=True,
        fde_weights={
            "privacy": 0.40,
            "cost": 0.20,       # Local is free — cost alone would always win without balancing capability
            "capability": 0.35, # Routes complex queries to capable cloud models over free-but-weak local
            "latency": 0.05,
        },
        privacy_providers=["ollama-llama3"],
    )

    # =============================================================================
    # WORKSHOP EXERCISE 2: Tune FDE Weights for Cost Optimization
    # WORKSHOP TODO: Uncomment the code below to prioritize cost over capability
    # =============================================================================
    # Reconfigure DIO to prioritize cost over capability
    # dio.fde_weights = {
    #     "privacy": 0.35,   # Still top priority for PII
    #     "cost": 0.35,      # Increased: prefer cheaper models
    #     "capability": 0.20,  # Decreased: less emphasis on performance
    #     "latency": 0.10,   # Same: fast responses still matter
    # }
    # print("💡 FDE weights updated to prioritize cost optimization!")
    # print()

    # Add providers with real adapters (falls back to mock when key absent)
    if openai_key:
        dio.add_provider(openai_provider, adapter=OpenAIProvider(openai_provider, api_key=openai_key))
    else:
        dio.add_provider(openai_provider)

    if google_key:
        dio.add_provider(gemini_provider, adapter=GeminiProvider(gemini_provider, api_key=google_key))
    else:
        dio.add_provider(gemini_provider)

    if claude_key:
        dio.add_provider(claude_provider, adapter=ClaudeProvider(claude_provider, api_key=claude_key))
    else:
        dio.add_provider(claude_provider)

    if ollama_url:
        dio.add_provider(local_provider, adapter=WebhostProvider(local_provider, base_url=ollama_url))
    else:
        dio.add_provider(local_provider)

    # Example 1: PII — email
    print("Example 1: Privacy-Sensitive Data (Email = PII)")
    prompt = "My email is john.doe@example.com. How can I reset my password?"
    response = dio.route(prompt)
    print(f"  Provider      : {response.provider} (forced local — PII detected)")
    print(f"  Privacy score : {response.metadata.get('privacy_score', 'N/A')}")
    print(f"  Response      : {response.content[:120]}...")
    print()

    # Example 2: PII — SSN
    print("Example 2: Privacy-Sensitive Data (SSN = PII)")
    prompt = "My SSN is 123-45-6789. Can you verify my identity?"
    response = dio.route(prompt)
    print(f"  Provider      : {response.provider} (forced local — PII detected)")
    print(f"  Privacy score : {response.metadata.get('privacy_score', 'N/A')}")
    print(f"  Response      : {response.content[:120]}...")
    print()

    # Example 3: Simple — cost-optimized
    print("Example 3: Simple Question (Cost Optimization)")
    prompt = "What is Python?"
    response = dio.route(prompt)
    model = dio.providers[response.provider].model
    print(f"  Provider      : {response.provider}  (model={model})")
    print(f"  Cost score    : {response.metadata.get('cost_score', 'N/A')}")
    print(f"  Capability    : {response.metadata.get('capability_score', 'N/A')}")
    print(f"  Response      : {response.content[:120]}...")
    print()

    # Example 4: Complex — capability-focused
    print("Example 4: Complex Reasoning")
    prompt = (
        "Explain the CAP theorem in distributed systems and provide "
        "real-world examples of systems that prioritize consistency vs availability."
    )
    response = dio.route(prompt)
    model = dio.providers[response.provider].model
    print(f"  Provider      : {response.provider}  (model={model})")
    print(f"  Cost score    : {response.metadata.get('cost_score', 'N/A')}")
    print(f"  Capability    : {response.metadata.get('capability_score', 'N/A')}")
    print(f"  Response      : {response.content[:120]}...")
    print()

    print("FDE Weights:")
    for k, v in dio.fde_weights.items():
        print(f"  {k:12}: {v*100:.0f}%")
    print()


def main():
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    claude_key = os.getenv("ANTHROPIC_API_KEY")
    ollama_url = os.getenv("OLLAMA_BASE_URL")

    missing = [
        ("OPENAI_API_KEY", "https://platform.openai.com/api-keys", openai_key),
        ("GOOGLE_API_KEY", "https://makersuite.google.com/app/apikey", google_key),
        ("ANTHROPIC_API_KEY", "https://console.anthropic.com/", claude_key),
    ]
    for name, url, val in missing:
        if not val:
            print(f"  {name} not set — using mock. Get key: {url}")
    if ollama_url:
        print(f"  OLLAMA_BASE_URL={ollama_url} — local provider will use WebhostProvider")
    else:
        print("  OLLAMA_BASE_URL not set — local provider using mock (set to enable real Ollama)")
    if any(not v for _, _, v in missing):
        print()

    demo_multi_provider(openai_key, google_key, claude_key, ollama_url=ollama_url)

    print("=" * 80)
    print("WORKSHOP COMPLETE")
    print("  model= field makes each model a first-class routing arm")
    print("  FDE selects the best model per query — privacy always wins")
    print("=" * 80)


if __name__ == "__main__":
    main()
