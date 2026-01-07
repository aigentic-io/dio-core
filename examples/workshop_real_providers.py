"""Workshop Example: Real Cloud Provider Integration.

Attendees will use their own API keys to build a working Smart Router.

Setup:
1. Get API keys (choose one or more):
   - OpenAI: https://platform.openai.com/api-keys
   - Google AI: https://makersuite.google.com/app/apikey
   - Anthropic Claude: https://console.anthropic.com/

2. Install dependencies:
   pip install openai google-generativeai anthropic python-dotenv

3. Add your API keys to .env file:
   OPENAI_API_KEY=your-key-here
   ANTHROPIC_API_KEY=your-key-here
   GOOGLE_API_KEY=your-key-here
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
from aigentic.providers.ollama import OllamaProvider


def main():
    # =============================================================================
    # STEP 1: Get API Keys from Environment
    # =============================================================================
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    claude_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_key:
        print("⚠️  OPENAI_API_KEY not set. Using mock provider.")
        print("   Get your key: https://platform.openai.com/api-keys")
        print()

    if not google_key:
        print("⚠️  GOOGLE_API_KEY not set. Using mock provider.")
        print("   Get your key: https://makersuite.google.com/app/apikey")
        print()

    if not claude_key:
        print("⚠️  ANTHROPIC_API_KEY not set. Using mock provider.")
        print("   Get your key: https://console.anthropic.com/")
        print()

    # =============================================================================
    # STEP 2: Configure Real Providers
    # =============================================================================

    # OpenAI GPT-4o-mini (Cloud) - Best for complex reasoning
    openai_provider = Provider(
        name="openai-gpt4",
        type="cloud",
        cost_per_input_token=0.005,
        cost_per_output_token=0.02,
        metadata={"vendor": "openai", "model": "gpt-4"}
    )

    # Google Gemini Pro (Cloud) - Good balance of cost/performance
    gemini_provider = Provider(
        name="gemini-2.5-flash",
        type="cloud",
        cost_per_input_token=0.005,
        cost_per_output_token=0.015,
        metadata={"vendor": "google", "model": "gemini-2.5-flash"}
    )

    # Claude (Cloud) - Good for reasoning and creativity
    claude_provider = Provider(
        name="claude-3-5-haiku",
        type="cloud",
        cost_per_input_token=0.005,
        cost_per_output_token=0.015,
        metadata={"vendor": "anthropic", "model": "claude-3-5-haiku"}
    )

    # Ollama Local (On-Prem) - Free, private, good for simple tasks
    local_provider = Provider(
        name="ollama-llama3.2",
        type="local",
        cost_per_input_token=0.0005,
        cost_per_output_token=0.002,
        metadata={"vendor": "ollama", "model": "llama3.2:3b"}
    )

    # =============================================================================
    # WORKSHOP EXERCISE 1: Initialize DIO with FDE (COMPLETED)
    # =============================================================================

    dio = DIO(
        use_fde=True,
        fde_weights={
            "privacy": 0.40,
            "cost": 0.25,
            "capability": 0.25,
            "latency": 0.10,
        },
        privacy_providers=["ollama-llama3.2"]  # Only local for PII
    )

    # =============================================================================
    # WORKSHOP EXERCISE 2: Tune FDE Weights for Cost Optimization
    # WORKSHOP TODO: Uncomment the code below to prioritize cost over capability
    # =============================================================================
    # Reconfigure DIO to prioritize cost (75%) over capability (15%)
    # dio.fde_weights = {
    #     "privacy": 0.40,   # Still top priority for PII
    #     "cost": 0.35,      # Increased: prefer cheaper models
    #     "capability": 0.15,  # Decreased: less emphasis on performance
    #     "latency": 0.10,   # Same: fast responses still matter
    # }
    # print("💡 FDE weights updated to prioritize cost optimization!")
    # print()

    # Add providers with real adapters
    if openai_key:
        dio.add_provider(
            openai_provider,
            adapter=OpenAIProvider(openai_provider, api_key=openai_key)
        )
    else:
        dio.add_provider(openai_provider)  # Mock adapter

    if google_key:
        dio.add_provider(
            gemini_provider,
            adapter=GeminiProvider(gemini_provider, api_key=google_key)
        )
    else:
        dio.add_provider(gemini_provider)  # Mock adapter

    if claude_key:
        dio.add_provider(
            claude_provider,
            adapter=ClaudeProvider(claude_provider, api_key=claude_key)
        )
    else:
        dio.add_provider(claude_provider)  # Mock adapter

    # Local provider (stub for now - real Ollama requires running server)
    dio.add_provider(local_provider)  # Mock adapter

    # =============================================================================
    # Test All Routing Scenarios with Real Providers
    # =============================================================================

    print("=" * 80)
    print("DIO WORKSHOP: SMART ROUTER WITH REAL CLOUD PROVIDERS")
    print("=" * 80)
    print()

    # Example 1: Privacy Policy - PII with email
    print("📝 Example 1: Privacy-Sensitive Data (Email = PII)")
    prompt = "My email is john.doe@example.com. How can I reset my password?"
    response = dio.route(prompt)
    print(f"  Prompt: '{prompt}'")
    print(f"  Provider: {response.provider} (forced local due to PII)")
    print(f"  FDE Score: {response.metadata.get('score', 'N/A')}")
    print(f"  Privacy Score: {response.metadata.get('privacy_score', 'N/A')}")
    print(f"  Response: {response.content[:150]}...")
    print()

    # Example 2: Privacy Policy - SSN
    print("📝 Example 2: Privacy-Sensitive Data (SSN = PII)")
    prompt = "My SSN is 123-45-6789. Can you verify my identity?"
    response = dio.route(prompt)
    print(f"  Prompt: '{prompt}'")
    print(f"  Provider: {response.provider} (forced local due to PII)")
    print(f"  FDE Score: {response.metadata.get('score', 'N/A')}")
    print(f"  Privacy Score: {response.metadata.get('privacy_score', 'N/A')}")
    print(f"  Response: {response.content[:150]}...")
    print()

    # Example 3: Simple question (cost-optimized)
    print("📝 Example 3: Simple Question (Cost Optimization)")
    prompt = "What is Python?"
    response = dio.route(prompt)
    print(f"  Prompt: '{prompt}'")
    print(f"  Provider: {response.provider}")
    print(f"  FDE Score: {response.metadata.get('score', 'N/A')}")
    print(f"  Privacy Score: {response.metadata.get('privacy_score', 'N/A')}")
    print(f"  Capability Score: {response.metadata.get('capability_score', 'N/A')}")
    print(f"  Response: {response.content[:150]}...")
    print(f"  💡 Routed based on: cost={dio.fde_weights['cost']}, capability={dio.fde_weights['capability']}")
    print()

    # Example 4: Complex reasoning (capability-focused)
    print("📝 Example 4: Complex Reasoning (Capability Priority)")
    prompt = "Explain the CAP theorem in distributed systems and provide real-world examples of systems that prioritize consistency vs availability."
    response = dio.route(prompt)
    print(f"  Prompt: '{prompt[:60]}...'")
    print(f"  Provider: {response.provider}")
    print(f"  FDE Score: {response.metadata.get('score', 'N/A')}")
    print(f"  Privacy Score: {response.metadata.get('privacy_score', 'N/A')}")
    print(f"  Capability Score: {response.metadata.get('capability_score', 'N/A')}")
    print(f"  Response: {response.content[:150]}...")
    print(f"  💡 After Exercise 2: Changing weights will affect this routing")
    print()

    # Example 5: Another simple query
    print("📝 Example 5: Another Simple Query")
    prompt = "Define machine learning"
    response = dio.route(prompt)
    print(f"  Prompt: '{prompt}'")
    print(f"  Provider: {response.provider}")
    print(f"  FDE Score: {response.metadata.get('score', 'N/A')}")
    print(f"  Privacy Score: {response.metadata.get('privacy_score', 'N/A')}")
    print(f"  Capability Score: {response.metadata.get('capability_score', 'N/A')}")
    print(f"  Response: {response.content[:150]}...")
    print()

    print("=" * 80)
    print("✅ WORKSHOP COMPLETE!")
    print("=" * 80)
    print("You've built a Smart Router that:")
    print("  ✓ Routes to cloud for complex queries")
    print("  ✓ Routes to local for simple/private queries")
    print("  ✓ Optimizes using FDE (Feature-Driven Evaluation)")
    print("  ✓ Enforces privacy constraints (PII always → local)")
    print("  ✓ Balances cost, capability, latency, and privacy")
    print()
    print("FDE Weights Used:")
    print(f"  • Privacy: {dio.fde_weights['privacy']*100}%")
    print(f"  • Cost: {dio.fde_weights['cost']*100}%")
    print(f"  • Capability: {dio.fde_weights['capability']*100}%")
    print(f"  • Latency: {dio.fde_weights['latency']*100}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
