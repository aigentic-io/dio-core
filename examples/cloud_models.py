"""Demo: model-tier routing with a single cloud vendor API key.

Shows how DIO routes between two models from the same vendor — cheap/fast
for simple queries, powerful for complex ones — using only one API key.

  Demo A — Gemini:  gemini-2.0-flash-lite (cheap) vs gemini-2.0-flash (frontier)
  Demo B — OpenAI:  gpt-4o-mini (cheap)          vs gpt-4o (frontier)

No local setup required — runs fully in mock mode without any API keys.

Setup:
    pip install google-genai openai python-dotenv

    # Add to .env (one key is enough to run that demo live):
    GOOGLE_API_KEY=your-key-here
    OPENAI_API_KEY=your-key-here

    python3 examples/cloud_models.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from aigentic.core import DIO, Provider

try:
    from aigentic.providers.gemini import GeminiProvider
except ImportError:
    GeminiProvider = None

try:
    from aigentic.providers.openai import OpenAIProvider
except ImportError:
    OpenAIProvider = None


def demo_single_gemini_api_key(gemini_key):
    """Demo A: model-level routing with a single Gemini API key.

    Registers gemini-2.0-flash-lite (cheap) and gemini-2.0-flash (frontier)
    as separate arms. FDE routes based on cost + capability.
    """
    mini = Provider(
        name="gemini-2.0-flash-lite",
        type="cloud",
        cost_per_input_token=0.000000075,   # $0.075/M tokens
        cost_per_output_token=0.0000003,    # $0.30/M tokens
        model="gemini-2.0-flash-lite",
        metadata={"vendor": "google"},
    )
    full = Provider(
        name="gemini-2.0-flash",
        type="cloud",
        cost_per_input_token=0.0000001,     # $0.10/M tokens
        cost_per_output_token=0.0000004,    # $0.40/M tokens
        model="gemini-2.0-flash",
        metadata={"vendor": "google"},
    )

    print("=" * 80)
    print("DEMO A: MODEL-TIER ROUTING (single Gemini API key)")
    print(f"  gemini-2.0-flash-lite  capability={mini.capability:.2f}  cheaper model → simple queries")
    print(f"  gemini-2.0-flash       capability={full.capability:.2f}  frontier model → complex queries")
    print(f"  {'Live mode (Gemini key found)' if gemini_key else 'Mock mode — set GOOGLE_API_KEY to run live'}")
    print("=" * 80)
    print()

    dio = DIO(
        use_fde=True,
        fde_weights={
            "privacy": 0.40,
            "cost": 0.15,       # Both models share the same vendor; cost gap is large
            "capability": 0.40, # Primary differentiator: cheap model for simple, frontier for complex
            "latency": 0.05,    # Both are cloud with similar latency
        },
    )

    if gemini_key and GeminiProvider:
        dio.add_provider(mini, adapter=GeminiProvider(mini, api_key=gemini_key))
        dio.add_provider(full, adapter=GeminiProvider(full, api_key=gemini_key))
    else:
        dio.add_provider(mini)
        dio.add_provider(full)

    _run_scenarios(dio, [
        ("What is Python?", "simple → expect gemini-2.0-flash-lite"),
        ("Define machine learning", "simple → expect gemini-2.0-flash-lite"),
        (
            "Explain the CAP theorem in distributed systems and compare "
            "Cassandra vs Spanner vs CockroachDB consistency models.",
            "complex → expect gemini-2.0-flash",
        ),
    ])


def demo_single_openai_api_key(openai_key):
    """Demo B: model-level routing with a single OpenAI API key.

    Registers gpt-4o-mini (cheap) and gpt-4o (frontier) as separate arms.
    FDE routes based on cost + capability.
    """
    mini = Provider(
        name="openai-gpt4o-mini",
        type="cloud",
        cost_per_input_token=0.00000015,    # $0.15/M tokens
        cost_per_output_token=0.0000006,    # $0.60/M tokens
        model="gpt-4o-mini",
        metadata={"vendor": "openai"},
    )
    full = Provider(
        name="openai-gpt4o",
        type="cloud",
        cost_per_input_token=0.0000025,     # $2.50/M tokens
        cost_per_output_token=0.00001,      # $10/M tokens
        model="gpt-4o",
        metadata={"vendor": "openai"},
    )

    print("=" * 80)
    print("DEMO B: MODEL-TIER ROUTING (single OpenAI API key)")
    print(f"  gpt-4o-mini  capability={mini.capability:.2f}  cheaper model → simple queries")
    print(f"  gpt-4o       capability={full.capability:.2f}  frontier model → complex queries")
    print(f"  {'Live mode (OpenAI key found)' if openai_key else 'Mock mode — set OPENAI_API_KEY to run live'}")
    print("=" * 80)
    print()

    dio = DIO(
        use_fde=True,
        fde_weights={
            "privacy": 0.40,
            "cost": 0.15,       # Both models share the same vendor; cost gap is large
            "capability": 0.40, # Primary differentiator: cheap model for simple, frontier for complex
            "latency": 0.05,    # Both are cloud with similar latency
        },
    )

    if openai_key and OpenAIProvider:
        dio.add_provider(mini, adapter=OpenAIProvider(mini, api_key=openai_key))
        dio.add_provider(full, adapter=OpenAIProvider(full, api_key=openai_key))
    else:
        dio.add_provider(mini)
        dio.add_provider(full)

    _run_scenarios(dio, [
        ("What is Python?", "simple → expect gpt-4o-mini"),
        ("Define machine learning", "simple → expect gpt-4o-mini"),
        (
            "Explain the CAP theorem in distributed systems and compare "
            "Cassandra vs Spanner vs CockroachDB consistency models.",
            "complex → expect gpt-4o",
        ),
    ])


def _run_scenarios(dio, scenarios):
    for prompt, note in scenarios:
        response = dio.route(prompt)
        model = dio.providers[response.provider].model
        cost_score = round(response.metadata.get("cost_score", 0), 1)
        cap_score = round(response.metadata.get("capability_score", 0), 1)
        print(f"  Prompt   : {prompt[:70]}{'...' if len(prompt) > 70 else ''}")
        print(f"  Routed to: {response.provider}  (model={model})")
        print(f"  Scores   : cost={cost_score}  capability={cap_score}  [{note}]")
        print()


def main():
    gemini_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not gemini_key:
        print("  GOOGLE_API_KEY not set — Demo A running in mock mode")
    if not openai_key:
        print("  OPENAI_API_KEY not set — Demo B running in mock mode")
    if not gemini_key or not openai_key:
        print()

    # demo_single_gemini_api_key(gemini_key)
    demo_single_openai_api_key(openai_key)

    print("=" * 80)
    print("KEY INSIGHT")
    print("  One API key is enough — register cheap + frontier models as separate arms.")
    print("  FDE routes simple queries to the cheap model, complex to the frontier.")
    print("  No local infrastructure required.")
    print("=" * 80)


if __name__ == "__main__":
    main()
