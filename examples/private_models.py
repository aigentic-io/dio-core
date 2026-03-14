"""Demo: model-tier routing with a single self-hosted Ollama instance.

Uses WebhostProvider to route between two local models on the same host:
  - llama3:latest  (lightweight, ~1050ms) → simple queries
  - gpt-oss:20b    (heavyweight, ~1530ms) → complex queries

Works offline — no cloud API keys required. FDE routes using capability
(Goldilocks scoring for simple tasks) and latency (capability-derived
estimate: larger model = slower inference).

Setup:
    # Point at your Ollama server (Tailscale, homelab, cloud VM, etc.)
    export OLLAMA_BASE_URL=http://localhost:11434   # optional: remote Ollama (e.g. Tailscale homelab)

    python3 examples/private_models.py

If OLLAMA_BASE_URL is not set, runs in mock mode (shows routing decisions
without making real HTTP calls).
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
from aigentic.providers.webhost import WebhostProvider

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

print("=" * 80)
print("DEMO: MODEL-TIER ROUTING (single Ollama host, two model sizes)")
print("  llama3:latest   capability=0.47   latency~1050ms   lightweight → simple queries")
print("  gpt-oss:20b     capability=0.74   latency~1530ms   heavyweight → complex queries")
print(f"  Ollama host: {OLLAMA_BASE_URL or '(mock mode — set OLLAMA_BASE_URL to run live)'}")
print("=" * 80)
print()

# Two models on the same Ollama host, registered as independent routing arms.
mini = Provider(
    name="llama3-latest",
    type="local",
    cost_per_million_input_token=0.01,    # ~$0.01/M tokens (electricity)
    cost_per_million_output_token=0.01,
    model="llama3:latest",
)
full = Provider(
    name="gpt-oss-20b",
    type="local",
    cost_per_million_input_token=0.02,    # ~$0.02/M tokens (larger model, more power)
    cost_per_million_output_token=0.02,
    model="gpt-oss:20b",
)

dio = DIO(
    use_fde=True,
    fde_weights={
        "privacy": 0.40,
        "cost": 0.10,       # Minimal — both models cost electricity only
        "capability": 0.30, # Routes complex queries to the larger model
        "latency": 0.20,    # Key tiebreaker: larger model = slower inference
    },
)

if OLLAMA_BASE_URL:
    dio.add_provider(mini, adapter=WebhostProvider(mini, base_url=OLLAMA_BASE_URL))
    dio.add_provider(full, adapter=WebhostProvider(full, base_url=OLLAMA_BASE_URL, timeout=5000))
else:
    dio.add_provider(mini)   # mock adapter — routing decisions still shown
    dio.add_provider(full)

scenarios = [
    ("What is Python?", "simple → expect llama3:latest"),
    ("Define machine learning", "simple → expect llama3:latest"),
    (
        "Explain the CAP theorem in distributed systems and compare "
        "Cassandra vs Spanner vs CockroachDB consistency models.",
        "complex → expect gpt-oss:20b",
    ),
]

for prompt, note in scenarios:
    response = dio.route(prompt)
    model = dio.providers[response.provider].model
    cap_score = round(response.metadata.get("capability_score", 0), 1)
    lat_score = round(response.metadata.get("latency_score", 0), 1)
    print(f"  Prompt   : {prompt[:70]}{'...' if len(prompt) > 70 else ''}")
    print(f"  Routed to: {response.provider}  (model={model})  cap={cap_score}  lat={lat_score}")
    print(f"  [{note}]")
    if OLLAMA_BASE_URL:
        content = response.content
        preview = content.replace('\n', ' ')[:120]
        print(f"  Response : {preview}{'...' if len(content) > 120 else ''}")
    print()

print("=" * 80)
print("KEY INSIGHT")
print("  Both models run on the same host — no cloud keys needed.")
print("  FDE routes using capability (Goldilocks: ~0.5 ideal for simple tasks)")
print("  and latency (auto-estimated from model size: 200 + capability*1800 ms).")
print("  Simple queries → fast 8B model; complex reasoning → heavyweight 20B.")
print("=" * 80)
