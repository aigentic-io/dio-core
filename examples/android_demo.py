"""Demo: Android-aware routing — no Android hardware required.

Simulates Android device contexts (battery, connectivity) as DIO routing
parameters. User identity (tier, policies) is server-resolved from the auth
token — not simulated here. Cost caps are passed explicitly to mirror what
the server would inject after resolving the token. Runs entirely from a
MacBook — all cloud providers fall back to mock when API keys are absent.

Android device tier matrix:
  gemini-nano   on-device (Pixel 9 AICore)   capability=0.35  latency=50ms   free
  phi-3-mini    on-device (mid-range)         capability=0.28  latency=120ms  free
  gemini-flash  cloud (cheap)                 capability=0.70  latency=300ms  paid
  gpt-4o        cloud (frontier)              capability=0.95  latency=800ms  paid

Setup (all optional — runs in mock mode without any keys):
    export OPENAI_API_KEY=...
    export GOOGLE_API_KEY=...
    export ANTHROPIC_API_KEY=...

    python3 examples/android_demo.py
"""

import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Suppress registry UserWarnings — capabilities are set explicitly below.
warnings.filterwarnings("ignore", category=UserWarning)

from aigentic.core import DIO, Provider

try:
    from aigentic.providers.openai import OpenAIProvider
except ImportError:
    OpenAIProvider = None

try:
    from aigentic.providers.gemini import GeminiProvider
except ImportError:
    GeminiProvider = None

# ── Providers: Android device tier matrix ─────────────────────────────────────
# Capabilities set explicitly (not from registry) for deterministic demo output.

gemini_nano = Provider(
    name="gemini-nano", type="local",
    cost_per_input_token=0.0, cost_per_output_token=0.0,
    capability=0.35, latency_ms=50,
    model="gemini-nano",       # Pixel 9 AICore — on-device, offline-capable
)
phi_mini = Provider(
    name="phi-3-mini", type="local",
    cost_per_input_token=0.0, cost_per_output_token=0.0,
    capability=0.28, latency_ms=120,
    model="phi-3-mini",        # mid-range Android — smaller on-device model
)
gemini_flash = Provider(
    name="gemini-flash", type="cloud",
    cost_per_input_token=0.0000001, cost_per_output_token=0.0000004,
    capability=0.70, latency_ms=300,
    model="gemini-2.0-flash",  # cost-effective cloud
)
gpt4o = Provider(
    name="gpt-4o", type="cloud",
    cost_per_input_token=0.0000025, cost_per_output_token=0.00001,
    capability=0.95, latency_ms=800,
    model="gpt-4o",            # frontier reasoning
)

# ── DIO setup ─────────────────────────────────────────────────────────────────
dio = DIO(
    use_fde=True,
    fde_weights={
        "privacy":    0.40,   # PII always forces on-device
        "cost":       0.20,   # tier caps enforce hard limits; weight balances the rest
        "capability": 0.30,   # routes complex queries to capable models when budget allows
        "latency":    0.10,   # on-device models win latency (50ms vs 800ms cloud)
    },
    privacy_providers=["gemini-nano", "phi-3-mini"],  # PII stays on-device
)

openai_key = os.getenv("OPENAI_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")

dio.add_provider(gemini_nano)   # always mock (no on-device Gemini Nano SDK in Python)
dio.add_provider(phi_mini)      # always mock

if openai_key and OpenAIProvider:
    dio.add_provider(gpt4o, adapter=OpenAIProvider(gpt4o, api_key=openai_key))
else:
    dio.add_provider(gpt4o)

if google_key and GeminiProvider:
    dio.add_provider(gemini_flash, adapter=GeminiProvider(gemini_flash, api_key=google_key))
else:
    dio.add_provider(gemini_flash)

# ── Helper ────────────────────────────────────────────────────────────────────
def _score_str(metadata: dict) -> str:
    p = round(metadata.get("privacy_score", 0), 1)
    c = round(metadata.get("cost_score", 0), 1)
    cap = round(metadata.get("capability_score", 0), 1)
    lat = round(metadata.get("latency_score", 0), 1)
    return f"privacy={p}  cost={c}  cap={cap}  lat={lat}"


def run_scenario(label: str, device_state: str, prompt: str, note: str, **fde_kwargs):
    print(f"  {label}")
    print(f"  Device : {device_state}")
    print(f"  Prompt : {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    response = dio.route(prompt, **fde_kwargs)
    provider = dio.providers[response.provider]
    print(f"  Routed : {response.provider}  (model={provider.model})")
    print(f"  Scores : {_score_str(response.metadata)}")
    print(f"  [{note}]")
    print()


# ── Demo ──────────────────────────────────────────────────────────────────────
print("=" * 80)
print("DEMO: ANDROID-AWARE ROUTING")
print("  All routing decisions made server-side — no Android hardware needed.")
print()
print(f"  {'Provider':<16}  {'Type':<7}  {'Capability':>10}  {'Latency':>9}  {'Cost':<10}")
print(f"  {'-'*16}  {'-'*7}  {'-'*10}  {'-'*9}  {'-'*10}")
for p in [gemini_nano, phi_mini, gemini_flash, gpt4o]:
    latency = f"{p.latency_ms}ms" if p.latency_ms else "~800ms"
    cost = "$0 (electricity)" if p.cost_per_input_token == 0 else f"${p.cost_per_input_token:.7f}/tok"
    print(f"  {p.name:<16}  {p.type:<7}  {p.capability:>10.2f}  {latency:>9}  {cost}")
live = []
if openai_key: live.append("OpenAI")
if google_key: live.append("Google")
mode = f"Live: {', '.join(live)} | Mock: rest" if live else "Mock mode (set API keys for live responses)"
print(f"\n  {mode}")
print("=" * 80)
print()

# Scenario 1: Low battery — force on-device regardless of query
run_scenario(
    label="Scenario 1: Low battery (15%)",
    device_state="battery=15%  connectivity=wifi",
    prompt="What is Python?",
    note="battery < 20 → require_local → gemini-nano (Goldilocks for simple)",
    require_local=True,
)

# Scenario 2: Offline — only local providers are eligible
run_scenario(
    label="Scenario 2: Offline mode",
    device_state="battery=80%  connectivity=offline",
    prompt="Explain the CAP theorem in distributed systems and compare consistency models.",
    note="offline → require_local → gemini-nano (higher capability wins for complex)",
    require_local=True,
)

# Scenario 3: Server-resolved cost cap (e.g. free tier) — blocks frontier model
run_scenario(
    label="Scenario 3: Cost-capped request (max_cost=$0.0001)",
    device_state="battery=85%  connectivity=wifi  max_cost=$0.0001 (server-resolved from tier)",
    prompt="Explain the CAP theorem in distributed systems and compare Cassandra vs Spanner.",
    note="max_cost=$0.0001 blocks gpt-4o ($0.00065/request) → gemini-flash",
    max_cost=0.0001,
)

# Scenario 4: Higher cost cap (e.g. premium tier) — frontier model wins
run_scenario(
    label="Scenario 4: Higher cost cap (max_cost=$0.005)",
    device_state="battery=85%  connectivity=wifi  max_cost=$0.005 (server-resolved from tier)",
    prompt="Explain the CAP theorem in distributed systems and compare Cassandra vs Spanner.",
    note="max_cost=$0.005 allows gpt-4o → highest capability wins for complex reasoning",
    max_cost=0.005,
)

# Scenario 5: PII detected — forced on-device regardless of connectivity/tier
# PII detector recognizes email, SSN, phone, credit card numbers.
run_scenario(
    label="Scenario 5: Message with PII (email detected)",
    device_state="battery=90%  connectivity=wifi  tier=premium",
    prompt="My email is patient@clinic.com and I've been having chest pain. What should I do?",
    note="PII detected (email) → forced to privacy_providers → gemini-nano (never leaves device)",
    # No kwargs — PII detection is automatic inside FDE
)

# Scenario 6: Optimal conditions + simple query — on-device wins via Goldilocks
run_scenario(
    label="Scenario 6: WiFi + premium + simple query",
    device_state="battery=90%  connectivity=wifi  tier=premium",
    prompt="What is Python?",
    note="simple query: gemini-nano (Goldilocks ~0.5 ideal) + free + fast beats cloud",
    max_cost=0.005,
)

print("=" * 80)
print("KEY INSIGHTS FOR ANDROID DEVELOPERS")
print("  1. Battery < 20%   → on-device (no HTTP drain, works in background)")
print("  2. Offline mode     → on-device only (Room for AI: cache-first pattern)")
print("  3. Free tier        → cost cap blocks frontier models automatically")
print("  4. PII / health data → stays on-device, never sent to cloud")
print("  5. Simple queries   → on-device sufficient (Goldilocks scoring)")
print("  6. Complex reasoning → cloud when budget and connectivity allow")
print()
print("  Deploy the DIO server (pip install aigentic[server]) and call it from")
print("  Kotlin via Retrofit — routing logic lives server-side, not in the APK.")
print("=" * 80)
