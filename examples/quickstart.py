"""DIO Quickstart Example"""

import sys
from pathlib import Path

# Add parent directory to path for development mode
sys.path.insert(0, str(Path(__file__).parent.parent))

from aigentic.core import DIO, Provider
from aigentic.core.provider import MockProvider
from aigentic.core.pii_detector import has_pii

# Setup providers with per-token pricing
cloud = Provider(name="bedrock", type="cloud", cost_per_input_token=0.01, cost_per_output_token=0.03, capability=0.9)
local = Provider(name="vllm-secure", type="local", cost_per_input_token=0.005, cost_per_output_token=0.015, capability=0.5)

# Initialize DIO
dio = DIO()
dio.add_provider(cloud)
dio.add_provider(local)

# =============================================================================
# WORKSHOP EXERCISE 1: Privacy-First Routing (COMPLETED)
# =============================================================================
# Add privacy policy
def privacy_rule(request):
    return "RESTRICTED" if has_pii(request.prompt) else "PUBLIC"

dio.add_policy(rule=privacy_rule, enforcement="strict")
# No explicit mappings needed! DIO automatically routes:
# - "RESTRICTED" → local providers (data sovereignty) — hard constraint
# - "PUBLIC" → allows advisory policies to refine, falls back to cloud

# =============================================================================
# WORKSHOP EXERCISE 2: Cost Optimizer Policy
# WORKSHOP TODO: Uncomment the code below (remove the # at start of each line)
# =============================================================================
# def cost_optimizer(request):
#     """Route simple queries to cheap local model, complex to cloud."""
#     prompt = request.prompt.lower()
#     simple_keywords = ["what is", "define", "explain briefly", "simple"]
#
#     # Check if prompt contains simple keywords
#     for keyword in simple_keywords:
#         if keyword in prompt:
#             return "SIMPLE"
#
#     return "COMPLEX"
#
# # Add the cost optimizer policy (advisory = can be overridden by privacy)
# dio.add_policy(rule=cost_optimizer, enforcement="advisory")
#
# # Map custom classifications to providers
# dio.set_classification_mapping("SIMPLE", local)   # Cheap queries → local
# dio.set_classification_mapping("COMPLEX", cloud)  # Complex queries → cloud

# =============================================================================
# WORKSHOP EXERCISE 3: Per-Token Cost Estimation
# WORKSHOP TODO: Uncomment the function below to see per-token cost breakdown
# =============================================================================
# Providers have separate input/output token costs. You can estimate the cost
# of a request before sending it — useful for budgeting and cost-aware routing.
#
# def show_cost_estimate(provider, input_tokens, output_tokens):
#     """Print a per-token cost breakdown for a provider."""
#     cost = provider.estimated_cost(input_tokens, output_tokens)
#     print(f"    Provider: {provider.name}")
#     print(f"    Input:  {input_tokens} tokens x ${provider.cost_per_input_token} = ${input_tokens * provider.cost_per_input_token:.4f}")
#     print(f"    Output: {output_tokens} tokens x ${provider.cost_per_output_token} = ${output_tokens * provider.cost_per_output_token:.4f}")
#     print(f"    Total estimated cost: ${cost:.4f}")

# =============================================================================
# Test All Routing Scenarios
# =============================================================================
print("=" * 80)
print("DIO SMART ROUTING EXAMPLES")
print("=" * 80)
print()

# --- Exercise 1 examples (always active) ---

print("📝 Example 1: Privacy Policy - SSN detected (PII → local)")
response = dio.route("My SSN is 123-45-6789")
print(f"  Prompt: 'My SSN is 123-45-6789'")
print(f"  Provider: {response.provider}")
print(f"  Content: {response.content}")
print(f"  💡 PII detected → strict RESTRICTED → forced to local provider")
print()

print("📝 Example 2: Privacy Policy - Email detected (PII → local)")
response = dio.route("Please contact me at user@example.com")
print(f"  Prompt: 'Please contact me at user@example.com'")
print(f"  Provider: {response.provider}")
print(f"  Content: {response.content}")
print(f"  💡 PII detected → strict RESTRICTED → forced to local provider")
print()

# --- Exercise 2 & 3 examples (behavior changes after uncommenting) ---

cost_optimizer_active = "cost_optimizer" in dir()
cost_estimation_active = "show_cost_estimate" in dir()

print("📝 Example 3: Cost Optimizer - Simple query should route to local")
response = dio.route("What is Python?")
print(f"  Prompt: 'What is Python?'")
print(f"  Provider: {response.provider}")
print(f"  Content: {response.content}")
if cost_optimizer_active:
    print(f"  💡 No PII → cost optimizer classifies SIMPLE → routed to local (cheaper)")
else:
    print(f"  💡 No PII → PUBLIC → routed to cloud (no cost optimizer active)")
    print(f"  💡 After Exercise 2: Will route to local via SIMPLE classification")
print()

print("📝 Example 4: Cost Optimizer - Complex query should route to cloud")
response = dio.route("Explain the CAP theorem in distributed systems with examples")
print(f"  Prompt: 'Explain the CAP theorem...'")
print(f"  Provider: {response.provider}")
print(f"  Content: {response.content}")
if cost_optimizer_active:
    print(f"  💡 No PII → cost optimizer classifies COMPLEX → routed to cloud (best capability)")
else:
    print(f"  💡 No PII → PUBLIC → routed to cloud (no cost optimizer active)")
    print(f"  💡 After Exercise 2: Complex queries will still route to cloud")
print()

print("📝 Example 5: Strict vs Advisory - PII overrides cost optimizer")
response = dio.route("Analyze the cryptographic security of SSN 123-45-6789 in distributed systems")
print(f"  Prompt: 'Analyze the cryptographic security of SSN 123-45-6789...'")
print(f"  Provider: {response.provider}")
if cost_optimizer_active:
    print(f"  💡 PII detected → strict RESTRICTED overrides advisory cost optimizer")
    print(f"  💡 Even though cost optimizer would classify COMPLEX → cloud, privacy wins")
else:
    print(f"  💡 PII detected → strict RESTRICTED → forced to local provider")
    print(f"  💡 After Exercise 2: Privacy will still override cost optimizer here")
print()

# --- Exercise 3 example ---

print("📝 Example 6: Per-Token Cost Estimation - Compare provider costs")
input_tokens = 50
output_tokens = 150
print(f"  Scenario: A typical query with ~{input_tokens} input and ~{output_tokens} output tokens")
if cost_estimation_active:
    print()
    show_cost_estimate(cloud, input_tokens, output_tokens)
    print()
    show_cost_estimate(local, input_tokens, output_tokens)
    print()
    savings = cloud.estimated_cost(input_tokens, output_tokens) - local.estimated_cost(input_tokens, output_tokens)
    print(f"  💡 Local is ${savings:.4f} cheaper per request")
    print(f"  💡 At 1000 requests/day, that's ${savings * 1000:.2f}/day saved by routing simple queries locally")
else:
    print(f"  Cloud estimated cost: ${cloud.estimated_cost(input_tokens, output_tokens):.4f}")
    print(f"  Local estimated cost: ${local.estimated_cost(input_tokens, output_tokens):.4f}")
    print(f"  💡 After Exercise 3: Will show full per-token cost breakdown")
print()

# --- Production features (always active) ---

# Simulate a cloud provider that fails (e.g., API outage, rate limit)
class FailingProvider(MockProvider):
    """Simulates a provider that is down."""
    def generate(self, prompt, **kwargs):
        raise ConnectionError("Service unavailable (simulated outage)")

print("📝 Example 7: Failover - Primary cloud provider is down")

# Set up a failing primary and a healthy backup
cloud_primary = Provider(name="bedrock-primary", type="cloud", cost_per_input_token=0.01, cost_per_output_token=0.03)
cloud_backup = Provider(name="bedrock-backup", type="cloud", cost_per_input_token=0.008, cost_per_output_token=0.025)

failover_dio = DIO()
failover_dio.add_provider(cloud_primary, adapter=FailingProvider(cloud_primary))
failover_dio.add_provider(cloud_backup)  # healthy mock

# Configure fallback: if primary raises ConnectionError, use backup
failover_dio.set_fallback(cloud_primary, cloud_backup, trigger=ConnectionError)

response = failover_dio.route("Explain quantum computing")
print(f"  Prompt: 'Explain quantum computing'")
print(f"  Primary provider: bedrock-primary (simulated outage)")
print(f"  Provider used: {response.provider}")
print(f"  Was fallback used: {response.was_fallback}")
print(f"  Fallback reason: {response.metadata.get('fallback_reason', 'N/A')}")
print(f"  Content: {response.content}")
print(f"  💡 Primary failed with ConnectionError → automatically fell back to backup")
print(f"  💡 In production: Handles API errors, rate limits, and downtime")
print()

print("=" * 80)
print("QUICKSTART COMPLETE!")
print("=" * 80)
print("You've learned:")
print("  1. Privacy-first routing (PII detection → local)")
print("  2. Cost optimization (simple vs complex queries) [Exercise 2]")
print("  3. Per-token cost estimation (input/output pricing) [Exercise 3]")
print("  4. Policy enforcement (strict overrides advisory)")
print("  5. Automatic failover for production resilience")
print("=" * 80)
