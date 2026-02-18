# DIO - Dynamic Intelligence Orchestrator

**Smart routing for cloud and on-premises AI models**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DIO is a production-ready framework that intelligently routes AI requests between cloud and local models based on privacy, cost, complexity, and performance requirements.

---

## 🚀 Quick Start

```bash
# Install
pip install -e ".[all]"

# Create .env file with your API key (optional)
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Run example
python3 examples/quickstart.py
```

**Output:**
```
Example 1: Sensitive data (contains PII)
  Provider: vllm-secure (local)
  Content: [PII stays on-premises]

Example 2: Normal query (no PII)
  Provider: bedrock (cloud)
  Content: [Uses best cloud model]
```

---

## ✨ Key Features

### 1. Privacy-First Routing

Automatic PII detection keeps sensitive data local:

```python
from aigentic.core import DIO, Provider
from aigentic.core.pii_detector import has_pii

cloud = Provider(name="gpt-4o-mini", type="cloud", cost_per_input_token=0.005, cost_per_output_token=0.02)
local = Provider(name="local-llm", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0)

dio = DIO()
dio.add_provider(cloud)
dio.add_provider(local)

def privacy_rule(request):
    return "RESTRICTED" if has_pii(request.prompt) else "PUBLIC"

dio.add_policy(rule=privacy_rule, enforcement="strict")

# PII stays local
print(dio.route("My SSN is 123-45-6789").provider)  # → local-llm

# Public queries use cloud
print(dio.route("What is Python?").provider)  # → gpt-4o-mini
```

**PII Detection (regex-based, no cloud calls):**
- SSN: `123-45-6789`
- Credit Cards: `1234-5678-9012-3456`
- Email: `user@example.com`
- Phone: `555-123-4567`

### 2. Federated Decision Engine (FDE)

Multi-factor routing that optimizes across privacy, cost, capability, and latency:

```python
dio = DIO(
    use_fde=True,
    fde_weights={
        "privacy": 0.40,     # Privacy-first (PII → local)
        "cost": 0.25,        # Cost optimization
        "capability": 0.25,  # Match complexity
        "latency": 0.10,     # Performance
    },
    privacy_providers=["ollama-llama3.2"]
)

# Automatically routes based on context
simple = dio.route("What is Python?")
# → ollama-llama3.2 (local, cost-efficient)

complex = dio.route("Explain CAP theorem with distributed consensus examples")
# → openai-gpt4o-mini (cloud, better capability)

pii = dio.route("My SSN is 123-45-6789")
# → ollama-llama3.2 (local, privacy constraint)
```

**How FDE Works:**
1. Analyze prompt (PII detection, complexity classification, token estimation)
2. Score each provider on 4 factors (cost scoring uses per-token input/output rates)
3. Weighted sum determines best provider
4. Returns provider + detailed scoring breakdown (including estimated cost)

### 3. Smart Defaults (Convention Over Configuration)

Common routing patterns work automatically without explicit mappings:

```python
# Smart defaults - no mapping needed!
def privacy_rule(request):
    return "RESTRICTED" if has_pii(request.prompt) else "PUBLIC"
    # RESTRICTED → automatically routes to type="local"
    # PUBLIC → automatically routes to type="cloud"

dio.add_policy(rule=privacy_rule, enforcement="strict")
```

**Built-in smart defaults:**
- `RESTRICTED` / `PRIVATE` → local providers
- `PUBLIC` → cloud providers

**Custom classifications need explicit mappings:**
```python
def cost_optimizer(request):
    return "SIMPLE" if is_simple(request.prompt) else "COMPLEX"

dio.add_policy(rule=cost_optimizer, enforcement="advisory")

# Set mappings for custom classifications
dio.set_classification_mapping("SIMPLE", local)
dio.set_classification_mapping("COMPLEX", cloud)
```

### 4. Real Cloud Provider Integration

Works with your existing API keys:

```python
from aigentic.providers.openai import OpenAIProvider
from aigentic.providers.gemini import GeminiProvider
from aigentic.providers.claude import ClaudeProvider

# Connect real APIs
dio.add_provider(
    cloud,
    adapter=OpenAIProvider(cloud, api_key=os.getenv("OPENAI_API_KEY"))
)

# Or use Gemini
dio.add_provider(
    cloud,
    adapter=GeminiProvider(cloud, api_key=os.getenv("GOOGLE_API_KEY"))
)

# Get real LLM responses
response = dio.route("Complex query")
print(response.content)  # Actual LLM output
```

**Supported providers:**
- OpenAI (GPT-4, GPT-4o-mini)
- Google Gemini
- Anthropic Claude
- Local Ollama
- Mock (for testing)

### 5. Extensible Policy Framework

Policies are just Python functions:

```python
def my_policy(request):
    # Your custom logic here
    if some_condition(request.prompt):
        return "CLASSIFICATION_A"
    return "CLASSIFICATION_B"

dio.add_policy(rule=my_policy, enforcement="strict")
```

**Policy enforcement levels:**
- `strict` - Must be satisfied, overrides other policies
- `advisory` - Suggestion, can be overridden by strict policies

**Multiple policies work together:**
```python
# Privacy policy (strict) + Cost optimizer (advisory)
dio.add_policy(rule=privacy_rule, enforcement="strict")
dio.add_policy(rule=cost_optimizer, enforcement="advisory")

# Strict policies take precedence
# Both policies evaluated, best match wins
```

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip

### Basic Installation

```bash
# Clone repository
git clone https://github.com/yourusername/dio-core.git
cd dio-core

# Install DIO (all providers)
pip install -e ".[all]"
```

### Installation Options

```bash
# Cloud providers only (OpenAI, Gemini, Claude)
pip install -e ".[cloud]"

# Local providers only (Ollama)
pip install -e ".[local]"

# Development tools
pip install -e ".[dev]"
```

### API Key Setup (Optional)

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
GOOGLE_API_KEY=your-key-here
```

DIO works with mock providers without API keys - perfect for learning and testing!

---

## 🎓 Workshop - DeveloperWeek 2026

**Building a Smart Router for Cloud & On-Prem AI**
**Duration:** 50 minutes
**Level:** Intermediate

### What You'll Build

By the end of the workshop, you'll have a working smart router with:

1. ✅ Privacy-first routing (PII stays local)
2. ✅ Cost optimization (simple queries to local)
3. ✅ Multi-policy system (both policies working together)

### Workshop Structure

**Part 1: Introduction & Demo (10 min)**
- The Cloud LLM Era isn't ending - it's evolving
- The "token tax" problem
- Live FDE demo

**Part 2: Hands-On Lab (30 min)**

*Exercise 1: Privacy Policy (10 min)*
- Pre-written privacy policy
- Run and observe PII detection
- Learn smart defaults

*Exercise 2: Cost Optimizer (20 min)*
- YOU write a custom policy
- Classify prompts as SIMPLE/COMPLEX
- Test and debug your implementation

**Part 3: Real APIs & Wrap-Up (10 min)**
- Connect real cloud providers
- See actual LLM responses
- Q&A and next steps

### Prerequisites

**Required:**
- Python 3.8+
- DIO installed (`pip install -e ".[all]"`)

**Highly Recommended:**
- API key from OpenAI, Gemini, or Claude
- `.env` file configured
- Provider packages installed

**Setup verification:**
```bash
python3 examples/quickstart.py
```

Expected output shows PII routing to local and public queries routing to cloud.

### Exercise Templates

**Exercise 1: Privacy Policy (Pre-written)**

```python
from aigentic.core import DIO, Provider
from aigentic.core.pii_detector import has_pii

cloud = Provider(name="gpt-4o-mini", type="cloud", cost_per_input_token=0.005, cost_per_output_token=0.02)
local = Provider(name="local-llm", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0)

dio = DIO()
dio.add_provider(cloud)
dio.add_provider(local)

def privacy_rule(request):
    return "RESTRICTED" if has_pii(request.prompt) else "PUBLIC"

dio.add_policy(rule=privacy_rule, enforcement="strict")

# Test it!
print(dio.route("My SSN is 123-45-6789").provider)  # local-llm
print(dio.route("What is Python?").provider)         # gpt-4o-mini
```

**Exercise 2: Cost Optimizer (YOU build this)**

```python
# TODO: Write your cost-optimization policy
def cost_optimizer(request):
    prompt = request.prompt.lower()
    simple_keywords = ["simple", "quick", "define", "what is", "explain briefly"]

    # YOUR CODE HERE: Check if any keyword is in prompt
    # If yes, return "SIMPLE"
    # If no, return "COMPLEX"

    return "???"  # Replace with your logic

# Add your policy
dio.add_policy(rule=cost_optimizer, enforcement="advisory")

# Set mappings
dio.set_classification_mapping("SIMPLE", local)
dio.set_classification_mapping("COMPLEX", cloud)

# Test it!
print(dio.route("What is Python?").provider)  # Should be local
print(dio.route("Explain distributed consensus").provider)  # Should be cloud
```

---

## 📖 Usage Examples

### Example 1: Basic Privacy Routing

```python
from aigentic.core import DIO, Provider
from aigentic.core.pii_detector import has_pii

# Setup providers
cloud = Provider(name="gpt-4o-mini", type="cloud", cost_per_input_token=0.005, cost_per_output_token=0.02)
local = Provider(name="local-llm", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0)

dio = DIO()
dio.add_provider(cloud)
dio.add_provider(local)

# Privacy policy
def privacy_rule(request):
    return "RESTRICTED" if has_pii(request.prompt) else "PUBLIC"

dio.add_policy(rule=privacy_rule, enforcement="strict")

# Test
pii_response = dio.route("My SSN is 123-45-6789")
print(f"Provider: {pii_response.provider}")  # local-llm
print(f"Content: {pii_response.content}")

public_response = dio.route("What is Python?")
print(f"Provider: {public_response.provider}")  # gpt-4o-mini
```

### Example 2: FDE Multi-Factor Routing

```python
from aigentic.core import DIO, Provider

cloud = Provider(name="gpt-4o-mini", type="cloud", cost_per_input_token=0.005, cost_per_output_token=0.02, capability=0.8)
local = Provider(name="llama3.2", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0, capability=0.4)

dio = DIO(
    use_fde=True,
    fde_weights={
        "privacy": 0.40,
        "cost": 0.25,
        "capability": 0.25,
        "latency": 0.10,
    },
    privacy_providers=["llama3.2"]
)

dio.add_provider(cloud)
dio.add_provider(local)

# Simple query → local (cost-efficient)
simple = dio.route("What is Python?")
print(f"Simple: {simple.provider}, Score: {simple.metadata.get('score')}")

# Complex query → cloud (better capability)
complex = dio.route("Explain CAP theorem in distributed systems")
print(f"Complex: {complex.provider}, Score: {complex.metadata.get('score')}")

# PII → local (privacy constraint)
pii = dio.route("My SSN is 123-45-6789")
print(f"PII: {pii.provider}, Score: {pii.metadata.get('score')}")
```

### Example 3: Real API Integration

```python
import os
from dotenv import load_dotenv
from aigentic.core import DIO, Provider
from aigentic.providers.openai import OpenAIProvider

load_dotenv()

cloud = Provider(name="gpt-4o-mini", type="cloud", cost_per_input_token=0.005, cost_per_output_token=0.02)
dio = DIO()

# Connect real OpenAI API
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    dio.add_provider(
        cloud,
        adapter=OpenAIProvider(cloud, api_key=openai_key)
    )
else:
    dio.add_provider(cloud)  # Uses mock if no key

# Get real response
response = dio.route("Explain quantum computing in 2 sentences")
print(response.content)  # Actual GPT-4o-mini response
```

### Example 4: Custom Multi-Policy System

```python
from aigentic.core import DIO, Provider
from aigentic.core.pii_detector import has_pii

cloud = Provider(name="gpt-4o-mini", type="cloud", cost_per_input_token=0.005, cost_per_output_token=0.02)
local = Provider(name="local-llm", type="local", cost_per_input_token=0.0, cost_per_output_token=0.0)

dio = DIO()
dio.add_provider(cloud)
dio.add_provider(local)

# Policy 1: Privacy (strict)
def privacy_rule(request):
    return "RESTRICTED" if has_pii(request.prompt) else "PUBLIC"

dio.add_policy(rule=privacy_rule, enforcement="strict")

# Policy 2: Cost optimizer (advisory)
def cost_optimizer(request):
    prompt = request.prompt.lower()
    simple_keywords = ["simple", "quick", "define", "what is"]

    for keyword in simple_keywords:
        if keyword in prompt:
            return "SIMPLE"
    return "COMPLEX"

dio.add_policy(rule=cost_optimizer, enforcement="advisory")
dio.set_classification_mapping("SIMPLE", local)
dio.set_classification_mapping("COMPLEX", cloud)

# Test all scenarios
test_cases = [
    "What is Python?",                    # SIMPLE → local
    "Explain distributed consensus",      # COMPLEX → cloud
    "My SSN is 123-45-6789",             # PII → local (strict wins)
    "Email me at test@example.com",      # PII → local (strict wins)
]

for prompt in test_cases:
    response = dio.route(prompt)
    print(f"'{prompt}' → {response.provider}")
```

---

## 🧪 Testing

DIO includes comprehensive tests:

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_pii.py -v
```

**Test Coverage:**
- ✅ PII Detection: 11/11 tests passing
- ✅ Routing Logic: 12/12 tests passing (includes smart defaults)
- ✅ Integration: 7/7 tests passing
- ✅ Cost Model: 6/6 tests passing (per-token estimation, FDE scoring)
- **Total: 36/36 tests passing**

---

## 🏗️ Architecture

### Core Components

```
dio-core/
├── aigentic/
│   ├── core/
│   │   ├── dio.py              # Main DIO class (2 modes: policy + FDE)
│   │   ├── fde.py              # Federated Decision Engine
│   │   ├── provider.py         # Provider abstraction
│   │   ├── pii_detector.py     # Privacy-first PII detection
│   │   ├── router.py           # Policy-based routing
│   │   └── response.py         # Response dataclass
│   ├── policies/
│   │   ├── privacy.py          # Privacy policy
│   │   └── fallback.py         # Fallback policy
│   └── providers/
│       ├── openai.py           # OpenAI integration
│       ├── gemini.py           # Google Gemini
│       ├── claude.py           # Anthropic Claude
│       ├── ollama.py           # Local Ollama
│       └── mock.py             # Testing/workshop
├── examples/
│   ├── quickstart.py           # Workshop quickstart (Exercises 1-3)
│   ├── workshop_setup.py       # Auto-detect & configure providers
│   └── workshop_real_providers.py  # Real cloud API integration
├── tests/
│   ├── test_pii.py             # PII detection tests
│   ├── test_router.py          # Routing logic tests
│   ├── test_integration.py     # End-to-end integration tests
│   └── test_cost.py            # Per-token cost model tests
├── pyproject.toml
├── requirements.txt
├── setup.py
└── README.md
```

### How It Works

1. **Request** → User submits prompt
2. **Policy Evaluation** → All policies evaluate the request
3. **Classification** → Request classified (e.g., RESTRICTED, SIMPLE, COMPLEX)
4. **Provider Selection** → Router selects best provider based on classification
5. **Execution** → Provider adapter generates response
6. **Response** → User receives result with metadata

### FDE Algorithm

The Federated Decision Engine scores each provider across 4 factors:

```python
overall_score = (
    privacy_score * 0.40 +      # PII detection, local preference
    cost_score * 0.25 +          # Per-token cost (input + output rates)
    capability_score * 0.25 +    # Model capability vs complexity
    latency_score * 0.10         # Response time
)
```

Cost scoring uses per-token input/output rates with estimated token counts based on prompt length and complexity. Capability scoring uses each provider's `capability` field (0.0-1.0) to differentiate models of the same type — e.g., GPT-4o-mini (0.8) vs Gemini Flash (0.6). **Provider with highest score wins.**

---

## 💡 Policy Ideas

### Business Logic
- **User tier routing**: Free → local, paid → premium cloud
- **Time-based**: Business hours → fast cloud, off-hours → local
- **Rate limiting**: After N requests → local for cost control
- **A/B testing**: Route 10% to experimental provider

### Domain-Specific
- **Healthcare**: PHI detection → HIPAA-compliant local
- **Finance**: High-value transactions → secure provider
- **Legal**: Contract review → specialized legal LLM
- **Code generation**: Different languages → specialized models

### Performance & Operations
- **SLA-based**: High-priority users → fastest provider
- **Load balancing**: Distribute across providers
- **Geo-routing**: Route based on user location
- **Failover**: Automatic fallback when primary fails

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check .
```

---

## 📚 Resources

- **GitHub:** https://github.com/yourusername/dio-core
- **Blog Series:**
  - [Part 1: The Cloud LLM Era Isn't Ending](https://medium.com/towards-aws/the-cloud-llm-era-isnt-ending-3d3f9cdfb5d7)
  - [Part 2: The Cloud LLM Era Isn't Ending (Part 2)](https://medium.com/stackademic/the-cloud-llm-era-isnt-ending-370de5a1c5bf)
  - [Part 3: The $8.4B Question - Feature-Driven Execution](https://medium.datadriveninvestor.com/the-8-4b-question-why-enterprise-ai-needs-feature-driven-execution-4412012a3264)
  - [Part 4: The $100B Code Red Problem](https://medium.com/towards-artificial-intelligence/the-100b-code-red-problem-66f0f76ce027)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built for the hybrid AI future where cloud and on-premises models work together intelligently.

**DeveloperWeek 2026** - Come build smart routers with us!

---

*Built with ❤️ for intelligent AI orchestration*
