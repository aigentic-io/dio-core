"""Tests for the DIO REST API server (aigentic.server).

Run: pytest tests/test_server.py -v
Requires: pip install -e ".[dev,server]"
"""

import warnings

import pytest
from fastapi.testclient import TestClient

from aigentic.core import DIO, Provider
from aigentic.server.app import app

# ── Test DIO fixture ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def test_dio():
    """Replace app.state.dio with a controlled two-provider test instance.

    local-mock  — type=local, cost=0, latency_ms=50  (privacy/local provider)
    cloud-mock  — type=cloud, cost=$0.003/1K tokens  (standard cloud)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dio = DIO(
            use_fde=True,
            fde_weights={"privacy": 0.40, "cost": 0.20, "capability": 0.30, "latency": 0.10},
            privacy_providers=["local-mock"],
        )
        dio.add_provider(Provider(
            name="local-mock", type="local",
            model="test-local", capability=0.4, latency_ms=50,
        ))
        dio.add_provider(Provider(
            name="cloud-mock", type="cloud",
            cost_per_input_token=0.000003, cost_per_output_token=0.000003,
            model="test-cloud", capability=0.9,
        ))
    app.state.dio = dio


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ── Health / providers ────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["fde"] is True
    names = [p["name"] for p in body["providers"]]
    assert "local-mock" in names
    assert "cloud-mock" in names


def test_providers_list(client):
    r = client.get("/providers")
    assert r.status_code == 200
    providers = {p["name"]: p for p in r.json()}
    assert "local-mock" in providers
    assert "cloud-mock" in providers
    assert providers["local-mock"]["type"] == "local"
    assert providers["cloud-mock"]["cost_per_input_token"] == pytest.approx(0.000003)


# ── /infer — request structure ────────────────────────────────────────────────

def test_infer_basic_text(client):
    r = client.post("/infer", json={
        "messages": [{"role": "user", "content": "What is Python?"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] in ("local-mock", "cloud-mock")
    assert body["routed_by"] == "fde"
    assert "score" in body["metadata"]


def test_infer_system_and_user_messages(client):
    r = client.post("/infer", json={
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ],
    })
    assert r.status_code == 200


def test_infer_missing_messages_rejected(client):
    """messages field is required — Pydantic must reject missing field."""
    r = client.post("/infer", json={})
    assert r.status_code == 422


def test_infer_invalid_role_rejected(client):
    r = client.post("/infer", json={
        "messages": [{"role": "unknown", "content": "Hello"}],
    })
    assert r.status_code == 422


def test_infer_temperature_and_max_tokens_accepted(client):
    """temperature and max_tokens pass through to the provider without error."""
    r = client.post("/infer", json={
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.3,
        "max_tokens": 256,
    })
    assert r.status_code == 200


# ── /infer — multimodal guard ─────────────────────────────────────────────────

def test_infer_image_only_rejected(client):
    """Pure image message must return 422 with clear multimodal error."""
    r = client.post("/infer", json={
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]}],
    })
    assert r.status_code == 422
    assert "Multimodal" in r.json()["detail"]["error"]


def test_infer_mixed_text_image_rejected(client):
    """Mixed text+image content must also be rejected."""
    r = client.post("/infer", json={
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe this:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]}],
    })
    assert r.status_code == 422


# ── /infer — routing decisions ────────────────────────────────────────────────

def test_infer_battery_routes_local(client):
    """battery_level < 20 must force local routing with descriptive reason."""
    r = client.post("/infer", json={
        "messages": [{"role": "user", "content": "Explain quantum computing"}],
        "client_context": {"battery_level": 15, "connectivity": "wifi"},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "local-mock"
    assert "Low battery (15%) — optimal for battery conservation" == body["metadata"]["reason"]


def test_infer_offline_routes_local(client):
    """connectivity=offline must force local routing with descriptive reason."""
    r = client.post("/infer", json={
        "messages": [{"role": "user", "content": "Hello"}],
        "client_context": {"connectivity": "offline"},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "local-mock"
    assert body["metadata"]["reason"] == "Offline — no network, routed to local provider"


def test_infer_offline_and_low_battery(client):
    """Both offline and battery < 20 must surface both triggers in the reason."""
    r = client.post("/infer", json={
        "messages": [{"role": "user", "content": "Hello"}],
        "client_context": {"connectivity": "offline", "battery_level": 8},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "local-mock"
    assert body["metadata"]["reason"] == "Offline and low battery (8%) — local provider selected"


def test_infer_pii_routes_to_privacy_provider(client):
    """PII content must route to the privacy provider with pii_detected metadata."""
    r = client.post("/infer", json={
        "messages": [{"role": "user", "content": "My SSN is 123-45-6789, please help."}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "local-mock"
    assert body["metadata"].get("pii_detected") is True
    assert "PII" in body["metadata"]["reason"]


def test_infer_require_local_explicit_override(client):
    """require_local=True must force local with 'explicit override' in reason."""
    r = client.post("/infer", json={
        "messages": [{"role": "user", "content": "Hello"}],
        "require_local": True,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "local-mock"
    assert body["metadata"]["reason"] == "require_local (explicit override)"


# ── /infer — tracing headers ─────────────────────────────────────────────────

def test_infer_request_id_echoed(client):
    """X-Request-ID from client must be echoed back in the response header."""
    r = client.post("/infer",
        json={"messages": [{"role": "user", "content": "Hello"}]},
        headers={"X-Request-ID": "test-trace-abc123"},
    )
    assert r.status_code == 200
    assert r.headers.get("x-request-id") == "test-trace-abc123"


def test_infer_server_generates_request_id(client):
    """When X-Request-ID is absent the server must generate and echo one."""
    r = client.post("/infer",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert r.status_code == 200
    assert r.headers.get("x-request-id")  # non-empty


# ── platform / connectivity optional (no assumed defaults) ────────────────────

def test_client_context_no_defaults(client):
    """platform and connectivity must be optional with no assumed defaults."""
    r = client.post("/infer", json={
        "messages": [{"role": "user", "content": "Hello"}],
        "client_context": {},  # empty context — no platform or connectivity
    })
    assert r.status_code == 200
