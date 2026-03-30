"""Tests for aigentic.registry.client — lookup logic and get_pricing() API."""

import aigentic.registry.client as _rc

_TEXT_PLAN = {
    "plan_name": "standard",
    "currency": "USD",
    "modality": ["text"],
    "base": {"tier": 0, "input": 2.50, "output": 10.00},
    "caching": {"unit": "per_million_tokens", "creation": None, "read": 1.25, "storage": None},
}

_IMAGE_PLAN = {
    "plan_name": "image",
    "currency": "USD",
    "modality": ["image"],
    "base": {"tier": 0, "input": 3.00, "output": 12.00},
    "caching": {"unit": "per_million_tokens", "creation": None, "read": None, "storage": None},
}

_SAMPLE_REGISTRY = {
    "models": {
        "gpt-4o": {
            "pricing": {"pricing_plans": [_TEXT_PLAN, _IMAGE_PLAN]},
        },
        "llama3": {
            "pricing": {"pricing_plans": [_TEXT_PLAN]},
        },
        "llama3-latest": {
            "pricing": {"pricing_plans": [_TEXT_PLAN]},
        },
    }
}


def _patch_cache(monkeypatch, data):
    monkeypatch.setattr(_rc, "_memory_cache", data)


class TestGetPricingGuards:
    def test_returns_none_when_cache_is_none(self, monkeypatch):
        _patch_cache(monkeypatch, None)
        assert _rc.get_pricing("gpt-4o") is None

    def test_returns_none_for_empty_model_name(self, monkeypatch):
        _patch_cache(monkeypatch, _SAMPLE_REGISTRY)
        assert _rc.get_pricing("") is None

    def test_returns_none_for_unknown_model(self, monkeypatch):
        _patch_cache(monkeypatch, _SAMPLE_REGISTRY)
        assert _rc.get_pricing("does-not-exist") is None


class TestGetPricingPlanSelection:
    def test_returns_matching_modality_plan(self, monkeypatch):
        _patch_cache(monkeypatch, _SAMPLE_REGISTRY)
        plan = _rc.get_pricing("gpt-4o", modality="image")
        assert plan is _IMAGE_PLAN

    def test_defaults_to_text_modality(self, monkeypatch):
        _patch_cache(monkeypatch, _SAMPLE_REGISTRY)
        plan = _rc.get_pricing("gpt-4o")
        assert plan is _TEXT_PLAN

    def test_falls_back_to_first_plan_when_modality_missing(self, monkeypatch):
        _patch_cache(monkeypatch, _SAMPLE_REGISTRY)
        # modality "video" not in any plan — falls back to first plan
        plan = _rc.get_pricing("gpt-4o", modality="video")
        assert plan is _TEXT_PLAN

    def test_model_name_case_insensitive(self, monkeypatch):
        _patch_cache(monkeypatch, _SAMPLE_REGISTRY)
        assert _rc.get_pricing("GPT-4O") is _TEXT_PLAN


class TestFourStepLookup:
    @property
    def models(self):
        return _SAMPLE_REGISTRY["models"]

    def test_exact_match(self):
        result = _rc._four_step_lookup(self.models, "gpt-4o")
        assert result is self.models["gpt-4o"]

    def test_gateway_prefix_stripped(self):
        # "openai/gpt-4o" → strip prefix → exact match "gpt-4o"
        result = _rc._four_step_lookup(self.models, "openai/gpt-4o")
        assert result is self.models["gpt-4o"]

    def test_ollama_tag_suffix_stripped(self):
        # "llama3:latest" → strip tag → exact match "llama3"
        result = _rc._four_step_lookup(self.models, "llama3:latest")
        assert result is self.models["llama3"]

    def test_colon_to_hyphen(self):
        # colon-to-hyphen step: "llama3:latest" → "llama3-latest"
        models = {"llama3-latest": _TEXT_PLAN}
        result = _rc._four_step_lookup(models, "llama3:latest")
        assert result is _TEXT_PLAN

    def test_longest_prefix_match(self):
        # "gpt-4o-2024-11-20" has no exact match — prefix falls back to "gpt-4o"
        result = _rc._four_step_lookup(self.models, "gpt-4o-2024-11-20")
        assert result is self.models["gpt-4o"]

    def test_returns_none_when_no_match(self):
        result = _rc._four_step_lookup(self.models, "completely-unknown-xyz")
        assert result is None

    def test_longer_key_wins_prefix_match(self):
        # "gpt-4o-mini" must match "gpt-4o" (longer key), not "gpt-4"
        models = {
            "gpt-4": {"pricing": {"pricing_plans": []}},
            "gpt-4o": {"pricing": {"pricing_plans": [_TEXT_PLAN]}},
        }
        result = _rc._four_step_lookup(models, "gpt-4o-mini")
        assert result is models["gpt-4o"]
