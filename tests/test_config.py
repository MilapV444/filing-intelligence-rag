"""Checks on the single configuration point (NFR-8) and credential handling (NFR-4)."""

import pytest

from risk_analytics import config


def test_every_configured_model_has_a_published_rate():
    """FR-18 and NFR-1 both depend on being able to price any call we make. A model
    configured without a rate would make the cost report silently wrong, which is
    worse than it being absent."""
    configured = {
        config.ROUTER_MODEL,
        config.REFLECTION_MODEL,
        config.VISION_MODEL,
        config.SPECIALIST_MODEL,
        config.SYNTHESIS_MODEL,
    }
    unpriced = configured - set(config.MODEL_RATES_USD_PER_MTOK)
    assert not unpriced, f"configured models with no published rate: {sorted(unpriced)}"


def test_cost_is_computed_from_published_rates():
    # 1M input + 1M output on Opus 5 at $5 / $25.
    assert config.cost_usd("claude-opus-5", 1_000_000, 1_000_000) == pytest.approx(30.0)
    # A realistic small call should be a fraction of a cent.
    assert config.cost_usd("claude-haiku-4-5", 2_000, 500) == pytest.approx(0.0045)
    assert config.cost_usd("claude-opus-5", 0, 0) == 0.0


def test_unpriced_model_raises_rather_than_reporting_zero():
    with pytest.raises(KeyError):
        config.cost_usd("some-model-we-never-configured", 1000, 1000)


def test_cheap_model_is_actually_cheaper_than_the_analysis_model():
    """The whole point of the split (DECISION-39162611). If someone edits the
    model constants and inverts this, the cost ceiling stops being reachable."""
    cheap = config.MODEL_RATES_USD_PER_MTOK[config.ROUTER_MODEL]
    analysis = config.MODEL_RATES_USD_PER_MTOK[config.SPECIALIST_MODEL]
    assert cheap["input"] < analysis["input"]
    assert cheap["output"] < analysis["output"]


def test_run_limits_are_set_to_the_approved_values():
    assert config.RUN_COST_CEILING_USD == 0.25  # NFR-1
    assert config.MAX_SPECIALISTS_PER_QUERY == 2  # FR-13
    assert config.MAX_RETRIEVAL_ITERATIONS >= 2  # FR-8 needs room to iterate at all


def test_api_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV_VAR, "test-key-value")
    assert config.api_key() == "test-key-value"


def test_api_key_is_re_read_rather_than_cached_at_import(monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV_VAR, "first")
    assert config.api_key() == "first"
    monkeypatch.setenv(config.API_KEY_ENV_VAR, "second")
    assert config.api_key() == "second"


@pytest.mark.parametrize("value", ["", "   "])
def test_missing_or_blank_key_raises(monkeypatch, value):
    monkeypatch.setenv(config.API_KEY_ENV_VAR, value)
    with pytest.raises(config.MissingCredentialError):
        config.api_key()


def test_missing_key_error_does_not_leak_any_key_material(monkeypatch):
    """NFR-4. The error is going to end up in logs and terminal scrollback."""
    monkeypatch.setenv(config.API_KEY_ENV_VAR, "")
    prefix = "sk-" + "ant-"  # assembled so this file holds no key-shaped literal
    monkeypatch.setenv("SOME_OTHER_SECRET", prefix + "should-never-appear")
    with pytest.raises(config.MissingCredentialError) as excinfo:
        config.api_key()
    assert prefix not in str(excinfo.value)
    assert config.API_KEY_ENV_VAR in str(excinfo.value)


def test_key_is_not_captured_in_module_state(monkeypatch):
    """A module-level constant holding the key would end up in tracebacks and
    in anything that dumps config."""
    monkeypatch.setenv(config.API_KEY_ENV_VAR, "sentinel-key-abc123")
    config.api_key()
    leaked = [
        name
        for name, value in vars(config).items()
        if isinstance(value, str) and "sentinel-key-abc123" in value
    ]
    assert not leaked, f"api key leaked into module attributes: {leaked}"
