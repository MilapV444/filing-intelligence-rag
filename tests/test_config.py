"""Checks on the single configuration point (NFR-8) and credential handling (NFR-4)."""

import pytest

from risk_analytics import config

# Key names are built from this constant rather than written as literals. A line
# like KEY="value" in a source file is exactly what the AC-11 credential scan
# looks for, and a test fixture should not need an exemption from that scan.
VAR = config.API_KEY_ENV_VAR


@pytest.fixture(autouse=True)
def isolate_env_file(tmp_path, monkeypatch):
    """Point config.ENV_FILE at a path that does not exist, for every test here.

    Without this, a real .env on the developer's machine satisfies api_key() and
    every test asserting a *missing* credential silently stops testing anything.
    That is not hypothetical: these tests passed until a real .env appeared on
    this machine, then three of them failed. A test outcome must not depend on
    whether the machine happens to be configured.
    """
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "isolated-absent.env")


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
    monkeypatch.setenv(VAR, "test-key-value")
    assert config.api_key() == "test-key-value"


def test_api_key_is_re_read_rather_than_cached_at_import(monkeypatch):
    monkeypatch.setenv(VAR, "first")
    assert config.api_key() == "first"
    monkeypatch.setenv(VAR, "second")
    assert config.api_key() == "second"


@pytest.mark.parametrize("value", ["", "   "])
def test_missing_or_blank_key_raises(monkeypatch, value):
    monkeypatch.setenv(VAR, value)
    with pytest.raises(config.MissingCredentialError):
        config.api_key()


def test_missing_key_error_does_not_leak_any_key_material(monkeypatch):
    """NFR-4. The error is going to end up in logs and terminal scrollback."""
    monkeypatch.setenv(VAR, "")
    prefix = "sk-" + "ant-"  # assembled so this file holds no key-shaped literal
    monkeypatch.setenv("SOME_OTHER_SECRET", prefix + "should-never-appear")
    with pytest.raises(config.MissingCredentialError) as excinfo:
        config.api_key()
    assert prefix not in str(excinfo.value)
    assert VAR in str(excinfo.value)


def test_key_is_not_captured_in_module_state(monkeypatch):
    """A module-level constant holding the key would end up in tracebacks and
    in anything that dumps config."""
    monkeypatch.setenv(VAR, "sentinel-key-abc123")
    config.api_key()
    leaked = [
        name
        for name, value in vars(config).items()
        if isinstance(value, str) and "sentinel-key-abc123" in value
    ]
    assert not leaked, f"api key leaked into module attributes: {leaked}"


# --- .env fallback ------------------------------------------------------------


def env_bytes(body: str, bom: bool = False, newline: str = "\n") -> bytes:
    return (b"\xef\xbb\xbf" if bom else b"") + (body + newline).encode("utf-8")


def _write_env(tmp_path, monkeypatch, raw: bytes):
    path = tmp_path / "written.env"
    path.write_bytes(raw)
    monkeypatch.setattr(config, "ENV_FILE", path)
    monkeypatch.delenv(VAR, raising=False)
    return path


def test_the_key_can_come_from_a_dotenv_file(tmp_path, monkeypatch):
    _write_env(tmp_path, monkeypatch, env_bytes(f"{VAR}=from-the-file"))
    assert config.api_key() == "from-the-file"


def test_a_byte_order_mark_does_not_hide_the_key(tmp_path, monkeypatch):
    """Windows PowerShell's `Set-Content -Encoding utf8` and Notepad both prepend
    a byte-order mark. Read as plain utf-8 the first key parses with an invisible
    prefix and never matches, so a correct-looking file yields "key is not set"
    with nothing visibly wrong. Observed on this project's own .env."""
    _write_env(tmp_path, monkeypatch, env_bytes(f"{VAR}=key-behind-a-bom", bom=True))
    assert config.api_key() == "key-behind-a-bom"


def test_crlf_line_endings_do_not_corrupt_the_value(tmp_path, monkeypatch):
    """Every Windows editor writes CRLF by default."""
    _write_env(tmp_path, monkeypatch, env_bytes(f"{VAR}=windows-line-ending", newline="\r\n"))
    assert config.api_key() == "windows-line-ending"


@pytest.mark.parametrize(
    "body,expected",
    [
        (f'{VAR}="quoted-value"', "quoted-value"),
        (f"{VAR}='single-quoted'", "single-quoted"),
        (f"  {VAR} = spaced-out  ", "spaced-out"),
    ],
)
def test_common_hand_written_variations_still_parse(tmp_path, monkeypatch, body, expected):
    _write_env(tmp_path, monkeypatch, env_bytes(body))
    assert config.api_key() == expected


def test_comments_and_blank_lines_are_ignored(tmp_path, monkeypatch):
    body = f"# the project key\n\nOTHER=irrelevant\n{VAR}=after-a-comment"
    _write_env(tmp_path, monkeypatch, env_bytes(body))
    assert config.api_key() == "after-a-comment"


def test_the_environment_wins_over_the_file(tmp_path, monkeypatch):
    """So a rotated key exported in the shell takes effect without editing the
    file, and a stale file cannot silently override it."""
    _write_env(tmp_path, monkeypatch, env_bytes(f"{VAR}=stale-file-value"))
    monkeypatch.setenv(VAR, "fresh-exported-value")
    assert config.api_key() == "fresh-exported-value"


def test_a_missing_file_is_not_an_error_just_a_missing_key(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)
    with pytest.raises(config.MissingCredentialError):
        config.api_key()


def test_the_error_names_the_dotenv_option(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)
    with pytest.raises(config.MissingCredentialError) as excinfo:
        config.api_key()
    assert ".env" in str(excinfo.value)
