"""The credential path from .env to the SDK client (NFR-4, NFR-8).

Regression cover for the bug the first live call found: `config.api_key()`
resolved a key from `.env`, then `anthropic.Anthropic()` was constructed with no
arguments and the SDK looked only at the process environment. The internal check
passed, the client had no credential, and the call failed with an SDK TypeError
about authentication over a `.env` that was perfectly correct.

Every other test in this project injects a fake client, so none of them ever
exercised the real construction path. These do.
"""

import pytest

from risk_analytics import config, llm

VAR = config.API_KEY_ENV_VAR


@pytest.fixture(autouse=True)
def isolate_credentials(tmp_path, monkeypatch):
    """No test here may see the developer's real key, from either source."""
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "absent.env")
    monkeypatch.delenv(VAR, raising=False)


def write_env(tmp_path, monkeypatch, value: str):
    path = tmp_path / "written.env"
    path.write_bytes(f"{VAR}={value}\n".encode("utf-8"))
    monkeypatch.setattr(config, "ENV_FILE", path)
    return path


def test_a_key_from_dotenv_reaches_the_constructed_client(tmp_path, monkeypatch):
    """The bug, stated as a test: nothing is in the environment, the key exists
    only in .env, and the client must still carry it."""
    write_env(tmp_path, monkeypatch, "key-only-in-the-file")
    client = llm.LLM().client()
    assert client.api_key == "key-only-in-the-file"


def test_a_key_from_the_environment_reaches_the_client(monkeypatch):
    monkeypatch.setenv(VAR, "key-from-the-environment")
    assert llm.LLM().client().api_key == "key-from-the-environment"


def test_the_environment_still_wins_over_the_file(tmp_path, monkeypatch):
    """A rotated key exported in the shell must take effect without editing the
    file, and the client must receive the winner, not the loser."""
    write_env(tmp_path, monkeypatch, "stale-file-key")
    monkeypatch.setenv(VAR, "fresh-environment-key")
    assert llm.LLM().client().api_key == "fresh-environment-key"


def test_a_missing_credential_raises_the_project_error_not_an_sdk_error():
    """The SDK's TypeError about resolving an authentication method says nothing
    about .env and sent us looking in the wrong place. Ours names both sources."""
    with pytest.raises(config.MissingCredentialError) as excinfo:
        llm.LLM().client()
    message = str(excinfo.value)
    assert VAR in message
    assert ".env" in message


def test_the_client_is_built_once_and_reused(tmp_path, monkeypatch):
    write_env(tmp_path, monkeypatch, "some-key")
    model = llm.LLM()
    assert model.client() is model.client()


def test_an_injected_client_is_used_untouched_and_needs_no_credential():
    """Every other suite depends on this: a fake client must never trigger
    credential resolution, or the tests would demand a real key."""
    sentinel = object()
    assert llm.LLM(client=sentinel).client() is sentinel


def test_importing_the_module_does_not_demand_a_credential():
    """The offline ingest path imports this transitively. Resolving at import
    would make classification and embedding require a key they never use."""
    llm.LLM()  # construction alone must not raise
    with pytest.raises(config.MissingCredentialError):
        llm.LLM().client()


def test_the_key_is_not_written_into_the_budget_or_usage_records(tmp_path, monkeypatch):
    """Usage records are serialised into the note's trace file (FR-18, NFR-7),
    which is written to disk and read by a human."""
    write_env(tmp_path, monkeypatch, "secret-key-value-xyz")
    model = llm.LLM()
    model.client()
    model.budget.record(llm.Usage(config.ROUTER_MODEL, 100, 10, "plan"))
    serialised = repr(model.budget.to_dict())
    assert "secret-key-value-xyz" not in serialised
