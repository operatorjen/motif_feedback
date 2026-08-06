from pathlib import Path
from types import SimpleNamespace

import pytest

from motif_feedback.models import RuntimeConfig
from motif_feedback.provider_catalog import (
    ProviderCatalogError,
    ProviderCatalogStore,
    ProviderRegistry,
)

SEED = """\
schema_version: 1
providers:
  - id: hosted
    label: Hosted example
    enabled: true
    base_url: https://models.example.test/v1
    api_key_env: HOSTED_EXAMPLE_API_KEY
    api_key_required: true
    models:
      - example-chat
    token_parameter: max_tokens
    supports_temperature: true
    supports_tools: true
    reasoning_effort: null
  - id: local
    label: Local model
    enabled: true
    base_url: http://host.docker.internal:11434/v1
    api_key_env: ""
    api_key_required: false
    models:
      - local-chat
    token_parameter: max_tokens
    supports_temperature: true
    supports_tools: true
    reasoning_effort: null
"""


def make_store(tmp_path: Path) -> ProviderCatalogStore:
    seed_path = tmp_path / "seed.yaml"
    seed_path.write_text(SEED, encoding="utf-8")
    store = ProviderCatalogStore(tmp_path / "state" / "providers.yaml", seed_path)
    store.initialize()
    return store


def fake_settings() -> SimpleNamespace:
    return SimpleNamespace(
        provider_api_key=lambda _provider: "",
        provider_base_url=lambda _provider: "",
    )


def test_catalog_initializes_and_keyless_local_provider_is_ready(tmp_path: Path):
    store = make_store(tmp_path)
    registry = ProviderRegistry(fake_settings(), store)

    assert store.path.exists()
    assert registry.ready("local") is True
    assert registry.ready("hosted") is False
    assert registry.public_profiles()[1]["models"] == ["local-chat"]


def test_legacy_openai_profile_inherits_responses_web_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    seed_path = tmp_path / "seed.yaml"
    seed_path.write_text(SEED.replace("id: hosted", "id: openai"), encoding="utf-8")
    store = ProviderCatalogStore(tmp_path / "state" / "providers.yaml", seed_path)
    store.initialize()
    registry = ProviderRegistry(fake_settings(), store)

    assert registry.profile("openai").web_search_mode == "responses"
    assert registry.profile("local").web_search_mode == "none"


def test_custom_provider_key_is_loaded_from_declared_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = make_store(tmp_path)
    registry = ProviderRegistry(fake_settings(), store)
    monkeypatch.setenv("HOSTED_EXAMPLE_API_KEY", "secret-value")

    assert registry.api_key("hosted") == "secret-value"
    assert registry.ready("hosted") is True
    assert "secret-value" not in str(registry.public_profiles())


def test_catalog_save_normalizes_entries_and_rejects_duplicate_ids(tmp_path: Path):
    store = make_store(tmp_path)
    saved = store.save(SEED.replace("Hosted example", "Configured service"))
    assert saved.providers[0].label == "Configured service"
    assert "Configured service" in store.yaml_text()

    duplicate = SEED.replace("id: local", "id: hosted")
    with pytest.raises(ProviderCatalogError, match="unique"):
        store.save(duplicate)


def test_catalog_rejects_credentials_embedded_in_provider_url(tmp_path: Path):
    store = make_store(tmp_path)
    unsafe = SEED.replace(
        "https://models.example.test/v1",
        "https://username:password@models.example.test/v1",
    )

    with pytest.raises(ProviderCatalogError, match="credentials"):
        store.save(unsafe)


def test_catalog_rejects_unknown_schema_versions(tmp_path: Path):
    store = make_store(tmp_path)

    with pytest.raises(ProviderCatalogError, match="schema_version 1"):
        store.save(SEED.replace("schema_version: 1", "schema_version: 2"))


def test_runtime_configuration_preserves_custom_provider_ids():
    runtime = RuntimeConfig(
        providers={
            "agent_a": "local",
            "agent_b": "hosted",
            "agent_c": "another-provider",
        },
        models={
            "agent_a": "local-chat",
            "agent_b": "example-chat",
            "agent_c": "custom-chat",
        },
    )

    assert runtime.providers == {
        "agent_a": "local",
        "agent_b": "hosted",
        "agent_c": "another-provider",
    }
