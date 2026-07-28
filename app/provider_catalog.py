from __future__ import annotations

import os
import re
import shutil
import threading
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .atomic_files import atomic_write_text
from .models import BUILTIN_PROVIDER_IDS, PROVIDER_ID_PATTERN

PROVIDER_CATALOG_SCHEMA_VERSION = 1
API_KEY_ENV_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


class ProviderCatalogError(ValueError):
    pass


class ProviderProfile(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=100)
    enabled: bool = True
    base_url: str = Field(min_length=1, max_length=2_048)
    api_key_env: str = Field(default="", max_length=100)
    api_key_required: bool = True
    models: list[str] = Field(default_factory=list, max_length=100)
    token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    supports_temperature: bool = True
    supports_tools: bool = True
    reasoning_effort: str | None = Field(default=None, max_length=40)
    web_search_mode: Literal["none", "responses"] | None = None
    web_search_context_size: Literal["low", "medium", "high"] = "medium"

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not PROVIDER_ID_PATTERN.fullmatch(normalized):
            raise ValueError(
                "Provider IDs must start with a lowercase letter and contain only "
                "lowercase letters, numbers, underscores, or hyphens."
            )
        return normalized

    @field_validator("label", "reasoning_effort", mode="before")
    @classmethod
    def normalize_optional_text(cls, value):
        if value is None:
            return None
        return str(value).strip()

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        try:
            parsed = urlsplit(normalized)
        except ValueError as exc:
            raise ValueError("Provider base URL is malformed.") from exc
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Provider base URL must be an http or https URL.")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Provider base URLs may not contain credentials.")
        if parsed.query or parsed.fragment:
            raise ValueError("Provider base URLs may not contain a query or fragment.")
        return normalized

    @field_validator("api_key_env")
    @classmethod
    def validate_key_environment_name(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized and not API_KEY_ENV_PATTERN.fullmatch(normalized):
            raise ValueError(
                "API key environment names must contain uppercase letters, numbers, "
                "and underscores."
            )
        return normalized

    @field_validator("models")
    @classmethod
    def normalize_models(cls, value: list[str]) -> list[str]:
        unique: list[str] = []
        for raw_model in value:
            model = str(raw_model).strip()
            if model and model not in unique:
                unique.append(model[:200])
        return unique

    @model_validator(mode="after")
    def require_key_environment_when_needed(self):
        if self.api_key_required and not self.api_key_env:
            raise ValueError(
                "Providers requiring an API key must declare api_key_env."
            )
        return self


class ProviderCatalogDocument(BaseModel):
    schema_version: int = PROVIDER_CATALOG_SCHEMA_VERSION
    providers: list[ProviderProfile] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_catalog(self):
        if self.schema_version != PROVIDER_CATALOG_SCHEMA_VERSION:
            raise ValueError(
                f"Provider catalog requires schema_version "
                f"{PROVIDER_CATALOG_SCHEMA_VERSION}."
            )
        identifiers = [profile.id for profile in self.providers]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Provider IDs must be unique.")
        if not any(profile.enabled for profile in self.providers):
            raise ValueError("At least one provider must be enabled.")
        return self


class ProviderCatalogStore:
    def __init__(self, path: Path, seed_path: Path) -> None:
        self.path = path
        self.seed_path = seed_path
        self._lock = threading.RLock()
        self._cached_signature: tuple[int, int] | None = None
        self._cached_catalog: ProviderCatalogDocument | None = None

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            shutil.copyfile(self.seed_path, self.path)
            self.path.chmod(0o600)
        self.load()

    def load(self) -> ProviderCatalogDocument:
        with self._lock:
            try:
                stat = self.path.stat()
                signature = (stat.st_mtime_ns, stat.st_size)
                if (
                    signature == self._cached_signature
                    and self._cached_catalog is not None
                ):
                    return self._cached_catalog.model_copy(deep=True)
                raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
                catalog = ProviderCatalogDocument.model_validate(raw)
            except (OSError, ValueError, yaml.YAMLError) as exc:
                raise ProviderCatalogError(f"Provider catalog is invalid: {exc}") from exc
            self._cached_signature = signature
            self._cached_catalog = catalog
            return catalog.model_copy(deep=True)

    def yaml_text(self) -> str:
        self.load()
        return self.path.read_text(encoding="utf-8")

    def save(self, yaml_text: str) -> ProviderCatalogDocument:
        try:
            raw = yaml.safe_load(yaml_text) or {}
            catalog = ProviderCatalogDocument.model_validate(raw)
        except (ValueError, yaml.YAMLError) as exc:
            raise ProviderCatalogError(f"Provider catalog is invalid: {exc}") from exc
        normalized = yaml.safe_dump(
            catalog.model_dump(),
            sort_keys=False,
            allow_unicode=True,
        )
        with self._lock:
            atomic_write_text(self.path, normalized)
            stat = self.path.stat()
            self._cached_signature = (stat.st_mtime_ns, stat.st_size)
            self._cached_catalog = catalog
        return catalog


class ProviderRegistry:
    def __init__(self, settings, store: ProviderCatalogStore) -> None:
        self.settings = settings
        self.store = store

    def profiles(self, *, enabled_only: bool = True) -> list[ProviderProfile]:
        profiles = [
            self._effective_profile(profile)
            for profile in self.store.load().providers
        ]
        if enabled_only:
            profiles = [profile for profile in profiles if profile.enabled]
        return profiles

    def profile(self, provider_id: str) -> ProviderProfile | None:
        normalized = str(provider_id).strip().lower()
        return next(
            (
                profile
                for profile in self.profiles(enabled_only=False)
                if profile.id == normalized
            ),
            None,
        )

    def api_key(self, provider_id: str) -> str:
        profile = self.profile(provider_id)
        if profile is None or not profile.api_key_env:
            return ""
        if profile.id in BUILTIN_PROVIDER_IDS:
            builtin = self.settings.provider_api_key(profile.id)
            if builtin:
                return builtin
        return os.environ.get(profile.api_key_env, "").strip()

    def ready(self, provider_id: str) -> bool:
        profile = self.profile(provider_id)
        if profile is None or not profile.enabled:
            return False
        return bool(profile.base_url) and (
            not profile.api_key_required or bool(self.api_key(profile.id))
        )

    def status(self) -> dict[str, bool]:
        return {
            profile.id: self.ready(profile.id)
            for profile in self.profiles(enabled_only=True)
        }

    def public_profiles(self) -> list[dict]:
        statuses = self.status()
        return [
            {
                "id": profile.id,
                "label": profile.label,
                "base_url": profile.base_url,
                "api_key_env": profile.api_key_env,
                "api_key_required": profile.api_key_required,
                "models": profile.models,
                "supports_tools": profile.supports_tools,
                "web_search_mode": profile.web_search_mode,
                "ready": statuses.get(profile.id, False),
            }
            for profile in self.profiles(enabled_only=True)
        ]

    def _effective_profile(self, profile: ProviderProfile) -> ProviderProfile:
        if profile.web_search_mode is None:
            profile = profile.model_copy(
                update={
                    "web_search_mode": (
                        "responses" if profile.id == "openai" else "none"
                    )
                }
            )
        if profile.id not in BUILTIN_PROVIDER_IDS:
            return profile
        if f"{profile.id.upper()}_BASE_URL" not in os.environ:
            return profile
        configured_base_url = self.settings.provider_base_url(profile.id)
        if not configured_base_url:
            return profile
        return profile.model_copy(update={"base_url": configured_base_url})
