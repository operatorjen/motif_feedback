from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .atomic_files import atomic_write_text
from .constants import (
    AGENT_FILE_HARD_MAX_BYTES,
    DEFAULT_AGENT_TURN_BEATS,
    DEFAULT_GLOBAL_MEMORY_CONTEXT_EVENTS,
    DEFAULT_LOCAL_MEMORY_CONTEXT_EVENTS,
    DEFAULT_PARTICIPATION_RETRIES,
    DEFAULT_PROVIDER_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_PROVIDER_RESPONSE_MAX_BYTES,
    DEFAULT_PROVIDER_TOOL_CALLS_PER_RESPONSE,
    DEFAULT_PROVIDER_TOOL_CALLS_PER_TURN,
    DEFAULT_RUNNER_REQUEST_MAX_BYTES,
    DEFAULT_RUNNER_RESPONSE_MAX_BYTES,
    DEFAULT_RUNNER_SOCKET_POLL_SECONDS,
    DEFAULT_RUNNER_SOCKET_READ_BYTES,
    DEFAULT_WEB_FETCH_CHUNK_BYTES,
    DEFAULT_WEB_FETCH_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_WEB_FETCH_USER_AGENT,
)
from .models import (
    AGENT_IDS,
    PROVIDER_NAMES,
    RuntimeConfig,
    normalize_provider_model,
)


class RuntimeNotConfiguredError(RuntimeError):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    workspace_root: Path = Field(default=Path("/workspace"), alias="WORKSPACE_ROOT")
    user_display_name: str = Field(
        default="User",
        min_length=1,
        max_length=80,
        alias="USER_DISPLAY_NAME",
    )
    moonshot_api_key: str = Field(default="", alias="MOONSHOT_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    moonshot_base_url: str = Field(default="https://api.moonshot.ai/v1", alias="MOONSHOT_BASE_URL")
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai",
        alias="GEMINI_BASE_URL",
    )
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    provider_timeout_seconds: float = Field(default=180, gt=0, alias="PROVIDER_TIMEOUT_SECONDS")
    provider_connect_timeout_seconds: float = Field(
        default=DEFAULT_PROVIDER_CONNECT_TIMEOUT_SECONDS,
        gt=0,
        alias="PROVIDER_CONNECT_TIMEOUT_SECONDS",
    )
    provider_participation_retries: int = Field(
        default=DEFAULT_PARTICIPATION_RETRIES,
        ge=0,
        le=10,
        alias="PROVIDER_PARTICIPATION_RETRIES",
    )
    provider_response_max_bytes: int = Field(
        default=DEFAULT_PROVIDER_RESPONSE_MAX_BYTES,
        gt=0,
        alias="PROVIDER_RESPONSE_MAX_BYTES",
    )
    provider_tool_calls_per_response: int = Field(
        default=DEFAULT_PROVIDER_TOOL_CALLS_PER_RESPONSE,
        ge=1,
        le=100,
        alias="PROVIDER_TOOL_CALLS_PER_RESPONSE",
    )
    provider_tool_calls_per_turn: int = Field(
        default=DEFAULT_PROVIDER_TOOL_CALLS_PER_TURN,
        ge=1,
        le=500,
        alias="PROVIDER_TOOL_CALLS_PER_TURN",
    )
    room_max_provider_requests: int = Field(
        default=64,
        ge=1,
        le=256,
        alias="ROOM_MAX_PROVIDER_REQUESTS",
    )
    room_max_elapsed_seconds: float = Field(
        default=900,
        gt=0,
        le=3_600,
        alias="ROOM_MAX_ELAPSED_SECONDS",
    )
    turn_trace_retention_days: int = Field(
        default=0,
        ge=0,
        le=3_650,
        alias="TURN_TRACE_RETENTION_DAYS",
    )
    max_context_messages: int = Field(default=30, alias="MAX_CONTEXT_MESSAGES")
    max_tool_rounds: int = Field(default=6, alias="MAX_TOOL_ROUNDS")
    max_upload_bytes: int = Field(default=5_242_880, alias="MAX_UPLOAD_BYTES")
    max_agent_write_bytes: int = Field(
        default=AGENT_FILE_HARD_MAX_BYTES,
        alias="MAX_AGENT_WRITE_BYTES",
    )
    web_fetch_timeout_seconds: float = Field(
        default=15, gt=0, alias="WEB_FETCH_TIMEOUT_SECONDS"
    )
    web_fetch_connect_timeout_seconds: float = Field(
        default=DEFAULT_WEB_FETCH_CONNECT_TIMEOUT_SECONDS,
        gt=0,
        alias="WEB_FETCH_CONNECT_TIMEOUT_SECONDS",
    )
    web_fetch_chunk_bytes: int = Field(
        default=DEFAULT_WEB_FETCH_CHUNK_BYTES,
        gt=0,
        alias="WEB_FETCH_CHUNK_BYTES",
    )
    web_fetch_user_agents: list[str] = Field(
        default_factory=lambda: [DEFAULT_WEB_FETCH_USER_AGENT],
        min_length=1,
        max_length=8,
        alias="WEB_FETCH_USER_AGENTS",
    )
    web_fetch_user_agent_attempts: int = Field(
        default=1,
        ge=1,
        le=2,
        alias="WEB_FETCH_USER_AGENT_ATTEMPTS",
    )
    web_fetch_max_bytes: int = Field(default=2_097_152, alias="WEB_FETCH_MAX_BYTES")
    web_fetch_max_text_chars: int = Field(default=60_000, alias="WEB_FETCH_MAX_TEXT_CHARS")
    web_fetch_max_redirects: int = Field(default=3, alias="WEB_FETCH_MAX_REDIRECTS")
    web_fetch_max_urls: int = Field(default=3, alias="WEB_FETCH_MAX_URLS")
    web_fetch_cache_seconds: int = Field(default=3600, alias="WEB_FETCH_CACHE_SECONDS")
    web_fetch_search_fallback: bool = Field(
        default=True,
        alias="WEB_FETCH_SEARCH_FALLBACK",
    )
    web_prompt_max_text_chars: int = Field(default=60_000, alias="WEB_PROMPT_MAX_TEXT_CHARS")
    runner_socket_path: Path = Field(
        default=Path("/workspace/runner/runner.sock"), alias="RUNNER_SOCKET_PATH"
    )
    runner_timeout_seconds: float = Field(default=65, gt=0, alias="RUNNER_TIMEOUT_SECONDS")
    runner_project_max_bytes: int = Field(
        default=8_000_000, gt=0, alias="RUNNER_PROJECT_MAX_BYTES"
    )
    runner_room_transcript_max_chars: int = Field(
        default=8_000, gt=0, alias="RUNNER_ROOM_TRANSCRIPT_MAX_CHARS"
    )
    runner_input_max_bytes: int = Field(
        default=16_000, gt=0, alias="RUNNER_INPUT_MAX_BYTES"
    )
    runner_input_message_max_bytes: int = Field(
        default=4_000, gt=0, alias="RUNNER_INPUT_MESSAGE_MAX_BYTES"
    )
    runner_argument_max_count: int = Field(
        default=24, gt=0, alias="RUNNER_ARGUMENT_MAX_COUNT"
    )
    runner_argument_max_chars: int = Field(
        default=500, gt=0, alias="RUNNER_ARGUMENT_MAX_CHARS"
    )
    runner_arguments_max_bytes: int = Field(
        default=4_000, gt=0, alias="RUNNER_ARGUMENTS_MAX_BYTES"
    )
    runner_request_max_bytes: int = Field(
        default=DEFAULT_RUNNER_REQUEST_MAX_BYTES,
        gt=0,
        alias="RUNNER_REQUEST_MAX_BYTES",
    )
    runner_response_max_bytes: int = Field(
        default=DEFAULT_RUNNER_RESPONSE_MAX_BYTES,
        gt=0,
        alias="RUNNER_RESPONSE_MAX_BYTES",
    )
    runner_socket_poll_seconds: float = Field(
        default=DEFAULT_RUNNER_SOCKET_POLL_SECONDS,
        gt=0,
        alias="RUNNER_SOCKET_POLL_SECONDS",
    )
    runner_socket_read_bytes: int = Field(
        default=DEFAULT_RUNNER_SOCKET_READ_BYTES,
        gt=0,
        alias="RUNNER_SOCKET_READ_BYTES",
    )
    runner_session_poll_seconds: float = Field(
        default=0.08, gt=0, alias="RUNNER_SESSION_POLL_SECONDS"
    )
    max_agent_turn_beats: int = Field(
        default=DEFAULT_AGENT_TURN_BEATS,
        ge=1,
        le=DEFAULT_AGENT_TURN_BEATS,
        alias="MAX_AGENT_TURN_BEATS",
    )
    local_memory_context_events: int = Field(
        default=DEFAULT_LOCAL_MEMORY_CONTEXT_EVENTS,
        ge=0,
        le=100,
        alias="LOCAL_MEMORY_CONTEXT_EVENTS",
    )
    global_memory_context_events: int = Field(
        default=DEFAULT_GLOBAL_MEMORY_CONTEXT_EVENTS,
        ge=0,
        le=100,
        alias="GLOBAL_MEMORY_CONTEXT_EVENTS",
    )

    @field_validator("user_display_name")
    @classmethod
    def normalize_user_display_name(cls, value: str) -> str:
        normalized = " ".join(str(value).split())
        if not normalized:
            raise ValueError("USER_DISPLAY_NAME may not be blank.")
        return normalized

    @field_validator("web_fetch_user_agents")
    @classmethod
    def normalize_web_fetch_user_agents(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for raw_value in value:
            user_agent = str(raw_value)
            if "\r" in user_agent or "\n" in user_agent:
                raise ValueError("WEB_FETCH_USER_AGENTS may not contain line breaks.")
            user_agent = " ".join(user_agent.split())
            if not user_agent:
                raise ValueError("WEB_FETCH_USER_AGENTS may not contain blank entries.")
            if len(user_agent) > 512:
                raise ValueError(
                    "WEB_FETCH_USER_AGENTS entries must be at most 512 characters."
                )
            if user_agent not in normalized:
                normalized.append(user_agent)
        if not normalized:
            raise ValueError("WEB_FETCH_USER_AGENTS must contain at least one entry.")
        return normalized

    @property
    def state_root(self) -> Path:
        return self.workspace_root / "state"

    @property
    def projects_root(self) -> Path:
        return self.workspace_root / "projects"

    @property
    def personas_root(self) -> Path:
        return self.state_root / "personas"

    @property
    def shared_root(self) -> Path:
        return self.state_root / "shared"

    @property
    def history_root(self) -> Path:
        return self.state_root / "history"

    @property
    def proposals_root(self) -> Path:
        return self.state_root / "proposals"

    @property
    def runtime_config_path(self) -> Path:
        return self.state_root / "runtime.yaml"

    @property
    def database_path(self) -> Path:
        return self.state_root / "motif.db"

    @property
    def provider_catalog_path(self) -> Path:
        return self.state_root / "providers.yaml"

    @property
    def seed_root(self) -> Path:
        return Path(__file__).resolve().parent / "seed"

    def provider_api_key(self, provider: str) -> str:
        keys = {
            "moonshot": self.moonshot_api_key,
            "gemini": self.gemini_api_key,
            "deepseek": self.deepseek_api_key,
            "openai": self.openai_api_key,
        }
        return keys.get(provider, "").strip()

    def provider_base_url(self, provider: str) -> str:
        urls = {
            "moonshot": self.moonshot_base_url,
            "gemini": self.gemini_base_url,
            "deepseek": self.deepseek_base_url,
            "openai": self.openai_base_url,
        }
        return urls.get(provider, "").rstrip("/")

    def provider_status(self) -> dict[str, bool]:
        return {provider: bool(self.provider_api_key(provider)) for provider in PROVIDER_NAMES}

    @property
    def agent_file_byte_limit(self) -> int:
        """Allow a lower configured limit, but never let agent files exceed 15 KB."""
        return min(max(1, self.max_agent_write_bytes), AGENT_FILE_HARD_MAX_BYTES)

    def load_runtime_config(self) -> RuntimeConfig | None:
        if not self.runtime_config_path.exists():
            return None
        try:
            raw = yaml.safe_load(
                self.runtime_config_path.read_text(encoding="utf-8")
            ) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"Runtime configuration is invalid: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("Runtime configuration must be a YAML mapping.")
        providers = raw.get("providers") if isinstance(raw.get("providers"), dict) else {}
        models = raw.get("models") if isinstance(raw.get("models"), dict) else {}
        migrated_providers: dict[str, str] = {}
        migrated_models: dict[str, str] = {}
        changed = "providers" not in raw or "models" not in raw
        for agent_id in AGENT_IDS:
            provider, model = normalize_provider_model(
                providers.get(agent_id, ""),
                models.get(agent_id, ""),
            )
            migrated_providers[agent_id] = provider
            migrated_models[agent_id] = model
            changed = (
                changed
                or model != models.get(agent_id)
                or provider != providers.get(agent_id)
            )
        raw["providers"] = migrated_providers
        raw["models"] = migrated_models
        try:
            config = RuntimeConfig.model_validate(raw)
        except ValueError as exc:
            raise ValueError(f"Runtime configuration is invalid: {exc}") from exc
        if changed:
            self.save_runtime_config(config)
        return config

    def require_runtime_config(self) -> RuntimeConfig:
        config = self.load_runtime_config()
        if config is None:
            raise RuntimeNotConfiguredError(
                "Setup is incomplete. Select a provider and model for each agent in Setup."
            )
        return config

    def save_runtime_config(self, config: RuntimeConfig) -> None:
        self.state_root.mkdir(parents=True, exist_ok=True)
        values = config.model_dump()
        ordered_values = {
            "providers": values.pop("providers"),
            "models": values.pop("models"),
            **values,
        }
        atomic_write_text(
            self.runtime_config_path,
            yaml.safe_dump(ordered_values, sort_keys=False, allow_unicode=True),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
