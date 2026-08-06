from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .constants import (
    CHAT_MESSAGE_MAX_CHARS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MAX_MAX_TOKENS,
    MAX_TEMPERATURE,
    MIN_MAX_TOKENS,
    MIN_TEMPERATURE,
    PERSONA_YAML_MAX_CHARS,
    PROJECT_NAME_MAX_CHARS,
    PROVIDER_CATALOG_YAML_MAX_CHARS,
    PROVIDER_MODEL_ID_MAX_CHARS,
    RUNNER_ARGUMENT_HARD_MAX_CHARS,
    RUNNER_ARGUMENT_HARD_MAX_COUNT,
    RUNNER_ARGUMENTS_HARD_MAX_BYTES,
    RUNNER_ID_MAX_CHARS,
    RUNNER_ID_MIN_CHARS,
    RUNNER_INPUT_HARD_MAX_BYTES,
    RUNNER_INPUT_MESSAGE_HARD_MAX_BYTES,
    RUNNER_PATH_MAX_CHARS,
    SHARED_CONTEXT_MAX_CHARS,
)

AGENT_IDS = ("agent_a", "agent_b", "agent_c")
ResearchMode = Literal["auto", "lead", "all", "off"]
ProviderName = str
BUILTIN_PROVIDER_IDS = ("moonshot", "gemini", "deepseek", "openai")
PROVIDER_NAMES = BUILTIN_PROVIDER_IDS
PROVIDER_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def normalize_provider_model(provider: str, model: str) -> tuple[ProviderName, str]:
    """Normalize legacy provider-prefixed model IDs without choosing a fallback."""
    normalized_provider = str(provider or "").strip().lower()
    normalized_model = str(model or "").strip()
    prefixes: tuple[tuple[str, ProviderName], ...] = (
        ("moonshotai/", "moonshot"),
        ("google/", "gemini"),
        ("deepseek/", "deepseek"),
        ("openai/", "openai"),
    )
    for prefix, inferred_provider in prefixes:
        if normalized_model.lower().startswith(prefix):
            return inferred_provider, normalized_model[len(prefix) :]
    return normalized_provider, normalized_model


class RuntimeOptions(BaseModel):
    default_research_mode: ResearchMode = "auto"
    room_default_participants: list[str] = Field(default_factory=lambda: list(AGENT_IDS))
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE, ge=MIN_TEMPERATURE, le=MAX_TEMPERATURE
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS, ge=MIN_MAX_TOKENS, le=MAX_MAX_TOKENS
    )

    @field_validator("room_default_participants")
    @classmethod
    def validate_participants(cls, value: list[str]) -> list[str]:
        unique = [agent_id for agent_id in AGENT_IDS if agent_id in value]
        return unique or list(AGENT_IDS)


class RuntimeConfig(RuntimeOptions):
    """A complete, user-saved provider and model configuration."""

    providers: dict[str, ProviderName]
    models: dict[str, str]

    @field_validator("models")
    @classmethod
    def validate_models(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for agent_id in AGENT_IDS:
            model = str(value.get(agent_id, "")).strip()
            if not model:
                raise ValueError(f"A model ID is required for {agent_id}.")
            if len(model) > PROVIDER_MODEL_ID_MAX_CHARS:
                raise ValueError(
                    f"Model IDs must be at most {PROVIDER_MODEL_ID_MAX_CHARS} characters."
                )
            normalized[agent_id] = model
        return normalized

    @field_validator("providers")
    @classmethod
    def validate_providers(cls, value: dict[str, str]) -> dict[str, ProviderName]:
        normalized: dict[str, ProviderName] = {}
        for agent_id in AGENT_IDS:
            provider = str(value.get(agent_id, "")).strip().lower()
            if not PROVIDER_ID_PATTERN.fullmatch(provider):
                raise ValueError(f"A valid provider ID is required for {agent_id}.")
            normalized[agent_id] = provider
        return normalized

class SetupUpdate(RuntimeConfig):
    """Validated runtime settings accepted by the setup endpoint."""


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=PROJECT_NAME_MAX_CHARS)


class FileSharingUpdate(BaseModel):
    path: str = Field(min_length=1, max_length=RUNNER_PATH_MAX_CHARS)
    shared_agent_edit: bool = False


class CodeRunRequest(BaseModel):
    run_id: str = Field(
        min_length=RUNNER_ID_MIN_CHARS,
        max_length=RUNNER_ID_MAX_CHARS,
        pattern=r"^[A-Za-z0-9-]+$",
    )
    path: str = Field(min_length=1, max_length=RUNNER_PATH_MAX_CHARS)
    arguments: list[str] = Field(
        default_factory=list, max_length=RUNNER_ARGUMENT_HARD_MAX_COUNT
    )
    stdin: str = Field(default="", max_length=RUNNER_INPUT_HARD_MAX_BYTES)

    @field_validator("arguments")
    @classmethod
    def validate_code_run_arguments(cls, value: list[str]) -> list[str]:
        if (
            sum(len(argument.encode("utf-8")) for argument in value)
            > RUNNER_ARGUMENTS_HARD_MAX_BYTES
        ):
            raise ValueError("Runner arguments exceed the combined size limit.")
        if any(
            "\x00" in argument or len(argument) > RUNNER_ARGUMENT_HARD_MAX_CHARS
            for argument in value
        ):
            raise ValueError(
                f"Each runner argument must be at most "
                f"{RUNNER_ARGUMENT_HARD_MAX_CHARS} characters."
            )
        return value

    @field_validator("stdin")
    @classmethod
    def validate_code_run_stdin(cls, value: str) -> str:
        if len(value.encode("utf-8")) > RUNNER_INPUT_HARD_MAX_BYTES:
            raise ValueError("Runner standard input exceeds the byte limit.")
        return value


class CodeRunInput(BaseModel):
    text: str = Field(default="", max_length=RUNNER_INPUT_MESSAGE_HARD_MAX_BYTES)
    append_newline: bool = True
    eof: bool = False

    @field_validator("text")
    @classmethod
    def validate_code_run_input(cls, value: str) -> str:
        if len(value.encode("utf-8")) > RUNNER_INPUT_MESSAGE_HARD_MAX_BYTES:
            raise ValueError("One runner input message exceeds the byte limit.")
        return value


class ChatRequest(BaseModel):
    turn_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=128,
        pattern=r"^[A-Za-z0-9-]+$",
    )
    project_id: str
    message: str = Field(min_length=1, max_length=CHAT_MESSAGE_MAX_CHARS)
    participants: list[str] = Field(default_factory=lambda: list(AGENT_IDS))
    research_mode: ResearchMode = "auto"

    @field_validator("participants")
    @classmethod
    def validate_chat_participants(cls, value: list[str]) -> list[str]:
        unique = [agent_id for agent_id in AGENT_IDS if agent_id in value]
        return unique or list(AGENT_IDS)


class PersonaEdit(BaseModel):
    yaml_text: str = Field(min_length=1, max_length=PERSONA_YAML_MAX_CHARS)


class SharedContextEdit(BaseModel):
    markdown_text: str = Field(min_length=1, max_length=SHARED_CONTEXT_MAX_CHARS)


class MotifStatusUpdate(BaseModel):
    status: Literal["active", "dormant", "rejected"]


class MotifPatternPreferenceUpdate(BaseModel):
    observer_agent_id: Literal["agent_a", "agent_b", "agent_c"]
    preference: Literal["notice", "follow", "test", "paused"]


class InteractionFeedbackUpdate(BaseModel):
    project_id: str = Field(min_length=1, max_length=120)
    message_id: str = Field(min_length=1, max_length=200)
    feedback_type: Literal[
        "useful_difference",
        "repetitive",
        "off_lens",
        "unsupported",
    ]
    active: bool = True


class BridgeMotifPacketRequest(BaseModel):
    motif_ids: list[str] = Field(default_factory=list, max_length=20)
    checkpoint_ids: list[str] = Field(default_factory=list, max_length=12)
    inquiry: str = Field(default="", max_length=12_000)
    human_note: str = Field(default="", max_length=4_000)


class ProviderCatalogEdit(BaseModel):
    yaml_text: str = Field(
        min_length=1,
        max_length=PROVIDER_CATALOG_YAML_MAX_CHARS,
    )


class EvidenceItem(BaseModel):
    event_id: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=2000)


class PersonaChange(BaseModel):
    path: str = Field(min_length=1, max_length=200)
    operation: Literal["replace", "append"]
    value: Any


class PersonaUpdate(BaseModel):
    agent_id: Literal["agent_a", "agent_b", "agent_c"]
    reason: str = Field(min_length=1, max_length=4000)
    evidence: list[EvidenceItem] = Field(min_length=1, max_length=20)
    changes: list[PersonaChange] = Field(min_length=1, max_length=20)
