from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from .execution_semantics import (
    AGENT_FINISHED_OPERATION,
    BEAT_FINISHED_OPERATION,
    MEMORY_COMMITTED_OPERATION,
    MESSAGE_COMMITTED_OPERATION,
    PROVIDER_COMPLETION_OPERATION,
)
from .models import ChatRequest, RuntimeConfig
from .providers import AgentCompletion
from .storage import Storage


def stable_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ExecutionLedger:
    """Durable, idempotent checkpoints for one room turn."""

    def __init__(
        self,
        storage: Storage,
        request: ChatRequest,
        runtime: RuntimeConfig,
    ) -> None:
        self.storage = storage
        self.request = request
        self.runtime = runtime
        self.enabled = bool(
            request.turn_id and callable(getattr(storage, "begin_turn_operation", None))
        )

    def operation_id(
        self,
        agent_id: str,
        turn_beat: int,
        operation_type: str,
    ) -> str:
        return f"{self.request.turn_id}:{agent_id}:{turn_beat}:{operation_type}"

    def begin(
        self,
        agent_id: str,
        turn_beat: int,
        operation_type: str,
        *,
        payload: dict | None = None,
    ) -> dict | None:
        if not self.enabled:
            return None
        operation_id = self.operation_id(agent_id, turn_beat, operation_type)
        fingerprint = stable_fingerprint(
            {
                "turn_id": self.request.turn_id,
                "project_id": self.request.project_id,
                "agent_id": agent_id,
                "turn_beat": turn_beat,
                "operation_type": operation_type,
                "provider": self.runtime.providers.get(agent_id),
                "model": self.runtime.models.get(agent_id),
            }
        )
        return self.storage.begin_turn_operation(
            operation_id=operation_id,
            turn_id=str(self.request.turn_id),
            project_id=self.request.project_id,
            agent_id=agent_id,
            turn_beat=turn_beat,
            operation_type=operation_type,
            request_fingerprint=fingerprint,
            payload=payload,
        )

    def complete(
        self,
        agent_id: str,
        turn_beat: int,
        operation_type: str,
        result: dict,
    ) -> dict:
        if not self.enabled:
            return result
        operation_id = self.operation_id(agent_id, turn_beat, operation_type)
        self.storage.complete_turn_operation(operation_id, result)
        return result

    def recover_completion(
        self,
        agent_id: str,
        turn_beat: int,
        *,
        enable_web_search: bool,
    ) -> AgentCompletion | None:
        operation = self.begin(
            agent_id,
            turn_beat,
            PROVIDER_COMPLETION_OPERATION,
            payload={
                "provider": self.runtime.providers[agent_id],
                "model": self.runtime.models[agent_id],
                "web_search": bool(enable_web_search),
            },
        )
        if not operation or operation["status"] != "completed":
            return None
        result = operation.get("result") or {}
        return AgentCompletion(
            content=str(result.get("content") or ""),
            annotations=list(result.get("annotations") or []),
            raw_message={},
            tool_events=list(result.get("tool_events") or []),
            locally_generated=bool(result.get("locally_generated")),
            continue_turn=bool(result.get("continue_turn")),
            usage={
                str(key): int(value)
                for key, value in (result.get("usage") or {}).items()
                if isinstance(value, int)
            },
            request_usage=[
                {
                    str(key): int(value)
                    for key, value in item.items()
                    if isinstance(value, int)
                }
                for item in (result.get("request_usage") or [])
                if isinstance(item, dict)
            ],
        )

    def checkpoint_completion(
        self,
        agent_id: str,
        turn_beat: int,
        completion: AgentCompletion,
    ) -> AgentCompletion:
        if self.enabled:
            self.complete(
                agent_id,
                turn_beat,
                PROVIDER_COMPLETION_OPERATION,
                {key: value for key, value in asdict(completion).items() if key != "raw_message"},
            )
        return completion

    def message_operation_id(self, agent_id: str, turn_beat: int) -> str | None:
        if not self.enabled:
            return None
        self.begin(agent_id, turn_beat, MESSAGE_COMMITTED_OPERATION)
        return self.operation_id(agent_id, turn_beat, MESSAGE_COMMITTED_OPERATION)

    def memory_operation_id(self, agent_id: str, turn_beat: int) -> str | None:
        if not self.enabled:
            return None
        self.begin(agent_id, turn_beat, MEMORY_COMMITTED_OPERATION)
        return self.operation_id(agent_id, turn_beat, MEMORY_COMMITTED_OPERATION)

    def mark_message_committed(
        self,
        agent_id: str,
        turn_beat: int,
        message_id: str,
    ) -> None:
        self.complete(
            agent_id,
            turn_beat,
            MESSAGE_COMMITTED_OPERATION,
            {"message_id": message_id},
        )

    def mark_memory_committed(
        self,
        agent_id: str,
        turn_beat: int,
        memory_event_id: str,
    ) -> None:
        self.complete(
            agent_id,
            turn_beat,
            MEMORY_COMMITTED_OPERATION,
            {"memory_event_id": memory_event_id},
        )

    def mark_beat_finished(self, agent_id: str, turn_beat: int) -> None:
        self.begin(agent_id, turn_beat, BEAT_FINISHED_OPERATION)
        self.complete(
            agent_id,
            turn_beat,
            BEAT_FINISHED_OPERATION,
            {"finished": True},
        )

    def mark_agent_finished(self, agent_id: str, final_beat: int) -> None:
        self.begin(agent_id, final_beat, AGENT_FINISHED_OPERATION)
        self.complete(
            agent_id,
            final_beat,
            AGENT_FINISHED_OPERATION,
            {"finished": True, "final_beat": final_beat},
        )
