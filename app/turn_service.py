from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from .models import ChatRequest, RuntimeConfig
from .orchestrator import Orchestrator
from .storage import ChatTurnConflictError, Storage

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]


class ChatTurnStateError(RuntimeError):
    pass


class ChatTurnBudgetError(ChatTurnStateError):
    pass


def chat_request_fingerprint(payload: ChatRequest) -> str:
    encoded = json.dumps(
        payload.model_dump(exclude={"turn_id"}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def room_result_payload(result) -> dict:
    return {
        "messages": result.messages,
        "research": result.research,
        "agent_failures": result.agent_failures,
        "web_sources": result.web_sources,
        "source_failures": result.source_failures,
    }


def _trace_event(event: dict, started: float) -> dict:
    trace = {
        "type": str(event.get("type") or "progress"),
        "elapsed_ms": round((time.monotonic() - started) * 1_000, 3),
    }
    for key in (
        "agent_id",
        "provider",
        "model",
        "round",
        "turn_beat",
        "url",
        "status_code",
        "attempt_count",
        "retrieval_method",
        "tool",
        "message_id",
    ):
        value = event.get(key)
        if value is not None and isinstance(value, (str, int, float, bool)):
            trace[key] = value
    message = event.get("message")
    if isinstance(message, dict):
        metadata = message.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("provider_usage"), dict):
            trace["provider_usage"] = metadata["provider_usage"]
    return trace


def _trace_summary(events: list[dict], duration_ms: float) -> dict:
    usage: dict[str, int] = {}
    for event in events:
        for key, value in event.get("provider_usage", {}).items():
            if isinstance(value, int):
                usage[key] = usage.get(key, 0) + value
    return {
        "duration_ms": round(duration_ms, 3),
        "provider_requests": sum(event.get("type") == "model_request" for event in events),
        "provider_usage": usage,
        "events": events,
    }


class TurnService:
    def __init__(
        self,
        settings,
        storage: Storage,
        orchestrator: Orchestrator,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.orchestrator = orchestrator

    async def execute(
        self,
        payload: ChatRequest,
        runtime: RuntimeConfig,
        progress_callback: ProgressCallback | None = None,
        *,
        resume: bool = False,
    ) -> dict:
        if payload.turn_id is None:
            payload = payload.model_copy(update={"turn_id": uuid.uuid4().hex})
        assert payload.turn_id is not None

        existing_user_message = None
        previous_events: list[dict] = []
        previous_duration_ms = 0.0
        if resume:
            payload, runtime, existing_user_message, previous_events, previous_duration_ms = (
                self._prepare_resume(payload.project_id, payload.turn_id)
            )
        else:
            try:
                turn = self.storage.begin_chat_turn(
                    payload.turn_id,
                    payload.project_id,
                    chat_request_fingerprint(payload),
                    request=payload.model_dump(),
                    runtime=runtime.model_dump(),
                )
            except ChatTurnConflictError as exc:
                raise ChatTurnStateError(str(exc)) from exc
            if not turn["created"]:
                if turn["status"] == "completed" and isinstance(turn["result"], dict):
                    return turn["result"]
                raise ChatTurnStateError(
                    "That room turn did not complete previously. Resume or accept its "
                    "partial result in TURNS."
                )

        started = time.monotonic()
        trace_events = [*previous_events]
        if resume:
            trace_events.append(
                {
                    "type": "turn_resume",
                    "elapsed_ms": round(previous_duration_ms, 3),
                }
            )
        provider_requests = sum(event.get("type") == "model_request" for event in trace_events)

        async def report(event: dict) -> None:
            nonlocal provider_requests
            if event.get("type") == "model_request":
                if provider_requests >= self.settings.room_max_provider_requests:
                    raise ChatTurnBudgetError(
                        "The room stopped before exceeding its provider-request budget."
                    )
                provider_requests += 1
            trace_events.append(_trace_event(event, started))
            if progress_callback is not None:
                await progress_callback(event)

        try:
            if not payload.participants:
                public_result = self._completed_resume_result(payload.project_id, payload.turn_id)
            else:
                async with asyncio.timeout(self.settings.room_max_elapsed_seconds):
                    kwargs = {"progress_callback": report}
                    if existing_user_message is not None:
                        kwargs["existing_user_message"] = existing_user_message
                    result = await self.orchestrator.chat(payload, runtime, **kwargs)
                public_result = room_result_payload(result)
                if resume:
                    public_result["messages"] = [
                        message
                        for message in self.storage.messages_for_turn(
                            payload.project_id, payload.turn_id
                        )
                        if message["role"] == "agent"
                    ]
            duration_ms = previous_duration_ms + (time.monotonic() - started) * 1_000
            self.storage.complete_chat_turn(
                payload.turn_id,
                public_result,
                _trace_summary(trace_events, duration_ms),
            )
            return public_result
        except asyncio.CancelledError:
            self._record_failure(
                payload.turn_id,
                "interrupted",
                "The streaming client disconnected before the room turn completed.",
                trace_events,
                previous_duration_ms,
                started,
            )
            raise
        except TimeoutError as exc:
            detail = "The room stopped after reaching its elapsed-time budget."
            self._record_failure(
                payload.turn_id,
                "failed",
                detail,
                trace_events,
                previous_duration_ms,
                started,
            )
            raise ChatTurnBudgetError(detail) from exc
        except Exception as exc:
            self._record_failure(
                payload.turn_id,
                "failed",
                str(exc),
                trace_events,
                previous_duration_ms,
                started,
            )
            raise

    def _prepare_resume(
        self,
        project_id: str,
        turn_id: str,
    ) -> tuple[ChatRequest, RuntimeConfig, dict | None, list[dict], float]:
        try:
            turn = self.storage.get_chat_turn(turn_id)
            if turn["project_id"] != project_id:
                raise ChatTurnConflictError("That turn belongs to a different project.")
            self.storage.resume_chat_turn(turn_id)
        except ChatTurnConflictError as exc:
            raise ChatTurnStateError(str(exc)) from exc
        request = ChatRequest.model_validate(turn["request"])
        runtime = RuntimeConfig.model_validate(turn["runtime"])
        messages = self.storage.messages_for_turn(project_id, turn_id)
        user_message = next(
            (message for message in messages if message["role"] == "user"),
            None,
        )
        operations = self.storage.list_turn_operations(turn_id)
        if operations:
            completed_agents = self.storage.completed_turn_agents(turn_id)
        else:
            completed_agents = {
                message["agent_id"]
                for message in messages
                if message["role"] == "agent" and message["agent_id"]
            }
        request = request.model_copy(
            update={
                "participants": [
                    agent_id
                    for agent_id in request.participants
                    if agent_id not in completed_agents
                ]
            }
        )
        trace = turn.get("trace") or {}
        return (
            request,
            runtime,
            user_message,
            list(trace.get("events") or []),
            float(trace.get("duration_ms") or 0),
        )

    def _completed_resume_result(self, project_id: str, turn_id: str) -> dict:
        messages = [
            message
            for message in self.storage.messages_for_turn(project_id, turn_id)
            if message["role"] == "agent"
        ]
        return {
            "messages": messages,
            "research": {
                "needs_search": False,
                "evidence_status": "resumed_from_stored_messages",
            },
            "agent_failures": [],
            "web_sources": [],
            "source_failures": [],
        }

    def _record_failure(
        self,
        turn_id: str,
        status: str,
        detail: str,
        events: list[dict],
        previous_duration_ms: float,
        started: float,
    ) -> None:
        duration_ms = previous_duration_ms + (time.monotonic() - started) * 1_000
        self.storage.fail_chat_turn(
            turn_id,
            status=status,
            detail=detail,
            trace=_trace_summary(events, duration_ms),
        )
