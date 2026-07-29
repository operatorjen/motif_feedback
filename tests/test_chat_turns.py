import asyncio
import json
from types import SimpleNamespace

import pytest

from app import main as main_module
from app.execution_ledger import ExecutionLedger
from app.models import ChatRequest, RuntimeConfig
from app.orchestrator import RoomResponse
from app.providers import AgentCompletion
from app.storage import Storage
from app.turn_service import ChatTurnBudgetError, TurnService


class CountingOrchestrator:
    def __init__(self):
        self.calls = 0

    async def chat(self, payload, _runtime, progress_callback=None):
        self.calls += 1
        if progress_callback is not None:
            await progress_callback(
                {
                    "type": "agent_complete",
                    "agent_id": "agent_a",
                    "provider": "moonshot",
                    "model": "model-a",
                    "message_id": "message-1",
                    "message": {
                        "metadata": {
                            "provider_usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 2,
                                "total_tokens": 12,
                            }
                        }
                    },
                }
            )
            await progress_callback({"type": "turn_complete"})
        return RoomResponse(
            messages=[{"id": "message-1", "content": payload.message}],
            research={"needs_search": False},
            agent_failures=[],
            web_sources=[],
            source_failures=[],
        )


def runtime_config():
    return RuntimeConfig(
        providers={
            "agent_a": "moonshot",
            "agent_b": "gemini",
            "agent_c": "deepseek",
        },
        models={
            "agent_a": "model-a",
            "agent_b": "model-b",
            "agent_c": "model-c",
        },
    )


def test_completed_turn_identifier_replays_without_calling_agents_again(
    tmp_path,
    monkeypatch,
):
    storage = Storage(tmp_path / "state" / "motif.db", tmp_path / "projects")
    storage.initialize()
    orchestrator = CountingOrchestrator()
    monkeypatch.setattr(main_module, "storage", storage)
    monkeypatch.setattr(main_module, "orchestrator", orchestrator)
    monkeypatch.setattr(
        main_module,
        "turn_service",
        TurnService(main_module.settings, storage, orchestrator),
    )
    payload = ChatRequest(
        turn_id="turn-replay-123",
        project_id="general",
        message="Preserve this turn.",
        participants=["agent_a"],
        research_mode="off",
    )

    first = asyncio.run(main_module._execute_chat_turn(payload, runtime_config()))
    replay = asyncio.run(main_module._execute_chat_turn(payload, runtime_config()))

    assert replay == first
    assert orchestrator.calls == 1
    stored = storage.get_chat_turn("turn-replay-123")
    assert stored["status"] == "completed"
    assert stored["trace"]["events"][0]["provider_usage"]["total_tokens"] == 12


class PartialThenResumeOrchestrator:
    def __init__(self, storage):
        self.storage = storage
        self.calls = []

    async def chat(
        self,
        payload,
        _runtime,
        progress_callback=None,
        existing_user_message=None,
    ):
        del progress_callback
        self.calls.append(
            {
                "participants": list(payload.participants),
                "existing_user": existing_user_message is not None,
            }
        )
        user = existing_user_message or self.storage.add_message(
            payload.project_id,
            "user",
            payload.message,
            metadata={"turn_id": payload.turn_id},
        )
        agent_id = payload.participants[0]
        message = self.storage.add_message(
            payload.project_id,
            "agent",
            f"{agent_id} finished",
            agent_id=agent_id,
            metadata={
                "turn_id": payload.turn_id,
                "user_message_id": user["id"],
            },
        )
        if len(self.calls) == 1:
            raise RuntimeError("simulated restart")
        return RoomResponse(
            messages=[message],
            research={"needs_search": False},
            agent_failures=[],
            web_sources=[],
            source_failures=[],
        )


def turn_settings(**overrides):
    return SimpleNamespace(
        room_max_provider_requests=overrides.get("requests", 64),
        room_max_elapsed_seconds=overrides.get("seconds", 30),
    )


def test_failed_turn_resumes_only_unfinished_agents_without_duplicate_user(tmp_path):
    storage = Storage(tmp_path / "state" / "motif.db", tmp_path / "projects")
    storage.initialize()
    orchestrator = PartialThenResumeOrchestrator(storage)
    service = TurnService(turn_settings(), storage, orchestrator)
    payload = ChatRequest(
        turn_id="turn-resume-123",
        project_id="general",
        message="Continue from stored progress.",
        participants=["agent_a", "agent_b"],
        research_mode="off",
    )

    with pytest.raises(RuntimeError, match="simulated restart"):
        asyncio.run(service.execute(payload, runtime_config()))
    result = asyncio.run(service.execute(payload, runtime_config(), resume=True))

    assert orchestrator.calls == [
        {"participants": ["agent_a", "agent_b"], "existing_user": False},
        {"participants": ["agent_b"], "existing_user": True},
    ]
    messages = storage.messages_for_turn("general", "turn-resume-123")
    assert [message["role"] for message in messages].count("user") == 1
    assert [message["agent_id"] for message in result["messages"]] == [
        "agent_a",
        "agent_b",
    ]
    assert storage.get_chat_turn("turn-resume-123")["status"] == "completed"


class OverBudgetOrchestrator:
    async def chat(self, _payload, _runtime, progress_callback=None):
        await progress_callback({"type": "model_request"})
        await progress_callback({"type": "model_request"})


def test_room_request_budget_fails_turn_before_extra_provider_request(tmp_path):
    storage = Storage(tmp_path / "state" / "motif.db", tmp_path / "projects")
    storage.initialize()
    service = TurnService(
        turn_settings(requests=1),
        storage,
        OverBudgetOrchestrator(),
    )
    payload = ChatRequest(
        turn_id="turn-budget-123",
        project_id="general",
        message="Do not run away.",
        participants=["agent_a"],
        research_mode="off",
    )

    with pytest.raises(ChatTurnBudgetError, match="provider-request budget"):
        asyncio.run(service.execute(payload, runtime_config()))

    turn = storage.get_chat_turn("turn-budget-123")
    assert turn["status"] == "failed"
    assert turn["trace"]["provider_requests"] == 1


class InterruptBetweenMessageAndMemory:
    def __init__(self, storage):
        self.storage = storage
        self.provider_calls = 0
        self.interrupt_once = True

    async def chat(
        self,
        payload,
        runtime,
        progress_callback=None,
        existing_user_message=None,
    ):
        del progress_callback
        ledger = ExecutionLedger(self.storage, payload, runtime)
        user = existing_user_message or self.storage.add_message(
            payload.project_id,
            "user",
            payload.message,
            metadata={"turn_id": payload.turn_id},
        )
        completion = ledger.recover_completion(
            "agent_a",
            1,
            enable_web_search=False,
        )
        if completion is None:
            self.provider_calls += 1
            completion = ledger.checkpoint_completion(
                "agent_a",
                1,
                AgentCompletion(
                    content="durable provider result",
                    annotations=[],
                    raw_message={},
                    tool_events=[],
                ),
            )
        operation_id = ledger.message_operation_id("agent_a", 1)
        message = self.storage.add_message(
            payload.project_id,
            "agent",
            completion.content,
            agent_id="agent_a",
            metadata={
                "turn_id": payload.turn_id,
                "user_message_id": user["id"],
                "turn_beat": 1,
            },
            operation_id=operation_id,
        )
        ledger.mark_message_committed("agent_a", 1, message["id"])
        if self.interrupt_once:
            self.interrupt_once = False
            raise RuntimeError("interrupted before memory commit")
        memory_operation_id = ledger.memory_operation_id("agent_a", 1)
        memory = self.storage.add_memory_event(
            payload.project_id,
            "agent_a",
            user["id"],
            outcome="response",
            trigger_text=payload.message,
            return_text=completion.content,
            actions=[],
            provider=runtime.providers["agent_a"],
            model=runtime.models["agent_a"],
            operation_id=memory_operation_id,
        )
        ledger.mark_memory_committed("agent_a", 1, memory["id"])
        ledger.mark_beat_finished("agent_a", 1)
        ledger.mark_agent_finished("agent_a", 1)
        return RoomResponse(
            messages=[message],
            research={"needs_search": False},
            agent_failures=[],
            web_sources=[],
            source_failures=[],
        )


def test_resume_continues_after_message_without_recalling_provider_or_duplicating(
    tmp_path,
    monkeypatch,
):
    storage = Storage(tmp_path / "state" / "motif.db", tmp_path / "projects")
    storage.initialize()
    orchestrator = InterruptBetweenMessageAndMemory(storage)
    service = TurnService(turn_settings(), storage, orchestrator)
    payload = ChatRequest(
        turn_id="turn-stage-resume-123",
        project_id="general",
        message="Commit each stage once.",
        participants=["agent_a"],
        research_mode="off",
    )

    with pytest.raises(RuntimeError, match="before memory commit"):
        asyncio.run(service.execute(payload, runtime_config()))
    monkeypatch.setattr(main_module, "storage", storage)
    diagnostic = main_module._public_chat_turn(storage.get_chat_turn(payload.turn_id))
    assert diagnostic["execution_stage"] == {
        "agent_id": "agent_a",
        "turn_beat": 1,
        "operation": "memory_committed",
        "status": "pending",
    }
    result = asyncio.run(service.execute(payload, runtime_config(), resume=True))

    assert orchestrator.provider_calls == 1
    messages = storage.messages_for_turn("general", payload.turn_id)
    assert [message["role"] for message in messages] == ["user", "agent"]
    assert len(storage.list_memory_events("general", "agent_a")) == 1
    assert [message["content"] for message in result["messages"]] == ["durable provider result"]
    operations = storage.list_turn_operations(payload.turn_id)
    assert operations[-1]["operation_type"] == "agent_finished"
    assert all(operation["status"] == "completed" for operation in operations)


def test_turn_listing_batches_operation_diagnostics(tmp_path, monkeypatch):
    storage = Storage(tmp_path / "state" / "motif.db", tmp_path / "projects")
    storage.initialize()
    for index in range(2):
        storage.begin_chat_turn(
            f"turn-batch-{index}",
            "general",
            f"fingerprint-{index}",
            request={
                "turn_id": f"turn-batch-{index}",
                "project_id": "general",
                "message": f"Batch turn {index}",
                "participants": ["agent_a"],
                "research_mode": "off",
            },
            runtime=runtime_config().model_dump(),
        )

    original_batch_reader = storage.list_turn_operations_for_turns
    batch_calls = []

    def read_batch(turn_ids):
        batch_calls.append(list(turn_ids))
        return original_batch_reader(turn_ids)

    monkeypatch.setattr(storage, "list_turn_operations_for_turns", read_batch)
    monkeypatch.setattr(
        storage,
        "list_turn_operations",
        lambda _turn_id: (_ for _ in ()).throw(
            AssertionError("turn listings must not issue per-turn operation queries")
        ),
    )
    monkeypatch.setattr(main_module, "storage", storage)

    turns = main_module.chat_turns("general", limit=100)

    assert len(turns) == 2
    assert batch_calls == [["turn-batch-1", "turn-batch-0"]]


def test_recovery_listing_returns_only_unresolved_resumable_turns(
    tmp_path,
    monkeypatch,
):
    storage = Storage(tmp_path / "state" / "motif.db", tmp_path / "projects")
    storage.initialize()
    request = {
        "turn_id": "turn-recoverable",
        "project_id": "general",
        "message": "Resume this turn.",
        "participants": ["agent_a"],
        "research_mode": "off",
    }
    runtime = runtime_config().model_dump()

    storage.begin_chat_turn(
        "turn-completed",
        "general",
        "completed-fingerprint",
        request={**request, "turn_id": "turn-completed"},
        runtime=runtime,
    )
    storage.complete_chat_turn("turn-completed", {"messages": []}, {})
    storage.begin_chat_turn(
        "turn-recoverable",
        "general",
        "recoverable-fingerprint",
        request=request,
        runtime=runtime,
    )
    storage.fail_chat_turn(
        "turn-recoverable",
        status="interrupted",
        detail="The process stopped.",
        trace={},
    )
    storage.begin_chat_turn("turn-legacy", "general", "legacy-fingerprint")
    storage.fail_chat_turn(
        "turn-legacy",
        status="failed",
        detail="This older turn has no replay state.",
        trace={},
    )

    monkeypatch.setattr(main_module, "storage", storage)

    turns = main_module.chat_turns("general", resumable_only=True)

    assert [turn["id"] for turn in turns] == ["turn-recoverable"]
    assert turns[0]["resumable"] is True


def test_turn_recovery_is_contextual_instead_of_an_inspector_tab():
    root = main_module.STATIC_ROOT
    html = (root / "index.html").read_text(encoding="utf-8")
    javascript = (root / "js" / "app.js").read_text(encoding="utf-8")

    assert 'id="turn-recovery"' in html
    assert 'data-tab="turns"' not in html
    assert 'id="tab-turns"' not in html
    assert "resumable_only=true" in javascript
    assert "loadTurnRecovery()" in javascript


def test_shared_chat_stream_emits_progress_and_result_in_order():
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(chat_lock=asyncio.Lock()))
    )

    async def execute(report):
        await report({"type": "agent_start", "agent_id": "agent_a"})
        return {"messages": [{"content": "done"}]}

    async def consume():
        response = main_module._chat_stream_response(request, execute)
        return [json.loads(chunk) async for chunk in response.body_iterator]

    events = asyncio.run(consume())

    assert events == [
        {"type": "agent_start", "agent_id": "agent_a"},
        {"type": "result", "messages": [{"content": "done"}]},
    ]
