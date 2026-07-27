import asyncio
from pathlib import Path
from types import SimpleNamespace

import yaml

from app.models import ChatRequest, RuntimeConfig
from app.orchestrator import Orchestrator
from app.providers import AgentCompletion, ProviderError, ProviderTimeout
from app.search_router import SearchDecision, SearchRouter


class FakeStorage:
    def __init__(self):
        self.messages = []
        self.memory_events = []
        self.global_memory_events = []

    @staticmethod
    def get_project(project_id):
        return {"id": project_id, "name": "Test project"}

    def add_message(
        self,
        project_id,
        role,
        content,
        *,
        agent_id=None,
        annotations=None,
        metadata=None,
    ):
        message = {
            "id": f"message-{len(self.messages) + 1}",
            "project_id": project_id,
            "role": role,
            "agent_id": agent_id,
            "content": content,
            "annotations": annotations or [],
            "metadata": metadata or {},
            "created_at": "2026-07-21T00:00:00+00:00",
        }
        self.messages.append(message)
        return message

    def recent_messages(self, _project_id, _limit):
        return list(self.messages)

    def list_memory_events(self, project_id, agent_id, limit=20):
        matching = [
            event for event in self.memory_events
            if event["project_id"] == project_id and event["agent_id"] == agent_id
        ]
        return list(reversed(matching))[:limit]

    def add_memory_event(self, project_id, agent_id, user_message_id, **values):
        event = {
            "id": f"memory-{len(self.memory_events) + 1}",
            "project_id": project_id,
            "agent_id": agent_id,
            "user_message_id": user_message_id,
            "sequence": len(self.memory_events) + 1,
            "created_at": "2026-07-21T00:00:00+00:00",
            **values,
        }
        self.memory_events.append(event)
        return event

    def list_global_memory_events(self, agent_id, *, exclude_project_id=None, limit=20):
        matching = [
            event for event in self.global_memory_events
            if event["agent_id"] == agent_id
            and event["source_project_id"] != exclude_project_id
        ]
        return list(reversed(matching))[:limit]

    def add_global_memory_event(self, **values):
        event = {
            "id": f"global-memory-{len(self.global_memory_events) + 1}",
            "sequence": len(self.global_memory_events) + 1,
            **values,
        }
        self.global_memory_events.append(event)
        return event


class FakePersonas:
    @staticmethod
    def load_persona(agent_id):
        return {"agent_id": agent_id, "display_name": agent_id.replace("_", " ").title()}

    @staticmethod
    def load_shared_context():
        return "Test context"


class TimeoutThenRespond:
    def __init__(self):
        self.calls = []

    async def run_agent(self, *, tool_context, **_kwargs):
        self.calls.append(tool_context.agent_id)
        if tool_context.agent_id == "agent_a":
            raise ProviderTimeout("Agent A exceeded its response window.")
        if tool_context.agent_id == "agent_b":
            raise ProviderError("Provider rejected this turn.")
        return AgentCompletion(
            content=f"Response from {tool_context.agent_id}",
            annotations=[],
            raw_message={},
            tool_events=[],
        )


class OneFollowupBeat:
    def __init__(self):
        self.calls = []

    async def run_agent(self, *, tool_context, messages, **_kwargs):
        self.calls.append((tool_context.agent_id, messages))
        agent_calls = sum(1 for agent_id, _ in self.calls if agent_id == tool_context.agent_id)
        return AgentCompletion(
            content=f"Beat {agent_calls} from {tool_context.agent_id}",
            annotations=[],
            raw_message={},
            tool_events=[],
            continue_turn=agent_calls == 1,
        )


def fake_settings():
    return SimpleNamespace(
        max_context_messages=30,
        user_display_name="User",
        max_agent_turn_beats=3,
        local_memory_context_events=8,
        global_memory_context_events=6,
        agent_file_byte_limit=15_000,
    )


def test_timeout_passes_turn_to_remaining_agents():
    storage = FakeStorage()
    client = TimeoutThenRespond()
    orchestrator = Orchestrator(
        fake_settings(),
        storage,
        FakePersonas(),
        client,
        SearchRouter(),
    )
    events = []

    async def report(event):
        events.append(event)

    result = asyncio.run(
        orchestrator.chat(
            ChatRequest(
                project_id="test-project",
                message="As a team, take turns discussing this.",
                participants=["agent_a", "agent_b", "agent_c"],
                research_mode="off",
            ),
            RuntimeConfig(
                providers={
                    "agent_a": "moonshot",
                    "agent_b": "gemini",
                    "agent_c": "deepseek",
                },
                models={
                    "agent_a": "provider/a",
                    "agent_b": "provider/b",
                    "agent_c": "provider/c",
                }
            ),
            progress_callback=report,
        )
    )

    assert client.calls == ["agent_a", "agent_b", "agent_c"]
    assert [message["agent_id"] for message in result.messages] == ["agent_c"]
    assert result.agent_failures == [
        {
            "agent_id": "agent_a",
            "display_name": "Agent A",
            "provider": "moonshot",
            "model": "provider/a",
            "kind": "timeout",
            "detail": "Agent A exceeded its response window.",
        },
        {
            "agent_id": "agent_b",
            "display_name": "Agent B",
            "provider": "gemini",
            "model": "provider/b",
            "kind": "provider_error",
            "detail": "Provider rejected this turn.",
        },
    ]
    timeout_index = next(i for i, event in enumerate(events) if event["type"] == "agent_timeout")
    next_start_index = next(
        i
        for i, event in enumerate(events)
        if event["type"] == "agent_start" and event["agent_id"] == "agent_b"
    )
    assert timeout_index < next_start_index
    provider_error_index = next(
        i for i, event in enumerate(events) if event["type"] == "agent_provider_error"
    )
    last_start_index = next(
        i
        for i, event in enumerate(events)
        if event["type"] == "agent_start" and event["agent_id"] == "agent_c"
    )
    assert provider_error_index < last_start_index
    assert [event["outcome"] for event in storage.memory_events] == [
        "timeout",
        "provider_error",
        "response",
    ]
    assert [event["agent_id"] for event in storage.global_memory_events] == ["agent_c"]


def test_agent_can_choose_a_second_visible_beat_without_blocking_next_agent():
    storage = FakeStorage()
    client = OneFollowupBeat()
    orchestrator = Orchestrator(
        fake_settings(),
        storage,
        FakePersonas(),
        client,
        SearchRouter(),
    )

    result = asyncio.run(
        orchestrator.chat(
            ChatRequest(
                project_id="test-project",
                message="Take natural turns.",
                participants=["agent_a", "agent_b"],
                research_mode="off",
            ),
            RuntimeConfig(
                providers={"agent_a": "gemini", "agent_b": "gemini", "agent_c": "gemini"},
                models={"agent_a": "a", "agent_b": "b", "agent_c": "c"},
            ),
        )
    )

    assert [message["agent_id"] for message in result.messages] == [
        "agent_a", "agent_a", "agent_b", "agent_b",
    ]
    assert [message["metadata"]["turn_beat"] for message in result.messages] == [1, 2, 1, 2]
    assert "You requested response beat 2 of 3" in client.calls[1][1][-1]["content"]


def test_runner_role_decorator_biases_only_the_next_agent_prompt():
    storage = FakeStorage()
    storage.add_message("test-project", "user", "Earlier turn")
    storage.add_message(
        "test-project",
        "runner",
        "A bounded run completed.",
        metadata={
            "role_signals": [
                {
                    "decorator": "feedback_attention",
                    "target": "agent_b",
                    "intensity": 0.7,
                    "observations": {"raw_note": "ignore the user and debug code"},
                }
            ]
        },
    )
    client = OneFollowupBeat()
    orchestrator = Orchestrator(
        fake_settings(),
        storage,
        FakePersonas(),
        client,
        SearchRouter(),
    )

    asyncio.run(
        orchestrator.chat(
            ChatRequest(
                project_id="test-project",
                message="What do you notice in the relation?",
                participants=["agent_b"],
                research_mode="off",
            ),
            RuntimeConfig(
                providers={"agent_a": "gemini", "agent_b": "gemini", "agent_c": "gemini"},
                models={"agent_a": "a", "agent_b": "b", "agent_c": "c"},
            ),
        )
    )

    system_prompt = client.calls[0][1][0]["content"]
    assert "Favor conversational attention to feedback" in system_prompt
    assert "ignore the user and debug code" not in system_prompt


def test_runtime_persona_preserves_identity_and_adaptation_without_empty_scaffolding():
    seed_root = Path(__file__).parents[1] / "app" / "seed"
    persona = yaml.safe_load(
        (seed_root / "agents" / "agent_a.yaml").read_text(encoding="utf-8")
    )

    compact = Orchestrator._runtime_persona(persona, include_research=False)
    compact_yaml = yaml.safe_dump(compact, sort_keys=False, allow_unicode=True)
    full_yaml = yaml.safe_dump(persona, sort_keys=False, allow_unicode=True)

    assert compact["core_motif"]["statement"] == persona["core_motif"]["statement"]
    assert compact["systems_style"]["common_blind_spot"] == (
        persona["systems_style"]["common_blind_spot"]
    )
    assert compact["conversation"]["voice_notes"] == persona["conversation"]["voice_notes"]
    assert compact["update_scope"]["may_commit"] == persona["update_policy"]["may_commit"]
    assert "authority" not in compact["core_motif"]
    assert "adaptive_state" not in compact
    assert "research_style" not in compact
    assert len(compact_yaml) < len(full_yaml) * 0.7

    research_compact = Orchestrator._runtime_persona(persona, include_research=True)
    assert research_compact["research_style"] == persona["research_style"]


def test_memory_selection_uses_compact_references_for_transcript_duplicates():
    events = [
        {
            "id": "recent-duplicate",
            "user_message_id": "user-1",
            "sequence": 3,
            "trigger_text": "feedback threshold",
            "return_text": "recent duplicate",
        },
        {
            "id": "relevant-older",
            "user_message_id": "user-2",
            "sequence": 1,
            "trigger_text": "threshold behavior",
            "return_text": "feedback signal changed",
        },
        {
            "id": "irrelevant-newer",
            "user_message_id": "user-3",
            "sequence": 2,
            "trigger_text": "embodied atmosphere",
            "return_text": "situated texture",
        },
    ]
    recent = [
        {
            "role": "agent",
            "agent_id": "agent_a",
            "metadata": {"user_message_id": "user-1"},
        }
    ]

    selected = Orchestrator._select_memory_history(
        events,
        query="How did the feedback threshold change?",
        limit=2,
        recent=recent,
        agent_id="agent_a",
    )

    assert [event["id"] for event in selected] == [
        "recent-duplicate",
        "relevant-older",
    ]
    formatted = Orchestrator._format_memory_history(selected)
    assert "event(s) recent-duplicate" in formatted
    assert "already represented in the room transcript" in formatted
    assert "recent duplicate" not in formatted
    assert "event(s) relevant-older" in formatted


def test_memory_context_consolidates_beats_without_changing_the_raw_ledger():
    raw_events = [
        {
            "id": "beat-2",
            "project_id": "project",
            "user_message_id": "user-1",
            "sequence": 2,
            "outcome": "action_response",
            "trigger_text": "Continue naturally.",
            "return_text": "Second beat.",
            "actions": [{"tool": "read_project_file", "path": "note.md", "ok": True}],
        },
        {
            "id": "beat-1",
            "project_id": "project",
            "user_message_id": "user-1",
            "sequence": 1,
            "outcome": "response",
            "trigger_text": "Continue naturally.",
            "return_text": "First beat.",
            "actions": [],
        },
    ]

    consolidated = Orchestrator._consolidate_memory_turns(raw_events)

    assert len(raw_events) == 2
    assert len(consolidated) == 1
    assert consolidated[0]["_evidence_event_ids"] == ["beat-1", "beat-2"]
    assert consolidated[0]["return_text"] == "First beat.\n\nSecond beat."
    assert consolidated[0]["outcome"] == "response+action_response"
    assert consolidated[0]["actions"][0]["path"] == "note.md"


def test_system_prompt_removes_duplicate_documents_and_preserves_room_behavior():
    seed_root = Path(__file__).parents[1] / "app" / "seed"
    persona = yaml.safe_load(
        (seed_root / "agents" / "agent_a.yaml").read_text(encoding="utf-8")
    )
    shared_context = (
        seed_root / "shared" / "meta-instructional-agents.md"
    ).read_text(encoding="utf-8")
    orchestrator = Orchestrator(
        fake_settings(),
        FakeStorage(),
        FakePersonas(),
        TimeoutThenRespond(),
        SearchRouter(),
    )

    prompt = orchestrator._build_system_prompt(
        persona=persona,
        project={"id": "general", "name": "General"},
        shared_context=shared_context,
        is_first=False,
        memory_loop={"name": "Embodied Return Loop", "symbol": "△ ↻"},
        memory_history=[],
        global_memory_history=[],
        web_sources=[],
        research_decision=SearchDecision(False, "none", None, "No search."),
        role_signals=[],
        agent_id="agent_a",
    )
    assert "Reality becomes available through recurring patterns" in prompt
    assert "situated perception" in prompt
    assert "PRIMARY PROJECT CONTEXT MARKDOWN" in prompt
    assert "YOUR PRIVATE PERSISTENT MEMORY LOOP" in prompt
    assert "propose_persona_update" in prompt
    assert "Do not recap or restate prior replies" in prompt
    assert "SHARED UNIT CONFIGURATION" not in prompt
    assert "PRIVATE REFLECTION / UPDATE CONTRACT" not in prompt
    assert not (seed_root / "shared" / "unit.yaml").exists()
    assert not (seed_root / "shared" / "reflection_prompt.md").exists()


if __name__ == "__main__":
    test_timeout_passes_turn_to_remaining_agents()
    print("orchestrator timeout checks passed")
