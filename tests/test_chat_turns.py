import asyncio

from app import main as main_module
from app.models import ChatRequest, RuntimeConfig
from app.orchestrator import RoomResponse
from app.storage import Storage


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
