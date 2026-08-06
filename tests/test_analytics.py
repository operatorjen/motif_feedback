import sqlite3
from pathlib import Path

from motif_feedback.storage import Storage


def test_prompt_usage_columns_are_added_to_existing_databases(tmp_path: Path):
    database = tmp_path / "state" / "motif.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE agent_prompt_runs (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL, turn_id TEXT NOT NULL,
                agent_id TEXT NOT NULL, turn_beat INTEGER NOT NULL,
                speaker_position INTEGER NOT NULL, provider TEXT NOT NULL,
                model TEXT NOT NULL, prompt_template_hash TEXT NOT NULL,
                persona_revision_hash TEXT NOT NULL, context_selector_version TEXT NOT NULL,
                status TEXT NOT NULL, message_id TEXT, prompt_tokens INTEGER,
                completion_tokens INTEGER, total_tokens INTEGER, output_chars INTEGER,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                UNIQUE(turn_id, agent_id, turn_beat)
            )
            """
        )

    storage = Storage(database, tmp_path / "projects")
    storage.initialize()
    with storage.connection() as connection:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(agent_prompt_runs)").fetchall()
        }

    assert {
        "cached_prompt_tokens",
        "reasoning_tokens",
        "provider_requests",
        "request_usage_json",
    } <= columns


def motif_observation(label: str, motif_id: str | None = None) -> dict:
    item = {
        "label": label,
        "description": f"{label} returns as an organization of the room.",
        "relation": "return" if motif_id else "emergence",
        "confidence": 0.8,
        "primary": True,
        "connections": [],
    }
    if motif_id:
        item["motif_id"] = motif_id
    return item


def begin_turn(storage: Storage, turn_id: str, project_id: str = "general") -> None:
    storage.begin_chat_turn(
        turn_id,
        project_id,
        f"fingerprint-{turn_id}",
        request={
            "turn_id": turn_id,
            "project_id": project_id,
            "message": "Observe the return.",
            "participants": ["agent_a"],
            "research_mode": "off",
        },
        runtime={
            "providers": {
                "agent_a": "openai",
                "agent_b": "openai",
                "agent_c": "openai",
            },
            "models": {
                "agent_a": "model-a",
                "agent_b": "model-b",
                "agent_c": "model-c",
            },
        },
    )


def record_prompt_run(
    storage: Storage,
    *,
    turn_id: str,
    exposures: list[dict],
    beat: int = 1,
) -> str:
    return storage.record_agent_prompt_run(
        project_id="general",
        turn_id=turn_id,
        agent_id="agent_a",
        turn_beat=beat,
        speaker_position=1,
        provider="openai",
        model="model-a",
        prompt_template_hash="prompt-hash",
        persona_revision_hash="persona-hash",
        context_selector_version="context-selector-v1",
        exposures=exposures,
    )


def test_analytics_records_context_feedback_and_prompted_motif_returns(tmp_path: Path):
    storage = Storage(tmp_path / "state" / "motif.db", tmp_path / "projects")
    storage.initialize()

    begin_turn(storage, "analytics-turn-1")
    user_one = storage.add_message(
        "general",
        "user",
        "Notice a recursive return.",
        metadata={"turn_id": "analytics-turn-1"},
    )
    first_run = record_prompt_run(
        storage,
        turn_id="analytics-turn-1",
        exposures=[
            {
                "context_kind": "recent_message",
                "source_id": user_one["id"],
                "source_project_id": "general",
                "prompt_section": "room_transcript",
                "rank": 1,
                "selection_reason": "recent_context_window",
                "source_version_hash": "message-hash",
                "estimated_chars": len(user_one["content"]),
            }
        ],
    )
    created = storage.record_motif_observations(
        project_id="general",
        observer_agent_id="agent_a",
        turn_id="analytics-turn-1",
        turn_beat=1,
        operation_id="analytics-turn-1:agent_a:1:tool:1:1",
        user_message_id=user_one["id"],
        observations=[motif_observation("Recursive return")],
    )
    first_response = storage.add_message(
        "general",
        "agent",
        "The return first appears here.",
        agent_id="agent_a",
        metadata={"turn_id": "analytics-turn-1", "turn_beat": 1},
    )
    storage.complete_agent_prompt_run(
        first_run,
        status="completed",
        message_id=first_response["id"],
        provider_usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cached_prompt_tokens": 40,
            "reasoning_tokens": 5,
        },
        provider_request_usage=[
            {"request": 1, "prompt_tokens": 60, "cached_prompt_tokens": 10},
            {"request": 2, "prompt_tokens": 40, "cached_prompt_tokens": 30},
        ],
        output_chars=len(first_response["content"]),
    )

    motif_id = created["primary_motif_id"]
    begin_turn(storage, "analytics-turn-2")
    user_two = storage.add_message(
        "general",
        "user",
        "The recursive return appears again.",
        metadata={"turn_id": "analytics-turn-2"},
    )
    second_run = record_prompt_run(
        storage,
        turn_id="analytics-turn-2",
        exposures=[
            {
                "context_kind": "own_motif",
                "source_id": motif_id,
                "source_project_id": "general",
                "prompt_section": "own_motif_hypotheses",
                "rank": 1,
                "selection_reason": "observer_motif_context",
                "source_version_hash": "motif-hash",
                "estimated_chars": 64,
            }
        ],
    )
    storage.record_motif_observations(
        project_id="general",
        observer_agent_id="agent_a",
        turn_id="analytics-turn-2",
        turn_beat=1,
        operation_id="analytics-turn-2:agent_a:1:tool:1:1",
        user_message_id=user_two["id"],
        observations=[motif_observation("Recursive return", motif_id)],
    )
    second_response = storage.add_message(
        "general",
        "agent",
        "The return has transformed.",
        agent_id="agent_a",
        metadata={"turn_id": "analytics-turn-2", "turn_beat": 1},
    )
    storage.complete_agent_prompt_run(
        second_run,
        status="completed",
        message_id=second_response["id"],
        provider_usage={"total_tokens": 90},
        output_chars=len(second_response["content"]),
    )
    storage.record_interaction_feedback(
        project_id="general",
        message_id=second_response["id"],
        feedback_type="useful_difference",
        active=True,
    )

    snapshot = storage.analytics_snapshot("general")

    assert snapshot["coverage"]["prompt_runs"] == 2
    assert snapshot["coverage"]["context_exposures"] == 2
    assert snapshot["agents"][0]["speaker_positions"] == {"1": 2}
    assert snapshot["agents"][0]["feedback"]["useful_difference"] == 1
    assert snapshot["agents"][0]["cached_prompt_tokens"] == 40
    assert snapshot["agents"][0]["reasoning_tokens"] == 5
    assert snapshot["agents"][0]["provider_requests"] == 2
    assert snapshot["motifs"]["return_exposure"] == {
        "prompted": 1,
        "unprompted": 1,
        "unknown": 0,
    }
    assert snapshot["recent_responses"][0]["feedback"] == ["useful_difference"]

    storage.record_interaction_feedback(
        project_id="general",
        message_id=second_response["id"],
        feedback_type="useful_difference",
        active=False,
    )
    revised = storage.analytics_snapshot("general")
    assert revised["agents"][0]["feedback"]["useful_difference"] == 0
    assert revised["recent_responses"][0]["feedback"] == []


def test_analytics_page_is_separate_and_linked_from_the_room():
    root = Path(__file__).parents[1] / "motif_feedback" / "static"
    room = (root / "index.html").read_text(encoding="utf-8")
    analytics = (root / "analytics.html").read_text(encoding="utf-8")
    source = (root / "js" / "analytics.js").read_text(encoding="utf-8")

    assert 'href="/analytics">ANALYTICS / DEBUG</a>' in room
    assert 'href="/">RETURN TO ROOM</a>' in analytics
    assert 'src="/static/js/analytics.js"' in analytics
    assert "<script>" not in analytics
    assert ".innerHTML" not in source
