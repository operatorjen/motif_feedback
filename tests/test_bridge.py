from pathlib import Path

from motif_feedback.bridge import build_motif_packet, build_turn_trace
from motif_feedback.storage import Storage


def make_storage(tmp_path: Path) -> tuple[Storage, dict]:
    storage = Storage(tmp_path / "state.db", tmp_path / "projects")
    storage.initialize()
    return storage, storage.create_project("Bridge field")


def test_motif_packet_preserves_observer_ownership_and_evidence(tmp_path: Path) -> None:
    storage, project = make_storage(tmp_path)
    user = storage.add_message(
        project["id"],
        "user",
        "The same threshold returned.",
        metadata={"turn_id": "bridge-turn"},
    )
    observed = storage.record_motif_observations(
        project_id=project["id"],
        observer_agent_id="agent_a",
        turn_id="bridge-turn",
        turn_beat=1,
        operation_id="bridge-turn:agent_a:1:tool:1",
        user_message_id=user["id"],
        observations=[
            {
                "label": "Felt threshold",
                "description": "A threshold organizes the return.",
                "relation": "emergence",
                "confidence": 0.8,
                "primary": True,
            }
        ],
    )
    agent = storage.add_message(
        project["id"],
        "agent",
        "The threshold is situated rather than universal.",
        agent_id="agent_a",
        metadata={"turn_id": "bridge-turn", "turn_beat": 1},
    )
    storage.attach_motif_response_evidence(
        project_id=project["id"],
        observer_agent_id="agent_a",
        turn_id="bridge-turn",
        turn_beat=1,
        message_id=agent["id"],
    )

    packet = build_motif_packet(
        storage,
        project["id"],
        motif_ids=[observed["primary_motif_id"]],
        checkpoint_ids=[],
        inquiry="What changes this relation?",
        human_note="Keep the situated difference visible.",
    )

    assert packet["schema_version"] == "motif-bridge/v1"
    assert packet["artifact_type"] == "motif_packet"
    assert packet["motifs"][0]["observer_agent_id"] == "agent_a"
    assert packet["motifs"][0]["events"][0]["evidence"][1]["message_id"] == agent["id"]
    assert "do not merge ownership" in packet["ownership_contract"]


def test_turn_trace_exposes_execution_without_runtime_configuration(tmp_path: Path) -> None:
    storage, project = make_storage(tmp_path)
    turn = storage.begin_chat_turn(
        "trace-turn",
        project["id"],
        "fingerprint",
        request={"message": "Test a return"},
        runtime={"providers": {}},
    )
    storage.add_message(
        project["id"],
        "user",
        "Test a return",
        metadata={"turn_id": turn["id"]},
    )

    trace = build_turn_trace(storage, project["id"], turn["id"])

    assert trace["artifact_type"] == "execution_trace"
    assert trace["turn"]["status"] == "running"
    assert trace["messages"][0]["content"] == "Test a return"
