from pathlib import Path

import pytest

from motif_feedback.agent_tools import AgentToolExecutor, ToolContext
from motif_feedback.file_tools import ProjectFileTools
from motif_feedback.motif_checkpoints import agent_pattern_checkpoints, project_motif_analysis
from motif_feedback.motif_trajectories import motif_sequence_summary
from motif_feedback.storage import Storage, StorageError


def make_storage(tmp_path: Path) -> tuple[Storage, dict]:
    storage = Storage(tmp_path / "state.db", tmp_path / "projects")
    storage.initialize()
    project = storage.create_project("Motif observations")
    return storage, project


def observation(
    label: str,
    *,
    primary: bool = True,
    motif_id: str | None = None,
    relation: str = "emergence",
) -> dict:
    item = {
        "label": label,
        "description": f"{label} organizes this conversational return.",
        "relation": relation,
        "confidence": 0.8,
        "primary": primary,
    }
    if motif_id:
        item["motif_id"] = motif_id
    return item


def record(
    storage: Storage,
    project_id: str,
    *,
    turn: int,
    observations: list[dict],
) -> dict:
    return storage.record_motif_observations(
        project_id=project_id,
        observer_agent_id="agent_a",
        turn_id=f"turn-motif-{turn}",
        turn_beat=1,
        operation_id=f"turn-motif-{turn}:agent_a:1:tool:1:1",
        user_message_id=f"message-{turn}",
        observations=observations,
    )


def test_motif_observations_are_idempotent_and_promote_only_after_return(tmp_path: Path):
    storage, project = make_storage(tmp_path)
    first = record(
        storage,
        project["id"],
        turn=1,
        observations=[observation("Observer midpoint")],
    )
    repeated = storage.record_motif_observations(
        project_id=project["id"],
        observer_agent_id="agent_a",
        turn_id="turn-motif-1",
        turn_beat=1,
        operation_id="turn-motif-1:agent_a:1:tool:1:1",
        user_message_id="message-1",
        observations=[observation("This payload is ignored during recovery")],
    )

    assert repeated == first
    motif_id = first["primary_motif_id"]
    candidate = storage.get_motif_detail(project["id"], motif_id)
    assert candidate["status"] == "candidate"
    assert candidate["support_count"] == 1
    assert candidate["distinct_turn_count"] == 1
    assert [event["event_type"] for event in candidate["events"]] == ["created"]

    second = record(
        storage,
        project["id"],
        turn=2,
        observations=[
            observation(
                "Observer midpoint",
                motif_id=motif_id,
                relation="return",
            )
        ],
    )
    supported = storage.get_motif_detail(project["id"], motif_id)
    assert second["observations"][0]["status"] == "supported"
    assert supported["support_count"] == 2
    assert supported["distinct_turn_count"] == 2
    assert [event["event_type"] for event in supported["events"]] == [
        "promoted",
        "created",
    ]
    sequence = storage.primary_motif_event_sequence(
        project["id"],
        "agent_a",
        limit=64,
    )
    assert [event["motif_id"] for event in sequence] == [
        motif_id,
        motif_id,
    ]


def test_same_turn_rephrasing_adds_an_alias_without_false_promotion(tmp_path: Path):
    storage, project = make_storage(tmp_path)
    created = storage.record_motif_observations(
        project_id=project["id"],
        observer_agent_id="agent_a",
        turn_id="same-turn",
        turn_beat=1,
        operation_id="same-turn:agent_a:1",
        user_message_id="same-turn-user",
        observations=[observation("Negotiated midpoint")],
    )
    motif_id = created["primary_motif_id"]
    storage.record_motif_observations(
        project_id=project["id"],
        observer_agent_id="agent_a",
        turn_id="same-turn",
        turn_beat=2,
        operation_id="same-turn:agent_a:2",
        user_message_id="same-turn-user",
        observations=[
            observation(
                "Conversational equilibrium",
                motif_id=motif_id,
                relation="transformation",
            )
        ],
    )

    candidate = storage.get_motif(project["id"], motif_id)
    assert candidate["label"] == "Negotiated midpoint"
    assert candidate["aliases"] == [
        "Negotiated midpoint",
        "Conversational equilibrium",
    ]
    assert candidate["support_count"] == 2
    assert candidate["distinct_turn_count"] == 1
    assert candidate["status"] == "candidate"
    assert storage.primary_motif_event_sequence(
        project["id"],
        "agent_a",
        limit=64,
        statuses={"supported", "active", "dormant"},
    ) == []

    resolved_by_alias = storage.record_motif_observations(
        project_id=project["id"],
        observer_agent_id="agent_a",
        turn_id="later-turn",
        turn_beat=1,
        operation_id="later-turn:agent_a:1",
        user_message_id="later-turn-user",
        observations=[observation("Conversational equilibrium", relation="return")],
    )
    assert resolved_by_alias["primary_motif_id"] == motif_id
    supported = storage.get_motif(project["id"], motif_id)
    assert supported["status"] == "supported"
    assert supported["distinct_turn_count"] == 2
    event_sequence = storage.primary_motif_event_sequence(
        project["id"],
        "agent_a",
        limit=64,
        statuses={"supported"},
    )
    assert [event["turn_id"] for event in event_sequence] == [
        "same-turn",
        "same-turn",
        "later-turn",
    ]
    assert {event["label"] for event in event_sequence} == {"Negotiated midpoint"}


def test_motif_observations_attach_both_sides_of_turn_evidence(tmp_path: Path):
    storage, project = make_storage(tmp_path)
    user_message = storage.add_message(
        project["id"],
        "user",
        "Can the agents meet at a midpoint?",
        metadata={"turn_id": "evidence-turn"},
    )
    created = storage.record_motif_observations(
        project_id=project["id"],
        observer_agent_id="agent_a",
        turn_id="evidence-turn",
        turn_beat=1,
        operation_id="evidence-turn:agent_a:1",
        user_message_id=user_message["id"],
        observations=[observation("Observer midpoint")],
    )
    agent_message = storage.add_message(
        project["id"],
        "agent",
        "The midpoint is negotiated rather than fixed.",
        agent_id="agent_a",
        metadata={"turn_id": "evidence-turn", "turn_beat": 1},
    )

    assert storage.attach_motif_response_evidence(
        project_id=project["id"],
        observer_agent_id="agent_a",
        turn_id="evidence-turn",
        turn_beat=1,
        message_id=agent_message["id"],
    ) == 1
    detail = storage.get_motif_detail(project["id"], created["primary_motif_id"])
    evidence = detail["events"][0]["evidence"]
    assert [(item["role"], item["excerpt"]) for item in evidence] == [
        ("user", "Can the agents meet at a midpoint?"),
        ("agent", "The midpoint is negotiated rather than fixed."),
    ]


def test_cross_observer_relations_remain_provisional_and_do_not_merge(tmp_path: Path):
    storage, project = make_storage(tmp_path)
    target = storage.record_motif_observations(
        project_id=project["id"],
        observer_agent_id="agent_b",
        turn_id="relation-turn-b",
        turn_beat=1,
        operation_id="relation-turn-b:agent_b:1",
        user_message_id="relation-user",
        observations=[observation("Metastable transition")],
    )
    source_observation = observation("Felt threshold")
    source_observation["connections"] = [
        {
            "motif_id": target["primary_motif_id"],
            "relation": "translation",
            "confidence": 0.72,
            "description": "Two observer-specific readings of the same turn may translate.",
        }
    ]
    source = storage.record_motif_observations(
        project_id=project["id"],
        observer_agent_id="agent_a",
        turn_id="relation-turn-a",
        turn_beat=1,
        operation_id="relation-turn-a:agent_a:1",
        user_message_id="relation-user",
        observations=[source_observation],
    )

    source_detail = storage.get_motif_detail(project["id"], source["primary_motif_id"])
    relation = source_detail["relations"][0]
    assert relation["relation"] == "translation"
    assert relation["source_label"] == "Felt threshold"
    assert relation["target_label"] == "Metastable transition"
    assert len(storage.list_motifs(project["id"])) == 2


def test_motifs_remain_observer_specific_and_user_lifecycle_is_append_only(tmp_path: Path):
    storage, project = make_storage(tmp_path)
    created = record(
        storage,
        project["id"],
        turn=1,
        observations=[observation("Recursive variation")],
    )
    motif_id = created["primary_motif_id"]

    with pytest.raises(StorageError, match="not owned"):
        storage.record_motif_observations(
            project_id=project["id"],
            observer_agent_id="agent_b",
            turn_id="turn-motif-b",
            turn_beat=1,
            operation_id="turn-motif-b:agent_b:1:tool:1:1",
            user_message_id="message-b",
            observations=[observation("Recursive variation", motif_id=motif_id)],
        )

    attempted_activation = observation(
        "Recursive variation",
        motif_id=motif_id,
        relation="return",
    )
    attempted_activation["proposed_status"] = "active"
    record(
        storage,
        project["id"],
        turn=2,
        observations=[attempted_activation],
    )
    assert storage.get_motif(project["id"], motif_id)["status"] == "supported"

    active = storage.set_motif_status(project["id"], motif_id, status="active")
    rejected = storage.set_motif_status(project["id"], motif_id, status="rejected")
    assert active["status"] == "active"
    assert rejected["status"] == "rejected"
    assert [event["status"] for event in rejected["events"][:2]] == [
        "rejected",
        "active",
    ]
    assert rejected["events"][-1]["event_type"] == "created"


def test_only_one_sparse_motif_batch_is_allowed_per_agent_beat(tmp_path: Path):
    storage, project = make_storage(tmp_path)
    record(
        storage,
        project["id"],
        turn=1,
        observations=[observation("Sparse observation")],
    )

    with pytest.raises(StorageError, match="already recorded"):
        storage.record_motif_observations(
            project_id=project["id"],
            observer_agent_id="agent_a",
            turn_id="turn-motif-1",
            turn_beat=1,
            operation_id="different-operation",
            user_message_id="message-1",
            observations=[observation("Another motif")],
        )


def test_motif_tool_is_a_recoverable_durable_turn_operation(tmp_path: Path):
    storage, project = make_storage(tmp_path)
    storage.begin_chat_turn(
        "turn-tool-motif",
        project["id"],
        "fingerprint",
        request={"message": "notice the motif"},
        runtime={"providers": {}},
    )
    executor = AgentToolExecutor(
        ProjectFileTools(storage, max_write_bytes=15_000, max_upload_bytes=100_000),
        object(),
    )
    context = ToolContext(
        agent_id="agent_a",
        project_id=project["id"],
        turn_id="turn-tool-motif",
        turn_beat=1,
        operation_id="turn-tool-motif:agent_a:1:tool:1:1",
        user_message_id="message-tool-motif",
    )
    arguments = {"observations": [observation("Durable observer hypothesis")]}

    first = executor.execute("record_motif_observations", arguments, context)
    second = executor.execute("record_motif_observations", arguments, context)

    assert second == first
    assert first["ok"] is True
    operations = storage.list_turn_operations("turn-tool-motif")
    assert [(item["operation_type"], item["status"]) for item in operations] == [
        ("tool:record_motif_observations", "completed")
    ]


def test_trajectory_summary_describes_recurrence_and_transition_variety():
    recurring_summary = motif_sequence_summary(["a", "a", "b", "a"])
    assert recurring_summary["sample_size"] == 4
    assert recurring_summary["recurrence_rate"] == 0.5
    assert recurring_summary["transition_diversity"] == 1.0


def test_sequence_patterns_require_support_across_distinct_turns():
    sequence = ["a", "b", "a", "b", "a", "b", "a", "b"]
    turns = [f"turn-{index}" for index in range(len(sequence))]
    summary = motif_sequence_summary(
        sequence,
        turn_ids=turns,
        labels={"a": "Threshold", "b": "Return"},
    )

    patterns = {tuple(item["labels"]): item for item in summary["frequent_patterns"]}
    returns = {tuple(item["labels"]): item for item in summary["return_patterns"]}
    assert patterns[("Threshold", "Return")]["distinct_turn_count"] == 4
    assert patterns[("Threshold", "Return", "Threshold")]["distinct_turn_count"] == 3
    assert returns[("Threshold", "Return", "Threshold")]["occurrence_count"] == 3

    same_turn = motif_sequence_summary(
        sequence,
        turn_ids=["one-turn"] * len(sequence),
        labels={"a": "Threshold", "b": "Return"},
    )
    assert same_turn["frequent_patterns"] == []
    assert same_turn["return_patterns"] == []


def test_established_patterns_become_user_controlled_agent_checkpoints(tmp_path: Path):
    storage, project = make_storage(tmp_path)
    motif_ids: dict[str, str] = {}
    for turn, label in enumerate(["Threshold", "Return"] * 4, start=1):
        item = observation(
            label,
            motif_id=motif_ids.get(label),
            relation="return" if label in motif_ids else "emergence",
        )
        result = record(
            storage,
            project["id"],
            turn=turn,
            observations=[item],
        )
        motif_ids.setdefault(label, result["primary_motif_id"])

    analysis = project_motif_analysis(storage, project["id"], ["agent_a"])
    assert analysis["trajectories"]["agent_a"]["established"]["sample_size"] == 8
    checkpoint = next(
        item
        for item in analysis["checkpoints"]
        if item["labels"] == ["Threshold", "Return"]
    )
    assert checkpoint["preference"] == "notice"
    assert checkpoint["distinct_turn_count"] == 4

    storage.set_motif_pattern_preference(
        project["id"],
        checkpoint["id"],
        observer_agent_id="agent_a",
        preference="test",
    )
    supplied = agent_pattern_checkpoints(storage, project["id"], "agent_a")
    assert next(item for item in supplied if item["id"] == checkpoint["id"])[
        "preference"
    ] == "test"

    storage.set_motif_pattern_preference(
        project["id"],
        checkpoint["id"],
        observer_agent_id="agent_a",
        preference="paused",
    )
    assert checkpoint["id"] not in {
        item["id"]
        for item in agent_pattern_checkpoints(storage, project["id"], "agent_a")
    }


def test_pattern_preferences_are_project_scoped_and_validated(tmp_path: Path):
    storage, project = make_storage(tmp_path)
    other = storage.create_project("Other project")
    pattern_key = "a" * 24
    storage.set_motif_pattern_preference(
        project["id"],
        pattern_key,
        observer_agent_id="agent_a",
        preference="follow",
    )

    assert storage.list_motif_pattern_preferences(project["id"])[0][
        "preference"
    ] == "follow"
    assert storage.list_motif_pattern_preferences(other["id"]) == []
    with pytest.raises(StorageError, match="Invalid motif pattern checkpoint"):
        storage.set_motif_pattern_preference(
            project["id"],
            "not-a-checkpoint",
            observer_agent_id="agent_a",
            preference="notice",
        )
    with pytest.raises(StorageError, match="Unknown motif pattern preference"):
        storage.set_motif_pattern_preference(
            project["id"],
            pattern_key,
            observer_agent_id="agent_a",
            preference="randomize",
        )


def test_initialize_removes_legacy_related_motif_column(tmp_path: Path):
    storage, _project = make_storage(tmp_path)
    with storage.connection() as connection:
        connection.execute(
            "ALTER TABLE motif_events ADD COLUMN "
            "related_motif_ids_json TEXT NOT NULL DEFAULT '[]'"
        )
        before = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(motif_events)").fetchall()
        }
    assert "related_motif_ids_json" in before

    storage.initialize()

    with storage.connection() as connection:
        after = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(motif_events)").fetchall()
        }
    assert "related_motif_ids_json" not in after
