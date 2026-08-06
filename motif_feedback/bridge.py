from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from .models import AGENT_IDS
from .motif_checkpoints import project_motif_analysis
from .storage import StorageError

BRIDGE_SCHEMA_VERSION = "motif-bridge/v1"


def _artifact_id(prefix: str, payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def build_motif_packet(
    storage,
    project_id: str,
    *,
    motif_ids: list[str],
    checkpoint_ids: list[str],
    inquiry: str,
    human_note: str,
) -> dict:
    project = storage.get_project(project_id)
    analysis = project_motif_analysis(storage, project_id, AGENT_IDS)
    available = storage.list_motifs(project_id, limit=100)
    available_by_id = {item["id"]: item for item in available}
    selected_ids = list(dict.fromkeys(motif_ids))
    if not selected_ids:
        selected_ids = [
            item["id"]
            for item in available
            if item.get("status") in {"supported", "active", "dormant"}
        ][:20]
    missing_motifs = [motif_id for motif_id in selected_ids if motif_id not in available_by_id]
    if missing_motifs:
        raise StorageError("One or more selected motifs do not belong to this project.")

    checkpoints_by_id = {item["id"]: item for item in analysis["checkpoints"]}
    selected_checkpoint_ids = list(dict.fromkeys(checkpoint_ids))
    missing_checkpoints = [
        checkpoint_id
        for checkpoint_id in selected_checkpoint_ids
        if checkpoint_id not in checkpoints_by_id
    ]
    if missing_checkpoints:
        raise StorageError("One or more selected checkpoints do not belong to this project.")

    motifs = [storage.get_motif_detail(project_id, motif_id) for motif_id in selected_ids]
    checkpoints = [checkpoints_by_id[item] for item in selected_checkpoint_ids]
    content = {
        "source_system": "motif_feedback",
        "project": {"id": project["id"], "name": project["name"]},
        "inquiry": inquiry.strip(),
        "human_note": human_note.strip(),
        "motifs": motifs,
        "checkpoints": checkpoints,
        "trajectories": analysis["trajectories"],
        "ownership_contract": (
            "Motifs remain project-scoped and owned by their observing agents. "
            "Relations are provisional and do not merge ownership or establish truth."
        ),
    }
    return {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "artifact_type": "motif_packet",
        "artifact_id": _artifact_id("packet", content),
        "created_at": datetime.now(UTC).isoformat(),
        **content,
    }


def build_turn_trace(storage, project_id: str, turn_id: str) -> dict:
    turn = storage.get_chat_turn(turn_id)
    if turn["project_id"] != project_id:
        raise StorageError("Chat turn not found.")
    operations = storage.list_turn_operations(turn_id)
    messages = storage.messages_for_turn(project_id, turn_id)
    content = {
        "source_system": "motif_feedback",
        "project_id": project_id,
        "turn": {
            key: turn.get(key)
            for key in (
                "id",
                "status",
                "resolution",
                "failure_detail",
                "started_at",
                "updated_at",
                "resolved_at",
            )
        },
        "operations": operations,
        "messages": messages,
    }
    return {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "artifact_type": "execution_trace",
        "artifact_id": _artifact_id("feedback_trace", content),
        "created_at": datetime.now(UTC).isoformat(),
        **content,
    }
