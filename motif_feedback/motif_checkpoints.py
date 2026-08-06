from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from .constants import MOTIF_TRAJECTORY_WINDOW
from .motif_trajectories import motif_sequence_summary


def motif_pattern_key(observer_agent_id: str, motif_ids: Iterable[str]) -> str:
    payload = json.dumps(
        [observer_agent_id, *motif_ids],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def project_motif_analysis(
    storage,
    project_id: str,
    agent_ids: Iterable[str],
) -> dict:
    """Build user-facing analytics and advisory checkpoints from motif events."""
    preferences = {
        item["pattern_key"]: item["preference"]
        for item in storage.list_motif_pattern_preferences(project_id)
    }
    trajectories = {}
    checkpoints = []
    for agent_id in agent_ids:
        observed_events = storage.primary_motif_event_sequence(
            project_id,
            agent_id,
            limit=MOTIF_TRAJECTORY_WINDOW,
        )
        established_events = storage.primary_motif_event_sequence(
            project_id,
            agent_id,
            limit=MOTIF_TRAJECTORY_WINDOW,
            statuses={"supported", "active", "dormant"},
        )
        observed = _summarize_events(observed_events)
        established = _summarize_events(established_events)
        trajectories[agent_id] = {
            "observed": observed,
            "established": established,
        }
        checkpoints.extend(
            _checkpoints_from_summary(
                agent_id,
                established,
                preferences,
            )
        )
    checkpoints.sort(
        key=lambda item: (
            item["preference"] == "paused",
            -item["distinct_turn_count"],
            -item["occurrence_count"],
            item["labels"],
        )
    )
    return {
        "trajectories": trajectories,
        "checkpoints": checkpoints,
    }


def agent_pattern_checkpoints(storage, project_id: str, agent_id: str) -> list[dict]:
    preferences = {
        item["pattern_key"]: item["preference"]
        for item in storage.list_motif_pattern_preferences(project_id)
    }
    established_events = storage.primary_motif_event_sequence(
        project_id,
        agent_id,
        limit=MOTIF_TRAJECTORY_WINDOW,
        statuses={"supported", "active", "dormant"},
    )
    return [
        checkpoint
        for checkpoint in _checkpoints_from_summary(
            agent_id,
            _summarize_events(established_events),
            preferences,
        )
        if checkpoint["preference"] != "paused"
    ][:3]


def _summarize_events(events: list[dict]) -> dict:
    labels = {
        event["motif_id"]: event["label"]
        for event in events
    }
    return motif_sequence_summary(
        [event["motif_id"] for event in events],
        turn_ids=[event["turn_id"] for event in events],
        labels=labels,
    )


def _checkpoints_from_summary(
    agent_id: str,
    summary: dict,
    preferences: dict[str, str],
) -> list[dict]:
    checkpoints = []
    seen_patterns: set[tuple[str, ...]] = set()
    patterns = [
        *summary["return_patterns"],
        *summary["frequent_patterns"],
    ]
    for pattern in patterns:
        motif_ids = tuple(str(item) for item in pattern["motif_ids"])
        if motif_ids in seen_patterns:
            continue
        seen_patterns.add(motif_ids)
        pattern_key = motif_pattern_key(agent_id, motif_ids)
        is_return = len(motif_ids) > 2 and motif_ids[0] == motif_ids[-1]
        checkpoints.append(
            {
                "id": pattern_key,
                "observer_agent_id": agent_id,
                "kind": "return_path" if is_return else "recurring_sequence",
                "motif_ids": list(motif_ids),
                "labels": pattern["labels"],
                "occurrence_count": pattern["occurrence_count"],
                "distinct_turn_count": pattern["distinct_turn_count"],
                "preference": preferences.get(pattern_key, "notice"),
            }
        )
        if len(seen_patterns) >= 6:
            break
    return checkpoints
