from __future__ import annotations

import hashlib
import json
import uuid
from collections import defaultdict
from typing import Any

from .storage_core import StorageError, utc_now

ANALYTICS_FEEDBACK_TYPES = {
    "useful_difference",
    "repetitive",
    "off_lens",
    "unsupported",
}
PROMPT_RUN_FINAL_STATUSES = {"completed", "failed", "discarded"}
PROMPT_RUN_STATUSES = {"prepared", *PROMPT_RUN_FINAL_STATUSES}
CONTEXT_KINDS = {
    "recent_message",
    "same_turn_message",
    "local_memory",
    "global_memory",
    "own_motif",
    "other_observer_motif",
    "pattern_checkpoint",
    "web_source",
    "role_signal",
}


def _stable_identifier(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_mapping(value: str | None) -> dict:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class AnalyticsRepositoryMixin:
    def record_agent_prompt_run(
        self,
        *,
        project_id: str,
        turn_id: str,
        agent_id: str,
        turn_beat: int,
        speaker_position: int,
        provider: str,
        model: str,
        prompt_template_hash: str,
        persona_revision_hash: str,
        context_selector_version: str,
        exposures: list[dict[str, Any]],
    ) -> str:
        if not turn_id:
            raise StorageError("Analytics prompt runs require a durable turn identifier.")
        if turn_beat < 1 or speaker_position < 1:
            raise StorageError("Invalid analytics turn position.")
        run_id = _stable_identifier([turn_id, agent_id, turn_beat])[:32]
        now = utc_now()
        with self._write_lock, self.connection() as connection:
            self._project_from_connection(connection, project_id)
            connection.execute(
                """
                INSERT INTO agent_prompt_runs(
                    id, project_id, turn_id, agent_id, turn_beat, speaker_position,
                    provider, model, prompt_template_hash, persona_revision_hash,
                    context_selector_version, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?, ?)
                ON CONFLICT(turn_id, agent_id, turn_beat) DO UPDATE SET
                    speaker_position = excluded.speaker_position,
                    provider = excluded.provider,
                    model = excluded.model,
                    prompt_template_hash = excluded.prompt_template_hash,
                    persona_revision_hash = excluded.persona_revision_hash,
                    context_selector_version = excluded.context_selector_version,
                    status = CASE
                        WHEN agent_prompt_runs.status IN ('completed', 'discarded')
                        THEN agent_prompt_runs.status
                        ELSE 'prepared'
                    END,
                    updated_at = excluded.updated_at
                """,
                (
                    run_id,
                    project_id,
                    turn_id,
                    agent_id,
                    turn_beat,
                    speaker_position,
                    provider,
                    model,
                    prompt_template_hash,
                    persona_revision_hash,
                    context_selector_version,
                    now,
                    now,
                ),
            )
            for exposure in exposures:
                kind = str(exposure.get("context_kind") or "")
                if kind not in CONTEXT_KINDS:
                    raise StorageError("Unknown analytics context kind.")
                source_id = str(exposure.get("source_id") or "").strip()
                prompt_section = str(exposure.get("prompt_section") or "").strip()
                if not source_id or not prompt_section:
                    raise StorageError("Analytics context exposures require a source and section.")
                exposure_id = _stable_identifier(
                    [run_id, kind, source_id, prompt_section]
                )[:32]
                connection.execute(
                    """
                    INSERT INTO context_exposures(
                        id, prompt_run_id, project_id, context_kind, source_id,
                        source_project_id, prompt_section, rank, selection_reason,
                        source_version_hash, estimated_chars, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(prompt_run_id, context_kind, source_id, prompt_section)
                    DO UPDATE SET
                        source_project_id = excluded.source_project_id,
                        rank = excluded.rank,
                        selection_reason = excluded.selection_reason,
                        source_version_hash = excluded.source_version_hash,
                        estimated_chars = excluded.estimated_chars
                    """,
                    (
                        exposure_id,
                        run_id,
                        project_id,
                        kind,
                        source_id[:200],
                        str(exposure.get("source_project_id") or "")[:120] or None,
                        prompt_section[:80],
                        max(0, int(exposure.get("rank") or 0)),
                        str(exposure.get("selection_reason") or "selected")[:120],
                        str(exposure.get("source_version_hash") or "")[:64],
                        max(0, int(exposure.get("estimated_chars") or 0)),
                        now,
                    ),
                )
        return run_id

    def complete_agent_prompt_run(
        self,
        prompt_run_id: str,
        *,
        status: str,
        message_id: str | None = None,
        provider_usage: dict[str, Any] | None = None,
        provider_request_usage: list[dict[str, Any]] | None = None,
        output_chars: int | None = None,
    ) -> None:
        if status not in PROMPT_RUN_FINAL_STATUSES:
            raise StorageError("Invalid final analytics prompt-run status.")
        usage = provider_usage or {}
        request_usage = provider_request_usage or []

        def optional_integer(key: str) -> int | None:
            value = usage.get(key)
            return int(value) if isinstance(value, int) and value >= 0 else None

        with self._write_lock, self.connection() as connection:
            updated = connection.execute(
                """
                UPDATE agent_prompt_runs
                SET status = ?, message_id = COALESCE(?, message_id),
                    prompt_tokens = COALESCE(?, prompt_tokens),
                    completion_tokens = COALESCE(?, completion_tokens),
                    total_tokens = COALESCE(?, total_tokens),
                    cached_prompt_tokens = COALESCE(?, cached_prompt_tokens),
                    reasoning_tokens = COALESCE(?, reasoning_tokens),
                    provider_requests = COALESCE(?, provider_requests),
                    request_usage_json = ?,
                    output_chars = COALESCE(?, output_chars),
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    message_id,
                    optional_integer("prompt_tokens"),
                    optional_integer("completion_tokens"),
                    optional_integer("total_tokens"),
                    optional_integer("cached_prompt_tokens"),
                    optional_integer("reasoning_tokens"),
                    len(request_usage),
                    json.dumps(request_usage, ensure_ascii=False),
                    max(0, int(output_chars)) if output_chars is not None else None,
                    utc_now(),
                    prompt_run_id,
                ),
            )
            if updated.rowcount != 1:
                raise StorageError("Analytics prompt run not found.")

    def record_interaction_feedback(
        self,
        *,
        project_id: str,
        message_id: str,
        feedback_type: str,
        active: bool,
    ) -> dict:
        if feedback_type not in ANALYTICS_FEEDBACK_TYPES:
            raise StorageError("Unknown interaction feedback type.")
        now = utc_now()
        with self._write_lock, self.connection() as connection:
            self._project_from_connection(connection, project_id)
            message = connection.execute(
                """
                SELECT agent_id FROM messages
                WHERE id = ? AND project_id = ? AND role = 'agent'
                """,
                (message_id, project_id),
            ).fetchone()
            if message is None or not message["agent_id"]:
                raise StorageError("Feedback can only be attached to an agent response.")
            record = {
                "id": uuid.uuid4().hex,
                "project_id": project_id,
                "message_id": message_id,
                "agent_id": message["agent_id"],
                "feedback_type": feedback_type,
                "active": bool(active),
                "created_at": now,
            }
            connection.execute(
                """
                INSERT INTO interaction_feedback_events(
                    id, project_id, message_id, agent_id,
                    feedback_type, active, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    project_id,
                    message_id,
                    record["agent_id"],
                    feedback_type,
                    int(bool(active)),
                    now,
                ),
            )
        return record

    def analytics_snapshot(self, project_id: str | None = None) -> dict:
        with self.connection() as connection:
            if project_id is not None:
                self._project_from_connection(connection, project_id)
            projects = [
                dict(row)
                for row in connection.execute(
                    "SELECT id, name FROM projects ORDER BY name COLLATE NOCASE, id"
                ).fetchall()
            ]
            project_names = {item["id"]: item["name"] for item in projects}
            parameters = (project_id,) if project_id is not None else ()

            def clause(column: str = "project_id", *, prefix: str = " WHERE ") -> str:
                return f"{prefix}{column} = ?" if project_id is not None else ""

            def count(table: str, column: str = "project_id") -> int:
                row = connection.execute(
                    f"SELECT COUNT(*) AS count FROM {table}{clause(column)}",
                    parameters,
                ).fetchone()
                return int(row["count"])

            coverage = {
                "projects": 1 if project_id is not None else len(projects),
                "messages": count("messages"),
                "agent_responses": int(
                    connection.execute(
                        "SELECT COUNT(*) AS count FROM messages "
                        "WHERE role = 'agent'"
                        + (" AND project_id = ?" if project_id is not None else ""),
                        parameters,
                    ).fetchone()["count"]
                ),
                "memory_events": count("agent_memory_events"),
                "global_memory_events": count(
                    "agent_global_memory_events",
                    "source_project_id",
                ),
                "chat_turns": count("chat_turns"),
                "prompt_runs": count("agent_prompt_runs"),
                "context_exposures": count("context_exposures"),
                "motifs": count("motifs"),
                "motif_events": count("motif_events"),
                "feedback_events": count("interaction_feedback_events"),
            }

            activity_rows = connection.execute(
                """
                SELECT substr(created_at, 1, 10) AS day, agent_id, COUNT(*) AS responses
                FROM messages
                WHERE role = 'agent'
                """
                + (" AND project_id = ?" if project_id is not None else "")
                + """
                GROUP BY substr(created_at, 1, 10), agent_id
                ORDER BY day, agent_id
                """,
                parameters,
            ).fetchall()
            activity_by_day: dict[str, dict[str, Any]] = {}
            for row in activity_rows:
                record = activity_by_day.setdefault(
                    row["day"],
                    {
                        "day": row["day"],
                        "agent_a": 0,
                        "agent_b": 0,
                        "agent_c": 0,
                    },
                )
                if row["agent_id"] in {"agent_a", "agent_b", "agent_c"}:
                    record[row["agent_id"]] = int(row["responses"])

            agent_rows = connection.execute(
                """
                SELECT agent_id, COUNT(*) AS responses,
                       ROUND(AVG(LENGTH(content)), 1) AS average_chars
                FROM messages
                WHERE role = 'agent'
                """
                + (" AND project_id = ?" if project_id is not None else "")
                + " GROUP BY agent_id ORDER BY agent_id",
                parameters,
            ).fetchall()
            agent_summary = {
                agent_id: {
                    "agent_id": agent_id,
                    "responses": 0,
                    "average_chars": 0,
                    "prompt_runs": 0,
                    "completed_prompt_runs": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cached_prompt_tokens": 0,
                    "reasoning_tokens": 0,
                    "provider_requests": 0,
                    "speaker_positions": {},
                    "feedback": {kind: 0 for kind in sorted(ANALYTICS_FEEDBACK_TYPES)},
                }
                for agent_id in ("agent_a", "agent_b", "agent_c")
            }
            for row in agent_rows:
                if row["agent_id"] in agent_summary:
                    agent_summary[row["agent_id"]].update(
                        {
                            "responses": int(row["responses"]),
                            "average_chars": float(row["average_chars"] or 0),
                        }
                    )

            run_rows = connection.execute(
                """
                SELECT agent_id, status, speaker_position,
                       prompt_tokens, completion_tokens, total_tokens,
                       cached_prompt_tokens, reasoning_tokens, provider_requests
                FROM agent_prompt_runs
                """
                + clause(),
                parameters,
            ).fetchall()
            for row in run_rows:
                summary = agent_summary.get(row["agent_id"])
                if summary is None:
                    continue
                summary["prompt_runs"] += 1
                summary["completed_prompt_runs"] += int(row["status"] == "completed")
                summary["prompt_tokens"] += int(row["prompt_tokens"] or 0)
                summary["completion_tokens"] += int(row["completion_tokens"] or 0)
                summary["total_tokens"] += int(row["total_tokens"] or 0)
                summary["cached_prompt_tokens"] += int(row["cached_prompt_tokens"] or 0)
                summary["reasoning_tokens"] += int(row["reasoning_tokens"] or 0)
                summary["provider_requests"] += int(row["provider_requests"] or 0)
                position = str(row["speaker_position"])
                summary["speaker_positions"][position] = (
                    summary["speaker_positions"].get(position, 0) + 1
                )

            feedback_rows = connection.execute(
                """
                SELECT event.id, event.message_id, event.agent_id,
                       event.feedback_type, event.active, event.created_at
                FROM interaction_feedback_events AS event
                """
                + (
                    "WHERE event.project_id = ? "
                    if project_id is not None
                    else ""
                )
                + "ORDER BY event.created_at, event.rowid",
                parameters,
            ).fetchall()
            latest_feedback: dict[tuple[str, str], dict] = {}
            for row in feedback_rows:
                latest_feedback[(row["message_id"], row["feedback_type"])] = dict(row)
            feedback_by_message: dict[str, list[str]] = defaultdict(list)
            for row in latest_feedback.values():
                if not bool(row["active"]):
                    continue
                feedback_by_message[row["message_id"]].append(row["feedback_type"])
                summary = agent_summary.get(row["agent_id"])
                if summary is not None:
                    summary["feedback"][row["feedback_type"]] += 1

            context_rows = connection.execute(
                """
                SELECT run.agent_id, exposure.context_kind,
                       COUNT(*) AS exposure_count,
                       SUM(exposure.estimated_chars) AS estimated_chars
                FROM context_exposures AS exposure
                JOIN agent_prompt_runs AS run ON run.id = exposure.prompt_run_id
                """
                + (
                    "WHERE run.project_id = ? "
                    if project_id is not None
                    else ""
                )
                + """
                GROUP BY run.agent_id, exposure.context_kind
                ORDER BY exposure.context_kind, run.agent_id
                """,
                parameters,
            ).fetchall()
            contexts = [
                {
                    "agent_id": row["agent_id"],
                    "context_kind": row["context_kind"],
                    "count": int(row["exposure_count"]),
                    "estimated_chars": int(row["estimated_chars"] or 0),
                }
                for row in context_rows
            ]

            motif_status_rows = connection.execute(
                """
                SELECT status, observer_agent_id, COUNT(*) AS motif_count
                FROM motifs
                """
                + clause()
                + " GROUP BY status, observer_agent_id ORDER BY status, observer_agent_id",
                parameters,
            ).fetchall()
            motif_statuses = [
                {
                    "status": row["status"],
                    "agent_id": row["observer_agent_id"],
                    "count": int(row["motif_count"]),
                }
                for row in motif_status_rows
            ]

            run_lookup_rows = connection.execute(
                """
                SELECT id, turn_id, agent_id, turn_beat
                FROM agent_prompt_runs
                """
                + clause(),
                parameters,
            ).fetchall()
            run_lookup = {
                (row["turn_id"], row["agent_id"], int(row["turn_beat"])): row["id"]
                for row in run_lookup_rows
            }
            exposure_rows = connection.execute(
                """
                SELECT prompt_run_id, source_id
                FROM context_exposures
                WHERE context_kind IN ('own_motif', 'other_observer_motif')
                """
                + (" AND project_id = ?" if project_id is not None else ""),
                parameters,
            ).fetchall()
            exposed_motifs: dict[str, set[str]] = defaultdict(set)
            for row in exposure_rows:
                exposed_motifs[row["prompt_run_id"]].add(row["source_id"])
            motif_event_rows = connection.execute(
                """
                SELECT motif_id, observer_agent_id, turn_id, turn_beat
                FROM motif_events
                WHERE actor_type = 'agent'
                """
                + (" AND project_id = ?" if project_id is not None else ""),
                parameters,
            ).fetchall()
            motif_return_exposure = {"prompted": 0, "unprompted": 0, "unknown": 0}
            for row in motif_event_rows:
                key = (
                    row["turn_id"],
                    row["observer_agent_id"],
                    int(row["turn_beat"] or 1),
                )
                run_id = run_lookup.get(key)
                if run_id is None:
                    motif_return_exposure["unknown"] += 1
                elif row["motif_id"] in exposed_motifs.get(run_id, set()):
                    motif_return_exposure["prompted"] += 1
                else:
                    motif_return_exposure["unprompted"] += 1

            turn_rows = connection.execute(
                "SELECT status, trace_json FROM chat_turns" + clause(),
                parameters,
            ).fetchall()
            turn_statuses: dict[str, int] = defaultdict(int)
            duration_total = 0.0
            duration_count = 0
            provider_requests = 0
            provider_usage: dict[str, int] = defaultdict(int)
            for row in turn_rows:
                turn_statuses[row["status"]] += 1
                trace = _json_mapping(row["trace_json"])
                duration = trace.get("duration_ms")
                if isinstance(duration, (int, float)):
                    duration_total += float(duration)
                    duration_count += 1
                requests = trace.get("provider_requests")
                if isinstance(requests, int):
                    provider_requests += requests
                for key, value in (trace.get("provider_usage") or {}).items():
                    if isinstance(value, int):
                        provider_usage[str(key)] += value
            reliability = {
                "turn_statuses": dict(turn_statuses),
                "average_duration_ms": (
                    round(duration_total / duration_count, 1) if duration_count else None
                ),
                "provider_requests": provider_requests,
                "provider_usage": dict(provider_usage),
            }

            recent_rows = connection.execute(
                """
                SELECT message.id, message.project_id, project.name AS project_name,
                       message.agent_id, message.content, message.created_at,
                       message.turn_id, message.metadata_json,
                       run.speaker_position
                FROM messages AS message
                JOIN projects AS project ON project.id = message.project_id
                LEFT JOIN agent_prompt_runs AS run ON run.message_id = message.id
                WHERE message.role = 'agent'
                """
                + (" AND message.project_id = ?" if project_id is not None else "")
                + """
                ORDER BY message.created_at DESC, message.rowid DESC
                LIMIT 36
                """,
                parameters,
            ).fetchall()
            recent_responses = []
            for row in recent_rows:
                metadata = _json_mapping(row["metadata_json"])
                recent_responses.append(
                    {
                        "id": row["id"],
                        "project_id": row["project_id"],
                        "project_name": row["project_name"],
                        "agent_id": row["agent_id"],
                        "excerpt": str(row["content"])[:360],
                        "created_at": row["created_at"],
                        "turn_id": row["turn_id"],
                        "turn_beat": metadata.get("turn_beat"),
                        "speaker_position": row["speaker_position"],
                        "feedback": sorted(feedback_by_message.get(row["id"], [])),
                    }
                )

            project_rows = connection.execute(
                """
                SELECT project.id, project.name,
                       SUM(CASE WHEN message.role = 'agent' THEN 1 ELSE 0 END) AS responses
                FROM projects AS project
                LEFT JOIN messages AS message ON message.project_id = project.id
                """
                + ("WHERE project.id = ? " if project_id is not None else "")
                + """
                GROUP BY project.id, project.name
                ORDER BY responses DESC, project.name COLLATE NOCASE
                """,
                parameters,
            ).fetchall()
            project_activity = [
                {
                    "project_id": row["id"],
                    "project_name": row["name"],
                    "responses": int(row["responses"] or 0),
                }
                for row in project_rows
            ]

        return {
            "scope": {
                "project_id": project_id,
                "project_name": project_names.get(project_id) if project_id else None,
            },
            "projects": projects,
            "coverage": coverage,
            "activity": list(activity_by_day.values()),
            "agents": list(agent_summary.values()),
            "contexts": contexts,
            "motifs": {
                "statuses": motif_statuses,
                "return_exposure": motif_return_exposure,
            },
            "reliability": reliability,
            "project_activity": project_activity,
            "recent_responses": recent_responses,
        }
