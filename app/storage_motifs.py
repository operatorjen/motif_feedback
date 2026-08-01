from __future__ import annotations

import json
import re
import sqlite3
import uuid
from typing import Any

from .constants import (
    DEFAULT_MOTIF_LIMIT,
    MAX_MOTIF_LIMIT,
    MOTIF_CONNECTION_MAX_ITEMS,
    MOTIF_DESCRIPTION_MAX_CHARS,
    MOTIF_LABEL_MAX_CHARS,
    MOTIF_OBSERVATIONS_PER_BEAT,
)
from .storage_core import StorageError, utc_now

MOTIF_STATUSES = {"candidate", "supported", "active", "dormant", "rejected"}
AGENT_RELATIONS = {
    "emergence",
    "return",
    "extension",
    "bridge",
    "contrast",
    "transformation",
}
MOTIF_CONNECTION_RELATIONS = {
    "possible_alignment",
    "translation",
    "contrast",
    "extension",
    "transformation",
    "shared_evidence",
}
USER_MOTIF_STATUSES = {"active", "dormant", "rejected"}
MOTIF_PATTERN_PREFERENCES = {"notice", "follow", "test", "paused"}
MOTIF_PATTERN_KEY_RE = re.compile(r"^[a-f0-9]{24}$")


def normalize_motif_label(label: str) -> str:
    normalized = re.sub(r"[^\w]+", " ", str(label).casefold(), flags=re.UNICODE)
    return " ".join(normalized.split())


class MotifRepositoryMixin:
    def list_motif_pattern_preferences(self, project_id: str) -> list[dict]:
        with self.connection() as connection:
            self._project_from_connection(connection, project_id)
            rows = connection.execute(
                """
                SELECT pattern_key, observer_agent_id, preference, updated_at
                FROM motif_pattern_preferences
                WHERE project_id = ?
                ORDER BY updated_at DESC, pattern_key
                """,
                (project_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_motif_pattern_preference(
        self,
        project_id: str,
        pattern_key: str,
        *,
        observer_agent_id: str,
        preference: str,
    ) -> dict:
        self.get_project(project_id)
        if not MOTIF_PATTERN_KEY_RE.fullmatch(pattern_key):
            raise StorageError("Invalid motif pattern checkpoint.")
        if preference not in MOTIF_PATTERN_PREFERENCES:
            raise StorageError("Unknown motif pattern preference.")
        now = utc_now()
        with self._write_lock, self.connection() as connection:
            connection.execute(
                """
                INSERT INTO motif_pattern_preferences(
                    project_id, pattern_key, observer_agent_id, preference, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(project_id, pattern_key) DO UPDATE SET
                    observer_agent_id = excluded.observer_agent_id,
                    preference = excluded.preference,
                    updated_at = excluded.updated_at
                """,
                (
                    project_id,
                    pattern_key,
                    observer_agent_id,
                    preference,
                    now,
                ),
            )
            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (now, project_id),
            )
        return {
            "pattern_key": pattern_key,
            "observer_agent_id": observer_agent_id,
            "preference": preference,
            "updated_at": now,
        }

    def record_motif_observations(
        self,
        *,
        project_id: str,
        observer_agent_id: str,
        turn_id: str,
        turn_beat: int,
        operation_id: str,
        user_message_id: str | None,
        observations: list[dict[str, Any]],
    ) -> dict:
        prepared = self._validate_motif_observations(observations)
        if not turn_id or not operation_id:
            raise StorageError("Motif observations require a durable turn operation.")

        with self._write_lock, self.connection() as connection:
            self._project_from_connection(connection, project_id)
            prior = connection.execute(
                """
                SELECT result_json FROM motif_observation_batches
                WHERE operation_id = ?
                """,
                (operation_id,),
            ).fetchone()
            if prior is not None:
                return json.loads(prior["result_json"])
            occupied = connection.execute(
                """
                SELECT operation_id FROM motif_observation_batches
                WHERE project_id = ? AND observer_agent_id = ?
                  AND turn_id = ? AND turn_beat = ?
                """,
                (project_id, observer_agent_id, turn_id, turn_beat),
            ).fetchone()
            if occupied is not None:
                raise StorageError("This agent already recorded motif observations for this beat.")

            batch_id = uuid.uuid4().hex
            created_at = utc_now()
            connection.execute(
                """
                INSERT INTO motif_observation_batches(
                    id, operation_id, project_id, observer_agent_id, turn_id,
                    turn_beat, user_message_id, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, '{}', ?)
                """,
                (
                    batch_id,
                    operation_id,
                    project_id,
                    observer_agent_id,
                    turn_id,
                    turn_beat,
                    user_message_id,
                    created_at,
                ),
            )
            results = []
            resolved_motif_ids: set[str] = set()
            for observation in prepared:
                recorded = self._record_one_motif(
                    connection,
                    batch_id=batch_id,
                    project_id=project_id,
                    observer_agent_id=observer_agent_id,
                    user_message_id=user_message_id,
                    turn_id=turn_id,
                    turn_beat=turn_beat,
                    observation=observation,
                    created_at=created_at,
                )
                if recorded["motif_id"] in resolved_motif_ids:
                    raise StorageError("A motif may appear only once in one observation batch.")
                resolved_motif_ids.add(recorded["motif_id"])
                results.append(recorded)
            result = {
                "ok": True,
                "batch_id": batch_id,
                "observations": results,
                "primary_motif_id": next(
                    item["motif_id"] for item in results if item["primary"]
                ),
            }
            connection.execute(
                "UPDATE motif_observation_batches SET result_json = ? WHERE id = ?",
                (json.dumps(result, ensure_ascii=False), batch_id),
            )
            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (created_at, project_id),
            )
        return result

    def get_motif_batch_result(self, operation_id: str) -> dict | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT result_json FROM motif_observation_batches
                WHERE operation_id = ?
                """,
                (operation_id,),
            ).fetchone()
        return json.loads(row["result_json"]) if row is not None else None

    def list_motifs(
        self,
        project_id: str,
        *,
        observer_agent_id: str | None = None,
        statuses: set[str] | None = None,
        limit: int = DEFAULT_MOTIF_LIMIT,
    ) -> list[dict]:
        safe_limit = min(max(int(limit), 1), MAX_MOTIF_LIMIT)
        clauses = ["project_id = ?"]
        parameters: list[Any] = [project_id]
        if observer_agent_id:
            clauses.append("observer_agent_id = ?")
            parameters.append(observer_agent_id)
        if statuses:
            invalid = set(statuses) - MOTIF_STATUSES
            if invalid:
                raise StorageError("Unknown motif status.")
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            parameters.extend(sorted(statuses))
        parameters.append(safe_limit)
        with self.connection() as connection:
            self._project_from_connection(connection, project_id)
            rows = connection.execute(
                f"""
                SELECT * FROM motifs
                WHERE {' AND '.join(clauses)}
                ORDER BY updated_at DESC, id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
            motifs = [self._row_to_motif(row) for row in rows]
            self._attach_aliases(connection, motifs)
        return motifs

    def get_motif(self, project_id: str, motif_id: str) -> dict:
        with self.connection() as connection:
            self._project_from_connection(connection, project_id)
            row = connection.execute(
                "SELECT * FROM motifs WHERE project_id = ? AND id = ?",
                (project_id, motif_id),
            ).fetchone()
            if row is None:
                raise StorageError("Motif not found.")
            motif = self._row_to_motif(row)
            self._attach_aliases(connection, [motif])
        return motif

    def get_motif_detail(self, project_id: str, motif_id: str) -> dict:
        motif = self.get_motif(project_id, motif_id)
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM motif_events
                WHERE project_id = ? AND motif_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (project_id, motif_id),
            ).fetchall()
            events = [self._row_to_motif_event(row) for row in rows]
            evidence_ids = list(
                dict.fromkeys(
                    message_id
                    for event in events
                    for message_id in event["evidence_message_ids"]
                )
            )
            evidence_by_id: dict[str, dict] = {}
            if evidence_ids:
                placeholders = ", ".join("?" for _ in evidence_ids)
                evidence_rows = connection.execute(
                    f"""
                    SELECT id, role, agent_id, content, created_at
                    FROM messages
                    WHERE project_id = ? AND id IN ({placeholders})
                    """,
                    [project_id, *evidence_ids],
                ).fetchall()
                evidence_by_id = {
                    row["id"]: {
                        "message_id": row["id"],
                        "role": row["role"],
                        "agent_id": row["agent_id"],
                        "excerpt": str(row["content"])[:800],
                        "created_at": row["created_at"],
                    }
                    for row in evidence_rows
                }
            relation_rows = connection.execute(
                """
                SELECT relation_event.*,
                       source.label AS source_label,
                       source.observer_agent_id AS source_observer_agent_id,
                       target.label AS target_label,
                       target.observer_agent_id AS target_observer_agent_id
                FROM motif_relation_events AS relation_event
                JOIN motifs AS source ON source.id = relation_event.source_motif_id
                JOIN motifs AS target ON target.id = relation_event.target_motif_id
                WHERE relation_event.project_id = ?
                  AND (
                      relation_event.source_motif_id = ?
                      OR relation_event.target_motif_id = ?
                  )
                ORDER BY relation_event.created_at DESC, relation_event.id DESC
                """,
                (project_id, motif_id, motif_id),
            ).fetchall()
        for event in events:
            event["evidence"] = [
                evidence_by_id[message_id]
                for message_id in event["evidence_message_ids"]
                if message_id in evidence_by_id
            ]
        motif["events"] = events
        motif["relations"] = [dict(row) for row in relation_rows]
        return motif

    def attach_motif_response_evidence(
        self,
        *,
        project_id: str,
        observer_agent_id: str,
        turn_id: str,
        turn_beat: int,
        message_id: str,
    ) -> int:
        """Attach the committed agent response to observations made during its model beat."""
        with self._write_lock, self.connection() as connection:
            message = connection.execute(
                """
                SELECT id FROM messages
                WHERE id = ? AND project_id = ? AND role = 'agent' AND agent_id = ?
                """,
                (message_id, project_id, observer_agent_id),
            ).fetchone()
            if message is None:
                raise StorageError("Motif response evidence message not found.")
            rows = connection.execute(
                """
                SELECT id, evidence_message_ids_json
                FROM motif_events
                WHERE project_id = ? AND observer_agent_id = ?
                  AND turn_id = ? AND turn_beat = ? AND actor_type = 'agent'
                """,
                (project_id, observer_agent_id, turn_id, turn_beat),
            ).fetchall()
            changed = 0
            for row in rows:
                evidence_ids = json.loads(row["evidence_message_ids_json"] or "[]")
                if message_id in evidence_ids:
                    continue
                evidence_ids.append(message_id)
                connection.execute(
                    """
                    UPDATE motif_events
                    SET evidence_message_ids_json = ?
                    WHERE id = ?
                    """,
                    (json.dumps(evidence_ids, ensure_ascii=False), row["id"]),
                )
                changed += 1
        return changed

    def set_motif_status(
        self,
        project_id: str,
        motif_id: str,
        *,
        status: str,
        actor_id: str = "user",
    ) -> dict:
        if status not in USER_MOTIF_STATUSES:
            raise StorageError("Motif status must be active, dormant, or rejected.")
        motif = self.get_motif(project_id, motif_id)
        now = utc_now()
        with self._write_lock, self.connection() as connection:
            connection.execute(
                "UPDATE motifs SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, motif_id),
            )
            connection.execute(
                """
                INSERT INTO motif_events(
                    id, batch_id, motif_id, project_id, observer_agent_id,
                    actor_type, actor_id, event_type, relation, primary_flag,
                    confidence, status, description, user_message_id,
                    turn_id, turn_beat, created_at
                ) VALUES (?, NULL, ?, ?, ?, 'user', ?, 'status_changed', NULL, 0,
                          NULL, ?, ?, NULL, NULL, NULL, ?)
                """,
                (
                    uuid.uuid4().hex,
                    motif_id,
                    project_id,
                    motif["observer_agent_id"],
                    actor_id,
                    status,
                    f"User set this motif to {status}.",
                    now,
                ),
            )
            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (now, project_id),
            )
        return self.get_motif_detail(project_id, motif_id)

    def primary_motif_event_sequence(
        self,
        project_id: str,
        observer_agent_id: str,
        *,
        limit: int,
        statuses: set[str] | None = None,
    ) -> list[dict]:
        safe_limit = min(max(int(limit), 1), MAX_MOTIF_LIMIT)
        if statuses and (set(statuses) - MOTIF_STATUSES):
            raise StorageError("Unknown motif status.")
        status_clause = ""
        parameters: list[Any] = [project_id, observer_agent_id]
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            status_clause = f" AND motif.status IN ({placeholders})"
            parameters.extend(sorted(statuses))
        parameters.append(safe_limit)
        with self.connection() as connection:
            self._project_from_connection(connection, project_id)
            rows = connection.execute(
                f"""
                SELECT motif_id, label, status, turn_id, turn_beat, created_at FROM (
                    SELECT event.motif_id, motif.label, motif.status,
                           event.turn_id, event.turn_beat, event.created_at,
                           event.rowid AS event_rowid
                    FROM motif_events AS event
                    JOIN motifs AS motif ON motif.id = event.motif_id
                    WHERE event.project_id = ? AND event.observer_agent_id = ?
                      AND event.actor_type = 'agent' AND event.primary_flag = 1
                      {status_clause}
                    ORDER BY event.created_at DESC, event.rowid DESC
                    LIMIT ?
                )
                ORDER BY created_at, event_rowid
                """,
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def _record_one_motif(
        self,
        connection: sqlite3.Connection,
        *,
        batch_id: str,
        project_id: str,
        observer_agent_id: str,
        user_message_id: str | None,
        turn_id: str,
        turn_beat: int,
        observation: dict[str, Any],
        created_at: str,
    ) -> dict:
        motif_id = observation.get("motif_id")
        existing = None
        if motif_id:
            existing = connection.execute(
                """
                SELECT * FROM motifs
                WHERE id = ? AND project_id = ? AND observer_agent_id = ?
                """,
                (motif_id, project_id, observer_agent_id),
            ).fetchone()
            if existing is None:
                raise StorageError(
                    "Referenced observations[].motif_id is not owned by this agent in this "
                    "project. Use only an ID from this agent's motif context; put another "
                    "observer's ID in connections[].motif_id."
                )
        else:
            existing = connection.execute(
                """
                SELECT motif.*
                FROM motif_aliases AS alias
                JOIN motifs AS motif ON motif.id = alias.motif_id
                WHERE alias.project_id = ? AND alias.observer_agent_id = ?
                  AND alias.normalized_alias = ?
                """,
                (project_id, observer_agent_id, observation["normalized_label"]),
            ).fetchone()

        confidence = observation["confidence"]
        if existing is None:
            motif_id = uuid.uuid4().hex
            support_count = 1
            distinct_turn_count = 1
            status = "candidate"
            event_type = "created"
            connection.execute(
                """
                INSERT INTO motifs(
                    id, project_id, observer_agent_id, normalized_label, label,
                    description, status, confidence, support_count,
                    distinct_turn_count, last_seen_turn_id,
                    first_seen_user_message_id, last_seen_user_message_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    motif_id,
                    project_id,
                    observer_agent_id,
                    observation["normalized_label"],
                    observation["label"],
                    observation["description"],
                    status,
                    confidence,
                    support_count,
                    distinct_turn_count,
                    turn_id,
                    user_message_id,
                    user_message_id,
                    created_at,
                    created_at,
                ),
            )
        else:
            motif_id = existing["id"]
            old_count = int(existing["support_count"])
            support_count = old_count + 1
            distinct_turn_count = int(existing["distinct_turn_count"])
            if existing["last_seen_turn_id"] != turn_id:
                distinct_turn_count += 1
            confidence = (
                (float(existing["confidence"]) * old_count) + confidence
            ) / support_count
            status = existing["status"]
            event_type = "reinforced"
            if status == "candidate" and distinct_turn_count >= 2:
                status = "supported"
                event_type = "promoted"
            connection.execute(
                """
                UPDATE motifs
                SET status = ?, confidence = ?, support_count = ?,
                    distinct_turn_count = ?, last_seen_turn_id = ?,
                    last_seen_user_message_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    confidence,
                    support_count,
                    distinct_turn_count,
                    turn_id,
                    user_message_id,
                    created_at,
                    motif_id,
                ),
            )
        self._add_motif_alias(
            connection,
            project_id=project_id,
            observer_agent_id=observer_agent_id,
            motif_id=motif_id,
            label=observation["label"],
            normalized_alias=observation["normalized_label"],
            created_at=created_at,
        )

        connections = observation["connections"]
        connection_targets = [item["motif_id"] for item in connections]
        if connection_targets:
            placeholders = ", ".join("?" for _ in connection_targets)
            rows = connection.execute(
                f"""
                SELECT id FROM motifs
                WHERE project_id = ? AND id IN ({placeholders})
                """,
                [project_id, *connection_targets],
            ).fetchall()
            if {row["id"] for row in rows} != set(connection_targets):
                raise StorageError("Connected motifs must belong to this project.")
            if motif_id in connection_targets:
                raise StorageError("A motif cannot be connected to itself.")

        connection.execute(
            """
            INSERT INTO motif_events(
                id, batch_id, motif_id, project_id, observer_agent_id,
                actor_type, actor_id, event_type, relation, primary_flag,
                confidence, status, description, evidence_message_ids_json,
                user_message_id, turn_id, turn_beat, created_at
            ) VALUES (?, ?, ?, ?, ?, 'agent', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                batch_id,
                motif_id,
                project_id,
                observer_agent_id,
                observer_agent_id,
                event_type,
                observation["relation"],
                int(observation["primary"]),
                observation["confidence"],
                status,
                observation["description"],
                json.dumps([user_message_id] if user_message_id else []),
                user_message_id,
                turn_id,
                turn_beat,
                created_at,
            ),
        )
        for motif_connection in connections:
            connection.execute(
                """
                INSERT INTO motif_relation_events(
                    id, batch_id, project_id, observer_agent_id,
                    source_motif_id, target_motif_id, relation,
                    confidence, description, turn_id, turn_beat, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    batch_id,
                    project_id,
                    observer_agent_id,
                    motif_id,
                    motif_connection["motif_id"],
                    motif_connection["relation"],
                    motif_connection["confidence"],
                    motif_connection["description"],
                    turn_id,
                    turn_beat,
                    created_at,
                ),
            )
        return {
            "motif_id": motif_id,
            "label": existing["label"] if existing is not None else observation["label"],
            "status": status,
            "support_count": support_count,
            "distinct_turn_count": distinct_turn_count,
            "primary": observation["primary"],
        }

    @staticmethod
    def _validate_motif_observations(
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not isinstance(observations, list) or not (
            1 <= len(observations) <= MOTIF_OBSERVATIONS_PER_BEAT
        ):
            raise StorageError(
                f"Record between 1 and {MOTIF_OBSERVATIONS_PER_BEAT} motif observations."
            )
        if sum(bool(item.get("primary")) for item in observations if isinstance(item, dict)) != 1:
            raise StorageError("Exactly one motif observation must be primary.")
        prepared = []
        identities: set[str] = set()
        for item in observations:
            if not isinstance(item, dict):
                raise StorageError("Each motif observation must be an object.")
            label = str(item.get("label") or "").strip()
            description = str(item.get("description") or "").strip()
            relation = str(item.get("relation") or "").strip()
            motif_id = str(item.get("motif_id") or "").strip() or None
            connections = item.get("connections") or []
            if not label or len(label) > MOTIF_LABEL_MAX_CHARS:
                raise StorageError(f"Motif labels must be 1-{MOTIF_LABEL_MAX_CHARS} characters.")
            if not description or len(description) > MOTIF_DESCRIPTION_MAX_CHARS:
                raise StorageError(
                    f"Motif descriptions must be 1-{MOTIF_DESCRIPTION_MAX_CHARS} characters."
                )
            if relation not in AGENT_RELATIONS:
                raise StorageError("Unknown motif relation.")
            try:
                confidence = float(item.get("confidence"))
            except (TypeError, ValueError) as exc:
                raise StorageError("Motif confidence must be a number.") from exc
            if not 0 <= confidence <= 1:
                raise StorageError("Motif confidence must be between 0 and 1.")
            if not isinstance(connections, list) or len(connections) > MOTIF_CONNECTION_MAX_ITEMS:
                raise StorageError("Too many motif connections.")
            prepared_connections = []
            connection_targets: set[str] = set()
            for motif_connection in connections:
                if not isinstance(motif_connection, dict):
                    raise StorageError("Each motif connection must be an object.")
                target_id = str(motif_connection.get("motif_id") or "").strip()
                connection_relation = str(
                    motif_connection.get("relation") or ""
                ).strip()
                connection_description = str(
                    motif_connection.get("description") or ""
                ).strip()
                try:
                    connection_confidence = float(motif_connection.get("confidence"))
                except (TypeError, ValueError) as exc:
                    raise StorageError("Motif connection confidence must be a number.") from exc
                if not target_id or target_id in connection_targets:
                    raise StorageError("Each connected motif must be named once.")
                if connection_relation not in MOTIF_CONNECTION_RELATIONS:
                    raise StorageError("Unknown motif connection relation.")
                if not connection_description or len(connection_description) > 600:
                    raise StorageError(
                        "Motif connection descriptions must be 1-600 characters."
                    )
                if not 0 <= connection_confidence <= 1:
                    raise StorageError(
                        "Motif connection confidence must be between 0 and 1."
                    )
                connection_targets.add(target_id)
                prepared_connections.append(
                    {
                        "motif_id": target_id,
                        "relation": connection_relation,
                        "description": connection_description,
                        "confidence": connection_confidence,
                    }
                )
            normalized_label = normalize_motif_label(label)
            if not normalized_label:
                raise StorageError("Motif label must contain letters or numbers.")
            identity = f"id:{motif_id}" if motif_id else f"label:{normalized_label}"
            if identity in identities:
                raise StorageError("A motif may appear only once in one observation batch.")
            identities.add(identity)
            prepared.append(
                {
                    "motif_id": motif_id,
                    "label": label,
                    "normalized_label": normalized_label,
                    "description": description,
                    "relation": relation,
                    "confidence": confidence,
                    "primary": bool(item.get("primary")),
                    "connections": prepared_connections,
                }
            )
        return prepared

    @staticmethod
    def _add_motif_alias(
        connection: sqlite3.Connection,
        *,
        project_id: str,
        observer_agent_id: str,
        motif_id: str,
        label: str,
        normalized_alias: str,
        created_at: str,
    ) -> None:
        claimed = connection.execute(
            """
            SELECT motif_id FROM motif_aliases
            WHERE project_id = ? AND observer_agent_id = ? AND normalized_alias = ?
            """,
            (project_id, observer_agent_id, normalized_alias),
        ).fetchone()
        if claimed is not None:
            if claimed["motif_id"] != motif_id:
                raise StorageError(
                    "This label is already an alias of another motif owned by this observer."
                )
            return
        connection.execute(
            """
            INSERT INTO motif_aliases(
                id, project_id, observer_agent_id, motif_id,
                normalized_alias, alias, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                project_id,
                observer_agent_id,
                motif_id,
                normalized_alias,
                label,
                created_at,
            ),
        )

    @staticmethod
    def _attach_aliases(
        connection: sqlite3.Connection,
        motifs: list[dict],
    ) -> None:
        if not motifs:
            return
        motif_by_id = {motif["id"]: motif for motif in motifs}
        placeholders = ", ".join("?" for _ in motif_by_id)
        rows = connection.execute(
            f"""
            SELECT motif_id, alias
            FROM motif_aliases
            WHERE motif_id IN ({placeholders})
            ORDER BY created_at, id
            """,
            list(motif_by_id),
        ).fetchall()
        for motif in motifs:
            motif["aliases"] = []
        for row in rows:
            motif_by_id[row["motif_id"]]["aliases"].append(row["alias"])

    @staticmethod
    def _row_to_motif(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["confidence"] = round(float(result["confidence"]), 3)
        return result

    @staticmethod
    def _row_to_motif_event(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["primary"] = bool(result.pop("primary_flag"))
        result["evidence_message_ids"] = json.loads(
            result.pop("evidence_message_ids_json") or "[]"
        )
        return result
