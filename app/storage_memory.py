from __future__ import annotations

import json
import re
import sqlite3
import uuid

from .constants import (
    DEFAULT_MEMORY_EVENT_LIMIT,
    STORED_MEMORY_QUERY_MAX_ROWS,
    STORED_MEMORY_RETURN_MAX_CHARS,
    STORED_MEMORY_RETURN_SUMMARY_MAX_CHARS,
    STORED_MEMORY_TRIGGER_MAX_CHARS,
    STORED_MEMORY_TRIGGER_SUMMARY_MAX_CHARS,
    STORED_SOURCE_PROJECT_NAME_MAX_CHARS,
)
from .storage_core import (
    ChatTurnConflictError,
    StorageError,
    utc_now,
)


class MemoryRepositoryMixin:
    @staticmethod
    def _compact_memory_text(text: str, max_chars: int) -> str:
        """Collapse whitespace and preserve both ends of a bounded return card."""
        compact = " ".join(str(text or "").split())
        if len(compact) <= max_chars:
            return compact
        if max_chars < 5:
            return compact[:max_chars]
        head_length = max(1, int((max_chars - 3) * 0.7))
        tail_length = max_chars - head_length - 3
        return f"{compact[:head_length]}...{compact[-tail_length:]}"

    def add_memory_event(
        self,
        project_id: str,
        agent_id: str,
        user_message_id: str,
        *,
        outcome: str,
        trigger_text: str,
        return_text: str,
        actions: list[dict],
        provider: str,
        model: str,
        operation_id: str | None = None,
    ) -> dict:
        timestamp = utc_now()
        with self._write_lock, self.connection() as connection:
            self._project_from_connection(connection, project_id)
            if operation_id is not None:
                existing = connection.execute(
                    """
                    SELECT id, project_id, agent_id, user_message_id, sequence,
                           outcome, trigger_text, return_text, actions_json,
                           provider, model, created_at
                    FROM agent_memory_events WHERE operation_id = ?
                    """,
                    (operation_id,),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["project_id"] != project_id
                        or existing["agent_id"] != agent_id
                        or existing["user_message_id"] != user_message_id
                    ):
                        raise ChatTurnConflictError(
                            "That memory operation belongs to different work."
                        )
                    return self._row_with_actions(existing)
            row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence
                FROM agent_memory_events WHERE project_id = ? AND agent_id = ?
                """,
                (project_id, agent_id),
            ).fetchone()
            sequence = int(row["next_sequence"])
            event_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO agent_memory_events(
                    id, project_id, agent_id, operation_id, user_message_id, sequence, outcome,
                    trigger_text, return_text, actions_json, provider, model, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    project_id,
                    agent_id,
                    operation_id,
                    user_message_id,
                    sequence,
                    outcome,
                    trigger_text[:STORED_MEMORY_TRIGGER_MAX_CHARS],
                    return_text[:STORED_MEMORY_RETURN_MAX_CHARS],
                    json.dumps(actions, ensure_ascii=False),
                    provider,
                    model,
                    timestamp,
                ),
            )
            if self._fts_available:
                connection.execute(
                    """
                    INSERT INTO agent_memory_fts(
                        event_id, project_id, agent_id, trigger_text, return_text
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        project_id,
                        agent_id,
                        trigger_text[:STORED_MEMORY_TRIGGER_MAX_CHARS],
                        return_text[:STORED_MEMORY_RETURN_MAX_CHARS],
                    ),
                )
        return {
            "id": event_id,
            "project_id": project_id,
            "agent_id": agent_id,
            "sequence": sequence,
            "outcome": outcome,
            "trigger_text": trigger_text[:STORED_MEMORY_TRIGGER_MAX_CHARS],
            "return_text": return_text[:STORED_MEMORY_RETURN_MAX_CHARS],
            "actions": actions,
            "provider": provider,
            "model": model,
            "created_at": timestamp,
        }

    def list_memory_events(
        self,
        project_id: str,
        agent_id: str,
        limit: int = DEFAULT_MEMORY_EVENT_LIMIT,
    ) -> list[dict]:
        if limit <= 0:
            self.get_project(project_id)
            return []
        safe_limit = min(max(limit, 1), STORED_MEMORY_QUERY_MAX_ROWS)
        with self.connection() as connection:
            self._project_from_connection(connection, project_id)
            rows = connection.execute(
                """
                SELECT id, project_id, agent_id, user_message_id, sequence, outcome,
                       trigger_text, return_text, actions_json, provider, model, created_at
                FROM agent_memory_events
                WHERE project_id = ? AND agent_id = ?
                ORDER BY sequence DESC LIMIT ?
                """,
                (project_id, agent_id, safe_limit),
            ).fetchall()
        return [self._row_with_actions(row) for row in rows]

    @staticmethod
    def _fts_query(text: str) -> str:
        terms = list(
            dict.fromkeys(token.lower() for token in re.findall(r"[A-Za-z0-9_]{2,}", text))
        )[:12]
        return " OR ".join(f'"{term}"' for term in terms)

    def search_memory_events(
        self,
        project_id: str,
        agent_id: str,
        query: str,
        *,
        limit: int,
    ) -> list[dict]:
        expression = self._fts_query(query)
        if not self._fts_available or not expression:
            return self.list_memory_events(project_id, agent_id, limit=limit)
        safe_limit = min(max(limit, 1), STORED_MEMORY_QUERY_MAX_ROWS)
        try:
            with self.connection() as connection:
                self._project_from_connection(connection, project_id)
                rows = connection.execute(
                    """
                    SELECT memory.id, memory.project_id, memory.agent_id,
                           memory.user_message_id, memory.sequence, memory.outcome,
                           memory.trigger_text, memory.return_text,
                           memory.actions_json, memory.provider, memory.model,
                           memory.created_at
                    FROM agent_memory_fts AS search
                    JOIN agent_memory_events AS memory
                      ON memory.id = search.event_id
                    WHERE agent_memory_fts MATCH ?
                      AND search.project_id = ? AND search.agent_id = ?
                    ORDER BY bm25(agent_memory_fts), memory.sequence DESC
                    LIMIT ?
                    """,
                    (expression, project_id, agent_id, safe_limit),
                ).fetchall()
                if not rows:
                    rows = connection.execute(
                        """
                        SELECT id, project_id, agent_id, user_message_id, sequence, outcome,
                               trigger_text, return_text, actions_json, provider, model, created_at
                        FROM agent_memory_events
                        WHERE project_id = ? AND agent_id = ?
                        ORDER BY sequence DESC LIMIT ?
                        """,
                        (project_id, agent_id, safe_limit),
                    ).fetchall()
        except sqlite3.OperationalError:
            return self.list_memory_events(project_id, agent_id, limit=limit)
        return [self._row_with_actions(row) for row in rows]

    def validate_agent_memory_evidence(
        self,
        project_id: str,
        agent_id: str,
        event_ids: list[str],
    ) -> list[dict]:
        """Resolve evidence visible to an agent in this project.

        Current-project events are visible directly. Successful events from another
        project are visible only when a provenance-labeled global continuity record
        exists for the same source event.
        """
        self.get_project(project_id)
        unique_ids = list(dict.fromkeys(str(event_id).strip() for event_id in event_ids))
        if not unique_ids or any(not event_id for event_id in unique_ids):
            raise StorageError("Persona-update evidence must reference stored memory events.")
        placeholders = ", ".join("?" for _ in unique_ids)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT memory.id, memory.project_id, memory.agent_id,
                       memory.user_message_id, memory.sequence, memory.outcome,
                       memory.created_at
                FROM agent_memory_events AS memory
                WHERE memory.agent_id = ?
                  AND memory.id IN ({placeholders})
                  AND (
                    memory.project_id = ?
                    OR EXISTS (
                        SELECT 1
                        FROM agent_global_memory_events AS global_memory
                        WHERE global_memory.source_memory_event_id = memory.id
                          AND global_memory.agent_id = memory.agent_id
                          AND global_memory.source_project_id != ?
                    )
                  )
                """,
                [agent_id, *unique_ids, project_id, project_id],
            ).fetchall()
        by_id = {row["id"]: dict(row) for row in rows}
        missing = [event_id for event_id in unique_ids if event_id not in by_id]
        if missing:
            raise StorageError(
                "Persona-update evidence was not found in this agent's visible memory."
            )
        return [by_id[event_id] for event_id in unique_ids]

    def add_global_memory_event(
        self,
        *,
        agent_id: str,
        source_project_id: str,
        source_project_name: str | None = None,
        source_memory_event_id: str,
        trigger_text: str,
        return_text: str,
        actions: list[dict],
        created_at: str | None = None,
    ) -> dict:
        """Store one compact, provenance-labeled return for cross-project continuity."""
        timestamp = created_at or utc_now()
        with self._write_lock, self.connection() as connection:
            source = connection.execute(
                """
                SELECT memory.agent_id, memory.project_id, projects.name AS project_name
                FROM agent_memory_events AS memory
                JOIN projects ON projects.id = memory.project_id
                WHERE memory.id = ?
                """,
                (source_memory_event_id,),
            ).fetchone()
            if (
                source is None
                or source["agent_id"] != agent_id
                or source["project_id"] != source_project_id
            ):
                raise StorageError(
                    "Global continuity provenance does not match its source memory event."
                )
            source_project_name = source["project_name"]
            existing = connection.execute(
                """
                SELECT id, agent_id, source_project_id, source_project_name,
                       source_memory_event_id, sequence, trigger_summary,
                       return_summary, actions_json, created_at
                FROM agent_global_memory_events WHERE source_memory_event_id = ?
                """,
                (source_memory_event_id,),
            ).fetchone()
            if existing is not None:
                return self._row_with_actions(existing)
            row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence
                FROM agent_global_memory_events WHERE agent_id = ?
                """,
                (agent_id,),
            ).fetchone()
            sequence = int(row["next_sequence"])
            event_id = uuid.uuid4().hex
            trigger_summary = self._compact_memory_text(
                trigger_text,
                STORED_MEMORY_TRIGGER_SUMMARY_MAX_CHARS,
            )
            return_summary = self._compact_memory_text(
                return_text,
                STORED_MEMORY_RETURN_SUMMARY_MAX_CHARS,
            )
            connection.execute(
                """
                INSERT INTO agent_global_memory_events(
                    id, agent_id, source_project_id, source_project_name,
                    source_memory_event_id, sequence, trigger_summary,
                    return_summary, actions_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    agent_id,
                    source_project_id,
                    source_project_name[:STORED_SOURCE_PROJECT_NAME_MAX_CHARS],
                    source_memory_event_id,
                    sequence,
                    trigger_summary,
                    return_summary,
                    json.dumps(actions, ensure_ascii=False),
                    timestamp,
                ),
            )
            if self._fts_available:
                connection.execute(
                    """
                    INSERT INTO agent_global_memory_fts(
                        event_id, agent_id, source_project_id, source_project_name,
                        trigger_summary, return_summary
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        agent_id,
                        source_project_id,
                        source_project_name[:STORED_SOURCE_PROJECT_NAME_MAX_CHARS],
                        trigger_summary,
                        return_summary,
                    ),
                )
        return {
            "id": event_id,
            "agent_id": agent_id,
            "source_project_id": source_project_id,
            "source_project_name": source_project_name[:STORED_SOURCE_PROJECT_NAME_MAX_CHARS],
            "source_memory_event_id": source_memory_event_id,
            "sequence": sequence,
            "trigger_summary": trigger_summary,
            "return_summary": return_summary,
            "actions": actions,
            "created_at": timestamp,
        }

    def list_global_memory_events(
        self,
        agent_id: str,
        *,
        exclude_project_id: str | None = None,
        limit: int = DEFAULT_MEMORY_EVENT_LIMIT,
    ) -> list[dict]:
        if limit <= 0:
            return []
        safe_limit = min(max(limit, 1), STORED_MEMORY_QUERY_MAX_ROWS)
        where = "agent_id = ?"
        values: list[object] = [agent_id]
        if exclude_project_id is not None:
            where += " AND source_project_id != ?"
            values.append(exclude_project_id)
        values.append(safe_limit)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT id, agent_id, source_project_id, source_project_name,
                       source_memory_event_id, sequence, trigger_summary,
                       return_summary, actions_json, created_at
                FROM agent_global_memory_events
                WHERE {where}
                ORDER BY sequence DESC LIMIT ?
                """,
                values,
            ).fetchall()
        return [self._row_with_actions(row) for row in rows]

    def list_global_memory_context_events(
        self,
        agent_id: str,
        *,
        exclude_project_id: str | None = None,
        limit: int = DEFAULT_MEMORY_EVENT_LIMIT,
    ) -> list[dict]:
        """Return internal prompt records with source-turn grouping metadata."""
        if limit <= 0:
            return []
        safe_limit = min(max(limit, 1), STORED_MEMORY_QUERY_MAX_ROWS)
        where = "global_memory.agent_id = ?"
        values: list[object] = [agent_id]
        if exclude_project_id is not None:
            where += " AND global_memory.source_project_id != ?"
            values.append(exclude_project_id)
        values.append(safe_limit)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT global_memory.id, global_memory.agent_id,
                       global_memory.source_project_id,
                       global_memory.source_project_name,
                       global_memory.source_memory_event_id,
                       global_memory.sequence, global_memory.trigger_summary,
                       global_memory.return_summary, global_memory.actions_json,
                       global_memory.created_at, source.user_message_id
                FROM agent_global_memory_events AS global_memory
                JOIN agent_memory_events AS source
                  ON source.id = global_memory.source_memory_event_id
                WHERE {where}
                ORDER BY global_memory.sequence DESC LIMIT ?
                """,
                values,
            ).fetchall()
        return [self._row_with_actions(row) for row in rows]

    def search_global_memory_context_events(
        self,
        agent_id: str,
        query: str,
        *,
        exclude_project_id: str | None = None,
        limit: int = DEFAULT_MEMORY_EVENT_LIMIT,
    ) -> list[dict]:
        expression = self._fts_query(query)
        if not self._fts_available or not expression:
            return self.list_global_memory_context_events(
                agent_id,
                exclude_project_id=exclude_project_id,
                limit=limit,
            )
        safe_limit = min(max(limit, 1), STORED_MEMORY_QUERY_MAX_ROWS)
        where = "search.agent_id = ?"
        values: list[object] = [expression, agent_id]
        if exclude_project_id is not None:
            where += " AND search.source_project_id != ?"
            values.append(exclude_project_id)
        values.append(safe_limit)
        try:
            with self.connection() as connection:
                rows = connection.execute(
                    f"""
                    SELECT global_memory.id, global_memory.agent_id,
                           global_memory.source_project_id,
                           global_memory.source_project_name,
                           global_memory.source_memory_event_id,
                           global_memory.sequence, global_memory.trigger_summary,
                           global_memory.return_summary,
                           global_memory.actions_json, global_memory.created_at,
                           source.user_message_id
                    FROM agent_global_memory_fts AS search
                    JOIN agent_global_memory_events AS global_memory
                      ON global_memory.id = search.event_id
                    JOIN agent_memory_events AS source
                      ON source.id = global_memory.source_memory_event_id
                    WHERE agent_global_memory_fts MATCH ? AND {where}
                    ORDER BY bm25(agent_global_memory_fts),
                             global_memory.sequence DESC
                    LIMIT ?
                    """,
                    values,
                ).fetchall()
                if not rows:
                    fallback_where = "global_memory.agent_id = ?"
                    fallback_values: list[object] = [agent_id]
                    if exclude_project_id is not None:
                        fallback_where += " AND global_memory.source_project_id != ?"
                        fallback_values.append(exclude_project_id)
                    fallback_values.append(safe_limit)
                    rows = connection.execute(
                        f"""
                        SELECT global_memory.id, global_memory.agent_id,
                               global_memory.source_project_id,
                               global_memory.source_project_name,
                               global_memory.source_memory_event_id,
                               global_memory.sequence, global_memory.trigger_summary,
                               global_memory.return_summary,
                               global_memory.actions_json, global_memory.created_at,
                               source.user_message_id
                        FROM agent_global_memory_events AS global_memory
                        JOIN agent_memory_events AS source
                          ON source.id = global_memory.source_memory_event_id
                        WHERE {fallback_where}
                        ORDER BY global_memory.sequence DESC LIMIT ?
                        """,
                        fallback_values,
                    ).fetchall()
        except sqlite3.OperationalError:
            return self.list_global_memory_context_events(
                agent_id,
                exclude_project_id=exclude_project_id,
                limit=limit,
            )
        return [self._row_with_actions(row) for row in rows]

    @staticmethod
    def _row_with_actions(row: sqlite3.Row) -> dict:
        data = dict(row)
        data["actions"] = json.loads(data.pop("actions_json"))
        return data

    def global_memory_stats(self, agent_id: str, *, exclude_project_id: str | None = None) -> dict:
        where = "agent_id = ?"
        values: list[object] = [agent_id]
        if exclude_project_id is not None:
            where += " AND source_project_id != ?"
            values.append(exclude_project_id)
        with self.connection() as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(*) AS event_count,
                       COUNT(DISTINCT source_project_id) AS project_count,
                       COALESCE(MAX(sequence), 0) AS latest_sequence
                FROM agent_global_memory_events WHERE {where}
                """,
                values,
            ).fetchone()
        return {"agent_id": agent_id, **dict(row)}

    def memory_stats(self, project_id: str) -> dict[str, dict]:
        with self.connection() as connection:
            self._project_from_connection(connection, project_id)
            rows = connection.execute(
                """
                SELECT agent_id, COUNT(*) AS event_count,
                       SUM(CASE WHEN outcome LIKE 'action%' THEN 1 ELSE 0 END) AS action_count,
                       SUM(CASE WHEN outcome IN ('timeout', 'no_response', 'provider_error') THEN 1 ELSE 0 END) AS failure_count,
                       MAX(sequence) AS latest_sequence
                FROM agent_memory_events WHERE project_id = ? GROUP BY agent_id
                """,
                (project_id,),
            ).fetchall()
        return {row["agent_id"]: dict(row) for row in rows}
