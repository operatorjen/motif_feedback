from __future__ import annotations

import json
import sqlite3
import uuid

from .constants import (
    DEFAULT_STORAGE_MESSAGE_LIMIT,
    MAX_PROJECT_MESSAGE_LIMIT,
    STORED_MEMORY_QUERY_MAX_ROWS,
)
from .storage_core import (
    ChatTurnConflictError,
    StorageError,
    utc_now,
)


class MessageRepositoryMixin:
    def add_message(
        self,
        project_id: str,
        role: str,
        content: str,
        *,
        agent_id: str | None = None,
        annotations: list[dict] | None = None,
        metadata: dict | None = None,
        operation_id: str | None = None,
    ) -> dict:
        project = self.get_project(project_id)
        del project
        timestamp = utc_now()
        message = {
            "id": uuid.uuid4().hex,
            "project_id": project_id,
            "role": role,
            "agent_id": agent_id,
            "content": content,
            "annotations": annotations or [],
            "metadata": metadata or {},
            "turn_id": (metadata or {}).get("turn_id"),
            "operation_id": operation_id,
            "created_at": timestamp,
        }
        with self._write_lock, self.connection() as connection:
            if operation_id is not None:
                existing = connection.execute(
                    """
                    SELECT id, project_id, role, agent_id, content,
                           annotations_json, metadata_json, created_at
                    FROM messages WHERE operation_id = ?
                    """,
                    (operation_id,),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["project_id"] != project_id
                        or existing["role"] != role
                        or existing["agent_id"] != agent_id
                    ):
                        raise ChatTurnConflictError(
                            "That message operation belongs to different work."
                        )
                    return self._row_to_message(existing)
            connection.execute(
                """
                INSERT INTO messages(
                    id, project_id, turn_id, operation_id, role, agent_id, content,
                    annotations_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message["id"],
                    project_id,
                    message["turn_id"],
                    operation_id,
                    role,
                    agent_id,
                    content,
                    json.dumps(message["annotations"], ensure_ascii=False),
                    json.dumps(message["metadata"], ensure_ascii=False),
                    timestamp,
                ),
            )
            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?", (timestamp, project_id)
            )
        return message

    def messages_for_turn(self, project_id: str, turn_id: str) -> list[dict]:
        with self.connection() as connection:
            self._project_from_connection(connection, project_id)
            rows = connection.execute(
                """
                SELECT id, project_id, role, agent_id, content,
                       annotations_json, metadata_json, created_at
                FROM messages
                WHERE project_id = ? AND turn_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (project_id, turn_id),
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def list_messages(
        self,
        project_id: str,
        limit: int = DEFAULT_STORAGE_MESSAGE_LIMIT,
    ) -> list[dict]:
        safe_limit = min(max(limit, 1), MAX_PROJECT_MESSAGE_LIMIT)
        with self.connection() as connection:
            self._project_from_connection(connection, project_id)
            rows = connection.execute(
                """
                SELECT id, project_id, role, agent_id, content,
                       annotations_json, metadata_json, created_at
                FROM (
                    SELECT rowid AS message_rowid, id, project_id, role, agent_id,
                           content, annotations_json, metadata_json, created_at
                    FROM messages
                    WHERE project_id = ?
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT ?
                )
                ORDER BY created_at ASC, message_rowid ASC
                """,
                (project_id, safe_limit),
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def update_message_metadata(
        self,
        project_id: str,
        message_id: str,
        updates: dict,
    ) -> None:
        with self._write_lock, self.connection() as connection:
            self._project_from_connection(connection, project_id)
            row = connection.execute(
                """
                SELECT metadata_json FROM messages
                WHERE project_id = ? AND id = ?
                """,
                (project_id, message_id),
            ).fetchone()
            if row is None:
                raise StorageError(f"Message {message_id!r} was not found.")
            try:
                metadata = json.loads(row["metadata_json"])
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            metadata.update(updates)
            connection.execute(
                """
                UPDATE messages SET metadata_json = ?
                WHERE project_id = ? AND id = ?
                """,
                (
                    json.dumps(metadata, ensure_ascii=False),
                    project_id,
                    message_id,
                ),
            )

    def recent_messages(self, project_id: str, limit: int) -> list[dict]:
        safe_limit = min(max(limit, 1), STORED_MEMORY_QUERY_MAX_ROWS)
        with self.connection() as connection:
            self._project_from_connection(connection, project_id)
            rows = connection.execute(
                """
                SELECT id, project_id, role, agent_id, content,
                       annotations_json, metadata_json, created_at
                FROM (
                    SELECT id, project_id, role, agent_id, content,
                           annotations_json, metadata_json, created_at
                    FROM messages
                    WHERE project_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                ORDER BY created_at ASC
                """,
                (project_id, safe_limit),
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "role": row["role"],
            "agent_id": row["agent_id"],
            "content": row["content"],
            "annotations": json.loads(row["annotations_json"]),
            "metadata": json.loads(row["metadata_json"]),
            "created_at": row["created_at"],
        }

