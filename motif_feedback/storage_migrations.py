from __future__ import annotations

import json

from .constants import (
    STORED_MEMORY_RETURN_SUMMARY_MAX_CHARS,
    STORED_MEMORY_TRIGGER_SUMMARY_MAX_CHARS,
    STORED_MESSAGE_MAX_CHARS,
)
from .storage_core import utc_now
from .tool_metadata import public_tool_arguments


class StorageMigrationMixin:
    def _run_once_migration(self, name: str, operation) -> None:
        with self.connection() as connection:
            applied = connection.execute(
                "SELECT 1 FROM schema_migrations WHERE name = ?",
                (name,),
            ).fetchone()
        if applied:
            return
        operation()
        with self._write_lock, self.connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO schema_migrations(name, applied_at)
                VALUES (?, ?)
                """,
                (name, utc_now()),
            )

    def _backfill_file_ownership(self) -> None:
        """Recover creators from historical tool events, then mark other files user-owned."""
        timestamp = utc_now()
        with self._write_lock, self.connection() as connection:
            rows = connection.execute(
                """
                SELECT project_id, agent_id, metadata_json
                FROM messages
                WHERE agent_id IS NOT NULL
                ORDER BY created_at ASC
                """
            ).fetchall()
            for row in rows:
                try:
                    metadata = json.loads(row["metadata_json"])
                except (TypeError, json.JSONDecodeError):
                    continue
                for event in metadata.get("tool_events", []):
                    if not isinstance(event, dict) or event.get("tool") != "write_project_file":
                        continue
                    result = event.get("result") if isinstance(event.get("result"), dict) else {}
                    arguments = (
                        event.get("arguments") if isinstance(event.get("arguments"), dict) else {}
                    )
                    if result.get("ok") is False:
                        continue
                    path = str(result.get("path") or arguments.get("path") or "").strip()
                    if not path:
                        continue
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO file_ownership(
                            project_id, path, owner_type, owner_id, created_at, updated_at
                        ) VALUES (?, ?, 'agent', ?, ?, ?)
                        """,
                        (row["project_id"], path, row["agent_id"], timestamp, timestamp),
                    )

            projects = connection.execute("SELECT id FROM projects").fetchall()
            for project in projects:
                root = self.projects_root / project["id"]
                if not root.exists():
                    continue
                for path in root.rglob("*"):
                    if path.is_symlink() or not path.is_file():
                        continue
                    relative = path.relative_to(root).as_posix()
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO file_ownership(
                            project_id, path, owner_type, owner_id, created_at, updated_at
                        ) VALUES (?, ?, 'user', NULL, ?, ?)
                        """,
                        (project["id"], relative, timestamp, timestamp),
                    )

    def _sanitize_tool_event_metadata(self) -> None:
        """Remove generated bodies from tool-event audit metadata written by older releases."""
        with self._write_lock, self.connection() as connection:
            rows = connection.execute(
                """
                SELECT id, metadata_json
                FROM messages
                WHERE metadata_json LIKE '%"tool_events"%'
                """
            ).fetchall()
            for row in rows:
                try:
                    metadata = json.loads(row["metadata_json"])
                except (TypeError, json.JSONDecodeError):
                    continue
                events = metadata.get("tool_events")
                if not isinstance(events, list):
                    continue
                changed = False
                for event in events:
                    if not isinstance(event, dict):
                        continue
                    arguments = event.get("arguments")
                    if not isinstance(arguments, dict):
                        continue
                    sanitized = public_tool_arguments(str(event.get("tool") or ""), arguments)
                    if sanitized != arguments:
                        event["arguments"] = sanitized
                        changed = True
                if changed:
                    connection.execute(
                        "UPDATE messages SET metadata_json = ? WHERE id = ?",
                        (json.dumps(metadata, ensure_ascii=False), row["id"]),
                    )

    def _backfill_memory_events(self) -> None:
        """Seed observable loop history from earlier stored agent responses once."""
        with self._write_lock, self.connection() as connection:
            user_rows = connection.execute(
                "SELECT id, content FROM messages WHERE role = 'user'"
            ).fetchall()
            user_messages = {row["id"]: row["content"] for row in user_rows}
            rows = connection.execute(
                """
                SELECT id, project_id, agent_id, content, metadata_json, created_at
                FROM messages WHERE role = 'agent' AND agent_id IS NOT NULL
                ORDER BY created_at ASC
                """
            ).fetchall()
            sequences: dict[tuple[str, str], int] = {}
            for row in rows:
                event_id = f"historical-{row['id']}"
                exists = connection.execute(
                    "SELECT 1 FROM agent_memory_events WHERE id = ?", (event_id,)
                ).fetchone()
                if exists:
                    continue
                key = (row["project_id"], row["agent_id"])
                if key not in sequences:
                    latest = connection.execute(
                        """
                        SELECT COALESCE(MAX(sequence), 0) AS latest
                        FROM agent_memory_events WHERE project_id = ? AND agent_id = ?
                        """,
                        key,
                    ).fetchone()
                    sequences[key] = int(latest["latest"])
                sequences[key] += 1
                try:
                    metadata = json.loads(row["metadata_json"])
                except (TypeError, json.JSONDecodeError):
                    metadata = {}
                user_message_id = str(metadata.get("user_message_id") or "")
                if user_message_id:
                    already_recorded = connection.execute(
                        """
                        SELECT 1 FROM agent_memory_events
                        WHERE project_id = ? AND agent_id = ? AND user_message_id = ?
                          AND return_text = ?
                        """,
                        (
                            row["project_id"],
                            row["agent_id"],
                            user_message_id,
                            row["content"][:STORED_MESSAGE_MAX_CHARS],
                        ),
                    ).fetchone()
                    if already_recorded:
                        continue
                actions = []
                for event in metadata.get("tool_events", []):
                    if not isinstance(event, dict):
                        continue
                    result = event.get("result") if isinstance(event.get("result"), dict) else {}
                    arguments = (
                        event.get("arguments") if isinstance(event.get("arguments"), dict) else {}
                    )
                    actions.append(
                        {
                            "tool": event.get("tool", ""),
                            "path": result.get("path") or arguments.get("path"),
                            "ok": result.get("ok") is not False,
                            "overwritten": bool(result.get("overwritten", False)),
                        }
                    )
                successful_action = any(action["ok"] for action in actions)
                connection.execute(
                    """
                    INSERT INTO agent_memory_events(
                        id, project_id, agent_id, user_message_id, sequence, outcome,
                        trigger_text, return_text, actions_json, provider, model, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'historical', '', ?)
                    """,
                    (
                        event_id,
                        row["project_id"],
                        row["agent_id"],
                        user_message_id,
                        sequences[key],
                        "action_response" if successful_action else "response",
                        user_messages.get(user_message_id, ""),
                        row["content"][:STORED_MESSAGE_MAX_CHARS],
                        json.dumps(actions, ensure_ascii=False),
                        row["created_at"],
                    ),
                )

    def _backfill_global_memory_events(self) -> None:
        """Create bounded cross-project return cards from successful historical turns."""
        with self._write_lock, self.connection() as connection:
            rows = connection.execute(
                """
                SELECT memory.id, memory.agent_id, memory.project_id, memory.trigger_text,
                       memory.return_text, memory.actions_json, memory.created_at,
                       projects.name AS project_name
                FROM agent_memory_events AS memory
                JOIN projects ON projects.id = memory.project_id
                WHERE memory.outcome IN ('response', 'action_response')
                ORDER BY memory.created_at ASC, memory.sequence ASC
                """
            ).fetchall()
            sequences: dict[str, int] = {}
            for row in rows:
                exists = connection.execute(
                    "SELECT 1 FROM agent_global_memory_events WHERE source_memory_event_id = ?",
                    (row["id"],),
                ).fetchone()
                if exists:
                    continue
                agent_id = row["agent_id"]
                if agent_id not in sequences:
                    latest = connection.execute(
                        """
                        SELECT COALESCE(MAX(sequence), 0) AS latest
                        FROM agent_global_memory_events WHERE agent_id = ?
                        """,
                        (agent_id,),
                    ).fetchone()
                    sequences[agent_id] = int(latest["latest"])
                sequences[agent_id] += 1
                connection.execute(
                    """
                    INSERT INTO agent_global_memory_events(
                        id, agent_id, source_project_id, source_project_name,
                        source_memory_event_id, sequence, trigger_summary,
                        return_summary, actions_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"global-{row['id']}",
                        agent_id,
                        row["project_id"],
                        row["project_name"],
                        row["id"],
                        sequences[agent_id],
                        self._compact_memory_text(
                            row["trigger_text"],
                            STORED_MEMORY_TRIGGER_SUMMARY_MAX_CHARS,
                        ),
                        self._compact_memory_text(
                            row["return_text"],
                            STORED_MEMORY_RETURN_SUMMARY_MAX_CHARS,
                        ),
                        row["actions_json"],
                        row["created_at"],
                    ),
                )

