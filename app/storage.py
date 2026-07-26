from __future__ import annotations

import json
import re
import shutil
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .constants import (
    DEFAULT_MEMORY_EVENT_LIMIT,
    DEFAULT_STORAGE_MESSAGE_LIMIT,
    DEFAULT_WEB_SOURCE_LIMIT,
    MAX_PROJECT_MESSAGE_LIMIT,
    MAX_WEB_SOURCE_LIMIT,
    PROJECT_ID_RANDOM_CHARS,
    PROJECT_SLUG_MAX_CHARS,
    SQLITE_TIMEOUT_SECONDS,
    STORED_MEMORY_QUERY_MAX_ROWS,
    STORED_MEMORY_RETURN_MAX_CHARS,
    STORED_MEMORY_RETURN_SUMMARY_MAX_CHARS,
    STORED_MEMORY_TRIGGER_MAX_CHARS,
    STORED_MEMORY_TRIGGER_SUMMARY_MAX_CHARS,
    STORED_MESSAGE_MAX_CHARS,
    STORED_PROJECT_NAME_MAX_CHARS,
    STORED_SOURCE_PROJECT_NAME_MAX_CHARS,
)
from .tool_metadata import public_tool_arguments

PROJECT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class StorageError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def slugify_project_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    slug = slug[:PROJECT_SLUG_MAX_CHARS] or "project"
    return f"{slug}-{uuid.uuid4().hex[:PROJECT_ID_RANDOM_CHARS]}"


class Storage:
    def __init__(self, database_path: Path, projects_root: Path) -> None:
        self.database_path = database_path
        self.projects_root = projects_root
        self._write_lock = threading.RLock()

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(
            self.database_path,
            timeout=SQLITE_TIMEOUT_SECONDS,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.projects_root.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    agent_id TEXT,
                    content TEXT NOT NULL,
                    annotations_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_project_created
                ON messages(project_id, created_at);

                CREATE TABLE IF NOT EXISTS file_ownership (
                    project_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    owner_type TEXT NOT NULL CHECK(owner_type IN ('user', 'agent')),
                    owner_id TEXT,
                    shared_agent_edit INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(project_id, path),
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS agent_memory_events (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    user_message_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    outcome TEXT NOT NULL,
                    trigger_text TEXT NOT NULL,
                    return_text TEXT NOT NULL,
                    actions_json TEXT NOT NULL DEFAULT '[]',
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(project_id, agent_id, sequence),
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_memory_project_agent_sequence
                ON agent_memory_events(project_id, agent_id, sequence DESC);

                CREATE TABLE IF NOT EXISTS agent_global_memory_events (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    source_project_id TEXT NOT NULL,
                    source_project_name TEXT NOT NULL,
                    source_memory_event_id TEXT NOT NULL UNIQUE,
                    sequence INTEGER NOT NULL,
                    trigger_summary TEXT NOT NULL,
                    return_summary TEXT NOT NULL,
                    actions_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    UNIQUE(agent_id, sequence)
                );

                CREATE INDEX IF NOT EXISTS idx_global_memory_agent_sequence
                ON agent_global_memory_events(agent_id, sequence DESC);

                CREATE TABLE IF NOT EXISTS web_sources (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    requested_url TEXT NOT NULL,
                    final_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content_text TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    byte_count INTEGER NOT NULL,
                    char_count INTEGER NOT NULL,
                    truncated INTEGER NOT NULL DEFAULT 0,
                    content_sha256 TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_web_sources_project_fetched
                ON web_sources(project_id, fetched_at DESC);

                CREATE INDEX IF NOT EXISTS idx_web_sources_project_requested
                ON web_sources(project_id, requested_url, fetched_at DESC);

                CREATE TABLE IF NOT EXISTS schema_migrations (
                    name TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                """
            )
            ownership_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(file_ownership)").fetchall()
            }
            if "shared_agent_edit" not in ownership_columns:
                connection.execute(
                    "ALTER TABLE file_ownership ADD COLUMN shared_agent_edit INTEGER NOT NULL DEFAULT 0"
                )
        if not self.list_projects():
            self.create_project("General", project_id="general")
        self._run_once_migration("backfill_file_ownership_v1", self._backfill_file_ownership)
        self._run_once_migration("backfill_memory_events_v1", self._backfill_memory_events)
        self._run_once_migration(
            "backfill_global_memory_events_v1",
            self._backfill_global_memory_events,
        )
        self._run_once_migration(
            "sanitize_tool_event_metadata_v1",
            self._sanitize_tool_event_metadata,
        )

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
                    arguments = event.get("arguments") if isinstance(event.get("arguments"), dict) else {}
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
                    arguments = event.get("arguments") if isinstance(event.get("arguments"), dict) else {}
                    actions.append({
                        "tool": event.get("tool", ""),
                        "path": result.get("path") or arguments.get("path"),
                        "ok": result.get("ok") is not False,
                        "overwritten": bool(result.get("overwritten", False)),
                    })
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

    def get_file_owner(self, project_id: str, path: str) -> dict | None:
        self.get_project(project_id)
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT owner_type, owner_id, shared_agent_edit, created_at, updated_at
                FROM file_ownership WHERE project_id = ? AND path = ?
                """,
                (project_id, path),
            ).fetchone()
        return dict(row) if row is not None else None

    def file_owners(self, project_id: str) -> dict[str, dict]:
        self.get_project(project_id)
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT project_id, path, owner_type, owner_id, shared_agent_edit,
                       created_at, updated_at
                FROM file_ownership WHERE project_id = ?
                """,
                (project_id,),
            ).fetchall()
        return {row["path"]: dict(row) for row in rows}

    def record_file_owner(
        self, project_id: str, path: str, owner_type: str, owner_id: str | None
    ) -> None:
        if owner_type not in {"user", "agent"}:
            raise StorageError("Invalid file owner type.")
        timestamp = utc_now()
        with self._write_lock, self.connection() as connection:
            connection.execute(
                """
                INSERT INTO file_ownership(
                    project_id, path, owner_type, owner_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, path) DO UPDATE SET
                    owner_type = excluded.owner_type,
                    owner_id = excluded.owner_id,
                    updated_at = excluded.updated_at
                """,
                (project_id, path, owner_type, owner_id, timestamp, timestamp),
            )

    def touch_file_owner(self, project_id: str, path: str) -> None:
        with self._write_lock, self.connection() as connection:
            connection.execute(
                "UPDATE file_ownership SET updated_at = ? WHERE project_id = ? AND path = ?",
                (utc_now(), project_id, path),
            )

    def set_file_sharing(self, project_id: str, path: str, allowed: bool) -> dict:
        owner = self.get_file_owner(project_id, path)
        if owner is None:
            raise StorageError("File ownership record not found.")
        if owner.get("owner_type") != "agent":
            raise StorageError("Only agent-created files can be shared between agents.")
        timestamp = utc_now()
        with self._write_lock, self.connection() as connection:
            connection.execute(
                """
                UPDATE file_ownership
                SET shared_agent_edit = ?, updated_at = ?
                WHERE project_id = ? AND path = ?
                """,
                (int(bool(allowed)), timestamp, project_id, path),
            )
        return {
            "project_id": project_id,
            "path": path,
            "owner_type": owner["owner_type"],
            "owner_id": owner.get("owner_id"),
            "shared_agent_edit": bool(allowed),
            "updated_at": timestamp,
        }

    def remove_file_owner(self, project_id: str, path: str) -> None:
        with self._write_lock, self.connection() as connection:
            connection.execute(
                "DELETE FROM file_ownership WHERE project_id = ? AND path = ?",
                (project_id, path),
            )

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
    ) -> dict:
        self.get_project(project_id)
        timestamp = utc_now()
        with self._write_lock, self.connection() as connection:
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
                    id, project_id, agent_id, user_message_id, sequence, outcome,
                    trigger_text, return_text, actions_json, provider, model, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    project_id,
                    agent_id,
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

    def add_global_memory_event(
        self,
        *,
        agent_id: str,
        source_project_id: str,
        source_project_name: str,
        source_memory_event_id: str,
        trigger_text: str,
        return_text: str,
        actions: list[dict],
        created_at: str | None = None,
    ) -> dict:
        """Store one compact, provenance-labeled return for cross-project continuity."""
        timestamp = created_at or utc_now()
        with self._write_lock, self.connection() as connection:
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
        return {
            "id": event_id,
            "agent_id": agent_id,
            "source_project_id": source_project_id,
            "source_project_name": source_project_name[
                :STORED_SOURCE_PROJECT_NAME_MAX_CHARS
            ],
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

    @staticmethod
    def _row_with_actions(row: sqlite3.Row) -> dict:
        data = dict(row)
        data["actions"] = json.loads(data.pop("actions_json"))
        return data

    def global_memory_stats(
        self, agent_id: str, *, exclude_project_id: str | None = None
    ) -> dict:
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

    def add_web_source(
        self,
        project_id: str,
        *,
        requested_url: str,
        final_url: str,
        title: str,
        content_text: str,
        content_type: str,
        byte_count: int,
        truncated: bool,
        content_sha256: str,
    ) -> dict:
        self.get_project(project_id)
        source_id = uuid.uuid4().hex
        fetched_at = utc_now()
        with self._write_lock, self.connection() as connection:
            connection.execute(
                """
                INSERT INTO web_sources(
                    id, project_id, requested_url, final_url, title, content_text,
                    content_type, byte_count, char_count, truncated,
                    content_sha256, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    project_id,
                    requested_url,
                    final_url,
                    title[:500],
                    content_text,
                    content_type[:200],
                    max(0, int(byte_count)),
                    len(content_text),
                    int(bool(truncated)),
                    content_sha256,
                    fetched_at,
                ),
            )
            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (fetched_at, project_id),
            )
        return self.get_web_source(project_id, source_id)

    def get_web_source(self, project_id: str, source_id: str) -> dict:
        self.get_project(project_id)
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT id, project_id, requested_url, final_url, title, content_text,
                       content_type, byte_count, char_count, truncated,
                       content_sha256, fetched_at
                FROM web_sources WHERE project_id = ? AND id = ?
                """,
                (project_id, source_id),
            ).fetchone()
        if row is None:
            raise StorageError("Web source not found.")
        return self._row_to_web_source(row)

    def latest_web_source(self, project_id: str, requested_url: str) -> dict | None:
        self.get_project(project_id)
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT id, project_id, requested_url, final_url, title, content_text,
                       content_type, byte_count, char_count, truncated,
                       content_sha256, fetched_at
                FROM web_sources
                WHERE project_id = ? AND requested_url = ?
                ORDER BY fetched_at DESC LIMIT 1
                """,
                (project_id, requested_url),
            ).fetchone()
        return self._row_to_web_source(row) if row is not None else None

    def list_web_sources(
        self,
        project_id: str,
        limit: int = DEFAULT_WEB_SOURCE_LIMIT,
    ) -> list[dict]:
        self.get_project(project_id)
        safe_limit = min(max(limit, 1), MAX_WEB_SOURCE_LIMIT)
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT id, project_id, requested_url, final_url, title,
                       content_type, byte_count, char_count, truncated,
                       content_sha256, fetched_at
                FROM web_sources WHERE project_id = ?
                ORDER BY fetched_at DESC LIMIT ?
                """,
                (project_id, safe_limit),
            ).fetchall()
        return [self._row_to_web_source(row) for row in rows]

    def delete_web_source(self, project_id: str, source_id: str) -> dict:
        source = self.get_web_source(project_id, source_id)
        with self._write_lock, self.connection() as connection:
            connection.execute(
                "DELETE FROM web_sources WHERE project_id = ? AND id = ?",
                (project_id, source_id),
            )
        return {"deleted": True, "id": source_id, "title": source["title"]}

    @staticmethod
    def _row_to_web_source(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["truncated"] = bool(result.get("truncated"))
        return result

    def validate_project_id(self, project_id: str) -> str:
        if not PROJECT_ID_PATTERN.fullmatch(project_id):
            raise StorageError("Invalid project identifier.")
        return project_id

    def create_project(self, name: str, project_id: str | None = None) -> dict:
        clean_name = " ".join(name.split()).strip()
        if not clean_name:
            raise StorageError("Project name is required.")
        identifier = self.validate_project_id(project_id or slugify_project_name(clean_name))
        timestamp = utc_now()
        with self._write_lock, self.connection() as connection:
            connection.execute(
                "INSERT INTO projects(id, name, created_at, updated_at) VALUES(?, ?, ?, ?)",
                (
                    identifier,
                    clean_name[:STORED_PROJECT_NAME_MAX_CHARS],
                    timestamp,
                    timestamp,
                ),
            )
        (self.projects_root / identifier).mkdir(parents=True, exist_ok=True)
        return {
            "id": identifier,
            "name": clean_name[:STORED_PROJECT_NAME_MAX_CHARS],
            "created_at": timestamp,
            "updated_at": timestamp,
        }

    def list_projects(self) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT id, name, created_at, updated_at FROM projects ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_project(self, project_id: str) -> dict:
        with self.connection() as connection:
            return self._project_from_connection(connection, project_id)

    def _project_from_connection(
        self,
        connection: sqlite3.Connection,
        project_id: str,
    ) -> dict:
        identifier = self.validate_project_id(project_id)
        row = connection.execute(
            "SELECT id, name, created_at, updated_at FROM projects WHERE id = ?",
            (identifier,),
        ).fetchone()
        if row is None:
            raise StorageError("Project not found.")
        return dict(row)

    def delete_project(self, project_id: str) -> dict:
        """Permanently remove one project and every project-scoped record and file."""
        project = self.get_project(project_id)
        identifier = project["id"]
        project_root = self.projects_root / identifier
        if project_root.is_symlink():
            raise StorageError("Refusing to delete a project stored through a symbolic link.")

        quarantine: Path | None = None
        with self._write_lock:
            if project_root.exists():
                quarantine = self.projects_root / f".deleting-{identifier}-{uuid.uuid4().hex}"
                project_root.rename(quarantine)

            try:
                with self.connection() as connection:
                    counts = {
                        "messages": connection.execute(
                            "SELECT COUNT(*) AS count FROM messages WHERE project_id = ?",
                            (identifier,),
                        ).fetchone()["count"],
                        "files": connection.execute(
                            "SELECT COUNT(*) AS count FROM file_ownership WHERE project_id = ?",
                            (identifier,),
                        ).fetchone()["count"],
                        "memory_events": connection.execute(
                            "SELECT COUNT(*) AS count FROM agent_memory_events WHERE project_id = ?",
                            (identifier,),
                        ).fetchone()["count"],
                        "global_memory_events": connection.execute(
                            """
                            SELECT COUNT(*) AS count FROM agent_global_memory_events
                            WHERE source_project_id = ?
                            """,
                            (identifier,),
                        ).fetchone()["count"],
                        "web_sources": connection.execute(
                            "SELECT COUNT(*) AS count FROM web_sources WHERE project_id = ?",
                            (identifier,),
                        ).fetchone()["count"],
                    }
                    connection.execute(
                        "DELETE FROM agent_global_memory_events WHERE source_project_id = ?",
                        (identifier,),
                    )
                    deleted = connection.execute(
                        "DELETE FROM projects WHERE id = ?", (identifier,)
                    )
                    if deleted.rowcount != 1:
                        raise StorageError("Project not found.")
                    remaining = int(
                        connection.execute(
                            "SELECT COUNT(*) AS count FROM projects"
                        ).fetchone()["count"]
                    )
            except Exception:
                if quarantine is not None and quarantine.exists() and not project_root.exists():
                    quarantine.rename(project_root)
                raise

            if quarantine is not None and quarantine.exists():
                shutil.rmtree(quarantine)

        fallback_project = None
        if remaining == 0:
            fallback_project = self.create_project("General", project_id="general")

        return {
            "deleted": True,
            "project": project,
            "deleted_records": {key: int(value) for key, value in counts.items()},
            "fallback_project": fallback_project,
        }

    def add_message(
        self,
        project_id: str,
        role: str,
        content: str,
        *,
        agent_id: str | None = None,
        annotations: list[dict] | None = None,
        metadata: dict | None = None,
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
            "created_at": timestamp,
        }
        with self._write_lock, self.connection() as connection:
            connection.execute(
                """
                INSERT INTO messages(
                    id, project_id, role, agent_id, content,
                    annotations_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message["id"],
                    project_id,
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
