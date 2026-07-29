from __future__ import annotations

import sqlite3

from .storage_core import (
    utc_now,
)


class StorageSchemaMixin:
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
                    turn_id TEXT,
                    operation_id TEXT,
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
                    operation_id TEXT,
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

                CREATE INDEX IF NOT EXISTS idx_memory_project_agent_user_message
                ON agent_memory_events(project_id, agent_id, user_message_id);

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
                    retrieval_method TEXT NOT NULL DEFAULT 'direct_http',
                    retrieval_attempts INTEGER NOT NULL DEFAULT 1,
                    fetched_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_web_sources_project_fetched
                ON web_sources(project_id, fetched_at DESC);

                CREATE INDEX IF NOT EXISTS idx_web_sources_project_requested
                ON web_sources(project_id, requested_url, fetched_at DESC);

                CREATE TABLE IF NOT EXISTS chat_turns (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(
                        status IN ('running', 'completed', 'failed', 'interrupted')
                    ),
                    result_json TEXT,
                    trace_json TEXT NOT NULL DEFAULT '{}',
                    failure_detail TEXT,
                    request_json TEXT,
                    runtime_json TEXT,
                    resolution TEXT,
                    resolved_at TEXT,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chat_turns_project_started
                ON chat_turns(project_id, started_at DESC);

                CREATE TABLE IF NOT EXISTS turn_operations (
                    id TEXT PRIMARY KEY,
                    turn_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    turn_beat INTEGER NOT NULL,
                    operation_type TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('started', 'completed')),
                    payload_json TEXT,
                    result_json TEXT,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    FOREIGN KEY(turn_id) REFERENCES chat_turns(id) ON DELETE CASCADE,
                    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_turn_operations_turn_agent
                ON turn_operations(turn_id, agent_id, turn_beat, operation_type);

                CREATE TABLE IF NOT EXISTS schema_migrations (
                    name TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                """
                UPDATE chat_turns
                SET status = 'interrupted', updated_at = ?
                WHERE status = 'running'
                """,
                (utc_now(),),
            )
            ownership_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(file_ownership)").fetchall()
            }
            message_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(messages)").fetchall()
            }
            if "turn_id" not in message_columns:
                connection.execute("ALTER TABLE messages ADD COLUMN turn_id TEXT")
                connection.execute(
                    """
                    UPDATE messages
                    SET turn_id = json_extract(metadata_json, '$.turn_id')
                    WHERE json_valid(metadata_json)
                    """
                )
            if "operation_id" not in message_columns:
                connection.execute("ALTER TABLE messages ADD COLUMN operation_id TEXT")
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_project_turn
                ON messages(project_id, turn_id, created_at)
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_operation
                ON messages(operation_id) WHERE operation_id IS NOT NULL
                """
            )
            memory_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(agent_memory_events)").fetchall()
            }
            if "operation_id" not in memory_columns:
                connection.execute("ALTER TABLE agent_memory_events ADD COLUMN operation_id TEXT")
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_operation
                ON agent_memory_events(operation_id) WHERE operation_id IS NOT NULL
                """
            )
            chat_turn_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(chat_turns)").fetchall()
            }
            for name, declaration in (
                ("request_json", "TEXT"),
                ("runtime_json", "TEXT"),
                ("resolution", "TEXT"),
                ("resolved_at", "TEXT"),
            ):
                if name not in chat_turn_columns:
                    connection.execute(f"ALTER TABLE chat_turns ADD COLUMN {name} {declaration}")
            if "shared_agent_edit" not in ownership_columns:
                connection.execute(
                    "ALTER TABLE file_ownership ADD COLUMN shared_agent_edit INTEGER NOT NULL DEFAULT 0"
                )
            web_source_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(web_sources)").fetchall()
            }
            if "retrieval_method" not in web_source_columns:
                connection.execute(
                    "ALTER TABLE web_sources ADD COLUMN "
                    "retrieval_method TEXT NOT NULL DEFAULT 'direct_http'"
                )
            if "retrieval_attempts" not in web_source_columns:
                connection.execute(
                    "ALTER TABLE web_sources ADD COLUMN "
                    "retrieval_attempts INTEGER NOT NULL DEFAULT 1"
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
        self._initialize_memory_fts()

    def _initialize_memory_fts(self) -> None:
        """Create a rebuildable FTS projection; lexical retrieval remains the fallback."""
        try:
            with self._write_lock, self.connection() as connection:
                connection.executescript(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS agent_memory_fts USING fts5(
                        event_id UNINDEXED,
                        project_id UNINDEXED,
                        agent_id UNINDEXED,
                        trigger_text,
                        return_text
                    );
                    CREATE VIRTUAL TABLE IF NOT EXISTS agent_global_memory_fts USING fts5(
                        event_id UNINDEXED,
                        agent_id UNINDEXED,
                        source_project_id UNINDEXED,
                        source_project_name,
                        trigger_summary,
                        return_summary
                    );
                    CREATE TRIGGER IF NOT EXISTS agent_memory_fts_delete
                    AFTER DELETE ON agent_memory_events BEGIN
                        DELETE FROM agent_memory_fts WHERE event_id = old.id;
                    END;
                    CREATE TRIGGER IF NOT EXISTS agent_global_memory_fts_delete
                    AFTER DELETE ON agent_global_memory_events BEGIN
                        DELETE FROM agent_global_memory_fts WHERE event_id = old.id;
                    END;
                    """
                )
                local_count = connection.execute(
                    "SELECT COUNT(*) AS count FROM agent_memory_events"
                ).fetchone()["count"]
                local_fts_count = connection.execute(
                    "SELECT COUNT(*) AS count FROM agent_memory_fts"
                ).fetchone()["count"]
                if local_count != local_fts_count:
                    connection.execute("DELETE FROM agent_memory_fts")
                    connection.execute(
                        """
                        INSERT INTO agent_memory_fts(
                            event_id, project_id, agent_id, trigger_text, return_text
                        )
                        SELECT id, project_id, agent_id, trigger_text, return_text
                        FROM agent_memory_events
                        """
                    )
                global_count = connection.execute(
                    "SELECT COUNT(*) AS count FROM agent_global_memory_events"
                ).fetchone()["count"]
                global_fts_count = connection.execute(
                    "SELECT COUNT(*) AS count FROM agent_global_memory_fts"
                ).fetchone()["count"]
                if global_count != global_fts_count:
                    connection.execute("DELETE FROM agent_global_memory_fts")
                    connection.execute(
                        """
                        INSERT INTO agent_global_memory_fts(
                            event_id, agent_id, source_project_id,
                            source_project_name, trigger_summary, return_summary
                        )
                        SELECT id, agent_id, source_project_id, source_project_name,
                               trigger_summary, return_summary
                        FROM agent_global_memory_events
                        """
                    )
            self._fts_available = True
        except sqlite3.OperationalError:
            self._fts_available = False
