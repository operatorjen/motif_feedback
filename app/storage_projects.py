from __future__ import annotations

import shutil
import sqlite3
import uuid
from pathlib import Path

from .constants import (
    STORED_PROJECT_NAME_MAX_CHARS,
)
from .storage_core import (
    PROJECT_ID_PATTERN,
    StorageError,
    slugify_project_name,
    utc_now,
)


class ProjectRepositoryMixin:
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
                        "chat_turns": connection.execute(
                            "SELECT COUNT(*) AS count FROM chat_turns WHERE project_id = ?",
                            (identifier,),
                        ).fetchone()["count"],
                        "turn_operations": connection.execute(
                            """
                            SELECT COUNT(*) AS count FROM turn_operations
                            WHERE project_id = ?
                            """,
                            (identifier,),
                        ).fetchone()["count"],
                    }
                    connection.execute(
                        "DELETE FROM agent_global_memory_events WHERE source_project_id = ?",
                        (identifier,),
                    )
                    deleted = connection.execute("DELETE FROM projects WHERE id = ?", (identifier,))
                    if deleted.rowcount != 1:
                        raise StorageError("Project not found.")
                    remaining = int(
                        connection.execute("SELECT COUNT(*) AS count FROM projects").fetchone()[
                            "count"
                        ]
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
