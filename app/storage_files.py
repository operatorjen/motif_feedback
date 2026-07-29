from __future__ import annotations

from .storage_core import (
    StorageError,
    utc_now,
)


class FileOwnershipRepositoryMixin:
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

