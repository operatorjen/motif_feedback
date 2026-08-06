from __future__ import annotations

import sqlite3
import uuid

from .constants import (
    DEFAULT_WEB_SOURCE_LIMIT,
    MAX_WEB_SOURCE_LIMIT,
)
from .storage_core import (
    StorageError,
    utc_now,
)


class WebSourceRepositoryMixin:
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
        retrieval_method: str = "direct_http",
        retrieval_attempts: int = 1,
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
                    content_sha256, retrieval_method, retrieval_attempts, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    str(retrieval_method)[:80] or "direct_http",
                    max(1, int(retrieval_attempts)),
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
                       content_sha256, retrieval_method, retrieval_attempts, fetched_at
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
                       content_sha256, retrieval_method, retrieval_attempts, fetched_at
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
                       content_sha256, retrieval_method, retrieval_attempts, fetched_at
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

