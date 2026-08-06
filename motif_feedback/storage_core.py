from __future__ import annotations

import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .constants import (
    PROJECT_ID_RANDOM_CHARS,
    PROJECT_SLUG_MAX_CHARS,
    SQLITE_TIMEOUT_SECONDS,
)

PROJECT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class StorageError(ValueError):
    pass


class ChatTurnConflictError(StorageError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def slugify_project_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    slug = slug[:PROJECT_SLUG_MAX_CHARS] or "project"
    return f"{slug}-{uuid.uuid4().hex[:PROJECT_ID_RANDOM_CHARS]}"


class StorageCore:
    def __init__(self, database_path: Path, projects_root: Path) -> None:
        self.database_path = database_path
        self.projects_root = projects_root
        self._write_lock = threading.RLock()
        self._fts_available = False

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

