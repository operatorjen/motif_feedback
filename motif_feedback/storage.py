from __future__ import annotations

from .storage_analytics import AnalyticsRepositoryMixin
from .storage_core import (
    ChatTurnConflictError,
    StorageCore,
    StorageError,
    slugify_project_name,
    utc_now,
)
from .storage_files import FileOwnershipRepositoryMixin
from .storage_memory import MemoryRepositoryMixin
from .storage_messages import MessageRepositoryMixin
from .storage_migrations import StorageMigrationMixin
from .storage_motifs import MotifRepositoryMixin
from .storage_projects import ProjectRepositoryMixin
from .storage_schema import StorageSchemaMixin
from .storage_turns import TurnRepositoryMixin
from .storage_web_sources import WebSourceRepositoryMixin

__all__ = [
    "ChatTurnConflictError",
    "Storage",
    "StorageError",
    "slugify_project_name",
    "utc_now",
]


class Storage(
    StorageSchemaMixin,
    StorageMigrationMixin,
    TurnRepositoryMixin,
    AnalyticsRepositoryMixin,
    FileOwnershipRepositoryMixin,
    MemoryRepositoryMixin,
    MotifRepositoryMixin,
    WebSourceRepositoryMixin,
    ProjectRepositoryMixin,
    MessageRepositoryMixin,
    StorageCore,
):
    """Stable storage facade composed from domain-focused SQLite repositories."""
