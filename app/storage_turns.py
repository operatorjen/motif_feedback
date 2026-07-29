from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

from .execution_semantics import AGENT_FINISHED_OPERATION
from .storage_core import (
    ChatTurnConflictError,
    StorageError,
    utc_now,
)


class TurnRepositoryMixin:
    def begin_chat_turn(
        self,
        turn_id: str,
        project_id: str,
        request_fingerprint: str,
        *,
        request: dict | None = None,
        runtime: dict | None = None,
    ) -> dict:
        """Create one durable turn record or return the matching existing record."""
        timestamp = utc_now()
        with self._write_lock, self.connection() as connection:
            self._project_from_connection(connection, project_id)
            row = connection.execute(
                """
                SELECT id, project_id, request_fingerprint, status, result_json,
                       trace_json, failure_detail, request_json, runtime_json,
                       resolution, resolved_at, started_at, updated_at
                FROM chat_turns WHERE id = ?
                """,
                (turn_id,),
            ).fetchone()
            if row is not None:
                existing = self._row_to_chat_turn(row)
                if (
                    existing["project_id"] != project_id
                    or existing["request_fingerprint"] != request_fingerprint
                ):
                    raise ChatTurnConflictError(
                        "That turn identifier is already attached to a different request."
                    )
                existing["created"] = False
                return existing
            connection.execute(
                """
                INSERT INTO chat_turns(
                    id, project_id, request_fingerprint, status, result_json,
                    trace_json, failure_detail, request_json, runtime_json,
                    resolution, resolved_at, started_at, updated_at
                ) VALUES (?, ?, ?, 'running', NULL, '{}', NULL, ?, ?, NULL, NULL, ?, ?)
                """,
                (
                    turn_id,
                    project_id,
                    request_fingerprint,
                    json.dumps(request, ensure_ascii=False) if request is not None else None,
                    json.dumps(runtime, ensure_ascii=False) if runtime is not None else None,
                    timestamp,
                    timestamp,
                ),
            )
        return {
            "id": turn_id,
            "project_id": project_id,
            "request_fingerprint": request_fingerprint,
            "status": "running",
            "result": None,
            "trace": {},
            "failure_detail": None,
            "request": request,
            "runtime": runtime,
            "resolution": None,
            "resolved_at": None,
            "started_at": timestamp,
            "updated_at": timestamp,
            "created": True,
        }

    def complete_chat_turn(
        self,
        turn_id: str,
        result: dict,
        trace: dict,
    ) -> dict:
        return self._finish_chat_turn(
            turn_id,
            status="completed",
            result=result,
            trace=trace,
            failure_detail=None,
        )

    def fail_chat_turn(
        self,
        turn_id: str,
        *,
        status: str,
        detail: str,
        trace: dict,
    ) -> dict:
        if status not in {"failed", "interrupted"}:
            raise StorageError("Invalid failed chat-turn status.")
        return self._finish_chat_turn(
            turn_id,
            status=status,
            result=None,
            trace=trace,
            failure_detail=" ".join(str(detail).split())[:4_000],
        )

    def _finish_chat_turn(
        self,
        turn_id: str,
        *,
        status: str,
        result: dict | None,
        trace: dict,
        failure_detail: str | None,
    ) -> dict:
        timestamp = utc_now()
        with self._write_lock, self.connection() as connection:
            updated = connection.execute(
                """
                UPDATE chat_turns
                SET status = ?, result_json = ?, trace_json = ?,
                    failure_detail = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    json.dumps(result, ensure_ascii=False) if result is not None else None,
                    json.dumps(trace, ensure_ascii=False),
                    failure_detail,
                    timestamp,
                    turn_id,
                ),
            )
            if updated.rowcount != 1:
                raise StorageError("Chat turn not found.")
        return self.get_chat_turn(turn_id)

    def get_chat_turn(self, turn_id: str) -> dict:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT id, project_id, request_fingerprint, status, result_json,
                       trace_json, failure_detail, request_json, runtime_json,
                       resolution, resolved_at, started_at, updated_at
                FROM chat_turns WHERE id = ?
                """,
                (turn_id,),
            ).fetchone()
        if row is None:
            raise StorageError("Chat turn not found.")
        return self._row_to_chat_turn(row)

    def list_chat_turns(self, project_id: str, limit: int = 100) -> list[dict]:
        safe_limit = min(max(int(limit), 1), 500)
        with self.connection() as connection:
            self._project_from_connection(connection, project_id)
            rows = connection.execute(
                """
                SELECT id, project_id, request_fingerprint, status, result_json,
                       trace_json, failure_detail, request_json, runtime_json,
                       resolution, resolved_at, started_at, updated_at
                FROM chat_turns
                WHERE project_id = ?
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (project_id, safe_limit),
            ).fetchall()
        return [self._row_to_chat_turn(row) for row in rows]

    def list_recoverable_chat_turns(self, project_id: str) -> list[dict]:
        """Return every unresolved turn that still has enough state to resume."""
        with self.connection() as connection:
            self._project_from_connection(connection, project_id)
            rows = connection.execute(
                """
                SELECT id, project_id, request_fingerprint, status, result_json,
                       trace_json, failure_detail, request_json, runtime_json,
                       resolution, resolved_at, started_at, updated_at
                FROM chat_turns
                WHERE project_id = ?
                  AND status IN ('failed', 'interrupted')
                  AND resolution IS NULL
                  AND request_json IS NOT NULL
                  AND runtime_json IS NOT NULL
                ORDER BY started_at DESC
                """,
                (project_id,),
            ).fetchall()
        return [self._row_to_chat_turn(row) for row in rows]

    def prune_chat_turn_traces(self, retention_days: int) -> int:
        """Prune diagnostics only when the operator explicitly configures retention."""
        if retention_days <= 0:
            return 0
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        with self._write_lock, self.connection() as connection:
            eligible_turns = [
                row["id"]
                for row in connection.execute(
                    """
                    SELECT id FROM chat_turns
                    WHERE updated_at < ?
                      AND (status = 'completed' OR resolution IS NOT NULL)
                    """,
                    (cutoff,),
                ).fetchall()
            ]
            if eligible_turns:
                placeholders = ", ".join("?" for _ in eligible_turns)
                connection.execute(
                    f"DELETE FROM turn_operations WHERE turn_id IN ({placeholders})",
                    eligible_turns,
                )
            updated = connection.execute(
                """
                UPDATE chat_turns
                SET trace_json = '{}'
                WHERE updated_at < ?
                  AND (status = 'completed' OR resolution IS NOT NULL)
                  AND trace_json != '{}'
                """,
                (cutoff,),
            )
            return int(updated.rowcount)

    def resume_chat_turn(self, turn_id: str) -> dict:
        timestamp = utc_now()
        with self._write_lock, self.connection() as connection:
            row = connection.execute(
                """
                SELECT status, resolution, request_json, runtime_json
                FROM chat_turns WHERE id = ?
                """,
                (turn_id,),
            ).fetchone()
            if row is None:
                raise StorageError("Chat turn not found.")
            if row["status"] not in {"failed", "interrupted"} or row["resolution"]:
                raise ChatTurnConflictError("Only an unresolved failed turn can be resumed.")
            if not row["request_json"] or not row["runtime_json"]:
                raise ChatTurnConflictError(
                    "This older turn does not contain the state required for resumption."
                )
            connection.execute(
                """
                UPDATE chat_turns
                SET status = 'running', failure_detail = NULL, updated_at = ?
                WHERE id = ?
                """,
                (timestamp, turn_id),
            )
        return self.get_chat_turn(turn_id)

    def resolve_chat_turn(self, turn_id: str, resolution: str) -> dict:
        if resolution != "accepted_partial":
            raise StorageError("Unsupported chat-turn resolution.")
        timestamp = utc_now()
        with self._write_lock, self.connection() as connection:
            updated = connection.execute(
                """
                UPDATE chat_turns
                SET resolution = ?, resolved_at = ?, updated_at = ?
                WHERE id = ? AND status IN ('failed', 'interrupted')
                  AND resolution IS NULL
                """,
                (resolution, timestamp, timestamp, turn_id),
            )
            if updated.rowcount != 1:
                raise ChatTurnConflictError(
                    "Only an unresolved failed turn can be accepted as partial."
                )
        return self.get_chat_turn(turn_id)

    def begin_turn_operation(
        self,
        *,
        operation_id: str,
        turn_id: str,
        project_id: str,
        agent_id: str,
        turn_beat: int,
        operation_type: str,
        request_fingerprint: str,
        payload: dict | None = None,
    ) -> dict:
        """Claim one deterministic operation or return its durable prior state."""
        timestamp = utc_now()
        with self._write_lock, self.connection() as connection:
            turn = connection.execute(
                "SELECT project_id FROM chat_turns WHERE id = ?",
                (turn_id,),
            ).fetchone()
            if turn is None or turn["project_id"] != project_id:
                raise StorageError("Chat turn not found.")
            row = connection.execute(
                """
                SELECT id, turn_id, project_id, agent_id, turn_beat,
                       operation_type, request_fingerprint, status,
                       payload_json, result_json, started_at, updated_at,
                       completed_at
                FROM turn_operations WHERE id = ?
                """,
                (operation_id,),
            ).fetchone()
            if row is not None:
                operation = self._row_to_turn_operation(row)
                if (
                    operation["turn_id"] != turn_id
                    or operation["project_id"] != project_id
                    or operation["agent_id"] != agent_id
                    or operation["turn_beat"] != turn_beat
                    or operation["operation_type"] != operation_type
                    or operation["request_fingerprint"] != request_fingerprint
                ):
                    raise ChatTurnConflictError(
                        "That operation identifier belongs to different work."
                    )
                operation["created"] = False
                return operation
            connection.execute(
                """
                INSERT INTO turn_operations(
                    id, turn_id, project_id, agent_id, turn_beat,
                    operation_type, request_fingerprint, status,
                    payload_json, result_json, started_at, updated_at,
                    completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'started', ?, NULL, ?, ?, NULL)
                """,
                (
                    operation_id,
                    turn_id,
                    project_id,
                    agent_id,
                    turn_beat,
                    operation_type,
                    request_fingerprint,
                    json.dumps(payload, ensure_ascii=False) if payload is not None else None,
                    timestamp,
                    timestamp,
                ),
            )
        return {
            "id": operation_id,
            "turn_id": turn_id,
            "project_id": project_id,
            "agent_id": agent_id,
            "turn_beat": turn_beat,
            "operation_type": operation_type,
            "request_fingerprint": request_fingerprint,
            "status": "started",
            "payload": payload,
            "result": None,
            "started_at": timestamp,
            "updated_at": timestamp,
            "completed_at": None,
            "created": True,
        }

    def complete_turn_operation(self, operation_id: str, result: dict) -> dict:
        timestamp = utc_now()
        encoded = json.dumps(result, ensure_ascii=False)
        with self._write_lock, self.connection() as connection:
            row = connection.execute(
                "SELECT status, result_json FROM turn_operations WHERE id = ?",
                (operation_id,),
            ).fetchone()
            if row is None:
                raise StorageError("Turn operation not found.")
            if row["status"] == "completed":
                existing = json.loads(row["result_json"] or "{}")
                if existing != result:
                    raise ChatTurnConflictError(
                        "That operation was already completed with a different result."
                    )
            else:
                connection.execute(
                    """
                    UPDATE turn_operations
                    SET status = 'completed', result_json = ?, updated_at = ?,
                        completed_at = ?
                    WHERE id = ?
                    """,
                    (encoded, timestamp, timestamp, operation_id),
                )
        return self.get_turn_operation(operation_id)

    def get_turn_operation(self, operation_id: str) -> dict:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT id, turn_id, project_id, agent_id, turn_beat,
                       operation_type, request_fingerprint, status,
                       payload_json, result_json, started_at, updated_at,
                       completed_at
                FROM turn_operations WHERE id = ?
                """,
                (operation_id,),
            ).fetchone()
        if row is None:
            raise StorageError("Turn operation not found.")
        return self._row_to_turn_operation(row)

    def list_turn_operations(self, turn_id: str) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT id, turn_id, project_id, agent_id, turn_beat,
                       operation_type, request_fingerprint, status,
                       payload_json, result_json, started_at, updated_at,
                       completed_at
                FROM turn_operations
                WHERE turn_id = ?
                ORDER BY started_at ASC, rowid ASC
                """,
                (turn_id,),
            ).fetchall()
        return [self._row_to_turn_operation(row) for row in rows]

    def list_turn_operations_for_turns(
        self,
        turn_ids: list[str],
    ) -> dict[str, list[dict]]:
        """Read operation histories for a bounded turn listing in one query."""
        unique_ids = list(dict.fromkeys(str(turn_id) for turn_id in turn_ids if turn_id))
        grouped = {turn_id: [] for turn_id in unique_ids}
        if not unique_ids:
            return grouped
        placeholders = ", ".join("?" for _ in unique_ids)
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT id, turn_id, project_id, agent_id, turn_beat,
                       operation_type, request_fingerprint, status,
                       payload_json, result_json, started_at, updated_at,
                       completed_at
                FROM turn_operations
                WHERE turn_id IN ({placeholders})
                ORDER BY started_at ASC, rowid ASC
                """,
                unique_ids,
            ).fetchall()
        for row in rows:
            grouped[str(row["turn_id"])].append(self._row_to_turn_operation(row))
        return grouped

    def completed_turn_agents(self, turn_id: str) -> set[str]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT agent_id
                FROM turn_operations
                WHERE turn_id = ? AND operation_type = ?
                  AND status = 'completed'
                """,
                (turn_id, AGENT_FINISHED_OPERATION),
            ).fetchall()
        return {str(row["agent_id"]) for row in rows}

    @staticmethod
    def _row_to_turn_operation(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "turn_id": row["turn_id"],
            "project_id": row["project_id"],
            "agent_id": row["agent_id"],
            "turn_beat": int(row["turn_beat"]),
            "operation_type": row["operation_type"],
            "request_fingerprint": row["request_fingerprint"],
            "status": row["status"],
            "payload": json.loads(row["payload_json"]) if row["payload_json"] else None,
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "started_at": row["started_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
        }

    @staticmethod
    def _row_to_chat_turn(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "request_fingerprint": row["request_fingerprint"],
            "status": row["status"],
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "trace": json.loads(row["trace_json"] or "{}"),
            "failure_detail": row["failure_detail"],
            "request": json.loads(row["request_json"]) if row["request_json"] else None,
            "runtime": json.loads(row["runtime_json"]) if row["runtime_json"] else None,
            "resolution": row["resolution"],
            "resolved_at": row["resolved_at"],
            "started_at": row["started_at"],
            "updated_at": row["updated_at"],
        }
