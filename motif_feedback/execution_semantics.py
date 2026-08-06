from __future__ import annotations

PROVIDER_COMPLETION_OPERATION = "provider_completion"
MESSAGE_COMMITTED_OPERATION = "message_committed"
MEMORY_COMMITTED_OPERATION = "memory_committed"
BEAT_FINISHED_OPERATION = "beat_finished"
AGENT_FINISHED_OPERATION = "agent_finished"

BEAT_OPERATION_SEQUENCE = (
    PROVIDER_COMPLETION_OPERATION,
    MESSAGE_COMMITTED_OPERATION,
    MEMORY_COMMITTED_OPERATION,
    BEAT_FINISHED_OPERATION,
)


def public_execution_stage(
    *,
    turn_status: str,
    participants: list[str],
    operations: list[dict],
) -> dict | None:
    """Project durable operations into the room recovery notice's current stage."""
    pending = next(
        (
            operation
            for operation in reversed(operations)
            if operation["status"] != "completed"
        ),
        None,
    )
    latest = pending or (operations[-1] if operations else None)
    if pending is None and latest is not None and turn_status in {"failed", "interrupted"}:
        try:
            stage_index = BEAT_OPERATION_SEQUENCE.index(latest["operation_type"])
        except ValueError:
            stage_index = -1
        if 0 <= stage_index < len(BEAT_OPERATION_SEQUENCE) - 1:
            latest = {
                **latest,
                "operation_type": BEAT_OPERATION_SEQUENCE[stage_index + 1],
                "status": "pending",
            }
        elif latest["operation_type"] == BEAT_OPERATION_SEQUENCE[-1]:
            latest = {
                **latest,
                "operation_type": AGENT_FINISHED_OPERATION,
                "status": "pending",
            }
        elif latest["operation_type"] == AGENT_FINISHED_OPERATION:
            finished_agents = {
                operation["agent_id"]
                for operation in operations
                if operation["operation_type"] == AGENT_FINISHED_OPERATION
                and operation["status"] == "completed"
            }
            remaining_agent = next(
                (agent_id for agent_id in participants if agent_id not in finished_agents),
                None,
            )
            if remaining_agent is not None:
                latest = {
                    **latest,
                    "agent_id": remaining_agent,
                    "turn_beat": 1,
                    "operation_type": BEAT_OPERATION_SEQUENCE[0],
                    "status": "pending",
                }
    if latest is None:
        return None
    return {
        "agent_id": latest["agent_id"],
        "turn_beat": latest["turn_beat"],
        "operation": latest["operation_type"],
        "status": latest["status"],
    }
