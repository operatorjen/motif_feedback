from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolPolicy:
    changes_state: bool = False
    interrupted_recovery: str = "none"


TOOL_POLICIES = {
    "record_motif_observations": ToolPolicy(
        changes_state=True,
        interrupted_recovery="read_motif_batch",
    ),
    "propose_persona_update": ToolPolicy(
        changes_state=True,
        interrupted_recovery="manual_review",
    ),
    "write_project_file": ToolPolicy(
        changes_state=True,
        interrupted_recovery="verify_content_hash",
    ),
}


def tool_changes_state(name: str) -> bool:
    return TOOL_POLICIES.get(name, ToolPolicy()).changes_state


def tool_recovery_strategy(name: str) -> str:
    return TOOL_POLICIES.get(name, ToolPolicy()).interrupted_recovery


def tool_request_fingerprint(name: str, arguments: dict[str, Any]) -> str:
    """Return one canonical identity for a tool request and its arguments."""
    encoded = json.dumps(
        {"tool": name, "arguments": arguments},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def public_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Retain useful audit fields without storing generated bodies."""
    public = dict(arguments)
    for key in ("content", "content_text", "markdown_text", "yaml_text"):
        value = public.pop(key, None)
        if value is not None:
            public[f"{key}_bytes"] = len(str(value).encode("utf-8"))
    if name == "propose_persona_update":
        changes = public.pop("changes", [])
        public["changes"] = [
            {
                key: change[key]
                for key in ("path", "operation")
                if isinstance(change, dict) and key in change
            }
            for change in changes
            if isinstance(change, dict)
        ]
        evidence = public.pop("evidence", [])
        public["evidence_count"] = len(evidence) if isinstance(evidence, list) else 0
    if name == "record_motif_observations":
        observations = public.pop("observations", [])
        public["observation_count"] = len(observations) if isinstance(observations, list) else 0
    return public
