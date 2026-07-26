from __future__ import annotations

from typing import Any


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
    return public
