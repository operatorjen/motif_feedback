from __future__ import annotations

import json
import math
import shutil
import threading
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from .atomic_files import atomic_write_text
from .config import Settings
from .constants import (
    ATTRACTOR_STRENGTH_PATH_PARTS,
    MAX_PERSONA_PROPOSALS,
    PERSONA_CHANGE_MAX_BYTES,
    PERSONA_CHANGE_MAX_COLLECTION_ITEMS,
    PERSONA_CHANGE_MAX_DEPTH,
    PERSONA_CHANGE_MAX_STRING_CHARS,
)
from .models import AGENT_IDS, PersonaUpdate

if TYPE_CHECKING:
    from .storage import Storage

PERSONA_SCHEMA_VERSION = 4
DEFAULT_RELATIONSHIP_EVIDENCE_EVENTS = 2
ACTIVE_CANDIDATE_STATUSES = {"dormant", "pending_user_review"}

AUTO_COMMIT_PREFIXES = (
    "current_position",
    "motif_expression",
    "relationship_memory",
    "continuity_training.current_cycle",
    "self_model.recurring_strengths",
    "self_model.recurring_distortions",
    "self_model.developmental_notes",
)

PROPOSAL_ONLY_PREFIXES = (
    "core_disposition",
    "systems_style",
    "research_style",
    "attractors",
    "conversation",
    "continuity_training.continuity_channels",
    "continuity_training.continuity_conditions",
)

AGENT_LOCKED_PREFIXES = (
    "agent_id",
    "core_motif",
    "source_context",
    "schema_version",
)


class PersonaUpdateError(ValueError):
    pass


def timestamp_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


class PersonaStore:
    def __init__(self, settings: Settings, storage: Storage | None = None) -> None:
        self.settings = settings
        self.storage = storage
        self._lock = threading.RLock()

    def initialize(self) -> None:
        roots = (
            self.settings.personas_root,
            self.settings.shared_root,
            self.settings.history_root,
            self.settings.proposals_root,
        )
        for root in roots:
            root.mkdir(parents=True, exist_ok=True)

        seed_agents = self.settings.seed_root / "agents"
        for agent_id in AGENT_IDS:
            destination = self.settings.personas_root / f"{agent_id}.yaml"
            seed = seed_agents / f"{agent_id}.yaml"
            if not destination.exists():
                shutil.copy2(seed, destination)
                destination.chmod(0o600)
            else:
                data = yaml.safe_load(destination.read_text(encoding="utf-8")) or {}
                if not isinstance(data, dict):
                    raise PersonaUpdateError("Persona YAML must contain a mapping.")
                self._validate_persona(agent_id, data)

        self._install_shared_file("meta-instructional-agents.md")

    def _install_shared_file(self, filename: str) -> None:
        """Install user-editable shared context on first startup."""
        source = self.settings.seed_root / "shared" / filename
        destination = self.settings.shared_root / filename
        if not destination.exists():
            shutil.copy2(source, destination)
            destination.chmod(0o600)

    def _agent_file(self, agent_id: str) -> Path:
        if agent_id not in AGENT_IDS:
            raise PersonaUpdateError("Unknown agent.")
        path = (self.settings.personas_root / f"{agent_id}.yaml").resolve(strict=False)
        path.relative_to(self.settings.personas_root.resolve())
        if path.is_symlink():
            raise PersonaUpdateError("Persona files may not be symbolic links.")
        return path

    def load_shared_context(self) -> str:
        path = self.settings.shared_root / "meta-instructional-agents.md"
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return "[No shared project-context markdown is installed.]"

    def save_user_shared_context(self, markdown_text: str) -> dict:
        normalized = markdown_text.strip() + "\n"
        if not normalized.strip():
            raise PersonaUpdateError("Shared context markdown may not be empty.")
        path = self.settings.shared_root / "meta-instructional-agents.md"
        history_dir = self.settings.history_root / "shared"
        history_dir.mkdir(parents=True, exist_ok=True)
        if path.exists():
            shutil.copy2(path, history_dir / f"{timestamp_id()}_meta-instructional-agents.md")
        atomic_write_text(path, normalized)
        record = {
            "timestamp": timestamp_id(),
            "actor": "user",
            "reason": "Manual shared project-context edit from local interface.",
            "file": "meta-instructional-agents.md",
        }
        with (history_dir / "changes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"ok": True, "markdown_text": normalized}

    def load_persona(self, agent_id: str) -> dict:
        path = self._agent_file(agent_id)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        self._validate_persona(agent_id, data)
        return data

    def get_persona_yaml(self, agent_id: str) -> str:
        return self._agent_file(agent_id).read_text(encoding="utf-8")

    def list_summaries(self) -> list[dict]:
        summaries: list[dict] = []
        for agent_id in AGENT_IDS:
            persona = self.load_persona(agent_id)
            core_motif = persona.get("core_motif", {})
            summaries.append(
                {
                    "agent_id": agent_id,
                    "display_name": persona.get("display_name", agent_id),
                    "archetype": persona.get("archetype", ""),
                    "version": persona.get("version", 1),
                    "schema_version": persona.get("schema_version", 0),
                    "summary": persona.get("core_disposition", {}).get("summary", ""),
                    "systems_orientation": persona.get("systems_style", {}).get("orientation", ""),
                    "core_motif_name": core_motif.get("name", ""),
                    "core_motif_symbol": core_motif.get("symbol", ""),
                    "core_motif_statement": core_motif.get("statement", ""),
                }
            )
        return summaries

    def save_user_edit(self, agent_id: str, yaml_text: str) -> dict:
        with self._lock:
            parsed = yaml.safe_load(yaml_text)
            if not isinstance(parsed, dict):
                raise PersonaUpdateError("Persona YAML must contain a mapping.")
            self._validate_persona(agent_id, parsed)
            previous = self.load_persona(agent_id)
            parsed["version"] = max(
                int(previous.get("version", 0)) + 1,
                int(parsed.get("version", 0)),
            )
            self._snapshot(agent_id, "user_edit")
            self._atomic_yaml_write(self._agent_file(agent_id), parsed)
            self._append_log(
                agent_id,
                {
                    "timestamp": timestamp_id(),
                    "actor": "user",
                    "reason": "Manual persona edit from local interface.",
                    "changes": [{"path": "*", "operation": "replace", "value": "manual_yaml_edit"}],
                },
            )
            self._reconcile_dormant_candidates(
                agent_id,
                previous,
                parsed,
                resolution_actor="user_edit",
            )
            return self.load_persona(agent_id)

    def submit_update(
        self,
        update: PersonaUpdate | dict,
        *,
        project_id: str | None = None,
    ) -> dict:
        normalized = update if isinstance(update, PersonaUpdate) else PersonaUpdate.model_validate(update)
        agent_id = normalized.agent_id

        with self._lock:
            current = self.load_persona(agent_id)
            revised = deepcopy(current)
            potentially_automatic_changes: list[dict] = []
            dormant_changes: list[dict] = []

            for change_model in normalized.changes:
                change = change_model.model_dump()
                path = change["path"]
                self._validate_change(current, change)
                if self._path_allowed(path, AGENT_LOCKED_PREFIXES):
                    raise PersonaUpdateError(
                        f"Agent may not update locked constitutional field: {path}"
                    )
                if self._path_allowed(path, AUTO_COMMIT_PREFIXES):
                    potentially_automatic_changes.append(change)
                elif self._path_allowed(path, PROPOSAL_ONLY_PREFIXES):
                    self._validate_attractor_delta(current, path, change["value"])
                    dormant_changes.append(change)
                else:
                    raise PersonaUpdateError(f"Agent may not update: {path}")

            verified_evidence = self._validate_evidence(
                agent_id,
                project_id,
                normalized.evidence,
            )
            auto_changes: list[dict] = []
            for change in potentially_automatic_changes:
                required_evidence = self._required_evidence_count(
                    current,
                    change["path"],
                )
                if len(verified_evidence) >= required_evidence:
                    auto_changes.append(change)
                else:
                    dormant_changes.append(change)

            operation_timestamp = timestamp_id()
            if auto_changes:
                self._snapshot(agent_id, operation_timestamp)
                for change in auto_changes:
                    self._set_path(revised, change["path"], change["value"], change["operation"])
                self._assert_core_motif_unchanged(current, revised)
                self._validate_persona(agent_id, revised)
                revised["version"] = int(revised.get("version", 0)) + 1
                self._atomic_yaml_write(self._agent_file(agent_id), revised)
                self._append_log(
                    agent_id,
                    {
                        "timestamp": operation_timestamp,
                        "actor": agent_id,
                        "reason": normalized.reason,
                        "evidence": verified_evidence,
                        "changes": auto_changes,
                    },
                )
                self._reconcile_dormant_candidates(
                    agent_id,
                    current,
                    revised,
                    resolution_actor="policy_commit",
                )

            proposal_path: Path | None = None
            if dormant_changes:
                proposal_path = self._save_dormant_candidate(
                    agent_id=agent_id,
                    reason=normalized.reason,
                    evidence=verified_evidence,
                    changes=dormant_changes,
                    current=current,
                    operation_timestamp=operation_timestamp,
                )

            return {
                "committed_change_count": len(auto_changes),
                "proposal_change_count": len(dormant_changes),
                "proposal_path": proposal_path.name if proposal_path else None,
            }

    def list_proposals(self) -> list[dict]:
        proposals: list[dict] = []
        for path in sorted(self.settings.proposals_root.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("status") not in ACTIVE_CANDIDATE_STATUSES:
                    continue
                data["file"] = path.name
                proposals.append(data)
            except (json.JSONDecodeError, OSError):
                continue
        return proposals[:MAX_PERSONA_PROPOSALS]

    def _validate_evidence(
        self,
        agent_id: str,
        project_id: str | None,
        evidence,
    ) -> list[dict]:
        if self.storage is None or not project_id:
            raise PersonaUpdateError(
                "Persona updates require project-scoped stored evidence."
            )
        evidence_items = [item.model_dump() for item in evidence]
        event_ids = [item["event_id"] for item in evidence_items]
        if len(set(event_ids)) != len(event_ids):
            raise PersonaUpdateError("Persona-update evidence IDs must be distinct.")
        try:
            resolved = self.storage.validate_agent_memory_evidence(
                project_id,
                agent_id,
                event_ids,
            )
        except ValueError as exc:
            raise PersonaUpdateError(str(exc)) from exc
        resolved_by_id = {item["id"]: item for item in resolved}
        return [
            {
                **item,
                "project_id": resolved_by_id[item["event_id"]]["project_id"],
                "sequence": resolved_by_id[item["event_id"]]["sequence"],
                "created_at": resolved_by_id[item["event_id"]]["created_at"],
            }
            for item in evidence_items
        ]

    @staticmethod
    def _candidate_fingerprint(agent_id: str, changes: list[dict]) -> str:
        canonical = json.dumps(
            {"agent_id": agent_id, "changes": changes},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def _required_evidence_count(cls, persona: dict, path: str) -> int:
        policy = persona.get("update_policy", {})
        if path.startswith("relationship_memory"):
            return max(
                1,
                int(
                    policy.get(
                        "minimum_supporting_events_for_relationship_change",
                        DEFAULT_RELATIONSHIP_EVIDENCE_EVENTS,
                    )
                ),
            )
        if path.startswith("attractors."):
            return max(
                1,
                int(policy.get("minimum_supporting_events_for_attractor_change", 5)),
            )
        return 1

    def _save_dormant_candidate(
        self,
        *,
        agent_id: str,
        reason: str,
        evidence: list[dict],
        changes: list[dict],
        current: dict,
        operation_timestamp: str,
    ) -> Path:
        fingerprint = self._candidate_fingerprint(agent_id, changes)
        candidate_path: Path | None = None
        candidate: dict | None = None
        proposed_paths = {change["path"] for change in changes}

        for path in sorted(self.settings.proposals_root.glob("*.json")):
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if (
                existing.get("agent_id") != agent_id
                or existing.get("status") not in ACTIVE_CANDIDATE_STATUSES
            ):
                continue
            if existing.get("fingerprint") == fingerprint:
                candidate_path = path
                candidate = existing
                continue
            existing_paths = {
                change.get("path")
                for change in existing.get("changes", [])
                if isinstance(change, dict)
            }
            if proposed_paths & existing_paths:
                existing["status"] = "superseded"
                existing["superseded_at"] = operation_timestamp
                existing["superseded_by"] = fingerprint
                atomic_write_text(
                    path,
                    json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
                )
                path.chmod(0o600)

        combined_evidence: dict[str, dict] = {}
        if candidate is not None:
            for item in candidate.get("evidence", []):
                if isinstance(item, dict) and item.get("event_id"):
                    combined_evidence[item["event_id"]] = item
        for item in evidence:
            combined_evidence[item["event_id"]] = item
        required_evidence = max(
            self._required_evidence_count(current, change["path"])
            for change in changes
        )
        verified_count = len(combined_evidence)

        if candidate is None:
            candidate_path = (
                self.settings.proposals_root
                / f"{operation_timestamp}_{agent_id}.json"
            )
            candidate = {
                "timestamp": operation_timestamp,
                "first_observed_at": operation_timestamp,
                "observation_count": 0,
            }
        candidate.update(
            {
                "status": "dormant",
                "agent_id": agent_id,
                "fingerprint": fingerprint,
                "reason": reason,
                "last_observed_at": operation_timestamp,
                "observation_count": int(candidate.get("observation_count", 0)) + 1,
                "evidence": list(combined_evidence.values()),
                "changes": changes,
                "governance": {
                    "required_evidence_events": required_evidence,
                    "verified_evidence_events": verified_count,
                    "eligible_for_user_incorporation": verified_count >= required_evidence,
                    "automatic_application": False,
                },
            }
        )
        assert candidate_path is not None
        atomic_write_text(
            candidate_path,
            json.dumps(candidate, indent=2, ensure_ascii=False) + "\n",
        )
        candidate_path.chmod(0o600)
        return candidate_path

    def _reconcile_dormant_candidates(
        self,
        agent_id: str,
        previous: dict,
        revised: dict,
        *,
        resolution_actor: str,
    ) -> None:
        reconciliation_timestamp = timestamp_id()
        for path in sorted(self.settings.proposals_root.glob("*.json")):
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if (
                candidate.get("agent_id") != agent_id
                or candidate.get("status") not in ACTIVE_CANDIDATE_STATUSES
            ):
                continue
            changes = [
                change
                for change in candidate.get("changes", [])
                if isinstance(change, dict) and change.get("path")
            ]
            if not changes:
                continue
            incorporated = all(
                self._change_is_present(revised, change)
                for change in changes
            )
            touched = any(
                self._get_path(previous, change["path"])[0]
                != self._get_path(revised, change["path"])[0]
                for change in changes
            )
            if not incorporated and not touched:
                continue
            candidate["status"] = (
                "incorporated"
                if incorporated
                else f"superseded_by_{resolution_actor}"
            )
            candidate["resolved_at"] = reconciliation_timestamp
            candidate["resolved_by"] = resolution_actor
            atomic_write_text(
                path,
                json.dumps(candidate, indent=2, ensure_ascii=False) + "\n",
            )
            path.chmod(0o600)

    @classmethod
    def _change_is_present(cls, persona: dict, change: dict) -> bool:
        existing, found = cls._get_path(persona, change["path"])
        if not found:
            return False
        if change.get("operation") == "append":
            return isinstance(existing, list) and change.get("value") in existing
        return existing == change.get("value")

    def clear_project_position(self, project_id: str) -> list[str]:
        """Remove a deleted project's structured pointer from active personas."""
        changed_agents: list[str] = []
        with self._lock:
            for agent_id in AGENT_IDS:
                persona = self.load_persona(agent_id)
                position = persona.get("current_position")
                if not isinstance(position, dict) or position.get("project_id") != project_id:
                    continue
                position["project_id"] = None
                position["stance"] = []
                position["active_questions"] = []
                position["recent_influences"] = []
                position["updated_at"] = datetime.now(UTC).isoformat()
                self._atomic_yaml_write(self._agent_file(agent_id), persona)
                changed_agents.append(agent_id)
        return changed_agents

    @staticmethod
    def _validate_persona(agent_id: str, data: dict) -> None:
        if data.get("agent_id") != agent_id:
            raise PersonaUpdateError("The persona's agent_id does not match the file being edited.")
        if int(data.get("schema_version", 0)) < PERSONA_SCHEMA_VERSION:
            raise PersonaUpdateError(
                f"Persona requires schema_version {PERSONA_SCHEMA_VERSION} or newer."
            )
        for key in (
            "display_name",
            "core_motif",
            "core_disposition",
            "systems_style",
            "attractors",
            "continuity_training",
        ):
            if key not in data:
                raise PersonaUpdateError(f"Persona is missing required field: {key}")
        if not isinstance(data.get("display_name"), str):
            raise PersonaUpdateError("Persona display_name must be text.")
        for key in (
            "core_disposition",
            "systems_style",
            "attractors",
            "continuity_training",
        ):
            if not isinstance(data.get(key), dict):
                raise PersonaUpdateError(f"Persona {key} must be a mapping.")
        motif = data.get("core_motif")
        if not isinstance(motif, dict) or not str(motif.get("name", "")).strip():
            raise PersonaUpdateError("Persona core_motif must have a name.")
        if not str(motif.get("statement", "")).strip():
            raise PersonaUpdateError("Persona core_motif must have a statement.")
        PersonaStore._validate_structured_persona_fields(data)

    @staticmethod
    def _validate_structured_persona_fields(data: dict) -> None:
        required_lists = {
            "motif_expression": (
                "current_form",
                "recent_perturbations",
                "retained_adaptations",
                "rejected_adaptations",
            ),
            "current_position": (
                "stance",
                "active_questions",
                "recent_influences",
            ),
            "self_model": (
                "recurring_strengths",
                "recurring_distortions",
                "developmental_notes",
            ),
        }
        for field, list_fields in required_lists.items():
            value = data.get(field)
            if value is None:
                continue
            if not isinstance(value, dict):
                raise PersonaUpdateError(f"Persona {field} must be a mapping.")
            missing = [key for key in list_fields if key not in value]
            if missing:
                raise PersonaUpdateError(
                    f"Persona {field} is missing required field: {missing[0]}"
                )
            if any(not isinstance(value[key], list) for key in list_fields):
                raise PersonaUpdateError(f"Persona {field} list fields must remain lists.")
        relationship_memory = data.get("relationship_memory")
        if relationship_memory is not None and not isinstance(relationship_memory, dict):
            raise PersonaUpdateError("Persona relationship_memory must be a mapping.")
        current_cycle = data.get("continuity_training", {}).get("current_cycle")
        if current_cycle is not None and not isinstance(current_cycle, dict):
            raise PersonaUpdateError(
                "Persona continuity_training.current_cycle must be a mapping."
            )

    @classmethod
    def _validate_change(cls, current: dict, change: dict) -> None:
        path = change["path"]
        value = change["value"]
        cls._validate_bounded_value(value)
        serialized = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(serialized) > PERSONA_CHANGE_MAX_BYTES:
            raise PersonaUpdateError(
                f"Persona change {path} exceeds the {PERSONA_CHANGE_MAX_BYTES}-byte limit."
            )

        existing, found = cls._get_path(current, path)
        if change["operation"] == "append":
            if not found or not isinstance(existing, list):
                raise PersonaUpdateError(f"{path} is not an existing list.")
            exemplar = next((item for item in existing if item is not None), None)
            if exemplar is not None and not cls._compatible_types(exemplar, value):
                raise PersonaUpdateError(
                    f"Persona change {path} does not match the existing list item type."
                )
            return
        if (
            found
            and existing is not None
            and value is not None
            and not cls._compatible_types(existing, value)
        ):
            raise PersonaUpdateError(
                f"Persona change {path} does not match the existing field type."
            )

    @classmethod
    def _validate_bounded_value(cls, value, depth: int = 0) -> None:
        if depth > PERSONA_CHANGE_MAX_DEPTH:
            raise PersonaUpdateError("Persona change nesting is too deep.")
        if value is None or isinstance(value, (bool, int)):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise PersonaUpdateError("Persona change numbers must be finite.")
            return
        if isinstance(value, str):
            if len(value) > PERSONA_CHANGE_MAX_STRING_CHARS:
                raise PersonaUpdateError(
                    "One persona change string exceeds the character limit."
                )
            return
        if isinstance(value, list):
            if len(value) > PERSONA_CHANGE_MAX_COLLECTION_ITEMS:
                raise PersonaUpdateError("Persona change list contains too many items.")
            for item in value:
                cls._validate_bounded_value(item, depth + 1)
            return
        if isinstance(value, dict):
            if len(value) > PERSONA_CHANGE_MAX_COLLECTION_ITEMS:
                raise PersonaUpdateError("Persona change mapping contains too many items.")
            for key, item in value.items():
                if not isinstance(key, str) or len(key) > 200:
                    raise PersonaUpdateError("Persona change mapping keys must be short text.")
                cls._validate_bounded_value(item, depth + 1)
            return
        raise PersonaUpdateError("Persona changes must contain JSON-compatible values.")

    @staticmethod
    def _get_path(data: dict, dotted_path: str) -> tuple[object, bool]:
        cursor: object = data
        for part in dotted_path.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                return None, False
            cursor = cursor[part]
        return cursor, True

    @staticmethod
    def _compatible_types(existing, revised) -> bool:
        if isinstance(existing, bool) or isinstance(revised, bool):
            return isinstance(existing, bool) and isinstance(revised, bool)
        if isinstance(existing, (int, float)) and isinstance(revised, (int, float)):
            return True
        return isinstance(revised, type(existing))

    @staticmethod
    def _path_allowed(path: str, prefixes: tuple[str, ...]) -> bool:
        return any(path == prefix or path.startswith(prefix + ".") for prefix in prefixes)

    @staticmethod
    def _set_path(data: dict, dotted_path: str, value, operation: str) -> None:
        parts = dotted_path.split(".")
        cursor = data
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        final = parts[-1]
        if operation == "replace":
            cursor[final] = value
            return
        if operation == "append":
            current = cursor.setdefault(final, [])
            if not isinstance(current, list):
                raise PersonaUpdateError(f"{dotted_path} is not a list.")
            current.append(value)
            return
        raise PersonaUpdateError("Unsupported update operation.")

    @staticmethod
    def _assert_core_motif_unchanged(current: dict, revised: dict) -> None:
        if current.get("core_motif") != revised.get("core_motif"):
            raise PersonaUpdateError("An agent update may not alter its core motif.")

    @staticmethod
    def _validate_attractor_delta(current: dict, path: str, new_value) -> None:
        if not path.startswith("attractors.") or not path.endswith(".strength"):
            return
        parts = path.split(".")
        if len(parts) != ATTRACTOR_STRENGTH_PATH_PARTS:
            raise PersonaUpdateError("Invalid attractor path.")
        current_strength = current.get("attractors", {}).get(parts[1], {}).get("strength")
        if current_strength is None:
            raise PersonaUpdateError("New attractors require manual review.")
        max_delta = float(
            current.get("update_policy", {}).get("max_attractor_delta_per_review", 0.05)
        )
        if abs(float(new_value) - float(current_strength)) > max_delta:
            raise PersonaUpdateError(
                f"Attractor change exceeds the allowed delta of {max_delta:.2f}."
            )

    def _snapshot(self, agent_id: str, label: str) -> None:
        history_dir = self.settings.history_root / agent_id
        history_dir.mkdir(parents=True, exist_ok=True)
        source = self._agent_file(agent_id)
        shutil.copy2(source, history_dir / f"{timestamp_id()}_{label}.yaml")

    def _append_log(self, agent_id: str, record: dict) -> None:
        history_dir = self.settings.history_root / agent_id
        history_dir.mkdir(parents=True, exist_ok=True)
        with (history_dir / "changes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _atomic_yaml_write(path: Path, data: dict) -> None:
        atomic_write_text(
            path,
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        )
