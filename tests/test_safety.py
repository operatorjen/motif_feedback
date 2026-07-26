import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from app.config import RuntimeNotConfiguredError, Settings
from app.file_tools import FileToolError, ProjectFileTools
from app.models import RuntimeConfig
from app.persona_store import PersonaStore, PersonaUpdateError
from app.search_router import SearchRouter
from app.storage import Storage


def make_services(tmp_path: Path):
    settings = Settings(WORKSPACE_ROOT=tmp_path)
    storage = Storage(settings.database_path, settings.projects_root)
    personas = PersonaStore(settings)
    personas.initialize()
    storage.initialize()
    tools = ProjectFileTools(storage, max_write_bytes=15000, max_upload_bytes=25000)
    return settings, storage, personas, tools


def test_agent_file_limit_cannot_be_configured_above_fifteen_kb(tmp_path: Path):
    settings = Settings(WORKSPACE_ROOT=tmp_path, MAX_AGENT_WRITE_BYTES=50000)
    assert settings.agent_file_byte_limit == 15000


def test_operational_configuration_defaults_preserve_existing_behavior(tmp_path: Path):
    settings = Settings(_env_file=None, WORKSPACE_ROOT=tmp_path)

    assert settings.provider_connect_timeout_seconds == 20
    assert settings.user_display_name == "User"
    assert settings.provider_participation_retries == 2
    assert settings.provider_response_max_bytes == 2_000_000
    assert settings.provider_tool_calls_per_response == 16
    assert settings.provider_tool_calls_per_turn == 48
    assert settings.max_agent_turn_beats == 3
    assert settings.local_memory_context_events == 8
    assert settings.global_memory_context_events == 6
    assert settings.web_fetch_connect_timeout_seconds == 10
    assert settings.web_fetch_chunk_bytes == 65_536
    assert settings.runner_timeout_seconds == 65
    assert settings.runner_project_max_bytes == 8_000_000
    assert settings.runner_room_transcript_max_chars == 8_000
    assert settings.runner_input_max_bytes == 16_000
    assert settings.runner_input_message_max_bytes == 4_000
    assert settings.runner_argument_max_count == 24
    assert settings.runner_argument_max_chars == 500
    assert settings.runner_arguments_max_bytes == 4_000


def test_operational_configuration_accepts_environment_style_overrides(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        WORKSPACE_ROOT=tmp_path,
        USER_DISPLAY_NAME="Open Source User",
        PROVIDER_PARTICIPATION_RETRIES=4,
        MAX_AGENT_TURN_BEATS=2,
        RUNNER_INPUT_MAX_BYTES=24_000,
        RUNNER_INPUT_MESSAGE_MAX_BYTES=6_000,
    )

    assert settings.provider_participation_retries == 4
    assert settings.user_display_name == "Open Source User"
    assert settings.max_agent_turn_beats == 2
    assert settings.runner_input_max_bytes == 24_000
    assert settings.runner_input_message_max_bytes == 6_000


def test_provider_status_includes_supported_non_default_provider(tmp_path: Path):
    settings = Settings(
        WORKSPACE_ROOT=tmp_path,
        MOONSHOT_API_KEY="moonshot-test-key",
        GEMINI_API_KEY="gemini-test-key",
        DEEPSEEK_API_KEY="deepseek-test-key",
        OPENAI_API_KEY="openai-test-key",
    )

    assert settings.provider_status() == {
        "moonshot": True,
        "gemini": True,
        "deepseek": True,
        "openai": True,
    }


def test_runtime_configuration_stays_absent_until_setup_is_saved(tmp_path: Path):
    settings = Settings(_env_file=None, WORKSPACE_ROOT=tmp_path)

    assert settings.load_runtime_config() is None
    assert not settings.runtime_config_path.exists()
    with pytest.raises(RuntimeNotConfiguredError, match="Setup is incomplete"):
        settings.require_runtime_config()


def test_runtime_configuration_preserves_existing_custom_selections(tmp_path: Path):
    settings = Settings(_env_file=None, WORKSPACE_ROOT=tmp_path)
    configured = RuntimeConfig(
        providers={
            "agent_a": "local-a",
            "agent_b": "local-b",
            "agent_c": "local-c",
        },
        models={
            "agent_a": "model-a",
            "agent_b": "model-b",
            "agent_c": "model-c",
        },
    )
    settings.save_runtime_config(configured)

    assert settings.load_runtime_config() == configured
    assert settings.require_runtime_config() == configured


def test_runtime_configuration_rejects_blank_models_and_invalid_providers():
    with pytest.raises(ValidationError, match="model ID is required"):
        RuntimeConfig(
            providers={
                "agent_a": "local",
                "agent_b": "local",
                "agent_c": "local",
            },
            models={
                "agent_a": "",
                "agent_b": "model-b",
                "agent_c": "model-c",
            },
        )

    with pytest.raises(ValidationError, match="valid provider ID is required"):
        RuntimeConfig(
            providers={
                "agent_a": "Not A Provider",
                "agent_b": "local",
                "agent_c": "local",
            },
            models={
                "agent_a": "model-a",
                "agent_b": "model-b",
                "agent_c": "model-c",
            },
        )


def test_runtime_configuration_migrates_legacy_prefixed_models(tmp_path: Path):
    settings = Settings(_env_file=None, WORKSPACE_ROOT=tmp_path)
    settings.state_root.mkdir(parents=True)
    settings.runtime_config_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "agent_a": "google/model-a",
                    "agent_b": "openai/model-b",
                    "agent_c": "deepseek/model-c",
                }
            }
        ),
        encoding="utf-8",
    )

    runtime = settings.load_runtime_config()

    assert runtime.providers == {
        "agent_a": "gemini",
        "agent_b": "openai",
        "agent_c": "deepseek",
    }
    assert runtime.models == {
        "agent_a": "model-a",
        "agent_b": "model-b",
        "agent_c": "model-c",
    }


def test_message_limit_returns_the_newest_messages_in_chat_order(tmp_path: Path):
    _, storage, _, _ = make_services(tmp_path)
    project = storage.create_project("Long conversation")
    for index in range(5):
        storage.add_message(project["id"], "runner", f"run-{index}")

    messages = storage.list_messages(project["id"], limit=3)

    assert [message["content"] for message in messages] == ["run-2", "run-3", "run-4"]


def test_project_files_cannot_escape_workspace(tmp_path: Path):
    _, storage, _, tools = make_services(tmp_path)
    project = storage.create_project("Safety")
    with pytest.raises(FileToolError):
        tools.write_file(project["id"], "../../outside.txt", "no")
    with pytest.raises(FileToolError):
        tools.write_file(project["id"], "/tmp/outside.txt", "no")


def test_agent_can_edit_own_file_without_turn_permission(tmp_path: Path):
    _, storage, _, tools = make_services(tmp_path)
    project = storage.create_project("Overwrite")
    tools.write_file(
        project["id"], "note.md", "first", actor_type="agent", actor_id="agent_a"
    )
    result = tools.write_file(
        project["id"], "note.md", "second", actor_type="agent", actor_id="agent_a"
    )
    assert result["overwritten"] is True
    assert tools.read_file(project["id"], "note.md")["content"] == "second"


def test_agent_file_limit_requires_consolidation(tmp_path: Path):
    _, storage, _, tools = make_services(tmp_path)
    project = storage.create_project("Bounded journal")
    tools.write_file(
        project["id"], "journal.md", "durable observation", actor_type="agent", actor_id="agent_a"
    )

    with pytest.raises(FileToolError) as failure:
        tools.write_file(
            project["id"], "journal.md", "x" * 15001,
            actor_type="agent", actor_id="agent_a",
        )

    assert failure.value.code == "agent_file_size_limit"
    assert failure.value.retryable is True
    assert "reframe" in str(failure.value).lower()
    assert tools.read_file(project["id"], "journal.md")["content"] == "durable observation"


def test_user_upload_keeps_its_separate_larger_limit(tmp_path: Path):
    _, storage, _, tools = make_services(tmp_path)
    project = storage.create_project("Large user source")

    result = tools.save_upload(project["id"], "source.txt", b"u" * 15000)

    assert result["bytes_written"] == 15000
    assert result["owner_type"] == "user"


def test_uploaded_raster_image_persists_and_has_preview_metadata(tmp_path: Path):
    settings, storage, _, tools = make_services(tmp_path)
    project = storage.create_project("Image artifacts")
    png = b"\x89PNG\r\n\x1a\n" + b"project-image"

    result = tools.save_upload(project["id"], "diagram.png", png)
    reloaded_storage = Storage(settings.database_path, settings.projects_root)
    reloaded_storage.initialize()
    reloaded_tools = ProjectFileTools(
        reloaded_storage, max_write_bytes=8000, max_upload_bytes=20000
    )
    files = reloaded_tools.list_files(project["id"])
    preview_path, media_type = reloaded_tools.preview_path(project["id"], "diagram.png")

    assert result["kind"] == "image"
    assert files[0]["path"] == "diagram.png"
    assert files[0]["kind"] == "image"
    assert preview_path.read_bytes() == png
    assert media_type == "image/png"


def test_agent_can_create_safe_svg_but_not_active_svg(tmp_path: Path):
    _, storage, _, tools = make_services(tmp_path)
    project = storage.create_project("Generated graphics")
    safe = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><circle cx="5" cy="5" r="4" fill="#f80"/></svg>'

    result = tools.write_file(
        project["id"], "motif.svg", safe, actor_type="agent", actor_id="agent_a"
    )
    _, media_type = tools.preview_path(project["id"], "motif.svg")

    assert result["path"] == "motif.svg"
    assert tools.list_files(project["id"])[0]["kind"] == "image"
    assert media_type == "image/svg+xml"
    with pytest.raises(FileToolError):
        tools.write_file(
            project["id"], "active.svg", '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
            actor_type="agent", actor_id="agent_a",
        )


def test_code_files_are_labeled_for_editor_preview(tmp_path: Path):
    _, storage, _, tools = make_services(tmp_path)
    project = storage.create_project("Code artifacts")
    tools.write_file(
        project["id"], "sample.py", "def example():\n    return 1",
        actor_type="agent", actor_id="agent_b",
    )

    assert tools.list_files(project["id"])[0]["kind"] == "code"
    assert tools.download_path(project["id"], "sample.py").read_text(encoding="utf-8").startswith("def example")


def test_runtime_artifacts_are_hidden_from_project_file_list(tmp_path: Path):
    _, storage, _, tools = make_services(tmp_path)
    project = storage.create_project("Runtime artifacts")
    tools.write_file(project["id"], "sample.py", "value = 1")
    cache = storage.projects_root / project["id"] / "__pycache__"
    cache.mkdir()
    (cache / "sample.cpython-313.pyc").write_bytes(b"generated bytecode")
    (storage.projects_root / project["id"] / ".DS_Store").write_bytes(b"finder metadata")

    assert [item["path"] for item in tools.list_files(project["id"])] == ["sample.py"]


def test_download_path_remains_confined_to_the_project(tmp_path: Path):
    _, storage, _, tools = make_services(tmp_path)
    project = storage.create_project("Downloads")
    tools.write_file(project["id"], "artifact.txt", "download me")

    assert tools.download_path(project["id"], "artifact.txt").name == "artifact.txt"
    with pytest.raises(FileToolError):
        tools.download_path(project["id"], "../motif.db")


def test_agent_cannot_edit_another_owners_file(tmp_path: Path):
    _, storage, _, tools = make_services(tmp_path)
    project = storage.create_project("Ownership")
    tools.write_file(
        project["id"], "agent-a.md", "first", actor_type="agent", actor_id="agent_a"
    )
    with pytest.raises(FileToolError):
        tools.write_file(
            project["id"], "agent-a.md", "changed", actor_type="agent", actor_id="agent_b"
        )
    tools.write_file(project["id"], "user.md", "uploaded")
    with pytest.raises(FileToolError):
        tools.write_file(
            project["id"], "user.md", "changed", actor_type="agent", actor_id="agent_a"
        )


def test_user_can_share_one_agent_file_without_transferring_ownership(tmp_path: Path):
    _, storage, _, tools = make_services(tmp_path)
    project = storage.create_project("Shared artifact")
    tools.write_file(
        project["id"], "shared.md", "created by A",
        actor_type="agent", actor_id="agent_a",
    )
    tools.write_file(
        project["id"], "private.md", "also created by A",
        actor_type="agent", actor_id="agent_a",
    )

    permission = tools.set_agent_sharing(project["id"], "shared.md", True)
    result = tools.write_file(
        project["id"], "shared.md", "revised by B",
        actor_type="agent", actor_id="agent_b",
    )

    assert permission["shared_agent_edit"] is True
    assert result["owner_id"] == "agent_a"
    assert result["shared_agent_edit"] is True
    assert storage.get_file_owner(project["id"], "shared.md")["owner_id"] == "agent_a"
    assert tools.read_file(project["id"], "shared.md")["content"] == "revised by B"
    with pytest.raises(FileToolError):
        tools.write_file(
            project["id"], "private.md", "B should not edit this",
            actor_type="agent", actor_id="agent_b",
        )


def test_revoking_shared_editing_restores_creator_only_rule(tmp_path: Path):
    _, storage, _, tools = make_services(tmp_path)
    project = storage.create_project("Revoked sharing")
    tools.write_file(
        project["id"], "draft.md", "owner copy",
        actor_type="agent", actor_id="agent_a",
    )
    tools.set_agent_sharing(project["id"], "draft.md", True)
    tools.set_agent_sharing(project["id"], "draft.md", False)

    with pytest.raises(FileToolError):
        tools.write_file(
            project["id"], "draft.md", "unauthorized revision",
            actor_type="agent", actor_id="agent_c",
        )
    assert storage.get_file_owner(project["id"], "draft.md")["shared_agent_edit"] == 0


def test_user_uploaded_file_cannot_be_shared_between_agents(tmp_path: Path):
    _, storage, _, tools = make_services(tmp_path)
    project = storage.create_project("User source")
    tools.write_file(project["id"], "source.md", "User source")

    with pytest.raises(FileToolError):
        tools.set_agent_sharing(project["id"], "source.md", True)


def test_user_can_delete_any_project_file(tmp_path: Path):
    _, storage, _, tools = make_services(tmp_path)
    project = storage.create_project("Removal")
    tools.write_file(
        project["id"], "agent-note.md", "content", actor_type="agent", actor_id="agent_a"
    )
    result = tools.delete_file(project["id"], "agent-note.md")
    assert result["deleted"] is True
    assert not (storage.projects_root / project["id"] / "agent-note.md").exists()
    assert storage.get_file_owner(project["id"], "agent-note.md") is None


def test_user_can_delete_confined_file_with_unlisted_extension(tmp_path: Path):
    _, storage, _, tools = make_services(tmp_path)
    project = storage.create_project("Generated artifact removal")
    cache = storage.projects_root / project["id"] / "__pycache__"
    cache.mkdir()
    artifact = cache / "module.cpython-313.pyc"
    artifact.write_bytes(b"generated bytecode")

    result = tools.delete_file(project["id"], "__pycache__/module.cpython-313.pyc")

    assert result["deleted"] is True
    assert not artifact.exists()
    assert not cache.exists()


def test_project_deletion_purges_files_conversations_sources_and_memory(tmp_path: Path):
    _, storage, personas, tools = make_services(tmp_path)
    target = storage.create_project("Delete all traces")
    survivor = storage.create_project("Keep this project")

    message = storage.add_message(target["id"], "user", "private project conversation")
    storage.add_message(survivor["id"], "user", "surviving conversation")
    tools.write_file(target["id"], "artifact.md", "private project file")
    local_memory = storage.add_memory_event(
        target["id"],
        "agent_a",
        message["id"],
        outcome="response",
        trigger_text="private trigger",
        return_text="private return",
        actions=[],
        provider="gemini",
        model="gemini-test",
    )
    storage.add_global_memory_event(
        agent_id="agent_a",
        source_project_id=target["id"],
        source_project_name=target["name"],
        source_memory_event_id=local_memory["id"],
        trigger_text=local_memory["trigger_text"],
        return_text=local_memory["return_text"],
        actions=[],
    )
    storage.add_web_source(
        target["id"],
        requested_url="https://example.com/private",
        final_url="https://example.com/private",
        title="Private source",
        content_text="private source snapshot",
        content_type="text/html",
        byte_count=23,
        truncated=False,
        content_sha256="a" * 64,
    )

    persona = personas.load_persona("agent_a")
    persona["current_position"]["project_id"] = target["id"]
    PersonaStore._atomic_yaml_write(personas._agent_file("agent_a"), persona)

    result = storage.delete_project(target["id"])
    cleared = personas.clear_project_position(target["id"])

    assert result["deleted"] is True
    assert result["deleted_records"] == {
        "messages": 1,
        "files": 1,
        "memory_events": 1,
        "global_memory_events": 1,
        "web_sources": 1,
    }
    assert not (storage.projects_root / target["id"]).exists()
    assert target["id"] not in {project["id"] for project in storage.list_projects()}
    assert storage.list_messages(survivor["id"])[0]["content"] == "surviving conversation"
    assert storage.list_global_memory_events("agent_a") == []
    assert cleared == ["agent_a"]
    assert personas.load_persona("agent_a")["current_position"]["project_id"] is None

    with storage.connection() as connection:
        for table, column in (
            ("messages", "project_id"),
            ("file_ownership", "project_id"),
            ("agent_memory_events", "project_id"),
            ("agent_global_memory_events", "source_project_id"),
            ("web_sources", "project_id"),
        ):
            count = connection.execute(
                f"SELECT COUNT(*) AS count FROM {table} WHERE {column} = ?",
                (target["id"],),
            ).fetchone()["count"]
            assert count == 0


def test_agent_memory_loops_are_private_and_sequenced(tmp_path: Path):
    _, storage, _, _ = make_services(tmp_path)
    project = storage.create_project("Memory")
    storage.add_memory_event(
        project["id"], "agent_a", "user-1", outcome="response",
        trigger_text="first", return_text="embodied return", actions=[],
        provider="gemini", model="gemini-test",
    )
    storage.add_memory_event(
        project["id"], "agent_b", "user-1", outcome="action_response",
        trigger_text="first", return_text="feedback return",
        actions=[{"tool": "read_project_file", "path": "note.md", "ok": True}],
        provider="deepseek", model="deepseek-test",
    )
    storage.add_memory_event(
        project["id"], "agent_a", "user-2", outcome="response",
        trigger_text="second", return_text="second embodied return", actions=[],
        provider="gemini", model="gemini-test",
    )
    storage.add_memory_event(
        project["id"], "agent_b", "user-2", outcome="provider_error",
        trigger_text="second", return_text="provider unavailable", actions=[],
        provider="deepseek", model="deepseek-test",
    )
    agent_a = storage.list_memory_events(project["id"], "agent_a")
    assert [event["sequence"] for event in agent_a] == [2, 1]
    assert all(event["agent_id"] == "agent_a" for event in agent_a)
    assert storage.memory_stats(project["id"])["agent_b"]["action_count"] == 1
    assert storage.memory_stats(project["id"])["agent_b"]["failure_count"] == 1


def test_project_scoped_reads_validate_and_query_with_one_connection(tmp_path: Path):
    _, storage, _, _ = make_services(tmp_path)
    project = storage.create_project("Efficient reads")
    storage.add_message(project["id"], "user", "hello")
    storage.add_memory_event(
        project["id"],
        "agent_a",
        "user-1",
        outcome="response",
        trigger_text="hello",
        return_text="return",
        actions=[],
        provider="gemini",
        model="gemini-test",
    )

    for read in (
        lambda: storage.list_messages(project["id"]),
        lambda: storage.recent_messages(project["id"], 30),
        lambda: storage.list_memory_events(project["id"], "agent_a"),
        lambda: storage.memory_stats(project["id"]),
    ):
        with patch.object(storage, "connection", wraps=storage.connection) as connection:
            assert read()
            assert connection.call_count == 1


def test_cross_project_memory_is_compact_provisional_and_source_labeled(tmp_path: Path):
    _, storage, _, _ = make_services(tmp_path)
    first = storage.create_project("First context")
    second = storage.create_project("Second context")
    local = storage.add_memory_event(
        first["id"], "agent_a", "user-1", outcome="response",
        trigger_text="trigger " * 100, return_text="begin " + ("return " * 200) + "end",
        actions=[{"tool": "read_project_file", "path": "note.md", "ok": True}],
        provider="gemini", model="gemini-test",
    )
    carried = storage.add_global_memory_event(
        agent_id="agent_a",
        source_project_id=first["id"],
        source_project_name=first["name"],
        source_memory_event_id=local["id"],
        trigger_text=local["trigger_text"],
        return_text=local["return_text"],
        actions=local["actions"],
        created_at=local["created_at"],
    )

    assert carried["source_project_name"] == "First context"
    assert len(carried["trigger_summary"]) <= 240
    assert len(carried["return_summary"]) <= 800
    assert carried["return_summary"].endswith("end")
    assert storage.list_global_memory_events(
        "agent_a", exclude_project_id=first["id"]
    ) == []
    visible_elsewhere = storage.list_global_memory_events(
        "agent_a", exclude_project_id=second["id"]
    )
    assert [event["source_project_id"] for event in visible_elsewhere] == [first["id"]]
    assert storage.global_memory_stats("agent_a")["project_count"] == 1
    assert storage.global_memory_stats(
        "agent_a", exclude_project_id=first["id"]
    )["event_count"] == 0


def test_agent_cannot_edit_another_persona(tmp_path: Path):
    _, _, personas, _ = make_services(tmp_path)
    with pytest.raises(PersonaUpdateError):
        personas.submit_update(
            {
                "agent_id": "agent_a",
                "reason": "invalid cross-agent edit",
                "evidence": [{"event_id": "e1", "summary": "test"}],
                "changes": [
                    {
                        "path": "agents.agent_b.core_disposition.summary",
                        "operation": "replace",
                        "value": "changed",
                    }
                ],
            }
        )


def test_agent_cannot_edit_core_motif(tmp_path: Path):
    _, _, personas, _ = make_services(tmp_path)
    before = personas.load_persona("agent_a")["core_motif"]
    with pytest.raises(PersonaUpdateError):
        personas.submit_update(
            {
                "agent_id": "agent_a",
                "reason": "attempted constitutional change",
                "evidence": [{"event_id": "e1", "summary": "test"}],
                "changes": [
                    {
                        "path": "core_motif.statement",
                        "operation": "replace",
                        "value": "new motif",
                    }
                ],
            }
        )
    assert personas.load_persona("agent_a")["core_motif"] == before


def test_agent_can_update_motif_expression(tmp_path: Path):
    _, _, personas, _ = make_services(tmp_path)
    result = personas.submit_update(
        {
            "agent_id": "agent_a",
            "reason": "durable return signal",
            "evidence": [{"event_id": "e1", "summary": "The user clarified observer position."}],
            "changes": [
                {
                    "path": "motif_expression.retained_adaptations",
                    "operation": "append",
                    "value": "Name the observer position before generalizing.",
                }
            ],
        }
    )
    assert result["committed_change_count"] == 1
    persona = personas.load_persona("agent_a")
    assert "Name the observer position" in persona["motif_expression"]["retained_adaptations"][0]


def test_agent_persona_update_requires_evidence(tmp_path: Path):
    _, _, personas, _ = make_services(tmp_path)

    with pytest.raises(ValidationError, match="evidence"):
        personas.submit_update(
            {
                "agent_id": "agent_a",
                "reason": "unsupported update",
                "evidence": [],
                "changes": [
                    {
                        "path": "motif_expression.retained_adaptations",
                        "operation": "append",
                        "value": "This should not be committed.",
                    }
                ],
            }
        )


def test_reflection_contract_uses_the_shared_seed_layout(tmp_path: Path):
    settings, _, personas, _ = make_services(tmp_path)

    assert (settings.seed_root / "shared" / "reflection_prompt.md").is_file()
    assert not (settings.seed_root / "reflection_prompt.md").exists()
    assert "Contract version: 5" in personas.load_reflection_contract()
    assert "Every change must cite one or more event IDs." in (
        personas.load_reflection_contract()
    )


def test_outdated_reflection_contract_is_snapshotted_and_migrated(tmp_path: Path):
    settings = Settings(WORKSPACE_ROOT=tmp_path)
    settings.shared_root.mkdir(parents=True, exist_ok=True)
    old_contract = "# Old reflection contract\n\n**Contract version: 4**\n"
    (settings.shared_root / "reflection_prompt.md").write_text(
        old_contract,
        encoding="utf-8",
    )

    personas = PersonaStore(settings)
    personas.initialize()

    assert "Contract version: 5" in personas.load_reflection_contract()
    snapshots = list(
        (settings.history_root / "migrations").glob("*_reflection_prompt.md")
    )
    assert len(snapshots) == 1
    assert snapshots[0].read_text(encoding="utf-8") == old_contract


def test_agent_persona_update_rejects_incompatible_field_shape(tmp_path: Path):
    _, _, personas, _ = make_services(tmp_path)

    with pytest.raises(PersonaUpdateError, match="field type"):
        personas.submit_update(
            {
                "agent_id": "agent_a",
                "reason": "malformed model output",
                "evidence": [{"event_id": "e1", "summary": "test"}],
                "changes": [
                    {
                        "path": "current_position.stance",
                        "operation": "replace",
                        "value": {"unexpected": "mapping"},
                    }
                ],
            }
        )

    assert isinstance(
        personas.load_persona("agent_a")["current_position"]["stance"],
        list,
    )


def test_agent_persona_update_rejects_oversized_nested_value(tmp_path: Path):
    _, _, personas, _ = make_services(tmp_path)

    with pytest.raises(PersonaUpdateError, match="character limit"):
        personas.submit_update(
            {
                "agent_id": "agent_a",
                "reason": "oversized model output",
                "evidence": [{"event_id": "e1", "summary": "test"}],
                "changes": [
                    {
                        "path": "relationship_memory.user.observations",
                        "operation": "append",
                        "value": "x" * 4_001,
                    }
                ],
            }
        )


def test_storage_backfills_are_recorded_as_one_time_migrations(tmp_path: Path):
    settings, storage, _, _ = make_services(tmp_path)
    with storage.connection() as connection:
        names = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM schema_migrations"
            ).fetchall()
        }
    assert names == {
        "backfill_file_ownership_v1",
        "backfill_memory_events_v1",
        "backfill_global_memory_events_v1",
        "sanitize_tool_event_metadata_v1",
    }

    reloaded = Storage(settings.database_path, settings.projects_root)

    def should_not_run():
        raise AssertionError("A completed startup migration ran again.")

    reloaded._backfill_file_ownership = should_not_run
    reloaded._backfill_memory_events = should_not_run
    reloaded._backfill_global_memory_events = should_not_run
    reloaded._sanitize_tool_event_metadata = should_not_run
    reloaded.initialize()


def test_storage_scrubs_generated_bodies_from_historical_tool_metadata(tmp_path: Path):
    settings, storage, _, _ = make_services(tmp_path)
    message = storage.add_message(
        "general",
        "agent",
        "done",
        agent_id="agent_a",
        metadata={
            "tool_events": [
                {
                    "tool": "write_project_file",
                    "arguments": {
                        "path": "private.md",
                        "content": "historical generated body",
                    },
                    "result": {"ok": True, "path": "private.md"},
                }
            ]
        },
    )
    with storage.connection() as connection:
        connection.execute(
            "DELETE FROM schema_migrations WHERE name = ?",
            ("sanitize_tool_event_metadata_v1",),
        )

    Storage(settings.database_path, settings.projects_root).initialize()

    with storage.connection() as connection:
        row = connection.execute(
            "SELECT metadata_json FROM messages WHERE id = ?",
            (message["id"],),
        ).fetchone()
    arguments = json.loads(row["metadata_json"])["tool_events"][0]["arguments"]
    assert arguments == {
        "path": "private.md",
        "content_bytes": len(b"historical generated body"),
    }


def test_shared_context_is_installed(tmp_path: Path):
    _, _, personas, _ = make_services(tmp_path)
    context = personas.load_shared_context()
    assert "The ⟁ Project" in context
    assert "The Phenomenologist" in context


def test_search_router_selects_cyberneticist_for_current_api_docs():
    decision = SearchRouter().decide(
        "Find the latest API documentation and current Docker version.",
        "auto",
        ["agent_a", "agent_b", "agent_c"],
    )
    assert decision.needs_search is True
    assert decision.scope == "lead"
    assert decision.lead_agent == "agent_b"


def test_search_router_selects_phenomenologist_for_embodied_research():
    decision = SearchRouter().decide(
        "Find recent research on interoception and embodied aesthetic experience.",
        "auto",
        ["agent_a", "agent_b", "agent_c"],
    )
    assert decision.lead_agent == "agent_a"


def test_search_router_selects_game_theorist_for_rules_and_policy():
    decision = SearchRouter().decide(
        "Find the current platform rules and policy incentives shaping player strategy.",
        "auto",
        ["agent_a", "agent_b", "agent_c"],
    )
    assert decision.lead_agent == "agent_c"


def test_search_router_escalates_deep_research():
    decision = SearchRouter().decide(
        "Do deep research and compare multiple sources on this policy.",
        "auto",
        ["agent_a", "agent_b", "agent_c"],
    )
    assert decision.needs_search is True
    assert decision.scope == "all"


def test_outdated_persona_schema_is_rejected(tmp_path: Path):
    settings = Settings(WORKSPACE_ROOT=tmp_path)
    settings.personas_root.mkdir(parents=True, exist_ok=True)
    outdated = {
        "agent_id": "agent_a",
        "display_name": "Outdated A",
        "schema_version": 3,
        "version": 2,
        "core_disposition": {"summary": "outdated"},
        "systems_style": {"orientation": "outdated"},
        "attractors": {},
        "continuity_training": {},
    }
    (settings.personas_root / "agent_a.yaml").write_text(
        yaml.safe_dump(outdated, sort_keys=False), encoding="utf-8"
    )
    personas = PersonaStore(settings)
    with pytest.raises(PersonaUpdateError, match="schema_version 4"):
        personas.initialize()
