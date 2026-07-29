import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app import main as main_module
from app.conversation_export import conversation_markdown
from app.storage import Storage


def test_complete_conversation_export_preserves_order_and_message_blocks(tmp_path):
    storage = Storage(tmp_path / "state" / "motif.db", tmp_path / "projects")
    storage.initialize()
    project = storage.create_project("Order of play")
    storage.add_message(
        project["id"],
        "user",
        "First move.",
        metadata={
            "turn_id": "turn-1",
            "web_source_failures": [
                {
                    "url": "https://example.com/blocked",
                    "detail": "The page returned HTTP 403.",
                }
            ],
        },
    )
    storage.add_message(
        project["id"],
        "agent",
        "Second move.",
        agent_id="agent_a",
        annotations=[
            {
                "type": "url_citation",
                "url_citation": {
                    "url": "https://example.com/source",
                    "title": "Example source",
                },
            },
            {
                "type": "url_citation",
                "url_citation": {
                    "url": "http://[malformed",
                    "title": "Malformed source",
                },
            },
        ],
        metadata={"turn_id": "turn-1", "turn_beat": 1},
    )
    storage.add_message(project["id"], "runner", "Third move.")

    assert [
        message["content"] for message in storage.list_messages(project["id"], limit=2)
    ] == ["Second move.", "Third move."]
    all_messages = storage.iter_conversation_messages(project["id"], batch_size=1)
    exported = "".join(
        conversation_markdown(
            project=project,
            messages=all_messages,
            agent_names={"agent_a": "The Phenomenologist"},
            user_display_name="User",
            exported_at="2026-07-29T12:00:00+00:00",
        )
    )

    assert exported.index("First move.") < exported.index("Second move.") < exported.index(
        "Third move."
    )
    assert "## 0001 · User" in exported
    assert "## 0002 · The Phenomenologist" in exported
    assert "## 0003 · Isolated Runner" in exported
    assert "**Turn:** `turn-1`" in exported
    assert "**Beat:** 1" in exported
    assert "Example source — https://example.com/source" in exported
    assert "Malformed source" not in exported
    assert "https://example.com/blocked — The page returned HTTP 403." in exported
    assert exported.count("\n---\n") == 2


def test_conversation_download_endpoint_streams_markdown_attachment(tmp_path, monkeypatch):
    storage = Storage(tmp_path / "state" / "motif.db", tmp_path / "projects")
    storage.initialize()
    project = storage.create_project("Export endpoint")
    storage.add_message(project["id"], "user", "Download me.")
    monkeypatch.setattr(main_module, "storage", storage)
    monkeypatch.setattr(
        main_module,
        "persona_store",
        SimpleNamespace(
            list_summaries=lambda: [
                {"agent_id": "agent_a", "display_name": "The Phenomenologist"}
            ]
        ),
    )

    response = main_module.download_project_conversation(project["id"])

    async def consume() -> str:
        parts = []
        async for chunk in response.body_iterator:
            parts.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
        return "".join(parts)

    exported = asyncio.run(consume())

    assert response.media_type == "text/markdown"
    assert response.headers["content-disposition"] == (
        f'attachment; filename="motif-feedback-{project["id"]}-conversation.md"'
    )
    assert "Download me." in exported


def test_conversation_download_rejects_unknown_project(tmp_path, monkeypatch):
    storage = Storage(tmp_path / "state" / "motif.db", tmp_path / "projects")
    storage.initialize()
    monkeypatch.setattr(main_module, "storage", storage)

    with pytest.raises(HTTPException) as exc_info:
        main_module.download_project_conversation("missing")

    assert exc_info.value.status_code == 404


def test_download_log_control_targets_complete_project_export():
    root = main_module.STATIC_ROOT
    html = (root / "index.html").read_text(encoding="utf-8")
    javascript = (root / "js" / "app.js").read_text(encoding="utf-8")

    assert 'id="download-log"' in html
    assert "/conversation.md" in javascript
    assert "Conversation log download started." in javascript
