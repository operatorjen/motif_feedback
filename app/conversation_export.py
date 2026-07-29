from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from urllib.parse import urlsplit


def conversation_export_filename(project_id: str) -> str:
    return f"motif-feedback-{project_id}-conversation.md"


def conversation_markdown(
    *,
    project: dict,
    messages: Iterable[dict],
    agent_names: dict[str, str],
    user_display_name: str,
    exported_at: str | None = None,
) -> Iterator[str]:
    """Stream a complete chronological conversation as separated Markdown blocks."""
    exported = exported_at or datetime.now(UTC).isoformat()
    project_name = _single_line(project.get("name") or project.get("id") or "Project")
    yield (
        "# Motif Feedback Conversation Log\n\n"
        f"**Project:** {project_name}  \n"
        f"**Exported:** {exported}\n\n"
    )

    found_message = False
    for sequence, message in enumerate(messages, start=1):
        found_message = True
        if sequence > 1:
            yield "\n---\n\n"
        yield _message_block(
            sequence,
            message,
            agent_names=agent_names,
            user_display_name=user_display_name,
        )

    if not found_message:
        yield "_This room has no stored messages._\n"


def _message_block(
    sequence: int,
    message: dict,
    *,
    agent_names: dict[str, str],
    user_display_name: str,
) -> str:
    role = str(message.get("role") or "system")
    agent_id = str(message.get("agent_id") or "")
    if role == "user":
        speaker = user_display_name
    elif role == "runner":
        speaker = "Isolated Runner"
    elif role == "agent":
        speaker = agent_names.get(agent_id, agent_id or "Agent")
    else:
        speaker = role.replace("_", " ").title()

    metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
    turn_id = message.get("turn_id") or metadata.get("turn_id")
    turn_beat = metadata.get("turn_beat")
    details = [f"**Time:** {_single_line(message.get('created_at') or 'unknown')}"]
    if turn_id:
        details.append(f"**Turn:** `{_inline_code(turn_id)}`")
    if isinstance(turn_beat, int):
        details.append(f"**Beat:** {turn_beat}")

    content = str(message.get("content") or "").strip() or "_No written content._"
    sections = [
        f"## {sequence:04d} · {_single_line(speaker)}",
        "  \n".join(details),
        content,
    ]

    sources = _message_sources(message)
    if sources:
        source_lines = ["**Sources:**"]
        source_lines.extend(
            f"- {_single_line(source['title'])} — {source['url']}" for source in sources
        )
        sections.append("\n".join(source_lines))

    failures = metadata.get("web_source_failures")
    if isinstance(failures, list):
        failure_lines = [
            (
                f"- {_single_line(failure.get('url') or 'URL')} — "
                f"{_single_line(failure.get('detail') or 'Retrieval failed.')}"
            )
            for failure in failures
            if isinstance(failure, dict)
        ]
        if failure_lines:
            sections.append("**Retrieval notes:**\n" + "\n".join(failure_lines))

    return "\n\n".join(sections).rstrip() + "\n"


def _message_sources(message: dict) -> list[dict[str, str]]:
    metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
    candidates: list[tuple[object, object]] = []
    for annotation in message.get("annotations") or []:
        if not isinstance(annotation, dict) or annotation.get("type") != "url_citation":
            continue
        citation = annotation.get("url_citation")
        if isinstance(citation, dict):
            candidates.append((citation.get("url"), citation.get("title")))
    for snapshot in metadata.get("web_sources") or []:
        if isinstance(snapshot, dict):
            candidates.append(
                (
                    snapshot.get("final_url") or snapshot.get("requested_url"),
                    snapshot.get("title"),
                )
            )

    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_url, raw_title in candidates:
        url = _single_line(raw_url)
        if not _is_web_url(url) or url in seen:
            continue
        seen.add(url)
        sources.append({"url": url, "title": _single_line(raw_title) or url})
    return sources


def _is_web_url(url: str) -> bool:
    try:
        return urlsplit(url).scheme in {"http", "https"}
    except ValueError:
        return False


def _single_line(value: object) -> str:
    return " ".join(str(value or "").split())


def _inline_code(value: object) -> str:
    return _single_line(value).replace("`", "ˋ")
