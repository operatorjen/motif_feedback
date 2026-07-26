from __future__ import annotations

import json
import queue
import re
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_RUNNER_CLIENT_TIMEOUT_SECONDS,
    DEFAULT_RUNNER_REQUEST_MAX_BYTES,
    DEFAULT_RUNNER_RESPONSE_MAX_BYTES,
    DEFAULT_RUNNER_SOCKET_POLL_SECONDS,
    DEFAULT_RUNNER_SOCKET_READ_BYTES,
)


class CodeRunnerError(RuntimeError):
    pass


ROLE_SIGNAL_LINE = re.compile(
    r"^__MOTIF_ROLE_SIGNAL__.*(?:\r?\n|$)",
    flags=re.MULTILINE,
)


def format_interactive_transcript(
    events: list[dict[str, Any]],
    *,
    user_label: str = "User",
) -> tuple[str, int]:
    """Render the server-owned stream/input event sequence for room history."""
    parts: list[str] = []
    input_count = 0
    for event in events:
        event_type = event.get("type")
        text = str(event.get("text", ""))
        if event_type == "stdin":
            input_count += 1
            if parts and not parts[-1].endswith(("\n", "\r")):
                parts.append("\n")
            visible_lines = text.rstrip("\r\n").splitlines() or [""]
            parts.append(f"[{user_label.upper()} INPUT]\n")
            parts.extend(f"› {line}\n" for line in visible_lines)
            continue
        if event_type not in {"stdout", "stderr"} or not text:
            continue
        visible_text = ROLE_SIGNAL_LINE.sub("", text)
        if not visible_text:
            continue
        if event_type == "stderr":
            if parts and not parts[-1].endswith(("\n", "\r")):
                parts.append("\n")
            parts.append("[STDERR]\n")
        parts.append(visible_text)
    return "".join(parts), input_count


class CodeRunnerClient:
    def __init__(
        self,
        socket_path: Path,
        timeout_seconds: float = DEFAULT_RUNNER_CLIENT_TIMEOUT_SECONDS,
        *,
        request_max_bytes: int = DEFAULT_RUNNER_REQUEST_MAX_BYTES,
        response_max_bytes: int = DEFAULT_RUNNER_RESPONSE_MAX_BYTES,
        socket_poll_seconds: float = DEFAULT_RUNNER_SOCKET_POLL_SECONDS,
        socket_read_bytes: int = DEFAULT_RUNNER_SOCKET_READ_BYTES,
    ) -> None:
        self.socket_path = socket_path
        self.timeout_seconds = timeout_seconds
        self.request_max_bytes = request_max_bytes
        self.response_max_bytes = response_max_bytes
        self.socket_poll_seconds = socket_poll_seconds
        self.socket_read_bytes = socket_read_bytes

    def run_python(
        self,
        path: str,
        files: dict[str, str],
        *,
        arguments: list[str] | None = None,
        stdin: str = "",
        cancel_event: threading.Event | None = None,
        input_queue: queue.Queue[dict] | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict:
        if not self.socket_path.exists():
            raise CodeRunnerError(
                "The isolated runner is not available. Start both services with: "
                "docker compose -f compose.yaml -f compose.dev.yaml up -d --build runner app"
            )
        request = {
            "action": "run_python",
            "path": path,
            "files": files,
            "arguments": arguments or [],
            "stdin": stdin,
        }
        encoded_request = json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n"
        if len(encoded_request) > self.request_max_bytes:
            raise CodeRunnerError("The selected project is too large for an isolated run.")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self.timeout_seconds)
                connection.connect(str(self.socket_path))
                connection.sendall(encoded_request)
                connection.settimeout(self.socket_poll_seconds)
                response_buffer = b""
                size = 0
                cancel_sent = False
                response_deadline = time.monotonic() + self.timeout_seconds
                final_result: dict | None = None
                while True:
                    if cancel_event is not None and cancel_event.is_set() and not cancel_sent:
                        connection.sendall(b'{"action":"cancel"}\n')
                        cancel_sent = True
                    if input_queue is not None:
                        while True:
                            try:
                                command = input_queue.get_nowait()
                            except queue.Empty:
                                break
                            connection.sendall(
                                json.dumps(command, ensure_ascii=False).encode("utf-8") + b"\n"
                            )
                    try:
                        chunk = connection.recv(self.socket_read_bytes)
                    except TimeoutError as exc:
                        if time.monotonic() >= response_deadline:
                            raise CodeRunnerError(
                                "The isolated runner did not return before its response deadline."
                            ) from exc
                        continue
                    if not chunk:
                        break
                    response_deadline = time.monotonic() + self.timeout_seconds
                    response_buffer += chunk
                    size += len(chunk)
                    if size > self.response_max_bytes:
                        raise CodeRunnerError("The isolated runner returned too much data.")
                    lines = response_buffer.split(b"\n")
                    response_buffer = lines.pop()
                    for raw_line in lines:
                        if not raw_line:
                            continue
                        try:
                            event = json.loads(raw_line.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise CodeRunnerError(
                                "The isolated runner returned an invalid event."
                            ) from exc
                        if event.get("type") == "result":
                            final_result = event
                        elif event_callback is not None:
                            event_callback(event)
                    if final_result is not None:
                        break
        except (OSError, TimeoutError) as exc:
            raise CodeRunnerError(f"Could not reach the isolated runner: {exc}") from exc
        if final_result is None:
            raise CodeRunnerError("The isolated runner ended without a final response.")
        final_result.pop("type", None)
        return final_result
