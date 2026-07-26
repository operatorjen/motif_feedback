from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from .code_runner import CodeRunnerClient, CodeRunnerError, format_interactive_transcript
from .config import Settings
from .file_tools import FileToolError, ProjectFileTools
from .models import CodeRunRequest
from .role_decorators import validate_role_signals
from .run_sessions import CodeRunSession
from .storage import Storage

RUNNER_TEXT_SUFFIXES = {
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".txt",
    ".md",
    ".csv",
}


class CodeRunValidationError(ValueError):
    def __init__(self, detail: str, *, status_code: int = 400) -> None:
        super().__init__(detail)
        self.status_code = status_code


class CodeRunCoordinator:
    """Prepare, execute, and record one user-approved isolated Python run."""

    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        file_tools: ProjectFileTools,
        runner: CodeRunnerClient,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.file_tools = file_tools
        self.runner = runner

    def create_session(self, project_id: str, payload: CodeRunRequest) -> CodeRunSession:
        self._validate_request(payload)
        initial_input_bytes = len(payload.stdin.encode("utf-8"))
        return CodeRunSession(
            project_id=project_id,
            path=payload.path,
            events=[{"type": "stdin", "text": payload.stdin}] if payload.stdin else [],
            input_bytes=initial_input_bytes,
        )

    async def run(
        self,
        project_id: str,
        payload: CodeRunRequest,
        session: CodeRunSession,
        *,
        run_lock: asyncio.Lock,
        is_disconnected: Callable[[], Awaitable[bool]],
    ) -> dict:
        files = self._project_snapshot(project_id, payload.path)

        def record_runner_event(event: dict) -> None:
            if event.get("type") not in {"stdout", "stderr"}:
                return
            text = str(event.get("text", ""))
            if text:
                session.record_output(event["type"], text)

        async with run_lock:
            runner_task = asyncio.create_task(
                asyncio.to_thread(
                    self.runner.run_python,
                    payload.path,
                    files,
                    arguments=payload.arguments,
                    stdin=payload.stdin,
                    cancel_event=session.cancel_event,
                    input_queue=session.input_queue,
                    event_callback=record_runner_event,
                )
            )
            while not runner_task.done():
                if await is_disconnected():
                    session.cancel_event.set()
                    break
                await asyncio.sleep(self.settings.runner_session_poll_seconds)
            result = await runner_task

        if result.get("error"):
            raise CodeRunnerError(str(result["error"]))
        return self._record_result(project_id, payload.path, session, result)

    def _validate_request(self, payload: CodeRunRequest) -> None:
        if len(payload.arguments) > self.settings.runner_argument_max_count:
            raise CodeRunValidationError(
                "Too many runner arguments. "
                f"Maximum: {self.settings.runner_argument_max_count}."
            )
        if any(
            len(argument) > self.settings.runner_argument_max_chars
            for argument in payload.arguments
        ):
            raise CodeRunValidationError(
                "A runner argument exceeds the configured character limit of "
                f"{self.settings.runner_argument_max_chars}."
            )
        argument_bytes = sum(
            len(argument.encode("utf-8")) for argument in payload.arguments
        )
        if argument_bytes > self.settings.runner_arguments_max_bytes:
            raise CodeRunValidationError(
                "Runner arguments exceed the configured combined byte limit of "
                f"{self.settings.runner_arguments_max_bytes}."
            )
        if len(payload.stdin.encode("utf-8")) > self.settings.runner_input_max_bytes:
            raise CodeRunValidationError(
                "Runner standard input exceeds the configured byte limit.",
                status_code=413,
            )

    def _project_snapshot(self, project_id: str, requested_path: str) -> dict[str, str]:
        path = self.file_tools.download_path(project_id, requested_path)
        if path.suffix.lower() != ".py":
            raise FileToolError("Only Python project files can run in the isolated runner.")

        files: dict[str, str] = {}
        total_bytes = 0
        for item in self.file_tools.list_files(project_id):
            relative_path = str(item["path"])
            if Path(relative_path).suffix.lower() not in RUNNER_TEXT_SUFFIXES:
                continue
            raw = self.file_tools.download_path(project_id, relative_path).read_bytes()
            total_bytes += len(raw)
            if total_bytes > self.settings.runner_project_max_bytes:
                raise FileToolError("The project is too large for an isolated run.")
            files[relative_path] = raw.decode("utf-8", errors="replace")
        return files

    def _record_result(
        self,
        project_id: str,
        path: str,
        session: CodeRunSession,
        result: dict,
    ) -> dict:
        role_signals = validate_role_signals(
            result.pop("role_signal_candidates", [])
        )
        result["role_signals"] = role_signals
        session_events, _cursor = session.snapshot()
        transcript, input_count = format_interactive_transcript(
            session_events,
            user_label=self.settings.user_display_name,
        )
        if input_count:
            result["transcript"] = transcript

        output = self._room_output(
            path,
            result,
            transcript=transcript,
            input_count=input_count,
            role_signals=role_signals,
        )
        room_output_limit = self.settings.runner_room_transcript_max_chars
        self.storage.add_message(
            project_id,
            "runner",
            output[:room_output_limit],
            metadata={
                "code_run": {
                    "path": path,
                    "ok": bool(result.get("ok")),
                    "return_code": result.get("return_code"),
                    "timed_out": bool(result.get("timed_out")),
                    "canceled": bool(result.get("canceled")),
                    "network": "disabled",
                    "output_truncated_for_room": len(output) > room_output_limit,
                    "interactive_input_count": input_count,
                },
                "role_signals": role_signals,
            },
        )
        return result

    def _room_output(
        self,
        path: str,
        result: dict,
        *,
        transcript: str,
        input_count: int,
        role_signals: list[dict],
    ) -> str:
        status = (
            "canceled"
            if result.get("canceled")
            else "timed out"
            if result.get("timed_out")
            else "completed"
            if result.get("return_code") == 0
            else f"exited with code {result.get('return_code')}"
        )
        return "\n".join(
            part
            for part in (
                f"{self.settings.user_display_name} approved one isolated run of {path}.",
                f"Result: {status}. Network: disabled. Workspace: temporary copy.",
                (
                    f"INTERACTIVE TRANSCRIPT:\n{transcript}"
                    if input_count
                    else f"STDOUT:\n{result.get('stdout', '')}"
                    if result.get("stdout")
                    else ""
                ),
                (
                    f"STDERR:\n{result.get('stderr', '')}"
                    if not input_count and result.get("stderr")
                    else ""
                ),
                (
                    "ROLE DECORATORS FOR THE NEXT USER MESSAGE:\n"
                    + "\n".join(
                        f"- {signal['label']} → {signal['target']} "
                        f"(intensity {signal['intensity']:.3g})"
                        for signal in role_signals
                    )
                    if role_signals
                    else ""
                ),
            )
            if part
        )
