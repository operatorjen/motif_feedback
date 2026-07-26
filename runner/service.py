from __future__ import annotations

import ctypes
import json
import os
import resource
import selectors
import signal
import socket
import socketserver
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path


def environment_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def environment_float(name: str, default: float, *, minimum: float = 0.01) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


SOCKET_PATH = Path(os.environ.get("RUNNER_SOCKET_PATH", "/runner/runner.sock"))
PYTHON_EXECUTABLE = "/usr/local/bin/python"
COPY_SUFFIXES = {
    ".py", ".json", ".yaml", ".yml", ".toml", ".ini", ".txt", ".md", ".csv",
}
MAX_PROJECT_COPY_BYTES = environment_int("RUNNER_PROJECT_MAX_BYTES", 8_000_000)
MAX_OUTPUT_BYTES = environment_int("RUNNER_OUTPUT_MAX_BYTES", 64_000)
MAX_REQUEST_BYTES = environment_int("RUNNER_REQUEST_MAX_BYTES", 8_500_000)
TIMEOUT_SECONDS = environment_float("RUNNER_EXECUTION_TIMEOUT_SECONDS", 60)
MAX_INPUT_BYTES = environment_int("RUNNER_INPUT_MAX_BYTES", 16_000)
MAX_INPUT_MESSAGE_BYTES = environment_int(
    "RUNNER_INPUT_MESSAGE_MAX_BYTES", 4_000
)
MAX_ARGUMENT_COUNT = environment_int("RUNNER_ARGUMENT_MAX_COUNT", 24)
MAX_ARGUMENT_CHARS = environment_int("RUNNER_ARGUMENT_MAX_CHARS", 500)
MAX_ARGUMENTS_BYTES = environment_int("RUNNER_ARGUMENTS_MAX_BYTES", 4_000)
MAX_ROLE_SIGNAL_CANDIDATES = environment_int(
    "RUNNER_ROLE_SIGNAL_MAX_COUNT", 16
)
CHILD_CPU_SECONDS = environment_int("RUNNER_CHILD_CPU_SECONDS", 5)
CHILD_FILE_MAX_BYTES = environment_int("RUNNER_CHILD_FILE_MAX_BYTES", 8_000_000)
CHILD_OPEN_FILE_LIMIT = environment_int("RUNNER_CHILD_OPEN_FILE_LIMIT", 32)
CHILD_MEMORY_MAX_BYTES = environment_int(
    "RUNNER_CHILD_MEMORY_MAX_BYTES", 256_000_000
)
CHILD_PROCESS_LIMIT = environment_int("RUNNER_CHILD_PROCESS_LIMIT", 16)
STREAM_READ_BYTES = environment_int("RUNNER_STREAM_READ_BYTES", 4_096)
CONTROL_READ_BYTES = environment_int("RUNNER_CONTROL_READ_BYTES", 4_096)
SELECT_POLL_SECONDS = environment_float("RUNNER_SELECT_POLL_SECONDS", 0.05)
ERROR_DETAIL_MAX_CHARS = environment_int("RUNNER_ERROR_DETAIL_MAX_CHARS", 2_000)
ROLE_HELPER_PATH = Path("motif_role.py")
ROLE_SIGNAL_PREFIX = "__MOTIF_ROLE_SIGNAL__"
PROJECT_LAUNCH_CODE = (
    "import runpy,sys;"
    "project_root,target,*script_args=sys.argv[1:];"
    "sys.path.insert(0,project_root);"
    "sys.argv=[target,*script_args];"
    "runpy.run_path(target,run_name='__main__')"
)
PR_SET_CHILD_SUBREAPER = 36
DESCENDANT_CLEANUP_ATTEMPTS = 10
DESCENDANT_CLEANUP_POLL_SECONDS = 0.01
ROLE_HELPER_SOURCE = f'''"""Bounded role signals for the local motif-feedback room.

Signals are validated again by the application. This module cannot grant tools,
change personas, or inject arbitrary prompt text.
"""
import json

_PREFIX = {ROLE_SIGNAL_PREFIX!r}
_DECORATORS = {{
    "embodied_attention",
    "feedback_attention",
    "strategic_attention",
    "integrative_attention",
    "playful_attention",
    "critical_attention",
}}
_TARGETS = {{"room", "agent_a", "agent_b", "agent_c"}}


def available():
    return tuple(sorted(_DECORATORS))


def emit(decorator, *, target="room", intensity=1.0, observations=None):
    if decorator not in _DECORATORS:
        raise ValueError("Unknown role decorator.")
    if target not in _TARGETS:
        raise ValueError("Role decorator target must be room or one agent id.")
    if not isinstance(observations or {{}}, dict):
        raise ValueError("Role decorator observations must be a mapping.")
    payload = {{
        "decorator": decorator,
        "target": target,
        "intensity": float(intensity),
        "observations": observations or {{}},
    }}
    print(_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)
'''


def confined_relative_path(relative_path: str, *, require_python: bool = False) -> Path:
    supplied = Path(relative_path)
    if supplied.is_absolute() or any(part in {"", ".", ".."} for part in supplied.parts):
        raise ValueError("Invalid project-relative path.")
    if supplied.suffix.lower() not in COPY_SUFFIXES:
        raise ValueError("Unsupported runner file type.")
    if require_python and supplied.suffix.lower() != ".py":
        raise ValueError("Only Python project files can run.")
    return supplied


def materialize_project(files: dict, destination: Path) -> None:
    if not isinstance(files, dict):
        raise ValueError("Runner files must be a path-to-text object.")
    total = 0
    for raw_path, content in sorted(files.items()):
        relative = confined_relative_path(str(raw_path))
        if relative == ROLE_HELPER_PATH:
            raise ValueError(f"{ROLE_HELPER_PATH} is reserved by the isolated runner.")
        if not isinstance(content, str):
            raise ValueError("Runner files must contain text.")
        encoded = content.encode("utf-8")
        total += len(encoded)
        if total > MAX_PROJECT_COPY_BYTES:
            raise ValueError("Project code exceeds the runner copy limit.")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded)
        target.chmod(0o444)
        for parent in target.parents:
            if parent == destination.parent:
                break
            parent.chmod(0o755)


def inject_role_helper(destination: Path) -> None:
    helper = destination / ROLE_HELPER_PATH
    helper.write_text(ROLE_HELPER_SOURCE, encoding="utf-8")
    helper.chmod(0o444)


def extract_role_signal_candidates(stdout: str) -> tuple[str, list[dict]]:
    visible_lines: list[str] = []
    candidates: list[dict] = []
    for line in stdout.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        if stripped.startswith(ROLE_SIGNAL_PREFIX):
            try:
                candidate = json.loads(stripped[len(ROLE_SIGNAL_PREFIX) :])
            except json.JSONDecodeError:
                visible_lines.append(line)
                continue
            if (
                isinstance(candidate, dict)
                and len(candidates) < MAX_ROLE_SIGNAL_CANDIDATES
            ):
                candidates.append(candidate)
            continue
        visible_lines.append(line)
    return "".join(visible_lines), candidates


def apply_child_limits() -> None:
    limits = {
        resource.RLIMIT_CPU: (CHILD_CPU_SECONDS, CHILD_CPU_SECONDS),
        resource.RLIMIT_FSIZE: (CHILD_FILE_MAX_BYTES, CHILD_FILE_MAX_BYTES),
        resource.RLIMIT_NOFILE: (CHILD_OPEN_FILE_LIMIT, CHILD_OPEN_FILE_LIMIT),
        resource.RLIMIT_CORE: (0, 0),
    }
    if hasattr(resource, "RLIMIT_AS"):
        limits[resource.RLIMIT_AS] = (
            CHILD_MEMORY_MAX_BYTES,
            CHILD_MEMORY_MAX_BYTES,
        )
    if hasattr(resource, "RLIMIT_NPROC"):
        limits[resource.RLIMIT_NPROC] = (CHILD_PROCESS_LIMIT, CHILD_PROCESS_LIMIT)
    for kind, value in limits.items():
        try:
            resource.setrlimit(kind, value)
        except (OSError, ValueError):
            # Some non-Linux development hosts do not permit every Linux
            # runner limit. Docker still enforces the container-level caps.
            continue


def enable_child_subreaper() -> bool:
    """Adopt orphaned grandchildren so detached runner processes remain killable."""
    if sys.platform != "linux":
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    except (AttributeError, OSError):
        return False
    return result == 0


def descendant_pids(root_pid: int) -> set[int]:
    """Return descendants visible in this Linux PID namespace."""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return set()
    children_by_parent: dict[int, set[int]] = {}
    for stat_path in proc_root.glob("[0-9]*/stat"):
        try:
            raw = stat_path.read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 1 :].split()
            pid = int(stat_path.parent.name)
            parent_pid = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
        children_by_parent.setdefault(parent_pid, set()).add(pid)

    descendants: set[int] = set()
    pending = list(children_by_parent.get(root_pid, ()))
    while pending:
        pid = pending.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        pending.extend(children_by_parent.get(pid, ()))
    return descendants


def terminate_new_descendants(baseline_pids: set[int]) -> None:
    """Kill and reap every process created under the runner service for one run."""
    if sys.platform != "linux":
        return
    current_pid = os.getpid()
    for _attempt in range(DESCENDANT_CLEANUP_ATTEMPTS):
        candidates = descendant_pids(current_pid) - baseline_pids
        if not candidates:
            return
        for pid in candidates:
            with suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        for pid in candidates:
            with suppress(ChildProcessError, ProcessLookupError):
                os.waitpid(pid, os.WNOHANG)
        time.sleep(DESCENDANT_CLEANUP_POLL_SECONDS)


def run_python(
    relative_path: str,
    files: dict,
    arguments: list[str] | None = None,
    stdin_text: str = "",
    control_socket: socket.socket | None = None,
    event_callback=None,
) -> dict:
    enable_child_subreaper()
    baseline_descendants = descendant_pids(os.getpid())
    relative = confined_relative_path(relative_path, require_python=True)
    if relative.as_posix() not in files:
        raise ValueError("The requested Python file was not included.")
    arguments = arguments or []
    if (
        not isinstance(arguments, list)
        or len(arguments) > MAX_ARGUMENT_COUNT
        or any(not isinstance(argument, str) for argument in arguments)
        or any(
            "\x00" in argument or len(argument) > MAX_ARGUMENT_CHARS
            for argument in arguments
        )
        or sum(len(argument.encode("utf-8")) for argument in arguments)
        > MAX_ARGUMENTS_BYTES
    ):
        raise ValueError("Invalid runner arguments.")
    if not isinstance(stdin_text, str) or len(stdin_text.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError("Runner input exceeds the size limit.")
    with tempfile.TemporaryDirectory(prefix="motif-run-") as temporary:
        Path(temporary).chmod(0o755)
        work_root = Path(temporary) / "project"
        work_root.mkdir()
        work_root.chmod(0o755)
        materialize_project(files, work_root)
        inject_role_helper(work_root)
        target = work_root / relative
        process = subprocess.Popen(
            [
                PYTHON_EXECUTABLE,
                "-I",
                "-u",
                "-B",
                "-c",
                PROJECT_LAUNCH_CODE,
                str(work_root),
                str(target),
                *arguments,
            ],
            cwd=work_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            preexec_fn=apply_child_limits,
            start_new_session=True,
        )
        stdout_bytes = bytearray()
        stderr_bytes = bytearray()
        output_truncated = False
        input_bytes = 0
        stdin_open = process.stdin is not None

        def send_input(text: str) -> None:
            nonlocal input_bytes, stdin_open
            if not stdin_open or process.stdin is None:
                return
            encoded = text.encode("utf-8")
            if len(encoded) > MAX_INPUT_MESSAGE_BYTES:
                return
            if input_bytes + len(encoded) > MAX_INPUT_BYTES:
                return
            input_bytes += len(encoded)
            try:
                process.stdin.write(encoded)
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                stdin_open = False

        def close_input() -> None:
            nonlocal stdin_open
            if not stdin_open or process.stdin is None:
                return
            with suppress(OSError):
                process.stdin.close()
            stdin_open = False

        if stdin_text:
            send_input(stdin_text)
        if control_socket is None:
            close_input()

        watcher = selectors.DefaultSelector()
        if process.stdout is not None:
            watcher.register(process.stdout, selectors.EVENT_READ, "stdout")
        if process.stderr is not None:
            watcher.register(process.stderr, selectors.EVENT_READ, "stderr")
        if control_socket is not None:
            control_socket.setblocking(False)
            watcher.register(control_socket, selectors.EVENT_READ, "control")

        deadline = time.monotonic() + TIMEOUT_SECONDS
        canceled = False
        timed_out = False
        return_code = None
        control_buffer = b""

        def capture_output(stream_name: str, chunk: bytes) -> None:
            nonlocal output_truncated
            destination = stdout_bytes if stream_name == "stdout" else stderr_bytes
            available = MAX_OUTPUT_BYTES - len(destination)
            if available <= 0:
                output_truncated = True
                return
            kept = chunk[:available]
            destination.extend(kept)
            if len(chunk) > available:
                output_truncated = True
            if event_callback is not None and kept:
                event_callback(
                    {
                        "type": stream_name,
                        "text": kept.decode("utf-8", errors="replace"),
                    }
                )

        try:
            while process.poll() is None and time.monotonic() < deadline and not canceled:
                for key, _mask in watcher.select(timeout=SELECT_POLL_SECONDS):
                    if key.data in {"stdout", "stderr"}:
                        chunk = os.read(key.fileobj.fileno(), STREAM_READ_BYTES)
                        if chunk:
                            capture_output(key.data, chunk)
                        else:
                            watcher.unregister(key.fileobj)
                        continue
                    try:
                        command_raw = control_socket.recv(CONTROL_READ_BYTES)
                    except BlockingIOError:
                        continue
                    if not command_raw:
                        canceled = True
                        break
                    control_buffer += command_raw
                    command_lines = control_buffer.split(b"\n")
                    control_buffer = command_lines.pop()
                    for raw_line in command_lines:
                        try:
                            command = json.loads(raw_line.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                        action = command.get("action")
                        if action == "cancel":
                            canceled = True
                            break
                        if action == "input" and isinstance(command.get("text"), str):
                            send_input(command["text"])
                        elif action == "close_stdin":
                            close_input()
            timed_out = process.poll() is None and not canceled
            return_code = process.poll()
        finally:
            # Kill the original process group even when its leader already
            # exited, then remove descendants that deliberately detached into
            # another session or process group.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            terminate_new_descendants(baseline_descendants)
            return_code = process.returncode if not timed_out and not canceled else return_code
            close_input()
            for stream_name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
                if stream is None:
                    continue
                try:
                    remainder = stream.read()
                except OSError:
                    remainder = b""
                if remainder:
                    capture_output(stream_name, remainder)
            watcher.close()
        visible_stdout, role_signal_candidates = extract_role_signal_candidates(
            bytes(stdout_bytes).decode("utf-8", errors="replace")
        )
        return {
            "ok": not timed_out and not canceled and return_code == 0,
            "path": relative_path,
            "return_code": return_code,
            "timed_out": timed_out,
            "canceled": canceled,
            "stdout": visible_stdout,
            "stderr": bytes(stderr_bytes).decode("utf-8", errors="replace"),
            "output_truncated": output_truncated,
            "role_signal_candidates": role_signal_candidates,
            "network": "disabled",
            "workspace": "temporary_copy",
        }


class RunnerHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            response = {"ok": False, "error": "Runner request is too large."}
        else:
            try:
                request = json.loads(raw.decode("utf-8"))
                if request.get("action") != "run_python":
                    raise ValueError("Unsupported runner action.")
                # Remove the public socket name while untrusted code is alive.
                # The already-connected application request remains usable.
                SOCKET_PATH.unlink(missing_ok=True)

                def send_event(event: dict) -> None:
                    self.wfile.write(
                        json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
                    )
                    self.wfile.flush()

                response = run_python(
                    str(request.get("path", "")),
                    request.get("files"),
                    request.get("arguments"),
                    str(request.get("stdin", "")),
                    self.request,
                    send_event,
                )
            except Exception as exc:
                response = {
                    "ok": False,
                    "error": str(exc)[:ERROR_DETAIL_MAX_CHARS],
                }
        response["type"] = "result"
        self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
        self.wfile.flush()


class RunnerServer(socketserver.UnixStreamServer):
    pass


def main() -> None:
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    while True:
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        with RunnerServer(str(SOCKET_PATH), RunnerHandler) as server:
            os.chmod(SOCKET_PATH, 0o660)
            server.handle_request()


if __name__ == "__main__":
    main()
