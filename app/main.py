from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .agent_tools import AgentToolExecutor
from .async_tasks import cancel_and_wait
from .code_runner import CodeRunnerClient, CodeRunnerError
from .config import RuntimeNotConfiguredError, get_settings
from .constants import (
    DEFAULT_MEMORY_EVENT_LIMIT,
    DEFAULT_PROJECT_MESSAGE_LIMIT,
    DEFAULT_WEB_SOURCE_LIMIT,
    MAX_MEMORY_EVENT_LIMIT,
    MAX_PROJECT_MESSAGE_LIMIT,
    MAX_WEB_SOURCE_LIMIT,
)
from .execution_semantics import public_execution_stage
from .file_tools import FileToolError, ProjectFileTools
from .memory_loops import memory_loop_for
from .models import (
    ChatRequest,
    CodeRunInput,
    CodeRunRequest,
    FileSharingUpdate,
    PersonaEdit,
    ProjectCreate,
    ProviderCatalogEdit,
    RuntimeConfig,
    RuntimeOptions,
    SetupUpdate,
    SharedContextEdit,
)
from .orchestrator import Orchestrator
from .persona_store import PersonaStore, PersonaUpdateError
from .provider_catalog import (
    ProviderCatalogError,
    ProviderCatalogStore,
    ProviderRegistry,
)
from .providers import DirectProviderClient, ProviderError
from .run_coordinator import CodeRunCoordinator, CodeRunValidationError
from .run_sessions import CodeRunSession
from .search_router import SearchRouter
from .security import LocalSecurityMiddleware, LocalSessionGuard
from .storage import ChatTurnConflictError, Storage, StorageError
from .turn_service import ChatTurnStateError, TurnService
from .web_sources import WebSourceService

settings = get_settings()
guard = LocalSessionGuard()
storage = Storage(settings.database_path, settings.projects_root)
provider_catalog_store = ProviderCatalogStore(
    settings.provider_catalog_path,
    settings.seed_root / "providers.yaml",
)
provider_registry = ProviderRegistry(settings, provider_catalog_store)
persona_store = PersonaStore(settings, storage)
file_tools = ProjectFileTools(
    storage,
    max_write_bytes=settings.agent_file_byte_limit,
    max_upload_bytes=settings.max_upload_bytes,
)
tool_executor = AgentToolExecutor(file_tools, persona_store)
provider_client = DirectProviderClient(
    settings,
    tool_executor,
    provider_registry,
)
web_source_service = WebSourceService(settings, storage)
code_runner = CodeRunnerClient(
    settings.runner_socket_path,
    timeout_seconds=settings.runner_timeout_seconds,
    request_max_bytes=settings.runner_request_max_bytes,
    response_max_bytes=settings.runner_response_max_bytes,
    socket_poll_seconds=settings.runner_socket_poll_seconds,
    socket_read_bytes=settings.runner_socket_read_bytes,
)
code_run_coordinator = CodeRunCoordinator(
    settings,
    storage,
    file_tools,
    code_runner,
)
orchestrator = Orchestrator(
    settings,
    storage,
    persona_store,
    provider_client,
    SearchRouter(),
    web_source_service,
)
turn_service = TurnService(settings, storage, orchestrator)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.workspace_root.mkdir(parents=True, exist_ok=True)
    provider_catalog_store.initialize()
    persona_store.initialize()
    storage.initialize()
    storage.prune_chat_turn_traces(settings.turn_trace_retention_days)
    settings.load_runtime_config()
    app.state.chat_lock = asyncio.Lock()
    app.state.code_run_lock = asyncio.Lock()
    app.state.code_run_sessions = {}
    app.state.code_run_sessions_lock = threading.Lock()
    try:
        yield
    finally:
        await provider_client.aclose()


app = FastAPI(
    title="Motif Feedback",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"]
)
app.add_middleware(LocalSecurityMiddleware, guard=guard)

STATIC_ROOT = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


async def _execute_chat_turn(
    payload: ChatRequest,
    runtime: RuntimeConfig,
    progress_callback=None,
) -> dict:
    return await turn_service.execute(
        payload,
        runtime,
        progress_callback,
    )


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


def _runtime_state() -> dict:
    runtime = settings.load_runtime_config()
    setup_complete = runtime is not None
    return {
        "setup_complete": setup_complete,
        "key_configured": setup_complete
        and all(provider_registry.ready(provider) for provider in set(runtime.providers.values())),
        "provider_status": provider_registry.status(),
        "provider_catalog": provider_registry.public_profiles(),
        "runtime": runtime.model_dump() if runtime is not None else None,
        "runtime_defaults": RuntimeOptions().model_dump(),
    }


def _require_runtime_config() -> RuntimeConfig:
    try:
        return settings.require_runtime_config()
    except RuntimeNotConfiguredError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/session")
def session() -> dict:
    return {
        "token": guard.token,
        **_runtime_state(),
        "agents": persona_store.list_summaries(),
        "projects": storage.list_projects(),
        "user_display_name": settings.user_display_name,
        "workspace_root": str(settings.workspace_root),
        "agent_file_max_bytes": settings.agent_file_byte_limit,
        "runner_input_max_bytes": settings.runner_input_max_bytes,
        "runner_input_message_max_bytes": settings.runner_input_message_max_bytes,
        "runner_argument_max_count": settings.runner_argument_max_count,
        "runner_argument_max_chars": settings.runner_argument_max_chars,
        "runner_arguments_max_bytes": settings.runner_arguments_max_bytes,
        "max_agent_turn_beats": settings.max_agent_turn_beats,
    }


@app.get("/api/setup")
def get_setup() -> dict:
    return _runtime_state()


@app.put("/api/setup")
def update_setup(payload: SetupUpdate) -> dict:
    runtime = payload
    enabled_providers = {profile.id for profile in provider_registry.profiles(enabled_only=True)}
    unknown = sorted(set(runtime.providers.values()) - enabled_providers)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                "Setup references providers that are not enabled in the catalog: "
                + ", ".join(unknown)
            ),
        )
    settings.save_runtime_config(runtime)
    return {
        "ok": True,
        "setup_complete": True,
        "runtime": runtime.model_dump(),
    }


@app.get("/api/provider-catalog")
def get_provider_catalog() -> dict:
    try:
        return {
            "yaml_text": provider_catalog_store.yaml_text(),
            "providers": provider_registry.public_profiles(),
        }
    except ProviderCatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/provider-catalog")
def update_provider_catalog(payload: ProviderCatalogEdit) -> dict:
    try:
        provider_catalog_store.save(payload.yaml_text)
        return {
            "ok": True,
            "yaml_text": provider_catalog_store.yaml_text(),
            "providers": provider_registry.public_profiles(),
        }
    except ProviderCatalogError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/projects")
def list_projects() -> list[dict]:
    return storage.list_projects()


@app.post("/api/projects")
def create_project(payload: ProjectCreate) -> dict:
    try:
        return storage.create_project(payload.name)
    except (StorageError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str, request: Request) -> dict:
    try:
        async with request.app.state.chat_lock:
            result = storage.delete_project(project_id)
            result["cleared_persona_positions"] = persona_store.clear_project_position(project_id)
            result["projects"] = storage.list_projects()
            return result
    except StorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/messages")
def project_messages(
    project_id: str,
    limit: int = Query(
        default=DEFAULT_PROJECT_MESSAGE_LIMIT,
        ge=1,
        le=MAX_PROJECT_MESSAGE_LIMIT,
    ),
) -> list[dict]:
    try:
        return storage.list_messages(project_id, limit=limit)
    except StorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _public_chat_turn(
    turn: dict,
    operations: list[dict] | None = None,
) -> dict:
    request_payload = turn.get("request") or {}
    trace = turn.get("trace") or {}
    if operations is None:
        operations = storage.list_turn_operations(turn["id"])
    return {
        "id": turn["id"],
        "project_id": turn["project_id"],
        "status": turn["status"],
        "resolution": turn.get("resolution"),
        "resumable": bool(
            turn["status"] in {"failed", "interrupted"}
            and not turn.get("resolution")
            and turn.get("request")
            and turn.get("runtime")
        ),
        "message": str(request_payload.get("message") or ""),
        "participants": request_payload.get("participants") or [],
        "research_mode": request_payload.get("research_mode"),
        "failure_detail": turn.get("failure_detail"),
        "started_at": turn["started_at"],
        "updated_at": turn["updated_at"],
        "resolved_at": turn.get("resolved_at"),
        "duration_ms": trace.get("duration_ms"),
        "provider_requests": trace.get("provider_requests", 0),
        "provider_usage": trace.get("provider_usage") or {},
        "operation_count": len(operations),
        "execution_stage": public_execution_stage(
            turn_status=turn["status"],
            participants=request_payload.get("participants") or [],
            operations=operations,
        ),
    }


@app.get("/api/chat-turns/{project_id}")
def chat_turns(project_id: str, limit: int = Query(default=100, ge=1, le=500)) -> list[dict]:
    try:
        turns = storage.list_chat_turns(project_id, limit=limit)
        operations_by_turn = storage.list_turn_operations_for_turns(
            [turn["id"] for turn in turns]
        )
        return [
            _public_chat_turn(turn, operations_by_turn.get(turn["id"], []))
            for turn in turns
        ]
    except StorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/chat-turns/{project_id}/{turn_id}/accept")
def accept_partial_chat_turn(project_id: str, turn_id: str) -> dict:
    try:
        turn = storage.get_chat_turn(turn_id)
        if turn["project_id"] != project_id:
            raise StorageError("Chat turn not found.")
        return _public_chat_turn(storage.resolve_chat_turn(turn_id, "accepted_partial"))
    except ChatTurnConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except StorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request) -> dict:
    runtime = _require_runtime_config()
    try:
        async with request.app.state.chat_lock:
            return await _execute_chat_turn(payload, runtime)
    except ChatTurnStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except StorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ProviderError, PersonaUpdateError, FileToolError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    runtime = _require_runtime_config()

    async def execute(report) -> dict:
        return await _execute_chat_turn(
            payload,
            runtime,
            progress_callback=report,
        )

    return _chat_stream_response(request, execute)


@app.post("/api/chat-turns/{project_id}/{turn_id}/resume/stream")
async def resume_chat_stream(
    project_id: str,
    turn_id: str,
    request: Request,
) -> StreamingResponse:
    runtime = _require_runtime_config()
    try:
        turn = storage.get_chat_turn(turn_id)
        if turn["project_id"] != project_id or not turn.get("request"):
            raise StorageError("Chat turn not found.")
        payload = ChatRequest.model_validate(turn["request"])
    except StorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def execute(report) -> dict:
        return await turn_service.execute(
            payload,
            runtime,
            report,
            resume=True,
        )

    return _chat_stream_response(request, execute)


def _chat_stream_response(
    request: Request,
    execute: Callable[[Callable[[dict], Awaitable[None]]], Awaitable[dict]],
) -> StreamingResponse:
    async def event_stream():
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        async def report(event: dict) -> None:
            await queue.put(event)

        async def run_chat() -> None:
            try:
                async with request.app.state.chat_lock:
                    result = await execute(report)
                await queue.put({"type": "result", **result})
            except ChatTurnStateError as exc:
                await queue.put({"type": "error", "detail": str(exc), "status": 409})
            except StorageError as exc:
                await queue.put({"type": "error", "detail": str(exc), "status": 404})
            except (ProviderError, PersonaUpdateError, FileToolError) as exc:
                await queue.put({"type": "error", "detail": str(exc), "status": 502})
            except Exception:
                await queue.put(
                    {
                        "type": "error",
                        "detail": "The local server encountered an unexpected error.",
                        "status": 500,
                    }
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_chat())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield json.dumps(event, ensure_ascii=False) + "\n"
        finally:
            await cancel_and_wait(task)

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/api/personas")
def personas() -> list[dict]:
    return persona_store.list_summaries()


@app.get("/api/personas/{agent_id}")
def get_persona(agent_id: str) -> dict:
    try:
        return {"agent_id": agent_id, "yaml_text": persona_store.get_persona_yaml(agent_id)}
    except PersonaUpdateError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/personas/{agent_id}")
def edit_persona(agent_id: str, payload: PersonaEdit) -> dict:
    try:
        updated = persona_store.save_user_edit(agent_id, payload.yaml_text)
        return {"ok": True, "persona": updated}
    except (PersonaUpdateError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/shared-context")
def get_shared_context() -> dict:
    return {"markdown_text": persona_store.load_shared_context()}


@app.put("/api/shared-context")
def edit_shared_context(payload: SharedContextEdit) -> dict:
    try:
        return persona_store.save_user_shared_context(payload.markdown_text)
    except PersonaUpdateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/proposals")
def proposals() -> list[dict]:
    return persona_store.list_proposals()


@app.get("/api/memory-loops/{project_id}/{agent_id}")
def memory_loop_data(
    project_id: str,
    agent_id: str,
    limit: int = Query(
        default=DEFAULT_MEMORY_EVENT_LIMIT,
        ge=1,
        le=MAX_MEMORY_EVENT_LIMIT,
    ),
) -> dict:
    try:
        definition = memory_loop_for(agent_id)
        events = storage.list_memory_events(project_id, agent_id, limit=limit)
        stats = storage.memory_stats(project_id).get(
            agent_id,
            {
                "agent_id": agent_id,
                "event_count": 0,
                "action_count": 0,
                "failure_count": 0,
                "latest_sequence": 0,
            },
        )
        global_events = storage.list_global_memory_events(
            agent_id, exclude_project_id=project_id, limit=limit
        )
        global_stats = storage.global_memory_stats(agent_id, exclude_project_id=project_id)
        return {
            "agent_id": agent_id,
            "definition": definition,
            "stats": stats,
            "events": events,
            "global_stats": global_stats,
            "global_events": global_events,
        }
    except (KeyError, StorageError) as exc:
        raise HTTPException(status_code=404, detail="Memory loop or project not found.") from exc


@app.get("/api/web-sources/{project_id}")
def list_web_sources(
    project_id: str,
    limit: int = Query(
        default=DEFAULT_WEB_SOURCE_LIMIT,
        ge=1,
        le=MAX_WEB_SOURCE_LIMIT,
    ),
) -> list[dict]:
    try:
        return storage.list_web_sources(project_id, limit=limit)
    except StorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/web-sources/{project_id}/{source_id}")
def get_web_source(project_id: str, source_id: str) -> dict:
    try:
        return storage.get_web_source(project_id, source_id)
    except StorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/web-sources/{project_id}/{source_id}")
def delete_web_source(project_id: str, source_id: str) -> dict:
    try:
        return storage.delete_web_source(project_id, source_id)
    except StorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/files/{project_id}")
def list_project_files(project_id: str) -> list[dict]:
    try:
        return file_tools.list_files(project_id)
    except (FileToolError, StorageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/files/{project_id}/read")
def read_project_file(project_id: str, path: str = Query(min_length=1)) -> dict:
    try:
        return file_tools.read_file(project_id, path)
    except (FileToolError, StorageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/files/{project_id}/preview")
def preview_project_file(project_id: str, path: str = Query(min_length=1)) -> FileResponse:
    try:
        image_path, media_type = file_tools.preview_path(project_id, path)
        return FileResponse(
            image_path,
            media_type=media_type,
            headers={"Content-Disposition": "inline", "Cache-Control": "no-store"},
        )
    except (FileToolError, StorageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/files/{project_id}/download")
def download_project_file(project_id: str, path: str = Query(min_length=1)) -> FileResponse:
    try:
        download_path = file_tools.download_path(project_id, path)
        return FileResponse(
            download_path,
            filename=download_path.name,
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )
    except (FileToolError, StorageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/files/{project_id}/sharing")
def update_project_file_sharing(project_id: str, payload: FileSharingUpdate) -> dict:
    try:
        return file_tools.set_agent_sharing(project_id, payload.path, payload.shared_agent_edit)
    except (FileToolError, StorageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/code-runs/{project_id}")
async def run_project_code(project_id: str, payload: CodeRunRequest, request: Request) -> dict:
    try:
        session = code_run_coordinator.create_session(project_id, payload)
    except CodeRunValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    with app.state.code_run_sessions_lock:
        if app.state.code_run_sessions:
            raise HTTPException(
                status_code=409,
                detail="Another isolated script is already running.",
            )
        app.state.code_run_sessions[payload.run_id] = session

    try:
        return await code_run_coordinator.run(
            project_id,
            payload,
            session,
            run_lock=app.state.code_run_lock,
            is_disconnected=request.is_disconnected,
        )
    except (FileToolError, StorageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CodeRunnerError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        with app.state.code_run_sessions_lock:
            app.state.code_run_sessions.pop(payload.run_id, None)


def _active_code_run(project_id: str, run_id: str) -> CodeRunSession:
    with app.state.code_run_sessions_lock:
        session = app.state.code_run_sessions.get(run_id)
    if session is None or session.project_id != project_id:
        raise HTTPException(status_code=404, detail="That isolated run is no longer active.")
    return session


@app.get("/api/code-runs/{project_id}/{run_id}")
def code_run_status(
    project_id: str,
    run_id: str,
    after: int = Query(default=0, ge=0),
) -> dict:
    session = _active_code_run(project_id, run_id)
    events, next_cursor = session.snapshot(after)
    return {"active": True, "events": events, "next": next_cursor}


@app.post("/api/code-runs/{project_id}/{run_id}/input")
def send_code_run_input(
    project_id: str,
    run_id: str,
    payload: CodeRunInput,
) -> dict:
    session = _active_code_run(project_id, run_id)
    if payload.eof:
        session.send({"action": "close_stdin"})
        return {"ok": True, "eof": True}
    if not payload.text:
        raise HTTPException(status_code=400, detail="Enter text or send EOF.")
    text = payload.text + ("\n" if payload.append_newline else "")
    encoded_size = len(text.encode("utf-8"))
    if encoded_size > settings.runner_input_message_max_bytes:
        raise HTTPException(
            status_code=413,
            detail="This input exceeds the configured per-message byte limit.",
        )
    if not session.record_input(text, encoded_size, settings.runner_input_max_bytes):
        raise HTTPException(
            status_code=413,
            detail="This run has reached its standard-input limit.",
        )
    session.send({"action": "input", "text": text})
    return {"ok": True, "bytes": encoded_size}


@app.delete("/api/code-runs/{project_id}/{run_id}")
def cancel_code_run(project_id: str, run_id: str) -> dict:
    session = _active_code_run(project_id, run_id)
    session.cancel()
    return {"ok": True, "canceling": True}


@app.post("/api/files/{project_id}/upload")
async def upload_project_file(
    project_id: str,
    file: Annotated[UploadFile, File()],
) -> dict:
    filename = file.filename or ""
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Upload exceeds the configured size limit.")
    try:
        return file_tools.save_upload(project_id, filename, content)
    except (FileToolError, StorageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/files/{project_id}")
def delete_project_file(project_id: str, path: str = Query(min_length=1)) -> dict:
    try:
        return file_tools.delete_file(project_id, path)
    except (FileToolError, StorageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
