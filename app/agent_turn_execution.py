from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .agent_tools import USER_TOOL_DEFINITIONS, ToolContext
from .execution_ledger import ExecutionLedger
from .memory_loops import memory_loop_for
from .models import ChatRequest, RuntimeConfig
from .orchestration_memory import MemoryContext
from .orchestration_prompts import PromptBuilder
from .orchestration_research import (
    record_search_evidence_failure,
    research_provenance,
)
from .providers import (
    AgentCompletion,
    ProviderError,
    ProviderNoResponse,
    ProviderTimeout,
)
from .search_router import SearchDecision

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]


class AgentTurnExecutor:
    """Runs and durably commits one agent's ordered response beats."""

    def __init__(
        self,
        *,
        settings,
        storage,
        persona_store,
        provider_client,
        prompt_builder: PromptBuilder,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.persona_store = persona_store
        self.provider_client = provider_client
        self.prompt_builder = prompt_builder

    async def run(
        self,
        *,
        agent_id: str,
        request: ChatRequest,
        runtime: RuntimeConfig,
        project: dict,
        project_context: str,
        shared_context: str,
        web_sources: list[dict],
        source_context: str,
        role_signals: list[dict],
        decision: SearchDecision,
        source_failures: list[dict],
        search_fallback_agent: str | None,
        recent: list[dict],
        public_sources: list[dict],
        user_message_id: str,
        visible_responses: list[dict],
        turn_transcript: list[dict],
        agent_failures: list[dict],
        progress_callback: ProgressCallback | None,
        execution_ledger: ExecutionLedger,
    ) -> str:
        persona = self.persona_store.load_persona(agent_id)
        display_name = persona.get("display_name", agent_id)
        web_search_enabled = agent_id == search_fallback_agent
        await self._emit(
            progress_callback,
            {
                "type": "agent_start",
                "agent_id": agent_id,
                "display_name": display_name,
            },
        )
        agent_recent = self._stable_recent_context(
            recent,
            turn_id=request.turn_id,
            agent_id=agent_id,
        )
        memory_history, global_memory_history = self._memory_histories(
            request=request,
            agent_id=agent_id,
            recent=agent_recent,
        )
        system_prompt = self.prompt_builder._build_system_prompt(
            persona=persona,
            project=project,
            project_context=project_context,
            shared_context=shared_context,
            is_first=not visible_responses,
            memory_loop=memory_loop_for(agent_id),
            memory_history=memory_history,
            global_memory_history=global_memory_history,
            web_sources=web_sources,
            research_decision=decision,
            source_failures=source_failures,
            web_search_enabled=web_search_enabled,
            search_fallback_agent=search_fallback_agent,
            role_signals=role_signals,
            agent_id=agent_id,
        )
        messages = self.prompt_builder._agent_messages(
            system_prompt,
            agent_recent + turn_transcript,
            web_sources,
            source_context=source_context,
        )
        tools = list(USER_TOOL_DEFINITIONS)

        try:
            completion = await self._completion_for_beat(
                agent_id=agent_id,
                display_name=display_name,
                request=request,
                runtime=runtime,
                messages=messages,
                tools=tools,
                enable_web_search=web_search_enabled,
                progress_callback=progress_callback,
                turn_beat=1,
                execution_ledger=execution_ledger,
            )
        except (ProviderTimeout, ProviderNoResponse, ProviderError) as exc:
            await self._record_provider_failure(
                exc=exc,
                agent_id=agent_id,
                display_name=display_name,
                request=request,
                runtime=runtime,
                user_message_id=user_message_id,
                agent_failures=agent_failures,
                turn_transcript=turn_transcript,
                progress_callback=progress_callback,
                execution_ledger=execution_ledger,
                turn_beat=1,
            )
            if web_search_enabled:
                await record_search_evidence_failure(
                    source_failures=source_failures,
                    runtime=runtime,
                    agent_id=agent_id,
                    detail=f"Search fallback failed: {exc}",
                    progress_callback=progress_callback,
                )
                execution_ledger.mark_agent_finished(agent_id, 1)
                return "search_failed"
            execution_ledger.mark_agent_finished(agent_id, 1)
            return "completed"

        if web_search_enabled and not PromptBuilder._annotation_sources(
            completion.annotations
        ):
            await record_search_evidence_failure(
                source_failures=source_failures,
                runtime=runtime,
                agent_id=agent_id,
                detail="Search fallback returned no cited evidence.",
                progress_callback=progress_callback,
            )
            execution_ledger.mark_agent_finished(agent_id, 1)
            return "search_uncited"

        stored = self._commit_completion(
            completion=completion,
            agent_id=agent_id,
            request=request,
            runtime=runtime,
            user_message_id=user_message_id,
            public_sources=public_sources,
            research_enabled=bool(web_sources) or web_search_enabled,
            research_provenance=(
                research_provenance(
                    source_failures,
                    completion,
                    provider=runtime.providers[agent_id],
                    model=runtime.models[agent_id],
                )
                if web_search_enabled
                else None
            ),
            turn_beat=1,
            execution_ledger=execution_ledger,
        )
        visible_responses.append(stored)
        turn_transcript.append(stored)
        await self._emit_agent_complete(
            stored,
            agent_id=agent_id,
            display_name=display_name,
            turn_beat=1,
            progress_callback=progress_callback,
        )
        final_beat = await self._run_followup_beats(
            completion=completion,
            initial_messages=messages,
            agent_id=agent_id,
            display_name=display_name,
            request=request,
            runtime=runtime,
            tools=tools,
            user_message_id=user_message_id,
            public_sources=public_sources,
            research_enabled=bool(web_sources) or web_search_enabled,
            visible_responses=visible_responses,
            turn_transcript=turn_transcript,
            agent_failures=agent_failures,
            progress_callback=progress_callback,
            execution_ledger=execution_ledger,
        )
        execution_ledger.mark_agent_finished(agent_id, final_beat)
        return "search_cited" if web_search_enabled else "completed"

    @staticmethod
    def _stable_recent_context(
        recent: list[dict],
        *,
        turn_id: str | None,
        agent_id: str,
    ) -> list[dict]:
        if not turn_id:
            return recent
        return [
            message
            for message in recent
            if not (
                message.get("agent_id") == agent_id
                and isinstance(message.get("metadata"), dict)
                and message["metadata"].get("turn_id") == turn_id
            )
        ]

    def _memory_histories(
        self,
        *,
        request: ChatRequest,
        agent_id: str,
        recent: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        local_limit = self.settings.local_memory_context_events
        local_candidate_limit = min(200, max(local_limit, local_limit * 4))
        local_reader = getattr(self.storage, "search_memory_events", None)
        local_candidates = (
            local_reader(
                request.project_id,
                agent_id,
                request.message,
                limit=local_candidate_limit,
            )
            if callable(local_reader)
            else self.storage.list_memory_events(
                request.project_id,
                agent_id,
                limit=local_candidate_limit,
            )
        )
        memory_history = MemoryContext._select_memory_history(
            MemoryContext._consolidate_memory_turns(local_candidates),
            query=request.message,
            limit=local_limit,
            recent=recent,
            agent_id=agent_id,
        )

        global_limit = self.settings.global_memory_context_events
        global_candidate_limit = min(200, max(global_limit, global_limit * 4))
        global_reader = getattr(
            self.storage,
            "search_global_memory_context_events",
            None,
        )
        if callable(global_reader):
            global_candidates = global_reader(
                agent_id,
                request.message,
                exclude_project_id=request.project_id,
                limit=global_candidate_limit,
            )
        else:
            fallback_reader = getattr(
                self.storage,
                "list_global_memory_context_events",
                self.storage.list_global_memory_events,
            )
            global_candidates = fallback_reader(
                agent_id,
                exclude_project_id=request.project_id,
                limit=global_candidate_limit,
            )
        global_history = MemoryContext._select_memory_history(
            MemoryContext._consolidate_memory_turns(global_candidates),
            query=request.message,
            limit=global_limit,
        )
        return memory_history, global_history

    async def _completion_for_beat(
        self,
        *,
        agent_id: str,
        display_name: str,
        request: ChatRequest,
        runtime: RuntimeConfig,
        messages: list[dict],
        tools: list[dict],
        enable_web_search: bool,
        progress_callback: ProgressCallback | None,
        turn_beat: int,
        execution_ledger: ExecutionLedger,
    ) -> AgentCompletion:
        completion = execution_ledger.recover_completion(
            agent_id,
            turn_beat,
            enable_web_search=enable_web_search,
        )
        if completion is not None:
            return completion

        async def report_agent_progress(event: dict[str, Any]) -> None:
            await self._emit(
                progress_callback,
                {
                    "agent_id": agent_id,
                    "display_name": display_name,
                    "provider": runtime.providers[agent_id],
                    "model": runtime.models[agent_id],
                    **event,
                },
            )

        completion = await self.provider_client.run_agent(
            provider=runtime.providers[agent_id],
            model=runtime.models[agent_id],
            messages=messages,
            tools=tools,
            tool_context=ToolContext(
                agent_id=agent_id,
                project_id=request.project_id,
                turn_id=request.turn_id,
                turn_beat=turn_beat,
            ),
            temperature=runtime.temperature,
            max_tokens=runtime.max_tokens,
            require_participation=True,
            enable_web_search=enable_web_search,
            progress_callback=report_agent_progress,
        )
        return execution_ledger.checkpoint_completion(
            agent_id,
            turn_beat,
            completion,
        )

    def _commit_completion(
        self,
        *,
        completion: AgentCompletion,
        agent_id: str,
        request: ChatRequest,
        runtime: RuntimeConfig,
        user_message_id: str,
        public_sources: list[dict],
        research_enabled: bool,
        research_provenance: dict[str, Any] | None,
        turn_beat: int,
        execution_ledger: ExecutionLedger,
    ) -> dict:
        content = completion.content.strip()
        metadata = {
            "turn_id": request.turn_id,
            "research_enabled": research_enabled,
            "tool_events": completion.tool_events,
            "user_message_id": user_message_id,
            "locally_generated_action_summary": completion.locally_generated,
            "web_sources": public_sources,
            "turn_beat": turn_beat,
        }
        if completion.usage:
            metadata["provider_usage"] = completion.usage
        if research_provenance is not None:
            metadata["research_provenance"] = research_provenance

        message_arguments = {
            "agent_id": agent_id,
            "annotations": completion.annotations,
            "metadata": metadata,
        }
        message_operation_id = execution_ledger.message_operation_id(
            agent_id,
            turn_beat,
        )
        if message_operation_id is not None:
            message_arguments["operation_id"] = message_operation_id
        stored = self.storage.add_message(
            request.project_id,
            "agent",
            content,
            **message_arguments,
        )
        execution_ledger.mark_message_committed(
            agent_id,
            turn_beat,
            stored["id"],
        )

        successful_actions = any(
            event.get("result", {}).get("ok") is not False for event in completion.tool_events
        )
        memory_event = self._record_memory(
            request=request,
            user_message_id=user_message_id,
            agent_id=agent_id,
            runtime=runtime,
            outcome="action_response" if successful_actions else "response",
            return_text=content,
            tool_events=completion.tool_events,
            operation_id=execution_ledger.memory_operation_id(
                agent_id,
                turn_beat,
            ),
        )
        execution_ledger.mark_memory_committed(
            agent_id,
            turn_beat,
            memory_event["id"],
        )
        execution_ledger.mark_beat_finished(agent_id, turn_beat)
        return stored

    async def _run_followup_beats(
        self,
        *,
        completion: AgentCompletion,
        initial_messages: list[dict],
        agent_id: str,
        display_name: str,
        request: ChatRequest,
        runtime: RuntimeConfig,
        tools: list[dict],
        user_message_id: str,
        public_sources: list[dict],
        research_enabled: bool,
        visible_responses: list[dict],
        turn_transcript: list[dict],
        agent_failures: list[dict],
        progress_callback: ProgressCallback | None,
        execution_ledger: ExecutionLedger,
    ) -> int:
        beat_number = 1
        beat_messages = [
            *initial_messages,
            {"role": "assistant", "content": completion.content.strip()},
        ]
        while completion.continue_turn and beat_number < self.settings.max_agent_turn_beats:
            beat_number += 1
            await self._emit(
                progress_callback,
                {
                    "type": "agent_followup_start",
                    "agent_id": agent_id,
                    "display_name": display_name,
                    "turn_beat": beat_number,
                },
            )
            followup_instruction = self._followup_instruction(beat_number)
            followup_messages = [
                *beat_messages,
                {"role": "user", "content": followup_instruction},
            ]
            try:
                completion = await self._completion_for_beat(
                    agent_id=agent_id,
                    display_name=display_name,
                    request=request,
                    runtime=runtime,
                    messages=followup_messages,
                    tools=tools,
                    enable_web_search=False,
                    progress_callback=progress_callback,
                    turn_beat=beat_number,
                    execution_ledger=execution_ledger,
                )
            except (ProviderTimeout, ProviderNoResponse, ProviderError) as exc:
                await self._record_provider_failure(
                    exc=exc,
                    agent_id=agent_id,
                    display_name=display_name,
                    request=request,
                    runtime=runtime,
                    user_message_id=user_message_id,
                    agent_failures=agent_failures,
                    progress_callback=progress_callback,
                    turn_beat=beat_number,
                    execution_ledger=execution_ledger,
                )
                break

            stored = self._commit_completion(
                completion=completion,
                agent_id=agent_id,
                request=request,
                runtime=runtime,
                user_message_id=user_message_id,
                public_sources=public_sources,
                research_enabled=research_enabled,
                research_provenance=None,
                turn_beat=beat_number,
                execution_ledger=execution_ledger,
            )
            visible_responses.append(stored)
            turn_transcript.append(stored)
            beat_messages.extend(
                [
                    {"role": "user", "content": followup_instruction},
                    {
                        "role": "assistant",
                        "content": completion.content.strip(),
                    },
                ]
            )
            await self._emit_agent_complete(
                stored,
                agent_id=agent_id,
                display_name=display_name,
                turn_beat=beat_number,
                progress_callback=progress_callback,
            )
        return beat_number

    def _followup_instruction(self, beat_number: int) -> str:
        maximum = self.settings.max_agent_turn_beats
        return (
            f"You requested response beat {beat_number} of {maximum}. Continue the same "
            "turn naturally after rereading what you just said and the room. Add only "
            "the thought, clarification, or conversational afterbeat that needed "
            "separate space. Do not repeat the prior beat. You may append "
            "[[CONTINUE_TURN]] once more only if another distinct beat is genuinely "
            "needed within this turn's configured limit."
        )

    async def _record_provider_failure(
        self,
        *,
        exc: ProviderError,
        agent_id: str,
        display_name: str,
        request: ChatRequest,
        runtime: RuntimeConfig,
        user_message_id: str,
        agent_failures: list[dict],
        progress_callback: ProgressCallback | None,
        execution_ledger: ExecutionLedger,
        turn_transcript: list[dict] | None = None,
        turn_beat: int = 1,
    ) -> None:
        if isinstance(exc, ProviderTimeout):
            kind, event_type = "timeout", "agent_timeout"
        elif isinstance(exc, ProviderNoResponse):
            kind, event_type = "no_response", "agent_no_response"
        else:
            kind, event_type = "provider_error", "agent_provider_error"
        detail = f"Follow-up beat {turn_beat} stopped: {exc}" if turn_beat > 1 else str(exc)
        failure = {
            "agent_id": agent_id,
            "display_name": display_name,
            "provider": runtime.providers[agent_id],
            "model": runtime.models[agent_id],
            "kind": kind,
            "detail": detail,
        }
        if turn_beat > 1:
            failure["turn_beat"] = turn_beat
        agent_failures.append(failure)
        memory_event = self._record_memory(
            request=request,
            user_message_id=user_message_id,
            agent_id=agent_id,
            runtime=runtime,
            outcome=kind,
            return_text=str(exc),
            tool_events=[],
            operation_id=execution_ledger.memory_operation_id(
                agent_id,
                turn_beat,
            ),
        )
        execution_ledger.mark_memory_committed(
            agent_id,
            turn_beat,
            memory_event["id"],
        )
        execution_ledger.mark_beat_finished(agent_id, turn_beat)
        if turn_transcript is not None:
            turn_transcript.append(
                {
                    "role": "system",
                    "content": self._failure_transcript(display_name, kind),
                }
            )
        await self._emit(progress_callback, {"type": event_type, **failure})

    def _record_memory(
        self,
        *,
        request: ChatRequest,
        user_message_id: str,
        agent_id: str,
        runtime: RuntimeConfig,
        outcome: str,
        return_text: str,
        tool_events: list[dict],
        operation_id: str | None,
    ) -> dict:
        actions = []
        for event in tool_events:
            result = event.get("result") if isinstance(event.get("result"), dict) else {}
            arguments = event.get("arguments") if isinstance(event.get("arguments"), dict) else {}
            actions.append(
                {
                    "tool": event.get("tool", ""),
                    "path": result.get("path") or arguments.get("path"),
                    "ok": result.get("ok") is not False,
                    "overwritten": bool(result.get("overwritten", False)),
                }
            )
        local_event = self.storage.add_memory_event(
            request.project_id,
            agent_id,
            user_message_id,
            outcome=outcome,
            trigger_text=request.message,
            return_text=return_text,
            actions=actions,
            provider=runtime.providers[agent_id],
            model=runtime.models[agent_id],
            operation_id=operation_id,
        )
        if outcome in {"response", "action_response"}:
            project = self.storage.get_project(request.project_id)
            self.storage.add_global_memory_event(
                agent_id=agent_id,
                source_project_id=request.project_id,
                source_project_name=project["name"],
                source_memory_event_id=local_event["id"],
                trigger_text=local_event["trigger_text"],
                return_text=local_event["return_text"],
                actions=local_event["actions"],
                created_at=local_event["created_at"],
            )
        return local_event

    @staticmethod
    def _failure_transcript(display_name: str, kind: str) -> str:
        if kind == "timeout":
            return (
                f"{display_name} timed out during this round. The room continued "
                "to the next selected agent."
            )
        if kind == "no_response":
            return (
                f"{display_name} returned neither speech nor a successful action after "
                "two retries. The room recorded the failed turn and continued."
            )
        return (
            f"{display_name}'s provider failed during this round. The error was "
            "recorded and the room continued to the next selected agent."
        )

    async def _emit_agent_complete(
        self,
        stored: dict,
        *,
        agent_id: str,
        display_name: str,
        turn_beat: int,
        progress_callback: ProgressCallback | None,
    ) -> None:
        await self._emit(
            progress_callback,
            {
                "type": "agent_complete",
                "agent_id": agent_id,
                "display_name": display_name,
                "message_id": stored["id"],
                "message": stored,
                "turn_beat": turn_beat,
            },
        )

    @staticmethod
    async def _emit(
        callback: ProgressCallback | None,
        event: dict[str, Any],
    ) -> None:
        if callback is not None:
            await callback(event)
