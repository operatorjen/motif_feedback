from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from .agent_tools import USER_TOOL_DEFINITIONS, ToolContext
from .execution_ledger import ExecutionLedger, stable_fingerprint
from .memory_loops import memory_loop_for
from .models import ChatRequest, RuntimeConfig
from .motif_checkpoints import agent_pattern_checkpoints
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
CONTEXT_SELECTOR_VERSION = "context-selector-v1"


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
        speaker_position: int,
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
        motif_reader = getattr(self.storage, "list_motifs", None)
        motif_context = (
            motif_reader(
                request.project_id,
                observer_agent_id=agent_id,
                statuses={"candidate", "supported", "active", "dormant"},
                limit=20,
            )
            if callable(motif_reader)
            else []
        )
        room_motif_context = (
            [
                motif
                for motif in motif_reader(
                    request.project_id,
                    statuses={"supported", "active"},
                    limit=40,
                )
                if motif.get("observer_agent_id") != agent_id
            ][:12]
            if callable(motif_reader)
            else []
        )
        pattern_checkpoints = (
            agent_pattern_checkpoints(
                self.storage,
                request.project_id,
                agent_id,
            )
            if callable(getattr(self.storage, "primary_motif_event_sequence", None))
            and callable(getattr(self.storage, "list_motif_pattern_preferences", None))
            else []
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
            motif_context=motif_context,
            room_motif_context=room_motif_context,
            pattern_checkpoints=pattern_checkpoints,
        )
        messages = self.prompt_builder._agent_messages(
            system_prompt,
            agent_recent + turn_transcript,
            web_sources,
            source_context=source_context,
        )
        tools = list(USER_TOOL_DEFINITIONS)
        base_exposures = self._context_exposures(
            project_id=request.project_id,
            recent=agent_recent,
            same_turn=turn_transcript,
            memory_history=memory_history,
            global_memory_history=global_memory_history,
            motif_context=motif_context,
            room_motif_context=room_motif_context,
            pattern_checkpoints=pattern_checkpoints,
            web_sources=web_sources,
            role_signals=role_signals,
        )
        prompt_template_hash = self._prompt_template_hash()
        persona_revision_hash = stable_fingerprint(persona)
        prompt_run_id = self._record_prompt_run(
            request=request,
            runtime=runtime,
            agent_id=agent_id,
            turn_beat=1,
            speaker_position=speaker_position,
            prompt_template_hash=prompt_template_hash,
            persona_revision_hash=persona_revision_hash,
            exposures=base_exposures,
        )

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
                user_message_id=user_message_id,
            )
        except (ProviderTimeout, ProviderNoResponse, ProviderError) as exc:
            self._finalize_prompt_run(prompt_run_id, status="failed")
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
            self._finalize_prompt_run(
                prompt_run_id,
                status="discarded",
                provider_usage=completion.usage,
                provider_request_usage=completion.request_usage,
                output_chars=len(completion.content),
            )
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
            prompt_run_id=prompt_run_id,
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
            speaker_position=speaker_position,
            prompt_template_hash=prompt_template_hash,
            persona_revision_hash=persona_revision_hash,
            base_exposures=base_exposures,
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
        user_message_id: str,
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
                user_message_id=user_message_id,
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
        prompt_run_id: str | None,
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
        evidence_attacher = getattr(
            self.storage,
            "attach_motif_response_evidence",
            None,
        )
        if callable(evidence_attacher) and request.turn_id:
            evidence_attacher(
                project_id=request.project_id,
                observer_agent_id=agent_id,
                turn_id=request.turn_id,
                turn_beat=turn_beat,
                message_id=stored["id"],
            )
        execution_ledger.mark_message_committed(
            agent_id,
            turn_beat,
            stored["id"],
        )

        successful_actions = any(
            event.get("tool") != "record_motif_observations"
            and event.get("result", {}).get("ok") is not False
            for event in completion.tool_events
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
        self._finalize_prompt_run(
            prompt_run_id,
            status="completed",
            message_id=stored["id"],
            provider_usage=completion.usage,
            provider_request_usage=completion.request_usage,
            output_chars=len(content),
        )
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
        speaker_position: int,
        prompt_template_hash: str,
        persona_revision_hash: str,
        base_exposures: list[dict[str, Any]],
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
            followup_exposures = [
                *base_exposures,
                *self._message_exposures(
                    turn_transcript,
                    context_kind="same_turn_message",
                    prompt_section="followup_turn_transcript",
                    selection_reason="same_turn_sequence",
                ),
            ]
            prompt_run_id = self._record_prompt_run(
                request=request,
                runtime=runtime,
                agent_id=agent_id,
                turn_beat=beat_number,
                speaker_position=speaker_position,
                prompt_template_hash=prompt_template_hash,
                persona_revision_hash=persona_revision_hash,
                exposures=followup_exposures,
            )
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
                    user_message_id=user_message_id,
                )
            except (ProviderTimeout, ProviderNoResponse, ProviderError) as exc:
                self._finalize_prompt_run(prompt_run_id, status="failed")
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
                prompt_run_id=prompt_run_id,
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

    @staticmethod
    def _prompt_template_hash() -> str:
        return stable_fingerprint(
            {
                "system_prompt_builder": inspect.getsource(PromptBuilder._build_system_prompt),
                "agent_message_builder": inspect.getsource(PromptBuilder._agent_messages),
                "followup_builder": inspect.getsource(AgentTurnExecutor._followup_instruction),
            }
        )

    def _record_prompt_run(
        self,
        *,
        request: ChatRequest,
        runtime: RuntimeConfig,
        agent_id: str,
        turn_beat: int,
        speaker_position: int,
        prompt_template_hash: str,
        persona_revision_hash: str,
        exposures: list[dict[str, Any]],
    ) -> str | None:
        recorder = getattr(self.storage, "record_agent_prompt_run", None)
        if not request.turn_id or not callable(recorder):
            return None
        return recorder(
            project_id=request.project_id,
            turn_id=request.turn_id,
            agent_id=agent_id,
            turn_beat=turn_beat,
            speaker_position=speaker_position,
            provider=runtime.providers[agent_id],
            model=runtime.models[agent_id],
            prompt_template_hash=prompt_template_hash,
            persona_revision_hash=persona_revision_hash,
            context_selector_version=CONTEXT_SELECTOR_VERSION,
            exposures=exposures,
        )

    def _finalize_prompt_run(
        self,
        prompt_run_id: str | None,
        *,
        status: str,
        message_id: str | None = None,
        provider_usage: dict[str, Any] | None = None,
        provider_request_usage: list[dict[str, Any]] | None = None,
        output_chars: int | None = None,
    ) -> None:
        finalizer = getattr(self.storage, "complete_agent_prompt_run", None)
        if prompt_run_id is None or not callable(finalizer):
            return
        finalizer(
            prompt_run_id,
            status=status,
            message_id=message_id,
            provider_usage=provider_usage,
            provider_request_usage=provider_request_usage,
            output_chars=output_chars,
        )

    def _context_exposures(
        self,
        *,
        project_id: str,
        recent: list[dict],
        same_turn: list[dict],
        memory_history: list[dict],
        global_memory_history: list[dict],
        motif_context: list[dict],
        room_motif_context: list[dict],
        pattern_checkpoints: list[dict],
        web_sources: list[dict],
        role_signals: list[dict],
    ) -> list[dict[str, Any]]:
        exposures = [
            *self._message_exposures(
                recent,
                context_kind="recent_message",
                prompt_section="room_transcript",
                selection_reason="recent_context_window",
            ),
            *self._message_exposures(
                same_turn,
                context_kind="same_turn_message",
                prompt_section="room_transcript",
                selection_reason="same_turn_sequence",
            ),
        ]
        groups = (
            (
                memory_history,
                "local_memory",
                "recent_loop_returns",
                "project_memory_selection",
            ),
            (
                global_memory_history,
                "global_memory",
                "cross_project_returns",
                "cross_project_memory_selection",
            ),
            (
                motif_context,
                "own_motif",
                "own_motif_hypotheses",
                "observer_motif_context",
            ),
            (
                room_motif_context,
                "other_observer_motif",
                "room_motif_hypotheses",
                "supported_room_motif_context",
            ),
            (
                pattern_checkpoints,
                "pattern_checkpoint",
                "pattern_checkpoints",
                "established_pattern_checkpoint",
            ),
            (
                web_sources,
                "web_source",
                "web_sources",
                "user_supplied_source",
            ),
            (
                role_signals,
                "role_signal",
                "role_decorators",
                "bounded_script_signal",
            ),
        )
        for items, kind, section, reason in groups:
            for rank, item in enumerate(items, start=1):
                source_id = str(
                    item.get("id")
                    or item.get("pattern_key")
                    or stable_fingerprint(item)[:32]
                )
                exposures.append(
                    self._exposure(
                        item,
                        context_kind=kind,
                        source_id=source_id,
                        source_project_id=(
                            item.get("project_id")
                            or item.get("source_project_id")
                            or project_id
                        ),
                        prompt_section=section,
                        rank=rank,
                        selection_reason=reason,
                    )
                )
        return exposures

    def _message_exposures(
        self,
        messages: list[dict],
        *,
        context_kind: str,
        prompt_section: str,
        selection_reason: str,
    ) -> list[dict[str, Any]]:
        return [
            self._exposure(
                message,
                context_kind=context_kind,
                source_id=str(message.get("id") or stable_fingerprint(message)[:32]),
                source_project_id=message.get("project_id"),
                prompt_section=prompt_section,
                rank=rank,
                selection_reason=selection_reason,
            )
            for rank, message in enumerate(messages, start=1)
        ]

    @staticmethod
    def _exposure(
        item: dict,
        *,
        context_kind: str,
        source_id: str,
        source_project_id: str | None,
        prompt_section: str,
        rank: int,
        selection_reason: str,
    ) -> dict[str, Any]:
        text_parts = [
            item.get("content"),
            item.get("return_text"),
            item.get("return_summary"),
            item.get("trigger_text"),
            item.get("trigger_summary"),
            item.get("description"),
            item.get("title"),
        ]
        estimated_chars = sum(len(str(value)) for value in text_parts if value)
        return {
            "context_kind": context_kind,
            "source_id": source_id,
            "source_project_id": source_project_id,
            "prompt_section": prompt_section,
            "rank": rank,
            "selection_reason": selection_reason,
            "source_version_hash": stable_fingerprint(item),
            "estimated_chars": estimated_chars,
        }

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
            if event.get("tool") == "record_motif_observations":
                continue
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
            self.storage.add_global_memory_event(
                agent_id=agent_id,
                source_project_id=request.project_id,
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
