from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import yaml

from .agent_turn_execution import AgentTurnExecutor
from .config import Settings
from .execution_ledger import ExecutionLedger
from .models import AGENT_IDS, ChatRequest, RuntimeConfig
from .orchestration_prompts import PromptBuilder
from .orchestration_research import search_fallback_eligible
from .persona_store import PersonaStore
from .providers import DirectProviderClient
from .role_decorators import pending_role_signals
from .search_router import SearchDecision, SearchRouter
from .storage import Storage
from .web_sources import WebSourceService


@dataclass
class RoomResponse:
    messages: list[dict]
    research: dict
    agent_failures: list[dict]
    web_sources: list[dict]
    source_failures: list[dict]


ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        persona_store: PersonaStore,
        provider_client: DirectProviderClient,
        search_router: SearchRouter,
        web_source_service: WebSourceService | None = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.persona_store = persona_store
        self.provider_client = provider_client
        self.search_router = search_router
        self.web_source_service = web_source_service
        self.prompt_builder = PromptBuilder(settings)
        self.agent_turn_executor = AgentTurnExecutor(
            settings=settings,
            storage=storage,
            persona_store=persona_store,
            provider_client=provider_client,
            prompt_builder=self.prompt_builder,
        )

    async def chat(
        self,
        request: ChatRequest,
        runtime: RuntimeConfig,
        progress_callback: ProgressCallback | None = None,
        existing_user_message: dict | None = None,
    ) -> RoomResponse:
        project = self.storage.get_project(request.project_id)
        participants = [agent_id for agent_id in AGENT_IDS if agent_id in request.participants]
        decision = self.search_router.decide(request.message, request.research_mode, participants)
        earlier = self.storage.recent_messages(
            request.project_id, self.settings.max_context_messages
        )
        role_signals = pending_role_signals(earlier)
        order = self._speaker_order(request.message, participants, decision, earlier)
        web_sources: list[dict] = []
        source_failures: list[dict] = []
        if self.web_source_service is not None:
            web_sources, source_failures = await self.web_source_service.collect_for_prompt(
                request.project_id,
                request.message,
                progress_callback,
            )
        search_fallback_failures = [
            failure for failure in source_failures if search_fallback_eligible(failure)
        ]
        search_fallback_agent = self._select_search_fallback_agent(
            order=order,
            runtime=runtime,
            decision=decision,
            source_failures=search_fallback_failures,
        )
        if search_fallback_agent is not None and order[0] != search_fallback_agent:
            order = [
                search_fallback_agent,
                *[agent_id for agent_id in order if agent_id != search_fallback_agent],
            ]
        research = {
            **decision.model_dump(),
            "search_fallback_agent": search_fallback_agent,
            "search_fallback_urls": [failure["url"] for failure in search_fallback_failures],
            "evidence_status": (
                "direct"
                if web_sources
                else "search_pending"
                if search_fallback_agent is not None
                else "unavailable"
                if source_failures
                else "not_requested"
            ),
            "agent_turns_skipped": bool(
                source_failures and not web_sources and search_fallback_agent is None
            ),
        }
        await self._emit_progress(
            progress_callback,
            {"type": "turn_start", "agents": order, "research": research},
        )
        if search_fallback_agent is not None:
            await self._emit_progress(
                progress_callback,
                {
                    "type": "source_search_fallback",
                    "agent_id": search_fallback_agent,
                    "urls": research["search_fallback_urls"],
                },
            )
        public_sources = [WebSourceService.public_source(source) for source in web_sources]
        user_metadata = {
            "turn_id": request.turn_id,
            "web_sources": public_sources,
            "web_source_failures": source_failures,
            "search_fallback_agent": search_fallback_agent,
            "evidence_status": research["evidence_status"],
            "agent_turns_skipped": research["agent_turns_skipped"],
        }
        if existing_user_message is None:
            user_message = self.storage.add_message(
                request.project_id,
                "user",
                request.message,
                metadata=user_metadata,
            )
        else:
            user_message = existing_user_message
            self.storage.update_message_metadata(
                request.project_id,
                user_message["id"],
                user_metadata,
            )
        if research["agent_turns_skipped"]:
            await self._emit_progress(
                progress_callback,
                {
                    "type": "source_no_evidence",
                    "detail": (
                        "No supplied page could be retrieved, so the selected agents "
                        "were not prompted."
                    ),
                },
            )
            return await self._complete_room(
                research=research,
                web_sources=public_sources,
                source_failures=source_failures,
                progress_callback=progress_callback,
            )
        recent = self.storage.recent_messages(
            request.project_id, self.settings.max_context_messages
        )
        visible_responses: list[dict] = []
        turn_transcript: list[dict] = []
        agent_failures: list[dict] = []
        shared_context = self.persona_store.load_shared_context()
        project_context = yaml.safe_dump(
            project,
            sort_keys=False,
            allow_unicode=True,
        )
        source_context = self.prompt_builder._format_web_sources(web_sources)
        execution_ledger = ExecutionLedger(self.storage, request, runtime)

        for agent_id in order:
            turn_status = await self.agent_turn_executor.run(
                agent_id=agent_id,
                request=request,
                runtime=runtime,
                project=project,
                project_context=project_context,
                shared_context=shared_context,
                web_sources=web_sources,
                source_context=source_context,
                role_signals=role_signals,
                decision=decision,
                source_failures=source_failures,
                search_fallback_agent=search_fallback_agent,
                recent=recent,
                public_sources=public_sources,
                user_message_id=user_message["id"],
                visible_responses=visible_responses,
                turn_transcript=turn_transcript,
                agent_failures=agent_failures,
                progress_callback=progress_callback,
                execution_ledger=execution_ledger,
            )
            if agent_id == search_fallback_agent:
                if turn_status == "search_cited":
                    research["evidence_status"] = (
                        "direct_and_search" if web_sources else "search_cited"
                    )
                elif turn_status in {"search_failed", "search_uncited"}:
                    research["evidence_status"] = "direct" if web_sources else "unavailable"
                    if not web_sources:
                        research["agent_turns_skipped"] = True
                self.storage.update_message_metadata(
                    request.project_id,
                    user_message["id"],
                    {
                        "web_source_failures": source_failures,
                        "evidence_status": research["evidence_status"],
                        "agent_turns_skipped": research["agent_turns_skipped"],
                    },
                )
            if turn_status in {"search_failed", "search_uncited"} and not web_sources:
                await self._emit_progress(
                    progress_callback,
                    {
                        "type": "source_no_evidence",
                        "detail": (
                            "The search fallback returned no cited evidence, so the "
                            "remaining agents were not prompted."
                        ),
                    },
                )
                break

        return await self._complete_room(
            messages=visible_responses,
            research=research,
            agent_failures=agent_failures,
            web_sources=public_sources,
            source_failures=source_failures,
            progress_callback=progress_callback,
        )

    async def _complete_room(
        self,
        *,
        research: dict,
        web_sources: list[dict],
        source_failures: list[dict],
        progress_callback: ProgressCallback | None,
        messages: list[dict] | None = None,
        agent_failures: list[dict] | None = None,
    ) -> RoomResponse:
        response = RoomResponse(
            messages=messages or [],
            research=research,
            agent_failures=agent_failures or [],
            web_sources=web_sources,
            source_failures=source_failures,
        )
        await self._emit_progress(progress_callback, {"type": "turn_complete"})
        return response

    @staticmethod
    async def _emit_progress(
        callback: ProgressCallback | None,
        event: dict[str, Any],
    ) -> None:
        if callback is not None:
            await callback(event)

    def _select_search_fallback_agent(
        self,
        *,
        order: list[str],
        runtime: RuntimeConfig,
        decision: SearchDecision,
        source_failures: list[dict],
    ) -> str | None:
        if (
            not source_failures
            or not decision.needs_search
            or not getattr(self.settings, "web_fetch_search_fallback", True)
        ):
            return None
        supports_web_search = getattr(
            self.provider_client,
            "supports_web_search",
            None,
        )
        if not callable(supports_web_search):
            return None
        return next(
            (agent_id for agent_id in order if supports_web_search(runtime.providers[agent_id])),
            None,
        )

    def _speaker_order(
        self,
        message: str,
        participants: list[str],
        decision: SearchDecision,
        recent: list[dict],
    ) -> list[str]:
        selected = [agent_id for agent_id in AGENT_IDS if agent_id in participants]
        if not selected:
            selected = list(AGENT_IDS)

        lowered = message.lower()
        aliases = {
            "agent_a": ("phenomenologist", "phenomenology", "agent a"),
            "agent_b": ("cyberneticist", "cybernetics", "agent b"),
            "agent_c": ("game theorist", "game theory", "agent c"),
        }
        explicit = next(
            (
                agent_id
                for agent_id in selected
                if any(alias in lowered for alias in aliases[agent_id])
            ),
            None,
        )
        first = explicit or decision.lead_agent

        if first is None:
            last_agent = next(
                (item.get("agent_id") for item in reversed(recent) if item.get("role") == "agent"),
                None,
            )
            if last_agent in selected:
                first = selected[(selected.index(last_agent) + 1) % len(selected)]
            else:
                first = selected[0]

        return [first, *[agent_id for agent_id in selected if agent_id != first]]
