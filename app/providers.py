from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import httpx

from .agent_tools import AgentToolExecutor, ToolContext
from .config import Settings
from .constants import PROVIDER_ERROR_DETAIL_MAX_CHARS
from .provider_catalog import ProviderRegistry
from .tool_metadata import public_tool_arguments

PROVIDER_TIMEOUT_STATUS_CODES = {408, 504, 524}


class ProviderError(RuntimeError):
    pass


class ProviderTimeout(ProviderError):
    """Raised when one direct-provider request times out."""


class ProviderNoResponse(ProviderError):
    """Raised when a model repeatedly returns neither speech nor a successful action."""


@dataclass
class AgentCompletion:
    content: str
    annotations: list[dict]
    raw_message: dict[str, Any]
    tool_events: list[dict]
    locally_generated: bool = False
    continue_turn: bool = False


class DirectProviderClient:
    """One client for catalog-defined OpenAI-compatible Chat Completions APIs."""

    def __init__(
        self,
        settings: Settings,
        tool_executor: AgentToolExecutor,
        provider_registry: ProviderRegistry | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.tool_executor = tool_executor
        self.provider_registry = provider_registry
        self._owns_http_client = http_client is None
        self.http_client = http_client

    async def aclose(self) -> None:
        if self._owns_http_client and self.http_client is not None:
            await self.http_client.aclose()
            self.http_client = None

    def _client(self) -> httpx.AsyncClient:
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(
                follow_redirects=False,
                trust_env=False,
            )
        return self.http_client

    async def run_agent(
        self,
        *,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_context: ToolContext,
        temperature: float,
        max_tokens: int,
        require_participation: bool = True,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> AgentCompletion:
        options = self._provider_options(provider)
        if options is None:
            raise ProviderError(
                f"Provider {provider!r} is not enabled in the provider catalog."
            )
        label = str(options["label"])
        key = self._provider_api_key(provider)
        if options["api_key_required"] and not key:
            key_env = str(options.get("api_key_env") or "the configured variable")
            raise ProviderError(
                f"{label} API key is missing. Add {key_env} to .env and restart "
                "the container."
            )
        if not model:
            raise ProviderError(
                f"No direct {label} model ID is configured for this agent."
            )

        conversation = [dict(message) for message in messages]
        tool_events: list[dict] = []
        available_tools = tools if options["supports_tools"] else []
        participation_retries = 0
        total_tool_calls = 0
        participation_retry_limit = self.settings.provider_participation_retries
        for round_index in range(self.settings.max_tool_rounds + 1):
            if progress_callback is not None:
                await progress_callback({"type": "model_request", "round": round_index + 1})
            payload = self._chat_payload(
                options=options,
                model=model,
                messages=conversation,
                tools=available_tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response_data = await self._post_chat(provider, payload, key)
            message, tool_calls = self._response_message(response_data, label)
            response_tool_limit = self.settings.provider_tool_calls_per_response
            turn_tool_limit = self.settings.provider_tool_calls_per_turn
            if len(tool_calls) > response_tool_limit:
                raise ProviderError(
                    f"{label} returned more than {response_tool_limit} tool calls at once."
                )
            total_tool_calls += len(tool_calls)
            if total_tool_calls > turn_tool_limit:
                raise ProviderError(
                    f"The agent exceeded the {turn_tool_limit}-tool-call turn limit."
                )

            if not tool_calls:
                completion, participation_retries = await self._finish_response(
                    message=message,
                    label=label,
                    conversation=conversation,
                    tool_events=tool_events,
                    require_participation=require_participation,
                    participation_retries=participation_retries,
                    participation_retry_limit=participation_retry_limit,
                    can_retry=round_index < self.settings.max_tool_rounds,
                    progress_callback=progress_callback,
                )
                if completion is not None:
                    return completion
                continue

            if round_index >= self.settings.max_tool_rounds:
                raise ProviderError("The agent exceeded the allowed number of tool rounds.")

            await self._execute_tool_calls(
                message=message,
                tool_calls=tool_calls,
                round_index=round_index,
                conversation=conversation,
                tool_events=tool_events,
                tool_context=tool_context,
                progress_callback=progress_callback,
            )

        raise ProviderError("Tool loop ended unexpectedly.")

    @staticmethod
    def _chat_payload(
        *,
        options: dict[str, Any],
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            str(options["token_parameter"]): max_tokens,
        }
        if options["supports_temperature"]:
            payload["temperature"] = temperature
        if options.get("reasoning_effort"):
            payload["reasoning_effort"] = options["reasoning_effort"]
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    @staticmethod
    def _response_message(
        response_data: dict[str, Any],
        label: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        choices = response_data.get("choices") or []
        if not isinstance(choices, list) or not choices:
            raise ProviderError(f"{label} returned no completion choices.")
        if not isinstance(choices[0], dict):
            raise ProviderError(f"{label} returned an invalid completion choice.")
        message = choices[0].get("message") or {}
        if not isinstance(message, dict):
            raise ProviderError(f"{label} returned an invalid completion message.")
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            raise ProviderError(f"{label} returned an invalid tool-call collection.")
        if any(not isinstance(call, dict) for call in tool_calls):
            raise ProviderError(f"{label} returned an invalid tool call.")
        return message, tool_calls

    async def _finish_response(
        self,
        *,
        message: dict[str, Any],
        label: str,
        conversation: list[dict[str, Any]],
        tool_events: list[dict],
        require_participation: bool,
        participation_retries: int,
        participation_retry_limit: int,
        can_retry: bool,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> tuple[AgentCompletion | None, int]:
        content = self._normalize_content(message.get("content"))
        content, continue_turn = self._extract_continue_turn(content)
        declined = self._is_pass(content)
        if (
            require_participation
            and (not content or declined)
            and participation_retries < participation_retry_limit
            and can_retry
        ):
            participation_retries += 1
            if content:
                conversation.append({"role": "assistant", "content": content})
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        "You are a selected participant and must take a visible turn. Do not pass, remain "
                        "silent, or return an empty answer. Respond conversationally from your own lens. "
                        "Agreement is allowed; if prior replies cover the topic, state what you agree with "
                        "and add a question, implication, or observation. If you used project tools, "
                        "synthesize what you did and mention any file you created or changed."
                    ),
                }
            )
            if progress_callback is not None:
                await progress_callback(
                    {
                        "type": "synthesizing" if tool_events else "participation_retry",
                        "attempt": participation_retries,
                    }
                )
            return None, participation_retries
        if require_participation and (not content or declined):
            successful_actions = [
                event
                for event in tool_events
                if event.get("result", {}).get("ok") is not False
            ]
            if successful_actions:
                return (
                    AgentCompletion(
                        content=self._action_fallback(successful_actions),
                        annotations=[],
                        raw_message={"local_action_fallback": True},
                        tool_events=tool_events,
                        locally_generated=True,
                    ),
                    participation_retries,
                )
            raise ProviderNoResponse(
                f"{label} returned no written response or successful action "
                f"after {participation_retry_limit} retries."
            )
        return (
            AgentCompletion(
                content,
                message.get("annotations") or [],
                message,
                tool_events,
                continue_turn=continue_turn,
            ),
            participation_retries,
        )

    async def _execute_tool_calls(
        self,
        *,
        message: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        round_index: int,
        conversation: list[dict[str, Any]],
        tool_events: list[dict],
        tool_context: ToolContext,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> None:
        conversation.append(
            {
                "role": "assistant",
                "content": self._normalize_content(message.get("content"))
                or "I’m using the available project tools to continue.",
                "tool_calls": tool_calls,
            }
        )
        for call in tool_calls:
            call_id = call.get("id") or f"tool-{round_index}-{len(tool_events)}"
            function = call.get("function") or {}
            name = function.get("name", "")
            arguments: dict[str, Any] = {}
            try:
                arguments = self.tool_executor.parse_arguments(
                    function.get("arguments", "{}")
                )
                result = self.tool_executor.execute(name, arguments, tool_context)
            except ValueError as exc:
                result = {
                    "ok": False,
                    "retryable": True,
                    "error": (
                        "The model produced malformed JSON for this tool call. Call the same "
                        "tool again with one complete, valid JSON object. Do not summarize the "
                        "failure as the final outcome."
                    ),
                    "detail": str(exc),
                }
            public_event = {
                "tool": name,
                "arguments": self._public_tool_arguments(name, arguments),
                "result": self._public_tool_result(name, result),
            }
            tool_events.append(public_event)
            if progress_callback is not None:
                await progress_callback({"type": "tool", **public_event})
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    @staticmethod
    def _public_tool_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
        public = dict(result)
        public.pop("content", None)
        public.pop("content_text", None)
        if name == "search_project_files" and isinstance(public.get("results"), list):
            public["results"] = [
                {key: item[key] for key in ("path", "score") if key in item}
                for item in public["results"] if isinstance(item, dict)
            ]
        return public

    @staticmethod
    def _public_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return public_tool_arguments(name, arguments)

    async def _post_chat(self, provider: str, payload: dict[str, Any], key: str) -> dict[str, Any]:
        options = self._provider_options(provider)
        label = str(options["label"]) if options else provider.title()
        base_url = self._provider_base_url(provider)
        if not base_url:
            raise ProviderError(f"No base URL is configured for {label}.")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        timeout = httpx.Timeout(
            self.settings.provider_timeout_seconds,
            connect=self.settings.provider_connect_timeout_seconds,
        )
        response_limit = self.settings.provider_response_max_bytes
        try:
            async with self._client().stream(
                "POST",
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            ) as response:
                declared_length = response.headers.get("content-length")
                if declared_length:
                    try:
                        if int(declared_length) > response_limit:
                            raise ProviderError(
                                f"{label} response exceeds the {response_limit}-byte limit."
                            )
                    except ValueError:
                        pass
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > response_limit:
                        raise ProviderError(
                            f"{label} response exceeds the {response_limit}-byte limit."
                        )
                status_code = response.status_code
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(
                f"{label} timed out before this agent returned a response."
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Could not reach {label}: {exc}") from exc

        decoded = bytes(body).decode("utf-8", errors="replace")
        if status_code >= 400:
            detail = decoded[:PROVIDER_ERROR_DETAIL_MAX_CHARS]
            with suppress(TypeError, ValueError):
                detail = self._format_error_detail(json.loads(decoded), detail)
            if status_code in PROVIDER_TIMEOUT_STATUS_CODES:
                raise ProviderTimeout(
                    f"{label} timed out while this agent was responding: {detail}"
                )
            raise ProviderError(f"{label} error {status_code}: {detail}")
        try:
            parsed = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"{label} returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise ProviderError(f"{label} returned an invalid JSON response shape.")
        return parsed

    def _provider_options(self, provider: str) -> dict[str, Any] | None:
        if self.provider_registry is not None:
            profile = self.provider_registry.profile(provider)
            if profile is None or not profile.enabled:
                return None
            return profile.model_dump()
        is_openai = provider == "openai"
        return {
            "id": provider,
            "label": provider.title(),
            "enabled": True,
            "api_key_env": f"{provider.upper()}_API_KEY",
            "api_key_required": True,
            "token_parameter": (
                "max_completion_tokens" if is_openai else "max_tokens"
            ),
            "supports_temperature": not is_openai,
            "supports_tools": True,
            "reasoning_effort": "none" if is_openai else None,
        }

    def _provider_api_key(self, provider: str) -> str:
        if self.provider_registry is not None:
            return self.provider_registry.api_key(provider)
        return self.settings.provider_api_key(provider)

    def _provider_base_url(self, provider: str) -> str:
        if self.provider_registry is not None:
            profile = self.provider_registry.profile(provider)
            return profile.base_url if profile and profile.enabled else ""
        return self.settings.provider_base_url(provider)

    @staticmethod
    def _format_error_detail(payload: dict[str, Any], fallback: str) -> str:
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            parts = [str(error.get(key)) for key in ("message", "type", "code", "param") if error.get(key) not in (None, "")]
            return (
                " / ".join(dict.fromkeys(parts))[:PROVIDER_ERROR_DETAIL_MAX_CHARS]
                or fallback
            )
        if error not in (None, ""):
            return str(error)[:PROVIDER_ERROR_DETAIL_MAX_CHARS]
        return fallback[:PROVIDER_ERROR_DETAIL_MAX_CHARS]

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(part for part in parts if part).strip()
        return ""

    @staticmethod
    def _is_pass(content: str) -> bool:
        normalized = content.strip().upper().rstrip(".!")
        return normalized in {"PASS", "[[PASS]]"}

    @staticmethod
    def _extract_continue_turn(content: str) -> tuple[str, bool]:
        marker = "[[CONTINUE_TURN]]"
        stripped = content.rstrip()
        if not stripped.endswith(marker):
            return content, False
        return stripped[: -len(marker)].rstrip(), True

    @staticmethod
    def _action_fallback(events: list[dict[str, Any]]) -> str:
        descriptions: list[str] = []
        for event in events:
            tool = event.get("tool", "project tool")
            result = event.get("result") if isinstance(event.get("result"), dict) else {}
            arguments = event.get("arguments") if isinstance(event.get("arguments"), dict) else {}
            path = result.get("path") or arguments.get("path")
            label = tool.replace("_", " ")
            descriptions.append(f"{label}{f' ({path})' if path else ''}")
        joined = ", ".join(dict.fromkeys(descriptions))
        return f"I completed my turn through project action: {joined}."
