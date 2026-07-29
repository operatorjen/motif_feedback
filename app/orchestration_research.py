from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .models import RuntimeConfig
from .orchestration_prompts import PromptBuilder
from .providers import AgentCompletion

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]


def search_fallback_eligible(failure: dict) -> bool:
    if "search_fallback_eligible" in failure:
        return failure.get("search_fallback_eligible") is True
    status_code = failure.get("status_code")
    return status_code in {401, 403, 429, 451} or (
        isinstance(status_code, int) and status_code >= 500
    )


async def record_search_evidence_failure(
    *,
    source_failures: list[dict],
    runtime: RuntimeConfig,
    agent_id: str,
    detail: str,
    progress_callback: ProgressCallback | None,
) -> None:
    failed_urls = [
        str(failure.get("url", ""))
        for failure in source_failures
        if failure.get("url")
        and search_fallback_eligible(failure)
        and failure.get("retrieval_method") == "direct_http"
    ]
    for url in failed_urls:
        failure = {
            "url": url,
            "detail": detail,
            "retrieval_method": "agent_search",
            "provider": runtime.providers[agent_id],
            "model": runtime.models[agent_id],
        }
        source_failures.append(failure)
        if progress_callback is not None:
            await progress_callback({"type": "source_search_no_evidence", **failure})


def research_provenance(
    source_failures: list[dict],
    completion: AgentCompletion,
    *,
    provider: str,
    model: str,
) -> dict[str, Any]:
    failures = [
        {
            key: failure[key]
            for key in (
                "url",
                "status_code",
                "attempt_count",
                "reason",
                "retrieval_method",
            )
            if key in failure
        }
        for failure in source_failures
        if search_fallback_eligible(failure)
    ]
    return {
        "method": "agent_search",
        "trigger": (
            "direct_http_403"
            if failures and all(item.get("status_code") == 403 for item in failures)
            else "recoverable_direct_read_failure"
        ),
        "provider": provider,
        "model": model,
        "direct_retrieval": failures,
        "citations": PromptBuilder._annotation_sources(completion.annotations),
    }
