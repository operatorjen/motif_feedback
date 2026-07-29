import asyncio
from types import SimpleNamespace

import httpx
import pytest

from app.agent_tools import AgentToolExecutor, ToolContext
from app.providers import DirectProviderClient, ProviderError, ProviderTimeout
from app.tool_metadata import (
    tool_changes_state,
    tool_recovery_strategy,
    tool_request_fingerprint,
)


class FakeToolExecutor:
    @staticmethod
    def parse_arguments(_raw):
        return {"path": "post.md"}

    @staticmethod
    def execute(_name, _arguments, _context):
        return {
            "ok": True,
            "path": "post.md",
            "content": "full private file contents",
            "truncated": False,
            "size_bytes": 26,
        }


def provider_settings(**overrides):
    values = {
        "max_tool_rounds": 6,
        "provider_api_key": lambda _provider: "test-key",
        "provider_base_url": lambda _provider: "https://provider.test/v1",
        "provider_timeout_seconds": 1,
        "provider_connect_timeout_seconds": 20,
        "provider_participation_retries": 2,
        "provider_response_max_bytes": 2_000_000,
        "provider_tool_calls_per_response": 16,
        "provider_tool_calls_per_turn": 48,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_empty_response_after_tools_gets_a_synthesis_retry():
    settings = provider_settings()
    client = DirectProviderClient(settings, FakeToolExecutor())
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "read_project_file",
                                        "arguments": '{"path":"post.md"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "", "tool_calls": []}}]},
            {
                "choices": [
                    {
                        "message": {
                            "content": "I read the post. Here is my conversational answer.",
                            "tool_calls": [],
                        }
                    }
                ]
            },
        ]
    )
    request_count = 0
    sent_payloads = []

    async def fake_post(_provider, payload, _key):
        nonlocal request_count
        request_count += 1
        sent_payloads.append(payload)
        return next(responses)

    progress = []

    async def report(event):
        progress.append(event)

    client._post_chat = fake_post
    completion = asyncio.run(
        client.run_agent(
            provider="moonshot",
            model="provider/model",
            messages=[{"role": "user", "content": "Review the post."}],
            tools=[{"type": "function"}],
            tool_context=ToolContext("agent_a", "general"),
            temperature=0.5,
            max_tokens=500,
            progress_callback=report,
        )
    )

    assert request_count == 3
    assert completion.content.startswith("I read the post")
    assert completion.tool_events[0]["result"]["path"] == "post.md"
    assert "content" not in completion.tool_events[0]["result"]
    assert any(event["type"] == "synthesizing" for event in progress)
    for payload in sent_payloads:
        assistant_messages = [
            message for message in payload["messages"] if message.get("role") == "assistant"
        ]
        assert all(str(message.get("content", "")).strip() for message in assistant_messages)


def test_provider_usage_is_accumulated_across_tool_rounds():
    settings = provider_settings()
    client = DirectProviderClient(settings, FakeToolExecutor())
    responses = iter(
        [
            {
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "read_project_file",
                                        "arguments": '{"path":"post.md"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
            {
                "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
                "choices": [
                    {
                        "message": {
                            "content": "Finished.",
                            "tool_calls": [],
                        }
                    }
                ],
            },
        ]
    )

    async def fake_post(_provider, _payload, _key):
        return next(responses)

    client._post_chat = fake_post
    completion = asyncio.run(
        client.run_agent(
            provider="moonshot",
            model="provider/model",
            messages=[{"role": "user", "content": "Read the post."}],
            tools=[{"type": "function"}],
            tool_context=ToolContext("agent_a", "general"),
            temperature=0.5,
            max_tokens=500,
        )
    )

    assert completion.usage == {
        "prompt_tokens": 30,
        "completion_tokens": 6,
        "total_tokens": 36,
    }


def test_openai_search_fallback_uses_responses_web_search():
    settings = provider_settings()
    client = DirectProviderClient(settings, FakeToolExecutor())
    sent_payloads = []

    async def fake_post(_provider, payload, _key):
        sent_payloads.append(payload)
        return {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Search-grounded answer.",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://example.com/source",
                                    "title": "Source",
                                    "start_index": 0,
                                    "end_index": 22,
                                }
                            ],
                        }
                    ],
                }
            ]
        }

    client._post_response = fake_post
    completion = asyncio.run(
        client.run_agent(
            provider="openai",
            model="gpt-5.2",
            messages=[{"role": "user", "content": "Search for this URL."}],
            tools=[],
            tool_context=ToolContext("agent_a", "general"),
            temperature=0.5,
            max_tokens=500,
            enable_web_search=True,
        )
    )

    assert sent_payloads[0]["tools"] == [
        {"type": "web_search", "search_context_size": "medium"}
    ]
    assert sent_payloads[0]["max_output_tokens"] == 500
    assert completion.annotations[0]["url_citation"]["title"] == "Source"


def test_public_search_results_do_not_persist_snippets():
    public = DirectProviderClient._public_tool_result(
        "search_project_files",
        {
            "ok": True,
            "results": [{"path": "post.md", "score": 3, "snippet": "private excerpt"}],
        },
    )

    assert public == {"ok": True, "results": [{"path": "post.md", "score": 3}]}

    source = DirectProviderClient._public_tool_result(
        "read_project_source",
        {"ok": True, "id": "source-1", "title": "Page", "content_text": "private snapshot"},
    )
    assert source == {"ok": True, "id": "source-1", "title": "Page"}


def test_public_write_arguments_do_not_persist_generated_content():
    public = DirectProviderClient._public_tool_arguments(
        "write_project_file",
        {
            "path": "artifact.md",
            "content": "private generated body",
        },
    )

    assert public == {
        "path": "artifact.md",
        "content_bytes": len(b"private generated body"),
    }


def test_tool_policy_uses_one_canonical_request_fingerprint():
    first = tool_request_fingerprint(
        "write_project_file",
        {"path": "note.md", "content": "same"},
    )
    reordered = tool_request_fingerprint(
        "write_project_file",
        {"content": "same", "path": "note.md"},
    )

    assert first == reordered
    assert tool_changes_state("write_project_file")
    assert tool_changes_state("propose_persona_update")
    assert not tool_changes_state("read_project_file")
    assert tool_recovery_strategy("write_project_file") == "verify_content_hash"
    assert tool_recovery_strategy("propose_persona_update") == "manual_review"


def test_provider_response_body_is_bounded():
    settings = provider_settings(
        provider_response_max_bytes=32,
    )

    async def scenario():
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                content=b'{"choices":[' + (b"x" * 64) + b"]}",
            )
        )
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = DirectProviderClient(
                settings,
                FakeToolExecutor(),
                http_client=http_client,
            )
            try:
                await client._post_chat(
                    "moonshot",
                    {"model": "provider/model"},
                    "test-key",
                )
            except Exception as exc:
                return exc
        raise AssertionError("Oversized provider response was accepted.")

    error = asyncio.run(scenario())
    assert "32-byte limit" in str(error)


def test_provider_tool_calls_are_bounded_per_response():
    settings = provider_settings(
        provider_tool_calls_per_response=1,
        provider_tool_calls_per_turn=4,
    )
    client = DirectProviderClient(settings, FakeToolExecutor())

    async def fake_post(_provider, _payload, _key):
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"id": "one", "function": {"name": "read_project_file"}},
                            {"id": "two", "function": {"name": "read_project_file"}},
                        ],
                    }
                }
            ]
        }

    client._post_chat = fake_post
    with pytest.raises(ProviderError, match="more than 1 tool calls"):
        asyncio.run(
            client.run_agent(
                provider="moonshot",
                model="provider/model",
                messages=[{"role": "user", "content": "Read the project."}],
                tools=[{"type": "function"}],
                tool_context=ToolContext("agent_a", "general"),
                temperature=0.5,
                max_tokens=500,
            )
        )


def test_provider_tool_calls_are_bounded_across_the_turn():
    settings = provider_settings(
        provider_tool_calls_per_response=2,
        provider_tool_calls_per_turn=1,
    )
    client = DirectProviderClient(settings, FakeToolExecutor())
    response = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "one",
                            "function": {
                                "name": "read_project_file",
                                "arguments": '{"path":"post.md"}',
                            },
                        }
                    ],
                }
            }
        ]
    }

    async def fake_post(_provider, _payload, _key):
        return response

    client._post_chat = fake_post
    with pytest.raises(ProviderError, match="1-tool-call turn limit"):
        asyncio.run(
            client.run_agent(
                provider="moonshot",
                model="provider/model",
                messages=[{"role": "user", "content": "Read the project."}],
                tools=[{"type": "function"}],
                tool_context=ToolContext("agent_a", "general"),
                temperature=0.5,
                max_tokens=500,
            )
        )


def test_multiline_tool_arguments_are_repaired_without_truncating_content():
    raw = '{"path":"journal.md","content":"first line\nsecond line\tindented"}'
    parsed = AgentToolExecutor.parse_arguments(raw)
    assert parsed == {
        "path": "journal.md",
        "content": "first line\nsecond line\tindented",
    }


def test_pass_is_retried_when_participation_is_required():
    settings = provider_settings()
    client = DirectProviderClient(settings, FakeToolExecutor())
    responses = iter([
        {"choices": [{"message": {"content": "[[PASS]]", "tool_calls": []}}]},
        {"choices": [{"message": {"content": "I agree, and here is what I notice.", "tool_calls": []}}]},
    ])

    async def fake_post(_provider, _payload, _key):
        return next(responses)

    client._post_chat = fake_post
    completion = asyncio.run(client.run_agent(
        provider="moonshot",
        model="kimi-k3",
        messages=[{"role": "user", "content": "Take your turn."}],
        tools=[],
        tool_context=ToolContext("agent_a", "general"),
        temperature=0.5,
        max_tokens=500,
    ))
    assert completion.content == "I agree, and here is what I notice."


def test_openai_uses_chat_completions_output_limit_field():
    settings = provider_settings()
    client = DirectProviderClient(settings, FakeToolExecutor())
    sent_payload = {}

    async def fake_post(provider, payload, _key):
        assert provider == "openai"
        sent_payload.update(payload)
        return {"choices": [{"message": {"content": "OpenAI response.", "tool_calls": []}}]}

    client._post_chat = fake_post
    completion = asyncio.run(client.run_agent(
        provider="openai",
        model="chat-latest",
        messages=[{"role": "user", "content": "Take your turn."}],
        tools=[{"type": "function"}],
        tool_context=ToolContext("agent_a", "general"),
        temperature=0.5,
        max_tokens=700,
    ))

    assert completion.content == "OpenAI response."
    assert sent_payload["max_completion_tokens"] == 700
    assert sent_payload["reasoning_effort"] == "none"
    assert "max_tokens" not in sent_payload
    assert "temperature" not in sent_payload


def test_http_timeout_is_classified_for_round_robin_recovery():
    settings = provider_settings()
    async def scenario():
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("test timeout", request=request)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = DirectProviderClient(
                settings,
                FakeToolExecutor(),
                http_client=http_client,
            )
            await client._post_chat(
                "moonshot",
                {"model": "provider/model"},
                "test-key",
            )

    try:
        asyncio.run(scenario())
    except ProviderTimeout as exc:
        assert "timed out" in str(exc).lower()
    else:
        raise AssertionError("Expected the HTTP timeout to become ProviderTimeout")


def test_provider_error_becomes_a_safe_useful_message():
    detail = DirectProviderClient._format_error_detail(
        {
            "error": {
                "message": "Provider returned error",
                "type": "invalid_request",
                "code": 400,
                "param": "temperature",
            },
        },
        "fallback",
    )

    assert "Provider returned error" in detail
    assert "invalid_request" in detail
    assert "temperature" in detail


if __name__ == "__main__":
    test_empty_response_after_tools_gets_a_synthesis_retry()
    test_pass_is_retried_when_participation_is_required()
    test_public_search_results_do_not_persist_snippets()
    test_http_timeout_is_classified_for_round_robin_recovery()
    test_provider_error_becomes_a_safe_useful_message()
    print("direct provider checks passed")
def test_continue_turn_marker_is_removed_before_display():
    content, continue_turn = DirectProviderClient._extract_continue_turn(
        "One complete thought.\n\n[[CONTINUE_TURN]]"
    )

    assert content == "One complete thought."
    assert continue_turn is True
