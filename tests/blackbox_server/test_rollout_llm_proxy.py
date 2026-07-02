from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from blackbox_server.proxy.rollout_llm_proxy import RolloutLLMProxy


class MockAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


def _make_proxy(
    sticky_header_name: str = "X-SMG-Routing-Key",
    *,
    max_steps: int | None = 100,
    default_temperature: float | None = None,
) -> RolloutLLMProxy:
    return RolloutLLMProxy(
        upstream_origin="http://127.0.0.1:30000",
        router_api_path="/v1",
        bound_session_id="sess-001",
        bound_instance_id="inst-001",
        sticky_header_name=sticky_header_name,
        max_steps=max_steps,
        default_temperature=default_temperature,
    )


def _sse_event(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def test_rollout_proxy_uses_bound_ids_for_headers_and_untagged_snapshot():
    proxy = _make_proxy()

    async def run_test() -> None:
        snapshot = await proxy._capture_snapshot()
        headers = proxy._build_upstream_headers(
            {
                "authorization": "Bearer x",
                "x-session-id": "bad-session",
                "X-Instance-Id": "bad-instance",
                "X-Turn-ID": "bad-turn",
            },
            is_chat=True,
        )
        assert headers["authorization"] == "Bearer x"
        assert headers["X-SMG-Routing-Key"] == "sess-001"
        assert headers["X-Session-Id"] == "sess-001"
        assert headers["X-Instance-Id"] == "inst-001"
        assert "X-Turn-Id" not in headers
        assert snapshot.session_id == "sess-001"
        assert snapshot.turn_id is None
        assert snapshot.backend_session_id is None

    asyncio.run(run_test())


def test_rollout_proxy_scopes_increment_steps_and_inject_turn_header():
    proxy = _make_proxy()

    async def run_test() -> None:
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        first = await proxy._capture_snapshot()
        second = await proxy._capture_snapshot()
        headers = proxy._build_upstream_headers({}, is_chat=True, turn_id=first.turn_id)
        await proxy.clear_turn()
        assert first.turn_id == "turn-001"
        assert first.step == 0
        assert second.step == 1
        assert second.backend_session_id == "oc-session-1"
        assert headers["X-Turn-Id"] == "turn-001"

    asyncio.run(run_test())


def test_rollout_proxy_matches_any_chat_completion_prefix():
    proxy = _make_proxy()

    assert proxy._is_chat_completion("POST", "v1/chat/completions") is True
    assert proxy._is_chat_completion("POST", "custom-prefix/chat/completions") is True
    assert proxy._is_chat_completion("GET", "v1/chat/completions") is False
    assert proxy._is_chat_completion("POST", "v1/responses") is False


def test_rollout_proxy_preserves_stream_options_without_logprob_injection():
    proxy = _make_proxy()
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=MockAsyncByteStream(
                [
                    _sse_event(
                        {"id": "resp-1", "choices": [{"delta": {"reasoning_content": "thinking"}}]}
                    ),
                    _sse_event({"id": "resp-1", "choices": [{"delta": {"content": "ok"}}]}),
                    b"data: [DONE]\n\n",
                ]
            ),
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "gpt-test",
                    "messages": [],
                    "stream": True,
                    "stream_options": {"include_usage": False, "extra": "keep-me"},
                },
            ) as response:
                body = await response.aread()
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        assert b'"reasoning_content": "thinking"' in body
        assert b'"content": "ok"' in body
        assert body.endswith(b"data: [DONE]\n\n")
        assert captured["payload"] == {
            "model": "gpt-test",
            "messages": [],
            "stream": True,
            "stream_options": {"include_usage": True, "extra": "keep-me"},
        }
        headers = captured["headers"]
        assert headers["x-smg-routing-key"] == "sess-001"
        assert headers["x-session-id"] == "sess-001"
        assert headers["x-instance-id"] == "inst-001"
        assert headers["x-turn-id"] == "turn-001"

    asyncio.run(run_test())


def test_rollout_proxy_records_stream_usage_profile():
    proxy = _make_proxy()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=MockAsyncByteStream(
                [
                    _sse_event({"id": "resp-1", "choices": [{"delta": {"content": "ok"}}]}),
                    _sse_event(
                        {
                            "id": "resp-1",
                            "choices": [],
                            "usage": {
                                "prompt_tokens": 3,
                                "completion_tokens": 2,
                                "total_tokens": 5,
                                "profile": {
                                    "chat.total_s": 1.25,
                                    "chat.generation_s": 1.0,
                                    "sglang.client_generate_post_s": 0.75,
                                },
                            },
                        }
                    ),
                    b"data: [DONE]\n\n",
                ]
            ),
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": True},
            ) as response:
                await response.aread()
        await proxy.drain_turn(timeout=1.0)
        profile = await proxy.turn_profile()
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        assert profile["proxy.llm_prompt_tokens"] == 3.0
        assert profile["proxy.llm_completion_tokens"] == 2.0
        assert profile["proxy.llm_total_tokens"] == 5.0
        assert profile["proxy.llm_chat_total_s"] == 1.25
        assert profile["proxy.llm_chat_generation_s"] == 1.0
        assert profile["proxy.llm_sglang_client_generate_post_s"] == 0.75

    asyncio.run(run_test())


def test_rollout_proxy_does_not_inject_logprobs_for_non_stream_requests():
    proxy = _make_proxy()
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                            "reasoning_content": "thinking",
                        }
                    }
                ],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": False},
            )
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"] == {
            "role": "assistant",
            "content": "ok",
            "reasoning_content": "thinking",
        }
        assert requests == [{"model": "gpt-test", "messages": [], "stream": False}]

    asyncio.run(run_test())


def test_rollout_proxy_injects_default_temperature_when_missing():
    proxy = _make_proxy(default_temperature=0.7)
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": False},
            )
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        assert requests == [
            {
                "model": "gpt-test",
                "messages": [],
                "stream": False,
                "temperature": 0.7,
            }
        ]

    asyncio.run(run_test())


def test_rollout_proxy_preserves_explicit_zero_temperature():
    proxy = _make_proxy(default_temperature=0.7)
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-test",
                    "messages": [],
                    "stream": False,
                    "temperature": 0.0,
                },
            )
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        assert requests == [
            {
                "model": "gpt-test",
                "messages": [],
                "stream": False,
                "temperature": 0.0,
            }
        ]

    asyncio.run(run_test())


def test_rollout_proxy_injects_default_temperature_for_stream_requests():
    proxy = _make_proxy(default_temperature=0.7)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=MockAsyncByteStream(
                [
                    _sse_event({"id": "resp-1", "choices": [{"delta": {"content": "ok"}}]}),
                    b"data: [DONE]\n\n",
                ]
            ),
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "gpt-test",
                    "messages": [],
                    "stream": True,
                    "stream_options": {"include_usage": False},
                },
            ) as response:
                body = await response.aread()
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        assert b'"content": "ok"' in body
        assert captured["payload"] == {
            "model": "gpt-test",
            "messages": [],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.7,
        }

    asyncio.run(run_test())


def test_rollout_proxy_allows_chat_steps_under_default_limit():
    proxy = _make_proxy()
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={
                "id": f"resp-{request_count}",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            first = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": False},
            )
            second = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": False},
            )
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert first.status_code == 200
        assert second.status_code == 200
        assert request_count == 2

    asyncio.run(run_test())


def test_rollout_proxy_allows_unlimited_chat_steps_when_max_steps_disabled():
    proxy = _make_proxy(max_steps=None)
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={
                "id": f"resp-{request_count}",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            for _ in range(3):
                response = await client.post(
                    "/v1/chat/completions",
                    json={"model": "gpt-test", "messages": [], "stream": False},
                )
                assert response.status_code == 200
        await proxy.drain_turn(timeout=1.0)
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert request_count == 3

    asyncio.run(run_test())


def test_rollout_proxy_rejects_chat_completion_after_max_steps_without_inflight_leak():
    proxy = _make_proxy(max_steps=1)
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={
                "id": f"resp-{request_count}",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            first = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": False},
            )
            second = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": False},
            )
        await proxy.drain_turn(timeout=1.0)
        recorded = await proxy.wait_for_max_steps_error(timeout=0.1)
        consumed = await proxy.consume_max_steps_error()
        cleared = await proxy.consume_max_steps_error()
        state = proxy.pause_state()
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert first.status_code == 200
        assert second.status_code == 429
        assert second.json() == {
            "error": {
                "message": "Turn exceeded max_steps.",
                "type": "rate_limit_error",
                "code": "max_steps_exceeded",
                "details": {
                    "max_steps": 1,
                    "attempted_step": 1,
                },
            }
        }
        assert recorded == {
            "error": "max_steps_exceeded",
            "message": "Turn exceeded max_steps.",
            "details": {
                "session_id": "sess-001",
                "turn_id": "turn-001",
                "max_steps": 1,
                "attempted_step": 1,
                "backend_message": "429 Turn exceeded max_steps.",
                "raw_error_code": "rate_limit_error",
            },
        }
        assert consumed == recorded
        assert cleared is None
        assert request_count == 1
        assert state["http_inflight_requests"] == 0

    asyncio.run(run_test())


def test_rollout_proxy_records_plain_context_overflow_response():
    proxy = _make_proxy()
    payload = {
        "error": "context_overflow",
        "message": "Dressage proxy context window overflow.",
        "details": {"phase": "input", "context_window": 4},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(413, json=payload)

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": False},
            )
        await proxy.drain_turn(timeout=1.0)
        recorded = await proxy.consume_context_overflow_error()
        state = proxy.pause_state()
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 413
        assert response.json() == payload
        assert recorded == payload
        assert state["http_inflight_requests"] == 0

    asyncio.run(run_test())


def test_rollout_proxy_records_stream_context_overflow_response():
    proxy = _make_proxy()
    payload = {
        "error": "context_overflow",
        "message": "Dressage proxy context window overflow.",
        "details": {"phase": "input_output", "context_window": 4},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            413,
            headers={"content-type": "application/json"},
            stream=MockAsyncByteStream([json.dumps(payload).encode("utf-8")]),
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": True},
            ) as response:
                body = await response.aread()
        await proxy.drain_turn(timeout=1.0)
        recorded = await proxy.consume_context_overflow_error()
        state = proxy.pause_state()
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 413
        assert json.loads(body.decode("utf-8")) == payload
        assert recorded == payload
        assert state["http_inflight_requests"] == 0

    asyncio.run(run_test())


def test_rollout_proxy_hides_plain_dressage_rollout_invalidated_response():
    proxy = _make_proxy()
    payload = {
        "detail": {
            "error": "generation_preempted",
            "message": "SGLang generation was interrupted.",
            "session_id": "sess-001",
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json=payload)

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": False},
            )
        await proxy.drain_turn(timeout=1.0)
        recorded = await proxy.consume_rollout_invalidated_error()
        state = proxy.pause_state()
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == ""
        assert recorded == payload["detail"]
        assert state["http_inflight_requests"] == 0

    asyncio.run(run_test())


def test_rollout_proxy_hides_stream_dressage_rollout_invalidated_response():
    proxy = _make_proxy()
    payload = {
        "detail": {
            "error": "trajectory_version_changed",
            "message": "SGLang weight version changed.",
            "session_id": "sess-001",
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502,
            headers={"content-type": "application/json"},
            stream=MockAsyncByteStream([json.dumps(payload).encode("utf-8")]),
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": True},
            ) as response:
                body = await response.aread()
        await proxy.drain_turn(timeout=1.0)
        recorded = await proxy.consume_rollout_invalidated_error()
        state = proxy.pause_state()
        await proxy.clear_turn()
        await proxy._client.aclose()

        assert response.status_code == 200
        assert b"chat.completion.chunk" in body
        assert b"data: [DONE]" in body
        assert recorded == payload["detail"]
        assert state["http_inflight_requests"] == 0

    asyncio.run(run_test())


def test_rollout_proxy_resets_max_steps_for_next_turn():
    proxy = _make_proxy(max_steps=1)
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={
                "id": f"resp-{request_count}",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )

    async def run_test() -> None:
        proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app),
            base_url="http://proxy",
        ) as client:
            await proxy.open_turn("turn-001", backend_session_id="oc-session-1")
            first = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": False},
            )
            await proxy.drain_turn(timeout=1.0)
            await proxy.clear_turn()

            await proxy.open_turn("turn-002", backend_session_id="oc-session-1")
            second = await client.post(
                "/v1/chat/completions",
                json={"model": "gpt-test", "messages": [], "stream": False},
            )
            await proxy.drain_turn(timeout=1.0)
            await proxy.clear_turn()
        await proxy._client.aclose()

        assert first.status_code == 200
        assert second.status_code == 200
        assert request_count == 2

    asyncio.run(run_test())
