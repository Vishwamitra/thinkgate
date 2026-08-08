from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from thinkgate.server import create_app


def _client() -> TestClient:
    return TestClient(create_app(upstream_base_url="http://fake-upstream:8000", backend="vllm"))


def test_chat_completions_returns_healthy_response_unmodified():
    with patch("thinkgate.routers.vllm_compat.VLLMClient") as mock_cls:
        mock_cls.return_value.chat = AsyncMock(
            return_value={"choices": [{"finish_reason": "stop", "message": {"content": "hi there"}}]}
        )

        response = _client().post(
            "/v1/chat/completions",
            json={"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hi there"
    mock_cls.return_value.chat.assert_awaited_once()


def test_chat_completions_heals_an_exhausted_response():
    with patch("thinkgate.routers.vllm_compat.VLLMClient") as mock_cls:
        mock_cls.return_value.chat = AsyncMock(
            side_effect=[
                {"choices": [{"finish_reason": "length", "message": {"content": ""}}]},
                {"choices": [{"finish_reason": "stop", "message": {"content": "healed answer"}}]},
            ]
        )

        response = _client().post(
            "/v1/chat/completions",
            json={
                "model": "Qwen/Qwen3-8B",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 200,
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "healed answer"
    assert mock_cls.return_value.chat.await_count == 2
    retry_payload = mock_cls.return_value.chat.await_args_list[1].args[0]
    assert retry_payload["max_tokens"] == 800


def test_stats_endpoint_reflects_recorded_requests():
    with patch("thinkgate.routers.vllm_compat.VLLMClient") as mock_cls:
        mock_cls.return_value.chat = AsyncMock(
            return_value={"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]}
        )
        client = _client()
        before = client.get("/stats").json()

        client.post(
            "/v1/chat/completions",
            json={"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "hi"}]},
        )

        after = client.get("/stats").json()

    assert after["requests_seen"] == before["requests_seen"] + 1


def test_streaming_request_returns_a_clear_501():
    response = _client().post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-8B",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )

    assert response.status_code == 501


def test_ollama_native_route_is_not_mounted_on_the_vllm_backend():
    """/api/chat is Ollama's own wire format -- it has no meaning against a
    vLLM upstream, so the vllm backend must not expose it."""
    response = _client().post(
        "/api/chat", json={"model": "Qwen/Qwen3-8B", "messages": []}
    )

    assert response.status_code == 404
