from unittest.mock import AsyncMock, patch

from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from thinkgate.server import create_app


def _client() -> TestClient:
    return TestClient(create_app(upstream_base_url="http://fake-upstream:11434"))


def test_chat_returns_healthy_response_unmodified():
    with patch("thinkgate.routers.ollama_native.OllamaClient") as mock_cls:
        mock_cls.return_value.chat = AsyncMock(
            return_value={"done_reason": "stop", "message": {"content": "hi there"}}
        )

        response = _client().post(
            "/api/chat",
            json={
                "model": "gemma4:12b",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

    assert response.status_code == 200
    assert response.json()["message"]["content"] == "hi there"
    mock_cls.return_value.chat.assert_awaited_once()


def test_chat_transparently_heals_an_exhausted_response():
    with patch("thinkgate.routers.ollama_native.OllamaClient") as mock_cls:
        mock_cls.return_value.chat = AsyncMock(
            side_effect=[
                {"done_reason": "length", "message": {"content": ""}},
                {"done_reason": "stop", "message": {"content": "healed answer"}},
            ]
        )

        response = _client().post(
            "/api/chat",
            json={
                "model": "gemma4:12b",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

    assert response.status_code == 200
    assert response.json()["message"]["content"] == "healed answer"
    assert mock_cls.return_value.chat.await_count == 2


def test_stats_endpoint_reflects_recorded_requests():
    with patch("thinkgate.routers.ollama_native.OllamaClient") as mock_cls:
        mock_cls.return_value.chat = AsyncMock(
            return_value={"done_reason": "stop", "message": {"content": "hi"}}
        )
        client = _client()
        before = client.get("/stats").json()

        client.post(
            "/api/chat",
            json={
                "model": "gemma4:12b",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

        after = client.get("/stats").json()

    assert after["requests_seen"] == before["requests_seen"] + 1


def test_streaming_requests_bypass_healing_and_go_to_passthrough():
    """Streaming can't go through forward_with_healing -- it should route
    straight to the raw passthrough instead."""
    fake_stream_response = JSONResponse({"passthrough": True})
    with (
        patch("thinkgate.routers.ollama_native.OllamaClient") as mock_cls,
        patch(
            "thinkgate.routers.ollama_native.stream_passthrough",
            new=AsyncMock(return_value=fake_stream_response),
        ) as mock_passthrough,
    ):
        _client().post(
            "/api/chat",
            json={
                "model": "gemma4:12b",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    mock_cls.return_value.chat.assert_not_called()
    mock_passthrough.assert_called_once()


def test_omitted_stream_field_also_goes_to_passthrough():
    """Ollama itself defaults to streaming when the field is missing, not
    to stream:false -- a request that never mentions `stream` must be
    treated the same as `stream: true`, or thinkgate ends up trying to
    parse Ollama's NDJSON chunks as a single JSON response and breaks."""
    fake_stream_response = JSONResponse({"passthrough": True})
    with (
        patch("thinkgate.routers.ollama_native.OllamaClient") as mock_cls,
        patch(
            "thinkgate.routers.ollama_native.stream_passthrough",
            new=AsyncMock(return_value=fake_stream_response),
        ) as mock_passthrough,
    ):
        _client().post(
            "/api/chat",
            json={"model": "gemma4:12b", "messages": [{"role": "user", "content": "hi"}]},
        )

    mock_cls.return_value.chat.assert_not_called()
    mock_passthrough.assert_called_once()
