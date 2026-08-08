from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from thinkgate.server import create_app


def _client(backend: str = "ollama") -> TestClient:
    return TestClient(create_app(upstream_base_url="http://fake-upstream:11434", backend=backend))


def _upstream_error(status_code: int, body: dict) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://fake-upstream:11434/api/chat")
    response = httpx.Response(status_code, json=body, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


def test_upstream_error_is_relayed_not_swallowed_into_a_500():
    """A wrong model name, or the upstream being down, isn't something to
    heal -- the caller should see the real status and body, not an opaque
    500 with no way to tell what actually went wrong."""
    with patch("thinkgate.routers.ollama_native.OllamaClient") as mock_cls:
        mock_cls.return_value.chat = AsyncMock(
            side_effect=_upstream_error(404, {"error": "model 'x' not found"})
        )

        response = _client().post(
            "/api/chat",
            json={
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )

    assert response.status_code == 404
    assert response.json() == {"error": "model 'x' not found"}


def test_upstream_error_is_relayed_on_the_vllm_backend_too():
    with patch("thinkgate.routers.vllm_compat.VLLMClient") as mock_cls:
        mock_cls.return_value.chat = AsyncMock(
            side_effect=_upstream_error(500, {"error": "internal server error"})
        )

        response = _client(backend="vllm").post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 500
    assert response.json() == {"error": "internal server error"}
