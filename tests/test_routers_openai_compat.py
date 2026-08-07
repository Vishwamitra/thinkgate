from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from thinkgate.routers.openai_compat import ollama_to_openai, openai_to_ollama
from thinkgate.server import create_app


def _client() -> TestClient:
    return TestClient(create_app(upstream_base_url="http://fake-upstream:11434"))


# --- translation helpers, tested directly ---


def test_openai_to_ollama_translates_max_tokens_into_options():
    payload = {
        "model": "gemma4:12b",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 200,
    }

    result = openai_to_ollama(payload)

    assert result["options"]["num_predict"] == 200
    assert result["stream"] is False  # OpenAI clients default stream to falsy too


def test_openai_to_ollama_preserves_an_explicit_think_choice():
    """An explicit `think` has to survive translation unchanged, or
    detect.py has no way to know the caller set it."""
    payload = {"model": "gemma4:12b", "messages": [], "think": True}

    result = openai_to_ollama(payload)

    assert result["think"] is True


def test_openai_to_ollama_omits_options_when_nothing_to_translate():
    payload = {"model": "gemma4:12b", "messages": []}

    result = openai_to_ollama(payload)

    assert "options" not in result


def test_ollama_to_openai_maps_content_and_finish_reason():
    ollama_response = {
        "model": "gemma4:12b",
        "done_reason": "stop",
        "message": {"content": "hello"},
        "prompt_eval_count": 10,
        "eval_count": 3,
    }

    result = ollama_to_openai(ollama_response, requested_model="gemma4:12b")

    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "hello"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
    }


# --- the actual route, through the healing layer ---


def test_chat_completions_heals_and_returns_openai_shaped_response():
    with patch("thinkgate.routers.openai_compat.OllamaClient") as mock_cls:
        mock_cls.return_value.chat = AsyncMock(
            side_effect=[
                {"done_reason": "length", "message": {"content": ""}},
                {
                    "model": "gemma4:12b",
                    "done_reason": "stop",
                    "message": {"content": "healed via openai route"},
                },
            ]
        )

        response = _client().post(
            "/v1/chat/completions",
            json={
                "model": "gemma4:12b",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "healed via openai route"
    assert mock_cls.return_value.chat.await_count == 2
    retry_payload = mock_cls.return_value.chat.await_args_list[1].args[0]
    assert retry_payload["think"] is False


def test_streaming_openai_request_returns_a_clear_501_not_a_broken_stream():
    """A byte passthrough here would hand the client a format it can't
    parse (Ollama-native vs. OpenAI SSE) -- an honest 501 beats that."""
    response = _client().post(
        "/v1/chat/completions",
        json={
            "model": "gemma4:12b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )

    assert response.status_code == 501
