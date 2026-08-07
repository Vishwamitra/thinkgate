from unittest.mock import AsyncMock, MagicMock, patch

from thinkgate.upstream.ollama import OllamaClient


async def test_chat_posts_payload_and_returns_parsed_json():
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {"done_reason": "stop", "message": {"content": "hi"}}

    fake_http_client = AsyncMock()
    fake_http_client.post.return_value = fake_response
    fake_http_client.__aenter__.return_value = fake_http_client
    fake_http_client.__aexit__.return_value = False

    with patch(
        "thinkgate.upstream.ollama.httpx.AsyncClient", return_value=fake_http_client
    ):
        client = OllamaClient(base_url="http://localhost:11434")
        result = await client.chat({"model": "gemma4:12b", "messages": []})

    fake_http_client.post.assert_called_once_with(
        "http://localhost:11434/api/chat", json={"model": "gemma4:12b", "messages": []}
    )
    assert result == {"done_reason": "stop", "message": {"content": "hi"}}


async def test_base_url_trailing_slash_is_stripped():
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {}

    fake_http_client = AsyncMock()
    fake_http_client.post.return_value = fake_response
    fake_http_client.__aenter__.return_value = fake_http_client
    fake_http_client.__aexit__.return_value = False

    with patch(
        "thinkgate.upstream.ollama.httpx.AsyncClient", return_value=fake_http_client
    ):
        client = OllamaClient(base_url="http://localhost:11434/")
        await client.chat({})

    called_url = fake_http_client.post.call_args.args[0]
    assert called_url == "http://localhost:11434/api/chat"
