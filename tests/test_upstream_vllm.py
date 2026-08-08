from unittest.mock import AsyncMock, MagicMock, patch

from thinkgate.upstream.vllm import VLLMClient


async def test_chat_posts_payload_and_returns_parsed_json():
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {
        "choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]
    }

    fake_http_client = AsyncMock()
    fake_http_client.post.return_value = fake_response
    fake_http_client.__aenter__.return_value = fake_http_client
    fake_http_client.__aexit__.return_value = False

    with patch(
        "thinkgate.upstream.vllm.httpx.AsyncClient", return_value=fake_http_client
    ):
        client = VLLMClient(base_url="http://localhost:8000")
        result = await client.chat({"model": "Qwen/Qwen3-8B", "messages": []})

    fake_http_client.post.assert_called_once_with(
        "http://localhost:8000/v1/chat/completions",
        json={"model": "Qwen/Qwen3-8B", "messages": []},
    )
    assert result == {"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]}


async def test_base_url_trailing_slash_is_stripped():
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {}

    fake_http_client = AsyncMock()
    fake_http_client.post.return_value = fake_response
    fake_http_client.__aenter__.return_value = fake_http_client
    fake_http_client.__aexit__.return_value = False

    with patch(
        "thinkgate.upstream.vllm.httpx.AsyncClient", return_value=fake_http_client
    ):
        client = VLLMClient(base_url="http://localhost:8000/")
        await client.chat({})

    called_url = fake_http_client.post.call_args.args[0]
    assert called_url == "http://localhost:8000/v1/chat/completions"
