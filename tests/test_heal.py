from unittest.mock import AsyncMock

from thinkgate.heal import forward_with_healing, forward_with_healing_vllm


def _healthy_response() -> dict:
    return {"done_reason": "stop", "message": {"content": "Hello, my friend."}}


def _exhausted_response() -> dict:
    return {"done_reason": "length", "message": {"content": ""}}


async def test_healthy_response_is_returned_unchanged_with_a_single_call():
    client = AsyncMock()
    client.chat.return_value = _healthy_response()

    response, healed = await forward_with_healing(client, {"model": "gemma4:12b"})

    assert response == _healthy_response()
    assert healed is False
    client.chat.assert_awaited_once()


async def test_exhausted_response_triggers_exactly_one_retry_with_think_false():
    client = AsyncMock()
    client.chat.side_effect = [_exhausted_response(), _healthy_response()]

    response, healed = await forward_with_healing(client, {"model": "gemma4:12b"})

    assert response == _healthy_response()
    assert healed is True
    assert client.chat.await_count == 2

    first_call_payload = client.chat.await_args_list[0].args[0]
    retry_payload = client.chat.await_args_list[1].args[0]
    assert "think" not in first_call_payload
    assert retry_payload["think"] is False
    # everything else about the original request is preserved on retry
    assert retry_payload["model"] == "gemma4:12b"


async def test_retry_is_bounded_to_one_attempt():
    client = AsyncMock()
    client.chat.side_effect = [_exhausted_response(), _exhausted_response()]

    response, healed = await forward_with_healing(client, {"model": "gemma4:12b"})

    assert response == _exhausted_response()
    assert healed is True
    assert client.chat.await_count == 2


async def test_caller_explicit_think_choice_skips_healing_entirely():
    """Same rule as detect.py: an explicit caller choice always wins, even
    if the response matches the failure signature exactly."""
    client = AsyncMock()
    client.chat.return_value = _exhausted_response()

    response, healed = await forward_with_healing(
        client, {"model": "gemma4:12b", "think": True}
    )

    assert response == _exhausted_response()
    assert healed is False
    client.chat.assert_awaited_once()


def _vllm_healthy_response() -> dict:
    return {"choices": [{"finish_reason": "stop", "message": {"content": "Hello, my friend."}}]}


def _vllm_exhausted_response() -> dict:
    return {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}


async def test_vllm_healthy_response_is_returned_unchanged_with_a_single_call():
    client = AsyncMock()
    client.chat.return_value = _vllm_healthy_response()

    response, healed = await forward_with_healing_vllm(
        client, {"model": "Qwen/Qwen3-8B", "max_tokens": 200}
    )

    assert response == _vllm_healthy_response()
    assert healed is False
    client.chat.assert_awaited_once()


async def test_vllm_exhausted_response_triggers_one_retry_with_bigger_budget():
    client = AsyncMock()
    client.chat.side_effect = [_vllm_exhausted_response(), _vllm_healthy_response()]

    response, healed = await forward_with_healing_vllm(
        client, {"model": "Qwen/Qwen3-8B", "max_tokens": 200}
    )

    assert response == _vllm_healthy_response()
    assert healed is True
    assert client.chat.await_count == 2

    retry_payload = client.chat.await_args_list[1].args[0]
    assert retry_payload["max_tokens"] == 800


async def test_vllm_without_max_tokens_is_not_healed():
    """No max_tokens on the original request means there's nothing to scale
    up, so healing is skipped rather than guessing at a number."""
    client = AsyncMock()
    client.chat.return_value = _vllm_exhausted_response()

    response, healed = await forward_with_healing_vllm(client, {"model": "Qwen/Qwen3-8B"})

    assert response == _vllm_exhausted_response()
    assert healed is False
    client.chat.assert_awaited_once()


async def test_vllm_retry_is_bounded_to_one_attempt():
    client = AsyncMock()
    client.chat.side_effect = [_vllm_exhausted_response(), _vllm_exhausted_response()]

    response, healed = await forward_with_healing_vllm(
        client, {"model": "Qwen/Qwen3-8B", "max_tokens": 200}
    )

    assert response == _vllm_exhausted_response()
    assert healed is True
    assert client.chat.await_count == 2


async def test_vllm_explicit_enable_thinking_skips_healing_entirely():
    client = AsyncMock()
    client.chat.return_value = _vllm_exhausted_response()

    response, healed = await forward_with_healing_vllm(
        client,
        {
            "model": "Qwen/Qwen3-8B",
            "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )

    assert response == _vllm_exhausted_response()
    assert healed is False
    client.chat.assert_awaited_once()
