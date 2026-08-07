from unittest.mock import AsyncMock

from thinkgate.heal import forward_with_healing


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
