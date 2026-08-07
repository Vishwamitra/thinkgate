import json
from pathlib import Path

from thinkgate.detect import is_thinking_exhausted

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_real_thinking_exhausted_response_is_detected():
    response = _load("ollama_chat_thinking_exhausted.json")

    assert is_thinking_exhausted(response, caller_set_think=False) is True


def test_real_healthy_response_is_not_flagged():
    response = _load("ollama_chat_healthy.json")

    assert is_thinking_exhausted(response, caller_set_think=False) is False


def test_real_ordinary_truncation_with_content_is_not_flagged():
    """A long answer that just ran out of room shouldn't get healed -- that
    would throw away a perfectly good partial response."""
    response = _load("ollama_chat_truncated_with_content.json")

    assert is_thinking_exhausted(response, caller_set_think=False) is False


def test_caller_explicit_think_choice_is_never_overridden():
    """An explicit caller choice wins even if the response otherwise
    matches the failure signature exactly."""
    response = {"done_reason": "length", "message": {"content": ""}}

    assert is_thinking_exhausted(response, caller_set_think=True) is False


def test_non_length_done_reason_is_never_flagged():
    response = {"done_reason": "stop", "message": {"content": ""}}

    assert is_thinking_exhausted(response, caller_set_think=False) is False


def test_missing_message_does_not_crash():
    """A minimal/malformed upstream response shouldn't raise; missing
    content counts the same as empty."""
    response = {"done_reason": "length"}

    assert is_thinking_exhausted(response, caller_set_think=False) is True


def test_whitespace_only_content_counts_as_empty():
    response = {"done_reason": "length", "message": {"content": "   \n  "}}

    assert is_thinking_exhausted(response, caller_set_think=False) is True
