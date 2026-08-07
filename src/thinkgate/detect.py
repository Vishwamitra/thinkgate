from __future__ import annotations

# A few stray characters before the cutoff is functionally the same failure
# as fully empty content. 8 bytes covers stray whitespace/punctuation
# without risking a match on a real short answer.
_EMPTY_CONTENT_THRESHOLD = 8


def is_thinking_exhausted(response: dict, caller_set_think: bool) -> bool:
    """Detects thinking silently eating the whole output budget."""
    # An explicit caller choice, either way, is never second-guessed.
    if caller_set_think:
        return False
    # done_reason == "length" means the token budget ran out, not a
    # natural stop.
    if response.get("done_reason") != "length":
        return False
    # Real partial content on a length cutoff is ordinary truncation, not
    # this bug, and must not trigger a retry.
    content = response.get("message", {}).get("content", "")
    return len(content.strip()) < _EMPTY_CONTENT_THRESHOLD
