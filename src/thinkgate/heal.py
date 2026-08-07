from __future__ import annotations

from thinkgate.detect import is_thinking_exhausted
from thinkgate.upstream.ollama import OllamaClient


async def forward_with_healing(client: OllamaClient, payload: dict) -> tuple[dict, bool]:
    # Always try the request unmodified first. Forcing think:false up front
    # would "fix" this but also throw away thinking on every request where
    # it was never a problem, so only pay for the retry when it's actually
    # needed. One retry max -- a payload that still fails with thinking off
    # is a different problem, not something to keep hammering on.
    caller_set_think = "think" in payload
    response = await client.chat(payload)

    if not is_thinking_exhausted(response, caller_set_think):
        return response, False

    healed_payload = {**payload, "think": False}
    healed_response = await client.chat(healed_payload)
    return healed_response, True
