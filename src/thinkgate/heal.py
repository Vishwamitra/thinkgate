from __future__ import annotations

from thinkgate.detect import is_thinking_exhausted, is_thinking_exhausted_openai_shape
from thinkgate.upstream.ollama import OllamaClient
from thinkgate.upstream.vllm import VLLMClient

_VLLM_BUDGET_MULTIPLIER = 4


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


async def forward_with_healing_vllm(client: VLLMClient, payload: dict) -> tuple[dict, bool]:
    # vLLM has no universal thinking-off switch -- the toggle name (and
    # whether one even exists) varies by model: Qwen3 uses enable_thinking,
    # granite uses thinking, DeepSeek-R1 distills have no switch at all.
    # A bigger token budget works regardless of which model is serving.
    caller_set_thinking = "enable_thinking" in payload.get("chat_template_kwargs", {})
    response = await client.chat(payload)

    if not is_thinking_exhausted_openai_shape(response, caller_set_thinking):
        return response, False

    if "max_tokens" not in payload:
        return response, False

    healed_payload = {**payload, "max_tokens": payload["max_tokens"] * _VLLM_BUDGET_MULTIPLIER}
    healed_response = await client.chat(healed_payload)
    return healed_response, True
