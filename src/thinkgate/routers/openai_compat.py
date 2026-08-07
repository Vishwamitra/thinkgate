from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from thinkgate.heal import forward_with_healing
from thinkgate.stats import stats
from thinkgate.upstream.ollama import OllamaClient

router = APIRouter()


def openai_to_ollama(payload: dict) -> dict:
    """Translates the OpenAI request fields thinkgate needs into Ollama's
    native /api/chat shape."""
    ollama_payload: dict = {
        "model": payload["model"],
        "messages": payload["messages"],
        "stream": payload.get("stream", False),
    }
    options = {}
    if "max_tokens" in payload:
        options["num_predict"] = payload["max_tokens"]
    if "temperature" in payload:
        options["temperature"] = payload["temperature"]
    if options:
        ollama_payload["options"] = options
    # Some OpenAI-compatible clients forward `think` as a vendor extension.
    # Pass it through untouched so detect.py still sees it as an explicit
    # caller choice, same as the native endpoint.
    if "think" in payload:
        ollama_payload["think"] = payload["think"]
    return ollama_payload


def ollama_to_openai(response: dict, requested_model: str) -> dict:
    """Maps an Ollama /api/chat response onto an OpenAI chat.completion --
    just the fields a typical client actually reads."""
    content = response.get("message", {}).get("content", "")
    finish_reason = response.get("done_reason") or "stop"
    prompt_tokens = response.get("prompt_eval_count", 0)
    completion_tokens = response.get("eval_count", 0)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response.get("model", requested_model),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    base_url = request.app.state.upstream_base_url
    ollama_payload = openai_to_ollama(payload)

    if ollama_payload.get("stream"):
        # Ollama's stream format and OpenAI's SSE aren't the same wire
        # format, so this can't just relay bytes like /api/chat does -- the
        # client would get data it can't parse. Erroring here beats a
        # passthrough that looks fine and isn't. SSE translation is v2.
        raise HTTPException(
            status_code=501,
            detail=(
                "Streaming is not yet supported on /v1/chat/completions. "
                "Use /api/chat (Ollama-native) for a streaming passthrough, "
                "or set stream=false here."
            ),
        )

    client = OllamaClient(base_url=base_url)
    response, healed = await forward_with_healing(client, ollama_payload)
    stats.record(healed)
    return JSONResponse(ollama_to_openai(response, requested_model=payload["model"]))
