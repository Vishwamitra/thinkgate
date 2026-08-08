from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from thinkgate.heal import forward_with_healing_vllm
from thinkgate.stats import stats
from thinkgate.upstream.vllm import VLLMClient

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    base_url = request.app.state.upstream_base_url

    if payload.get("stream"):
        raise HTTPException(
            status_code=501,
            detail="Streaming is not yet supported. Set stream=false.",
        )

    client = VLLMClient(base_url=base_url)
    response, healed = await forward_with_healing_vllm(client, payload)
    stats.record(healed)
    return JSONResponse(response)
