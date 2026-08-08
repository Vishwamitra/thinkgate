from __future__ import annotations

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from thinkgate.heal import forward_with_healing
from thinkgate.stats import stats
from thinkgate.upstream.ollama import OllamaClient

router = APIRouter()


@router.post("/api/chat")
async def chat(request: Request):
    payload = await request.json()
    base_url = request.app.state.upstream_base_url

    # Ollama itself defaults to streaming when the field is omitted, not to
    # stream:false -- match that default rather than silently treating a
    # missing field as non-streaming and crashing trying to parse NDJSON
    # chunks as one JSON body.
    if payload.get("stream", True):
        return await stream_passthrough(base_url, "/api/chat", payload)

    client = OllamaClient(base_url=base_url)
    response, healed = await forward_with_healing(client, payload)
    stats.record(healed)
    return JSONResponse(response)


async def stream_passthrough(base_url: str, path: str, payload: dict) -> StreamingResponse:
    """Streaming isn't healed in v1 (see README) -- just relay the raw
    upstream stream so streaming callers still work."""

    async def relay():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", f"{base_url}{path}", json=payload
            ) as upstream_response:
                async for chunk in upstream_response.aiter_bytes():
                    yield chunk

    return StreamingResponse(relay(), media_type="application/x-ndjson")
