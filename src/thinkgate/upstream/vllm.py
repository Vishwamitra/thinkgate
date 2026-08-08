from __future__ import annotations

import httpx


class VLLMClient:
    """Thin async client for vLLM's OpenAI-compatible /v1/chat/completions.
    Takes and returns raw dicts in that shape -- no translation needed since
    vLLM already speaks it natively."""

    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def chat(self, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions", json=payload
            )
            response.raise_for_status()
            return response.json()
