from __future__ import annotations

import httpx


class OllamaClient:
    """Thin async client for Ollama's native /api/chat. Takes and returns
    raw dicts in Ollama's own shape -- translation happens in the routers,
    not here."""

    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def chat(self, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()
