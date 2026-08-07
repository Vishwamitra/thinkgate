from __future__ import annotations

import os

from fastapi import FastAPI

from thinkgate.routers import ollama_native, openai_compat
from thinkgate.stats import stats

DEFAULT_UPSTREAM = "http://localhost:11434"
DEFAULT_PORT = 11435


def create_app(upstream_base_url: str = DEFAULT_UPSTREAM) -> FastAPI:
    app = FastAPI(
        title="thinkgate",
        description="Detects and heals thinking-model responses silently "
        "emptied by exhausted token budgets.",
    )
    app.state.upstream_base_url = upstream_base_url

    app.include_router(ollama_native.router)
    app.include_router(openai_compat.router)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "upstream": upstream_base_url}

    @app.get("/stats")
    async def get_stats() -> dict:
        return stats.as_dict()

    return app


app = create_app(os.environ.get("THINKGATE_UPSTREAM", DEFAULT_UPSTREAM))


def main() -> None:
    import uvicorn

    port = int(os.environ.get("THINKGATE_PORT", DEFAULT_PORT))
    uvicorn.run(app, host="0.0.0.0", port=port)
