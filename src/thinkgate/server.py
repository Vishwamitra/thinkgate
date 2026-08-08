from __future__ import annotations

import os

from fastapi import FastAPI

from thinkgate.routers import ollama_native, openai_compat, vllm_compat
from thinkgate.stats import stats

DEFAULT_UPSTREAM = "http://localhost:11434"
DEFAULT_PORT = 11435
DEFAULT_BACKEND = "ollama"


def create_app(upstream_base_url: str = DEFAULT_UPSTREAM, backend: str = DEFAULT_BACKEND) -> FastAPI:
    app = FastAPI(
        title="thinkgate",
        description="Detects and heals thinking-model responses silently "
        "emptied by exhausted token budgets.",
    )
    app.state.upstream_base_url = upstream_base_url

    if backend == "ollama":
        app.include_router(ollama_native.router)
        app.include_router(openai_compat.router)
    elif backend == "vllm":
        app.include_router(vllm_compat.router)
    else:
        raise ValueError(f"Unknown THINKGATE_BACKEND {backend!r}, expected 'ollama' or 'vllm'")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "backend": backend, "upstream": upstream_base_url}

    @app.get("/stats")
    async def get_stats() -> dict:
        return stats.as_dict()

    return app


app = create_app(
    os.environ.get("THINKGATE_UPSTREAM", DEFAULT_UPSTREAM),
    os.environ.get("THINKGATE_BACKEND", DEFAULT_BACKEND),
)


def main() -> None:
    import uvicorn

    port = int(os.environ.get("THINKGATE_PORT", DEFAULT_PORT))
    uvicorn.run(app, host="0.0.0.0", port=port)
