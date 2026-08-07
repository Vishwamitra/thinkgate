# thinkgate

A reverse proxy for [Ollama](https://ollama.com) that detects and heals the
"thinking model returns nothing" failure: a reasoning-capable model burns its
entire output-token budget on a hidden chain-of-thought and comes back with
`done_reason: "length"` and empty content — no error, just silence.

## Why

Same failure signature, independently rediscovered across a pile of
unrelated projects:

- [LangChain #36936](https://github.com/langchain-ai/langchain/issues/36936)
- [CrewAI #3031](https://github.com/crewAIInc/crewAI/issues/3031)
- [openclaw #46680](https://github.com/openclaw/openclaw/issues/46680) (and a regression at [#73366](https://github.com/openclaw/openclaw/issues/73366))
- [Zammad #5984](https://github.com/zammad/zammad/issues/5984)
- [NVIDIA/NemoClaw #246](https://github.com/NVIDIA/NemoClaw/issues/246)
- [NousResearch/hermes-agent #46131](https://github.com/NousResearch/hermes-agent/issues/46131)
- [ManifoldKit #487](https://github.com/ManifoldKit/ManifoldKit/issues/487)
- [Graphify #820](https://github.com/Graphify/Graphify/issues/820)
- Ollama's own repo: [#16184](https://github.com/ollama/ollama/issues/16184), [#16583](https://github.com/ollama/ollama/issues/16583), [#14798](https://github.com/ollama/ollama/issues/14798)

We hit it ourselves twice: once in a local SARIF-triage tool, and once in
[HKUDS/LightRAG](https://github.com/HKUDS/LightRAG), where entity extraction
silently returned nothing for an entire document batch. Root cause and fix
are up as [LightRAG issue #3597](https://github.com/HKUDS/LightRAG/issues/3597)
/ [PR #3599](https://github.com/HKUDS/LightRAG/pull/3599).

The one existing point-solution we found, `ollama-nothink-proxy`, has 1 star,
0 forks, is macOS-only, and takes the blunt approach: permanently force
`think: false` on a pre-configured list of model aliases. That gives up
thinking's actual benefit everywhere, and requires knowing in advance which
models need it. thinkgate instead watches every response for the failure
signature and only intervenes when it's about to happen — no per-model
config, and thinking still runs normally the rest of the time.

## How it works

```
client ──▶ thinkgate ──▶ Ollama
             │
             ├─ forwards the request unchanged
             ├─ done_reason == "length" and content is ~empty,
             │  and the caller didn't explicitly set `think`?
             │     └─ yes → retry once with think: false → return that
             └─     no  → return the original response
```

Detection is behavioral, not a hardcoded model list — see
[`src/thinkgate/detect.py`](src/thinkgate/detect.py). An explicit `think`
from the caller is always respected and never overridden.

Both endpoint families a client already expects are supported, so pointing
something at thinkgate is just a base-URL swap:

- `/api/chat` — Ollama-native (includes a streaming passthrough — not yet healed, see Limitations)
- `/v1/chat/completions` — OpenAI-compatible, for LangChain/CrewAI/the `openai` SDK/etc.

## Quickstart

```bash
git clone https://github.com/Vishwamitra/thinkgate.git
cd thinkgate
uv tool install .          # or: pip install .

thinkgate                  # listens on :11435, proxies to localhost:11434 by default
```

In another terminal, send the same request you'd send straight to Ollama —
just point it at thinkgate's port instead:

```bash
curl http://localhost:11435/api/chat -d '{
  "model": "your-thinking-model",
  "messages": [{"role": "user", "content": "Say hello in exactly three words."}]
}'
```

A request that would come back empty against raw Ollama comes back with real
content through thinkgate. No client-side changes required.

### Point an existing client at it

Anything that speaks the OpenAI SDK shape just needs a `base_url` swap:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11435/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="your-thinking-model",
    messages=[{"role": "user", "content": "Say hello in exactly three words."}],
)
```

Same idea for LangChain's `ChatOpenAI(base_url=...)`, CrewAI's LLM config, or
any other OpenAI-compatible client. Swap the URL, keep the code.

### Configuration

| env var | default | purpose |
| --- | --- | --- |
| `THINKGATE_UPSTREAM` | `http://localhost:11434` | the real Ollama server to proxy to |
| `THINKGATE_PORT` | `11435` | port thinkgate itself listens on |

### Stats

```bash
curl http://localhost:11435/stats
# {"requests_seen": 12, "requests_healed": 3}
```

## Docker

```bash
docker compose -f examples/docker-compose.yml up
```

Runs thinkgate alongside Ollama, pre-wired together — see
[`examples/docker-compose.yml`](examples/docker-compose.yml).

## Limitations

These are deliberate cuts for v1, not oversights.

- **Streaming isn't healed.** `/api/chat` passes a streaming request
  straight through to Ollama unchanged; buffering a whole stream just to
  decide whether to retry it isn't worth the added latency for v1.
  `/v1/chat/completions` returns `501` on a streaming request rather than a
  silently broken passthrough, since Ollama's stream format and OpenAI's SSE
  aren't the same wire format (see
  [`openai_compat.py`](src/thinkgate/routers/openai_compat.py)). Streaming
  healing is the natural v2.
- **Ollama only.** The detect → retry → heal shape is backend-agnostic by
  design, but v1 only ships the Ollama adapter (the one we've directly
  proven the bug against). vLLM/LM Studio adapters are additive, not a
  rewrite.
- **No dashboard.** Observability is the `/stats` endpoint plus log lines.
  Enough to prove the value; a UI can come later if it's actually needed.

## Development

```bash
uv sync
uv run pytest
```

23 tests, using real captured Ollama fixtures under `tests/fixtures/` for the
response shapes that matter, rather than hand-written stubs.

## License

MIT — see [LICENSE](LICENSE).
