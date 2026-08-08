# thinkgate

A reverse proxy for [Ollama](https://ollama.com) and [vLLM](https://docs.vllm.ai)
that detects and heals the "thinking model returns nothing" failure: a
reasoning-capable model burns its entire output-token budget on a hidden
chain-of-thought and comes back with `done_reason: "length"` and empty
content — no error, just silence.

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

(vLLM follows the same detect → retry shape, but heals by raising the token
budget instead of disabling thinking — see the vLLM backend section below
for why.)

Which endpoints are active depends on `THINKGATE_BACKEND`:

- **`ollama`** (default) — `/api/chat` (native, includes a streaming
  passthrough — not yet healed, see Limitations) and `/v1/chat/completions`
  (OpenAI-compatible, translated to Ollama's shape)
- **`vllm`** — `/v1/chat/completions` only, since vLLM already speaks that
  shape natively

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
  "messages": [{"role": "user", "content": "Say hello in exactly three words."}],
  "stream": false
}'
```

A request that would come back empty against raw Ollama comes back with real
content through thinkgate. No client-side changes required.

`stream: false` matters here — Ollama itself defaults to streaming when the
field is omitted, and healing only applies to non-streaming requests (see
Limitations). thinkgate mirrors Ollama's own default, so an omitted `stream`
field gets passed straight through unhealed, same as it would talk to Ollama
directly.

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
| `THINKGATE_UPSTREAM` | `http://localhost:11434` | the real Ollama/vLLM server to proxy to |
| `THINKGATE_PORT` | `11435` | port thinkgate itself listens on |
| `THINKGATE_BACKEND` | `ollama` | `ollama` or `vllm` |

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
[`examples/docker-compose.yml`](examples/docker-compose.yml). The compose
example is Ollama-only for now; for vLLM, point `THINKGATE_UPSTREAM` at
your own vLLM deployment with `THINKGATE_BACKEND=vllm` set.

## vLLM backend

Point thinkgate at a vLLM OpenAI-compatible server instead of Ollama:

```bash
THINKGATE_BACKEND=vllm THINKGATE_UPSTREAM=http://localhost:8000 thinkgate
```

Only `/v1/chat/completions` is active in this mode — vLLM doesn't speak
Ollama's native API, so there's no `/api/chat` to expose.

The healing strategy is different here. vLLM has no universal "disable
thinking" switch: it's per-model (Qwen3 uses `enable_thinking`, granite
uses `thinking`, and DeepSeek-R1 distills have no switch at all). So
instead of forcing thinking off, thinkgate retries once with `max_tokens`
multiplied by 4 — that works no matter which model is actually being
served. Healing is skipped if the original request didn't set `max_tokens`
(nothing to scale up) or if it already set `chat_template_kwargs.enable_thinking`
itself.

**Not yet verified against a live vLLM instance.** Everything else in this
project was proven against a real running model before being called done;
this wasn't, because vLLM's mainline support is CUDA-first and there's no
NVIDIA GPU on the machine it was built on. The adapter is written against
vLLM's [documented response shape](https://docs.vllm.ai/en/latest/features/reasoning_outputs/)
and covered by tests using hand-built fixtures (`tests/fixtures/vllm_*.json`
— clearly separate from the real captured Ollama ones). If you run this
against a real vLLM server and something's off, please open an issue.

## Limitations

These are deliberate cuts for v1, not oversights.

- **Streaming isn't healed**, on either backend. `/api/chat` (Ollama)
  passes a streaming request straight through unchanged; buffering a whole
  stream just to decide whether to retry it isn't worth the added latency
  for v1. `/v1/chat/completions` returns `501` on a streaming request
  rather than a silently broken passthrough, since neither backend's native
  stream and OpenAI's SSE are the same wire format. Streaming healing is
  the natural v2.
- **vLLM support is unverified against a live instance** — see the vLLM
  backend section above for why, and what's been checked instead.
- **No OpenAI backend yet.** OpenAI's reasoning models have no user-facing
  thinking-off switch either, so the realistic fix there looks closer to
  vLLM's (retry with a bigger budget) than Ollama's — but that's unproven
  without a real repro against OpenAI's API, so it's not built rather than
  built and guessed at.
- **No dashboard.** Observability is the `/stats` endpoint plus log lines.
  Enough to prove the value; a UI can come later if it's actually needed.

## Development

```bash
uv sync
uv run pytest
```

42 tests. Ollama fixtures are real captured responses; vLLM fixtures are
hand-built against vLLM's documented shape (see Limitations) — both live
under `tests/fixtures/`.

## License

MIT — see [LICENSE](LICENSE).
